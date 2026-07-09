//! The corpus engine: incremental `index` + hybrid `search`, tying together
//! discovery, chunking, the freshness manifest, the tantivy index, and the
//! optional embedder/vector store.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Serialize;

use crate::chunk::chunk_file;
use crate::config::CorpusConfig;
use crate::discovery::discover;
use crate::embed::Embedder;
use crate::index::TextIndex;
use crate::manifest::{content_hash, Manifest};
use crate::search::{reciprocal_rank_fusion, snippet, SearchMode, SearchResult};
use crate::store::{ChunkStore, VectorStore};

/// Outcome of an `index` run — the incremental-update report.
#[derive(Debug, Clone, Serialize, Default)]
pub struct IndexReport {
    /// Files (re)indexed because they were new or changed.
    pub indexed: usize,
    /// Files skipped because their content hash was unchanged.
    pub skipped_unchanged: usize,
    /// Files dropped because they vanished from the repo.
    pub removed: usize,
    /// Total chunks in the index after this run.
    pub chunks: usize,
    /// Chunks embedded during this run (0 when embeddings are off).
    pub embedded: usize,
}

/// One repo's corpus. Owns the resolved paths + config; the embedder is
/// injected (live rig or a stub) so indexing/search is backend-agnostic.
pub struct Corpus {
    repo: PathBuf,
    data_dir: PathBuf,
    config: CorpusConfig,
    embedder: Option<Arc<dyn Embedder>>,
}

impl Corpus {
    /// Construct a corpus for `repo` with `data_dir`, `config`, and an optional
    /// embedder (present ⇔ semantic search is enabled for this run).
    pub fn new(
        repo: PathBuf,
        data_dir: PathBuf,
        config: CorpusConfig,
        embedder: Option<Arc<dyn Embedder>>,
    ) -> Self {
        Self {
            repo,
            data_dir,
            config,
            embedder,
        }
    }

    fn manifest_path(&self) -> PathBuf {
        self.data_dir.join("manifest.json")
    }
    fn chunks_path(&self) -> PathBuf {
        self.data_dir.join("chunks.json")
    }
    fn vectors_path(&self) -> PathBuf {
        self.data_dir.join("vectors.json")
    }

    /// Incrementally (re)index the repo. Unchanged files are skipped by content
    /// hash; changed/new files are re-chunked (and re-embedded if enabled);
    /// deleted files' chunks are dropped. All state (manifest, tantivy, chunk
    /// store, vector store) is reconciled against the current file set.
    pub async fn index(&self, include_override: Option<Vec<String>>) -> anyhow::Result<IndexReport> {
        std::fs::create_dir_all(&self.data_dir)?;
        let include = include_override.unwrap_or_else(|| self.config.include.clone());

        let mut manifest = Manifest::load(&self.manifest_path());
        let mut chunks = ChunkStore::load(&self.chunks_path());
        let mut vectors = VectorStore::load(&self.vectors_path());

        let text = TextIndex::open(&self.data_dir)?;
        let mut writer = text.writer()?;
        let path_field = text.path_field();

        let discovered = discover(&self.repo, &include)?;
        let seen: std::collections::HashSet<String> =
            discovered.iter().map(|d| d.relpath.clone()).collect();

        let mut report = IndexReport::default();

        // ── new / changed / unchanged ────────────────────────────────────────
        for doc in &discovered {
            let bytes = match std::fs::read(&doc.abs) {
                Ok(b) => b,
                Err(e) => {
                    tracing::warn!(path = %doc.relpath, error = %e, "skipping unreadable file");
                    continue;
                }
            };
            let hash = content_hash(&bytes);

            if manifest.is_unchanged(&doc.relpath, &hash) {
                report.skipped_unchanged += 1;
                continue;
            }

            // Changed or new: drop any prior chunks for this path, then re-add.
            if let Some(old_ids) = manifest.remove(&doc.relpath) {
                for id in old_ids {
                    chunks.remove(&id);
                    vectors.remove(&id);
                }
            }
            TextIndex::delete_path(&writer, path_field, &doc.relpath);

            let source = String::from_utf8_lossy(&bytes);
            let file_chunks = chunk_file(&doc.relpath, &source);
            let mut chunk_ids = Vec::with_capacity(file_chunks.len());

            for chunk in &file_chunks {
                text.add_chunk(&writer, chunk)?;
                if let Some(embedder) = &self.embedder {
                    let embed_text = embed_input(chunk);
                    match embedder.embed(&embed_text).await {
                        Ok(vec) => {
                            vectors.insert(&chunk.id, vec);
                            report.embedded += 1;
                        }
                        Err(e) => {
                            // Fail-fast: a half-embedded corpus silently degrades
                            // semantic recall. Surface it rather than limp on.
                            anyhow::bail!("embedding failed for {}: {e}", chunk.id);
                        }
                    }
                }
                chunk_ids.push(chunk.id.clone());
                chunks.insert(chunk.clone());
            }

            manifest.upsert(&doc.relpath, hash, chunk_ids);
            report.indexed += 1;
        }

        // ── deletions (in manifest but no longer on disk) ────────────────────
        let stale: Vec<String> = manifest
            .files
            .keys()
            .filter(|rel| !seen.contains(*rel))
            .cloned()
            .collect();
        for rel in stale {
            if let Some(old_ids) = manifest.remove(&rel) {
                for id in old_ids {
                    chunks.remove(&id);
                    vectors.remove(&id);
                }
            }
            TextIndex::delete_path(&writer, path_field, &rel);
            report.removed += 1;
        }

        writer.commit()?;

        // Record the embedding model so a later run can detect a model change.
        if let Some(embedder) = &self.embedder {
            vectors.model = Some(embedder.model_id());
            vectors.dims = embedder.dimensions();
        }

        report.chunks = chunks.len();
        manifest.save(&self.manifest_path())?;
        chunks.save(&self.chunks_path())?;
        vectors.save(&self.vectors_path())?;

        Ok(report)
    }

    /// Hybrid / text / semantic search. Returns up to `k` ranked results.
    pub async fn search(
        &self,
        query: &str,
        k: usize,
        mode: SearchMode,
    ) -> anyhow::Result<Vec<SearchResult>> {
        let chunks = ChunkStore::load(&self.chunks_path());
        if chunks.is_empty() {
            return Ok(Vec::new());
        }
        let text = TextIndex::open(&self.data_dir)?;

        // Over-fetch each lane so fusion has depth to work with.
        let fetch = (k * 4).max(k);

        let text_ids: Vec<String> = match mode {
            SearchMode::Semantic => Vec::new(),
            _ => text
                .search(query, fetch)?
                .into_iter()
                .map(|h| h.chunk_id)
                .collect(),
        };

        let semantic_ids: Vec<String> = match mode {
            SearchMode::Text => Vec::new(),
            _ => self.semantic_ids(query, fetch).await?,
        };

        // Semantic explicitly requested but unavailable → surface it.
        if mode == SearchMode::Semantic && semantic_ids.is_empty() && self.embedder.is_none() {
            anyhow::bail!(
                "semantic search requires embeddings to be enabled and an index built \
                 with embeddings (run corpus_index with embeddings: true)"
            );
        }

        let ranked_ids: Vec<String> = match mode {
            SearchMode::Text => text_ids,
            SearchMode::Semantic => semantic_ids,
            SearchMode::Hybrid => {
                if semantic_ids.is_empty() {
                    text_ids // text-only fallback when no vectors
                } else {
                    reciprocal_rank_fusion(&[text_ids, semantic_ids])
                        .into_iter()
                        .map(|(id, _)| id)
                        .collect()
                }
            }
        };

        // For text/semantic single-lane, keep the lane's own score; for hybrid,
        // recompute a positional score so the surfaced number is meaningful.
        let mut results = Vec::new();
        for (rank, id) in ranked_ids.into_iter().take(k).enumerate() {
            if let Some(chunk) = chunks.get(&id) {
                results.push(SearchResult {
                    path: chunk.path.clone(),
                    heading_path: chunk.heading_path.clone(),
                    snippet: snippet(&chunk.text, 240),
                    // Rank-descending score in (0,1]; stable across modes.
                    score: 1.0 / (rank as f32 + 1.0),
                });
            }
        }
        Ok(results)
    }

    /// The semantic lane: embed the query, cosine-KNN over the vector store.
    /// Empty when embeddings are off or no vectors are persisted.
    async fn semantic_ids(&self, query: &str, k: usize) -> anyhow::Result<Vec<String>> {
        let Some(embedder) = &self.embedder else {
            return Ok(Vec::new());
        };
        let vectors = VectorStore::load(&self.vectors_path());
        if vectors.is_empty() {
            return Ok(Vec::new());
        }
        let q = embedder.embed(query).await?;
        Ok(vectors.knn(&q, k).into_iter().map(|(id, _)| id).collect())
    }
}

/// The text handed to the embedder for a chunk: heading path + body, so the
/// vector reflects where the chunk sits, not just its prose.
fn embed_input(chunk: &crate::chunk::Chunk) -> String {
    if chunk.heading_path.is_empty() {
        chunk.text.clone()
    } else {
        format!("{}\n\n{}", chunk.heading_path.join(" > "), chunk.text)
    }
}

/// Convenience: build a [`Corpus`] for `repo`, resolving the data dir + config
/// and constructing the embedder iff `embeddings` is enabled (config or the
/// per-call override). Returns the corpus plus whether embeddings are active.
pub fn build(
    repo: &Path,
    embeddings_override: Option<bool>,
) -> anyhow::Result<Corpus> {
    let data_dir = crate::config::resolve_data_dir(repo);
    let mut config = CorpusConfig::load(&data_dir);
    if let Some(on) = embeddings_override {
        config.embeddings = on;
    }

    let embedder: Option<Arc<dyn Embedder>> = if config.embeddings {
        let provider = config
            .embed_provider
            .clone()
            .ok_or_else(|| anyhow::anyhow!("embeddings enabled but embed_provider is not set"))?;
        let model = config
            .embed_model
            .clone()
            .ok_or_else(|| anyhow::anyhow!("embeddings enabled but embed_model is not set"))?;
        let dims = config
            .embed_dims
            .ok_or_else(|| anyhow::anyhow!("embeddings enabled but embed_dims is not set"))?;
        let e = crate::embed::RigEmbedder::from_config(&provider, &model, dims)?;
        Some(Arc::new(e))
    } else {
        None
    };

    Ok(Corpus::new(repo.to_path_buf(), data_dir, config, embedder))
}
