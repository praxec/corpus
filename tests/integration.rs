//! End-to-end tests for the corpus engine: text search, the incremental
//! freshness contract, and the semantic path (via a network-free stub embedder,
//! so the plumbing is exercised without a live provider or spend).

use std::path::Path;
use std::sync::Arc;

use corpus::config::CorpusConfig;
use corpus::corpus::Corpus;
use corpus::embed::{Embedder, StubEmbedder};
use corpus::search::SearchMode;

fn write(repo: &Path, rel: &str, body: &str) {
    let path = repo.join(rel);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(path, body).unwrap();
}

fn text_corpus(repo: &Path, data: &Path) -> Corpus {
    Corpus::new(
        repo.to_path_buf(),
        data.to_path_buf(),
        CorpusConfig::default(),
        None,
    )
}

#[tokio::test]
async fn text_index_and_search_end_to_end() {
    let repo = tempfile::tempdir().unwrap();
    let data = tempfile::tempdir().unwrap();
    write(
        repo.path(),
        "docs/config.md",
        "# Configuration\n\n## Executor kinds\n\nThe executor kinds are llm and tool.\n",
    );
    write(
        repo.path(),
        "README.md",
        "# Project\n\nA gateway for governed workflows.\n",
    );

    let corpus = text_corpus(repo.path(), data.path());
    let report = corpus.index(None).await.unwrap();
    assert_eq!(report.indexed, 2);
    assert_eq!(report.skipped_unchanged, 0);
    assert_eq!(report.removed, 0);
    assert!(report.chunks >= 2);
    assert_eq!(report.embedded, 0);

    let hits = corpus
        .search("executor kinds", 5, SearchMode::Text)
        .await
        .unwrap();
    assert!(!hits.is_empty());
    assert_eq!(hits[0].path, "docs/config.md");
    assert_eq!(
        hits[0].heading_path,
        vec!["Configuration", "Executor kinds"]
    );
    assert!(hits[0].snippet.contains("executor kinds"));
}

#[tokio::test]
async fn freshness_skips_unchanged_reindexes_changed_and_drops_deleted() {
    let repo = tempfile::tempdir().unwrap();
    let data = tempfile::tempdir().unwrap();
    write(repo.path(), "a.md", "# A\n\nalpha content about widgets\n");
    write(repo.path(), "b.md", "# B\n\nbeta content about gadgets\n");

    let corpus = text_corpus(repo.path(), data.path());

    // First index: both new.
    let r1 = corpus.index(None).await.unwrap();
    assert_eq!(r1.indexed, 2);
    assert_eq!(r1.skipped_unchanged, 0);

    // Re-index with no changes: both skipped by content hash.
    let r2 = corpus.index(None).await.unwrap();
    assert_eq!(r2.indexed, 0);
    assert_eq!(r2.skipped_unchanged, 2);
    assert_eq!(r2.removed, 0);

    // Change only a.md: exactly one reindexed, one skipped.
    write(
        repo.path(),
        "a.md",
        "# A\n\nalpha content about sprockets now\n",
    );
    let r3 = corpus.index(None).await.unwrap();
    assert_eq!(r3.indexed, 1);
    assert_eq!(r3.skipped_unchanged, 1);
    assert_eq!(r3.removed, 0);

    // The new content is searchable; the old content is gone.
    let sprockets = corpus
        .search("sprockets", 5, SearchMode::Text)
        .await
        .unwrap();
    assert!(!sprockets.is_empty());
    let widgets = corpus.search("widgets", 5, SearchMode::Text).await.unwrap();
    assert!(widgets.is_empty(), "stale chunk should have been dropped");

    // Delete b.md: reported removed, chunks dropped.
    std::fs::remove_file(repo.path().join("b.md")).unwrap();
    let r4 = corpus.index(None).await.unwrap();
    assert_eq!(r4.removed, 1);
    assert_eq!(r4.indexed, 0);
    let gadgets = corpus.search("gadgets", 5, SearchMode::Text).await.unwrap();
    assert!(gadgets.is_empty(), "deleted file's chunks should be gone");
}

#[tokio::test]
async fn semantic_and_hybrid_search_via_stub_embedder() {
    let repo = tempfile::tempdir().unwrap();
    let data = tempfile::tempdir().unwrap();
    write(
        repo.path(),
        "guide.md",
        "# Guide\n\n## Configuring executors\n\nSet executor kinds in the config file.\n",
    );
    write(
        repo.path(),
        "recipes.md",
        "# Recipes\n\n## Baking bread\n\nMix flour water yeast salt and bake.\n",
    );

    let embedder: Arc<dyn Embedder> = Arc::new(StubEmbedder::new(256));
    let corpus = Corpus::new(
        repo.path().to_path_buf(),
        data.path().to_path_buf(),
        CorpusConfig::default(),
        Some(embedder),
    );

    let report = corpus.index(None).await.unwrap();
    assert!(report.embedded > 0, "chunks should be embedded");
    assert_eq!(report.embedded, report.chunks);

    // Semantic: a query sharing tokens with the executor section should rank it
    // first via cosine over the stub's bag-of-words vectors.
    let sem = corpus
        .search("how to configure executor kinds", 5, SearchMode::Semantic)
        .await
        .unwrap();
    assert!(!sem.is_empty());
    assert_eq!(sem[0].path, "guide.md");

    // Hybrid fuses text + semantic and still surfaces the right doc.
    let hybrid = corpus
        .search("configure executor kinds", 5, SearchMode::Hybrid)
        .await
        .unwrap();
    assert!(!hybrid.is_empty());
    assert_eq!(hybrid[0].path, "guide.md");
}

#[tokio::test]
async fn hybrid_falls_back_to_text_when_no_vectors() {
    let repo = tempfile::tempdir().unwrap();
    let data = tempfile::tempdir().unwrap();
    write(
        repo.path(),
        "a.md",
        "# A\n\ncontent about telemetry pipelines\n",
    );

    // Index without an embedder (no vectors persisted).
    let corpus = text_corpus(repo.path(), data.path());
    corpus.index(None).await.unwrap();

    // Hybrid with no embedder must not error — it falls back to text.
    let hits = corpus
        .search("telemetry", 5, SearchMode::Hybrid)
        .await
        .unwrap();
    assert!(!hits.is_empty());
}
