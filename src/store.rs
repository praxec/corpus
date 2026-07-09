//! Persisted chunk + vector stores.
//!
//! - [`ChunkStore`] (`chunks.json`): the source of truth for chunk text and
//!   metadata (path, heading_path). Both text-search and semantic-search
//!   results resolve their `snippet` / `heading_path` from here, so retrieval
//!   metadata lives in exactly one place.
//! - [`VectorStore`] (`vectors.json`): chunk_id → embedding vector, plus the
//!   embedding model/dims it was built with (so a model change is detectable).
//!   Loaded fully into memory; queried by brute-force cosine (no ANN — a
//!   repo's doc corpus is small).

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::chunk::Chunk;

/// chunk_id → chunk. Persisted as `chunks.json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChunkStore {
    #[serde(default)]
    pub chunks: BTreeMap<String, Chunk>,
}

impl ChunkStore {
    pub fn load(path: &Path) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|raw| serde_json::from_str(&raw).ok())
            .unwrap_or_default()
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        save_json(path, self)
    }

    pub fn insert(&mut self, chunk: Chunk) {
        self.chunks.insert(chunk.id.clone(), chunk);
    }

    pub fn remove(&mut self, id: &str) {
        self.chunks.remove(id);
    }

    pub fn get(&self, id: &str) -> Option<&Chunk> {
        self.chunks.get(id)
    }

    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }
}

/// chunk_id → embedding vector. Persisted as `vectors.json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VectorStore {
    /// The model these vectors were built with (`provider/model`), for drift
    /// detection. `None` for an empty store.
    #[serde(default)]
    pub model: Option<String>,
    /// The vector dimensionality.
    #[serde(default)]
    pub dims: usize,
    #[serde(default)]
    pub vectors: BTreeMap<String, Vec<f32>>,
}

impl VectorStore {
    pub fn load(path: &Path) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|raw| serde_json::from_str(&raw).ok())
            .unwrap_or_default()
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        save_json(path, self)
    }

    pub fn insert(&mut self, id: &str, vector: Vec<f32>) {
        self.vectors.insert(id.to_string(), vector);
    }

    pub fn remove(&mut self, id: &str) {
        self.vectors.remove(id);
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Brute-force cosine KNN: the `k` chunk_ids most similar to `query`,
    /// as `(chunk_id, cosine)` sorted by descending similarity.
    pub fn knn(&self, query: &[f32], k: usize) -> Vec<(String, f32)> {
        let mut scored: Vec<(String, f32)> = self
            .vectors
            .iter()
            .map(|(id, v)| (id.clone(), cosine(query, v)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }
}

/// Cosine similarity of two equal-length vectors. Returns `0.0` for a length
/// mismatch or a zero-norm vector (both are "no signal", never a panic).
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn save_json<T: Serialize>(path: &Path, value: &T) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let body = serde_json::to_string(value)?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, body)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_of_identical_is_one() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_of_orthogonal_is_zero() {
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    }

    #[test]
    fn cosine_length_mismatch_is_zero() {
        assert_eq!(cosine(&[1.0, 2.0], &[1.0]), 0.0);
    }

    #[test]
    fn knn_ranks_by_similarity() {
        let mut vs = VectorStore::default();
        vs.insert("far", vec![-1.0, 0.0]);
        vs.insert("near", vec![1.0, 0.1]);
        vs.insert("mid", vec![0.5, 1.0]);
        let top = vs.knn(&[1.0, 0.0], 2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, "near");
    }
}
