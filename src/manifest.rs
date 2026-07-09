//! The freshness manifest — the incremental-update contract.
//!
//! Keyed by **relative path from the repo root**, each entry records the
//! file's content hash (blake3) and the ids of the chunks it produced. On
//! re-index we:
//!
//! - compute each discovered file's hash;
//! - **skip** files whose hash matches the manifest (`skipped_unchanged`);
//! - re-chunk (and re-embed) only new/changed files;
//! - **drop** chunks for files that vanished from the repo (`removed`).
//!
//! The manifest is the single source of truth for "what is currently indexed",
//! so the tantivy index, the chunk store, and the vector store are always
//! reconciled against it.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Per-file manifest entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileEntry {
    /// blake3 content hash (hex).
    pub hash: String,
    /// Ids of the chunks produced from this file.
    pub chunk_ids: Vec<String>,
}

/// The persisted manifest: relpath → entry.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Manifest {
    #[serde(default)]
    pub files: BTreeMap<String, FileEntry>,
}

impl Manifest {
    /// Load from `<data_dir>/manifest.json`, or an empty manifest if absent
    /// or unreadable (a corrupt manifest triggers a full rebuild, which is
    /// safe — the reconcile logic re-derives everything).
    pub fn load(path: &Path) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|raw| serde_json::from_str(&raw).ok())
            .unwrap_or_default()
    }

    /// Persist atomically (write temp + rename) to avoid a torn manifest.
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let body = serde_json::to_string_pretty(self)?;
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, body)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// The recorded hash for `relpath`, if any.
    pub fn hash_of(&self, relpath: &str) -> Option<&str> {
        self.files.get(relpath).map(|e| e.hash.as_str())
    }

    /// True when `relpath` is present with exactly this hash (→ skip).
    pub fn is_unchanged(&self, relpath: &str, hash: &str) -> bool {
        self.hash_of(relpath) == Some(hash)
    }

    /// Record (insert or replace) a file's hash + chunk ids.
    pub fn upsert(&mut self, relpath: &str, hash: String, chunk_ids: Vec<String>) {
        self.files
            .insert(relpath.to_string(), FileEntry { hash, chunk_ids });
    }

    /// Remove a file, returning its chunk ids (so the caller can drop them
    /// from the index / stores).
    pub fn remove(&mut self, relpath: &str) -> Option<Vec<String>> {
        self.files.remove(relpath).map(|e| e.chunk_ids)
    }
}

/// The blake3 content hash of `bytes`, hex-encoded. This is the freshness key:
/// unchanged bytes → unchanged hash → skipped file.
pub fn content_hash(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unchanged_content_yields_unchanged_hash() {
        let a = content_hash(b"# Title\n\nhello");
        let b = content_hash(b"# Title\n\nhello");
        assert_eq!(a, b);
    }

    #[test]
    fn changed_content_yields_different_hash() {
        let a = content_hash(b"hello");
        let b = content_hash(b"hello!");
        assert_ne!(a, b);
    }

    #[test]
    fn is_unchanged_tracks_the_recorded_hash() {
        let mut m = Manifest::default();
        let h = content_hash(b"body");
        m.upsert("a.md", h.clone(), vec!["a.md#0".into()]);
        assert!(m.is_unchanged("a.md", &h));
        assert!(!m.is_unchanged("a.md", "deadbeef"));
        assert!(!m.is_unchanged("missing.md", &h));
    }

    #[test]
    fn remove_returns_chunk_ids() {
        let mut m = Manifest::default();
        m.upsert("a.md", "h".into(), vec!["a.md#0".into(), "a.md#1".into()]);
        let ids = m.remove("a.md").unwrap();
        assert_eq!(ids, vec!["a.md#0", "a.md#1"]);
        assert!(m.remove("a.md").is_none());
    }

    #[test]
    fn round_trips_through_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("manifest.json");
        let mut m = Manifest::default();
        m.upsert("a.md", "h1".into(), vec!["a.md#0".into()]);
        m.save(&path).unwrap();
        let loaded = Manifest::load(&path);
        assert_eq!(loaded.hash_of("a.md"), Some("h1"));
    }
}
