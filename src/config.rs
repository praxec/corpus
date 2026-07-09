//! Corpus configuration.
//!
//! Precedence (low → high): built-in defaults, an optional on-disk
//! `<data_dir>/config.json`, process env vars, then per-call MCP tool
//! arguments. This mirrors praxec: a small config surface, embeddings
//! **off by default** (opt-in to avoid spend), provider keys via env
//! (optionally seeded from `~/.praxec/providers.env`).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The default doc globs. Configurable; respected relative to the repo root.
pub const DEFAULT_INCLUDE: &[&str] = &["**/*.md", "**/*.mdx", "**/*.txt", "**/*.adoc"];

/// Resolved corpus configuration for one repo.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusConfig {
    /// Include globs (relative to repo root). Defaults to [`DEFAULT_INCLUDE`].
    #[serde(default = "default_include")]
    pub include: Vec<String>,

    /// Whether semantic embeddings are enabled. Default `false` (text-only).
    #[serde(default)]
    pub embeddings: bool,

    /// Embedding provider slug (`openai` | `gemini` | `ollama` | `openrouter`),
    /// matching praxec-embeddings. Only consulted when `embeddings` is true.
    #[serde(default)]
    pub embed_provider: Option<String>,

    /// Embedding model id (provider-specific). Only consulted when
    /// `embeddings` is true.
    #[serde(default)]
    pub embed_model: Option<String>,

    /// Embedding dimensionality. Only consulted when `embeddings` is true.
    #[serde(default)]
    pub embed_dims: Option<usize>,
}

fn default_include() -> Vec<String> {
    DEFAULT_INCLUDE.iter().map(|s| s.to_string()).collect()
}

impl Default for CorpusConfig {
    fn default() -> Self {
        Self {
            include: default_include(),
            embeddings: false,
            embed_provider: None,
            embed_model: None,
            embed_dims: None,
        }
    }
}

impl CorpusConfig {
    /// Load config for a repo: start from `<data_dir>/config.json` if present,
    /// else defaults, then overlay env vars. Per-call tool-argument overrides
    /// are applied by the caller (they are request-scoped, not persisted).
    pub fn load(data_dir: &Path) -> Self {
        let mut cfg = Self::from_file(&data_dir.join("config.json")).unwrap_or_default();
        cfg.apply_env();
        cfg
    }

    fn from_file(path: &Path) -> Option<Self> {
        let raw = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&raw).ok()
    }

    /// Overlay environment variables:
    /// - `CORPUS_EMBEDDINGS` = `1`/`true`/`on` → enable embeddings.
    /// - `CORPUS_EMBED_PROVIDER`, `CORPUS_EMBED_MODEL`, `CORPUS_EMBED_DIMS`.
    /// - `CORPUS_INCLUDE` = comma-separated globs.
    fn apply_env(&mut self) {
        if let Ok(v) = std::env::var("CORPUS_EMBEDDINGS") {
            self.embeddings = matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "on");
        }
        if let Ok(v) = std::env::var("CORPUS_EMBED_PROVIDER") {
            if !v.trim().is_empty() {
                self.embed_provider = Some(v.trim().to_string());
            }
        }
        if let Ok(v) = std::env::var("CORPUS_EMBED_MODEL") {
            if !v.trim().is_empty() {
                self.embed_model = Some(v.trim().to_string());
            }
        }
        if let Ok(v) = std::env::var("CORPUS_EMBED_DIMS") {
            if let Ok(d) = v.trim().parse::<usize>() {
                self.embed_dims = Some(d);
            }
        }
        if let Ok(v) = std::env::var("CORPUS_INCLUDE") {
            let globs: Vec<String> = v
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            if !globs.is_empty() {
                self.include = globs;
            }
        }
    }
}

/// Resolve the data dir for a repo. Precedence:
/// 1. `$CORPUS_DATA_DIR` if set + non-empty (used verbatim).
/// 2. `<repo>/.corpus`.
pub fn resolve_data_dir(repo: &Path) -> PathBuf {
    if let Ok(p) = std::env::var("CORPUS_DATA_DIR") {
        if !p.trim().is_empty() {
            return PathBuf::from(p);
        }
    }
    repo.join(".corpus")
}

/// Seed provider API keys from praxec's `~/.praxec/providers.env` convention
/// into the process env (existing env vars win, matching praxec). Best-effort:
/// a missing file is not an error. Precedence for the path:
/// `$PRAXEC_PROVIDER_KEYS_FILE`, then `~/.praxec/providers.env`.
pub fn seed_provider_keys() {
    let path = match std::env::var("PRAXEC_PROVIDER_KEYS_FILE") {
        Ok(p) if !p.trim().is_empty() => PathBuf::from(p),
        _ => match dirs::home_dir() {
            Some(d) => d.join(".praxec").join("providers.env"),
            None => return,
        },
    };
    let raw = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(_) => return,
    };
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((k, v)) = line.split_once('=') {
            let (k, v) = (k.trim(), v.trim());
            if std::env::var(k).is_err() {
                // SAFETY: called once at startup, before any worker threads
                // read the environment.
                unsafe { std::env::set_var(k, v) };
            }
        }
    }
}
