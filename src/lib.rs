//! corpus — a minimal docs-RAG MCP server.
//!
//! Indexes a repo's documentation (markdown-aware chunking) and serves hybrid
//! retrieval over an always-on BM25 index (tantivy) plus an optional,
//! opt-in semantic lane (rig embeddings + brute-force cosine KNN). Incremental
//! by content hash (blake3), keyed by repo-relative path.
//!
//! Two MCP tools: `corpus_index` and `corpus_search`.

#![cfg_attr(not(test), warn(clippy::unwrap_used))]

pub mod chunk;
pub mod config;
pub mod corpus;
pub mod discovery;
pub mod embed;
pub mod index;
pub mod manifest;
pub mod search;
pub mod server;
pub mod store;

pub use corpus::{build, Corpus, IndexReport};
pub use search::{SearchMode, SearchResult};
pub use server::CorpusServer;
