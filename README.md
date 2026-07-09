# corpus

A minimal **docs-RAG MCP server** in Rust. It indexes a repository's
documentation and serves hybrid retrieval — always-on lexical search (BM25 via
[tantivy]) plus an optional, opt-in semantic lane ([rig] embeddings + brute-force
cosine KNN). Incremental by content hash (blake3), keyed by repo-relative path.

It is a standalone sibling to the praxec tools (`fmeca`, `cpm-planner`,
`log-analyzer`): one small binary, no vector DB, no ANN index, no background
service. State lives in a per-repo `.corpus/` data dir.

## Tools

### `corpus_index`

```jsonc
{
  "repo_path": "/abs/path/to/repo",     // required
  "include":   ["**/*.md", "**/*.txt"], // optional glob override
  "embeddings": false                   // optional: enable semantic embedding this run
}
```

→ `{ indexed, skipped_unchanged, removed, chunks, embedded }`

Incremental: unchanged files (by blake3 hash) are **skipped**, changed/new files
are re-chunked (and re-embedded if enabled), and chunks for deleted files are
**dropped**.

### `corpus_search`

```jsonc
{
  "query":     "how do executor kinds work", // required
  "repo_path": "/abs/path/to/repo",           // required
  "k":         8,                              // optional (default 8)
  "mode":      "hybrid"                        // hybrid | text | semantic (default hybrid)
}
```

→ `[{ path, heading_path, snippet, score }]`, ranked.

- `text` — BM25 only (always available).
- `semantic` — cosine KNN over embeddings (requires an index built with
  `embeddings: true`).
- `hybrid` — Reciprocal Rank Fusion of BM25 + semantic; falls back to text when
  no vectors are present.

## Architecture

| Concern            | Approach                                                                 |
|--------------------|--------------------------------------------------------------------------|
| Discovery          | `ignore` crate walk (respects `.gitignore`) + configurable include globs |
| Chunking           | `pulldown-cmark`; split on headings, carry a `heading_path`, size-cap with overlap |
| Freshness          | blake3 content hash per file, keyed by **relpath**, in `manifest.json`   |
| Text search        | tantivy BM25 over `heading` + `body`, persisted under `.corpus/tantivy/` |
| Semantic (opt-in)  | rig embeddings → `vectors.json`, in-memory brute-force cosine KNN         |
| Hybrid ranking     | Reciprocal Rank Fusion (RRF, k=60)                                        |

Default include globs: `**/*.md`, `**/*.mdx`, `**/*.txt`, `**/*.adoc`.

### Data dir layout (`<repo>/.corpus/`)

```
manifest.json   relpath -> { hash, chunk_ids }   (the freshness contract)
chunks.json     chunk_id -> { path, heading_path, text }
vectors.json    chunk_id -> [f32]   (+ model/dims; only when embeddings are on)
tantivy/        the BM25 index
config.json     optional persisted config (see below)
```

Override the data dir with `CORPUS_DATA_DIR`.

## Configuration

Precedence (low → high): built-in defaults → `<data_dir>/config.json` → env
vars → per-call tool arguments.

```jsonc
// <repo>/.corpus/config.json
{
  "include": ["**/*.md", "**/*.txt"],
  "embeddings": false,
  "embed_provider": "openai",   // openai | gemini | ollama | openrouter
  "embed_model": "text-embedding-3-small",
  "embed_dims": 1536
}
```

Env overrides: `CORPUS_EMBEDDINGS`, `CORPUS_EMBED_PROVIDER`,
`CORPUS_EMBED_MODEL`, `CORPUS_EMBED_DIMS`, `CORPUS_INCLUDE` (comma-separated),
`CORPUS_DATA_DIR`.

### Embeddings (matches praxec)

The semantic lane uses [rig] (`rig-core` 0.38) exactly like `praxec-embeddings`:
providers `openai | gemini | ollama | openrouter`, each built with
`Client::from_env().embedding_model_with_ndims(model, dims)`. Provider API keys
come from the process env and are seeded at startup from praxec's
`~/.praxec/providers.env` convention (override with `PRAXEC_PROVIDER_KEYS_FILE`;
existing env vars win). Example models: `openai/text-embedding-3-small` (1536),
`gemini/text-embedding-004` (768), `ollama/nomic-embed-text` (768, local/keyless).

Embeddings are **off by default** to avoid spend. Enable per-run
(`corpus_index { embeddings: true }`) or via config/env. The model id + dims are
stored in `vectors.json` so a model change is detectable.

## Run / wire as an MCP server

```bash
cargo build --release
```

The binary speaks MCP over **stdio** (logs go to stderr). Wire it into an
MCP client, e.g.:

```jsonc
{
  "mcpServers": {
    "corpus": {
      "command": "/abs/path/to/corpus/target/release/corpus"
    }
  }
}
```

Then, from the client:

1. `corpus_index { "repo_path": "/abs/path/to/repo" }`
2. `corpus_search { "query": "...", "repo_path": "/abs/path/to/repo" }`

For semantic search, index once with `embeddings: true` (and provider keys in
env or `~/.praxec/providers.env`), then search with `mode: "semantic"` or
`"hybrid"`.

## Development

```bash
cargo build
cargo clippy --all-targets
cargo test
```

Tests cover markdown chunking, the freshness manifest, BM25 index/delete, cosine
+ RRF, and end-to-end index/search — including the semantic + hybrid path via a
deterministic network-free stub embedder (no live provider or spend needed to
exercise the plumbing).

[tantivy]: https://github.com/quickwit-oss/tantivy
[rig]: https://github.com/0xPlaygrounds/rig
