//! Embeddings — the optional semantic path.
//!
//! Matches **praxec-embeddings** exactly: the rig crate (`rig-core` 0.38)
//! treats embeddings as first-class, independent of any chat SDK. rig's
//! `EmbeddingModel` is not object-safe (associated `Client` + const), so we
//! enum-dispatch across the same four providers praxec supports
//! (`openai` | `gemini` | `ollama` | `openrouter`), each built with
//! `Client::from_env().embedding_model_with_ndims(model, dims)`. Provider keys
//! come from the process env (seeded from `~/.praxec/providers.env`, the praxec
//! convention).
//!
//! [`Embedder`] is the object-safe seam the rest of corpus depends on, so the
//! indexer/search code is agnostic to whether the backend is live rig or the
//! deterministic [`StubEmbedder`] used in tests (no network / no spend).

use async_trait::async_trait;

/// Object-safe embedding seam. Implemented by the live [`RigEmbedder`] and the
/// test [`StubEmbedder`].
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Embed `text` into a dense vector of length [`dimensions`](Self::dimensions).
    async fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>>;
    /// The embedding dimensionality.
    fn dimensions(&self) -> usize;
    /// `provider/model` identifier, for vector-store drift detection.
    fn model_id(&self) -> String;
}

// ── rig-backed embedder (matches praxec-embeddings) ──────────────────────────

use rig::client::{EmbeddingsClient, ProviderClient};
use rig::embeddings::EmbeddingModel as _;
use rig::providers::{gemini, ollama, openai, openrouter};

/// The provider-specific rig embedding model. rig's `EmbeddingModel` is not
/// object-safe, so we enum-dispatch (same pattern as praxec-embeddings).
enum Inner {
    OpenAi(openai::EmbeddingModel),
    Gemini(gemini::embedding::EmbeddingModel),
    Ollama(ollama::EmbeddingModel),
    OpenRouter(openrouter::embedding::EmbeddingModel),
}

/// A rig-backed [`Embedder`].
pub struct RigEmbedder {
    inner: Inner,
    dims: usize,
    provider: String,
    model: String,
}

impl RigEmbedder {
    /// Build an embedder for `(provider, model, dims)`. The provider client is
    /// constructed from the process env (keys seeded from `providers.env`).
    /// `provider` is one of `openai` | `gemini` | `ollama` | `openrouter`.
    pub fn from_config(provider: &str, model: &str, dims: usize) -> anyhow::Result<Self> {
        let inner = match provider {
            "openai" => Inner::OpenAi(
                openai::Client::from_env()
                    .map_err(|e| anyhow::anyhow!("{e}"))?
                    .embedding_model_with_ndims(model, dims),
            ),
            "gemini" => Inner::Gemini(
                gemini::Client::from_env()
                    .map_err(|e| anyhow::anyhow!("{e}"))?
                    .embedding_model_with_ndims(model, dims),
            ),
            "ollama" => Inner::Ollama(
                ollama::Client::from_env()
                    .map_err(|e| anyhow::anyhow!("{e}"))?
                    .embedding_model_with_ndims(model, dims),
            ),
            "openrouter" => Inner::OpenRouter(
                openrouter::Client::from_env()
                    .map_err(|e| anyhow::anyhow!("{e}"))?
                    .embedding_model_with_ndims(model, dims),
            ),
            other => anyhow::bail!(
                "unsupported embedding provider '{other}' \
                 (expected openai | gemini | ollama | openrouter)"
            ),
        };
        Ok(Self {
            inner,
            dims,
            provider: provider.to_string(),
            model: model.to_string(),
        })
    }
}

#[async_trait]
impl Embedder for RigEmbedder {
    async fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let emb = match &self.inner {
            Inner::OpenAi(m) => m.embed_text(text).await,
            Inner::Gemini(m) => m.embed_text(text).await,
            Inner::Ollama(m) => m.embed_text(text).await,
            Inner::OpenRouter(m) => m.embed_text(text).await,
        }
        .map_err(|e| anyhow::anyhow!("embedding backend failed: {e}"))?;
        // praxec's index is f32; rig returns f64.
        let vec: Vec<f32> = emb.vec.into_iter().map(|v| v as f32).collect();
        if vec.len() != self.dims {
            anyhow::bail!(
                "dimension mismatch from {}/{}: got {}, expected {}",
                self.provider,
                self.model,
                vec.len(),
                self.dims
            );
        }
        Ok(vec)
    }

    fn dimensions(&self) -> usize {
        self.dims
    }

    fn model_id(&self) -> String {
        format!("{}/{}", self.provider, self.model)
    }
}

// ── deterministic stub embedder (tests / offline plumbing proof) ─────────────

/// A deterministic, network-free embedder for tests and offline verification
/// of the semantic plumbing. It produces a bag-of-words hashed vector so that
/// texts sharing tokens land near each other under cosine — enough to prove
/// index → embed → cosine-KNN end-to-end without a live provider.
pub struct StubEmbedder {
    dims: usize,
}

impl StubEmbedder {
    pub fn new(dims: usize) -> Self {
        Self { dims }
    }
}

#[async_trait]
impl Embedder for StubEmbedder {
    async fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let mut v = vec![0.0f32; self.dims];
        for token in text.split(|c: char| !c.is_alphanumeric()) {
            if token.is_empty() {
                continue;
            }
            let lower = token.to_ascii_lowercase();
            let mut h: u64 = 1469598103934665603; // FNV-1a offset basis
            for b in lower.bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(1099511628211);
            }
            let idx = (h % self.dims as u64) as usize;
            v[idx] += 1.0;
        }
        Ok(v)
    }

    fn dimensions(&self) -> usize {
        self.dims
    }

    fn model_id(&self) -> String {
        format!("stub/hash-{}", self.dims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::cosine;

    #[tokio::test]
    async fn stub_is_deterministic() {
        let e = StubEmbedder::new(64);
        let a = e.embed("executor kinds llm tool").await.unwrap();
        let b = e.embed("executor kinds llm tool").await.unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 64);
    }

    #[tokio::test]
    async fn stub_shared_tokens_are_more_similar() {
        let e = StubEmbedder::new(128);
        let q = e.embed("how do I configure the executor").await.unwrap();
        let near = e
            .embed("configure the executor kinds in the config")
            .await
            .unwrap();
        let far = e.embed("unrelated banana zebra sunshine").await.unwrap();
        assert!(cosine(&q, &near) > cosine(&q, &far));
    }

    #[test]
    fn rig_rejects_unknown_provider() {
        let r = RigEmbedder::from_config("frobnicate", "x", 768);
        assert!(r.is_err());
        // RigEmbedder isn't Debug (rig models aren't), so inspect via `.err()`.
        let msg = r.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(msg.contains("unsupported"), "got: {msg}");
    }
}
