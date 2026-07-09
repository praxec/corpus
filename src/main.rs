//! corpus MCP server binary.
//!
//! Serves the two-tool MCP surface over stdio (matching praxec's serve path).
//! Provider keys are seeded from `~/.praxec/providers.env` at startup so the
//! optional rig embedder can authenticate without extra wiring.

use anyhow::Context;
use corpus::server::CorpusServer;
use rmcp::ServiceExt;
use rmcp::transport::stdio;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Log to stderr — stdout is the MCP transport.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    // Seed provider API keys from the praxec convention (env still wins).
    corpus::config::seed_provider_keys();

    tracing::info!("starting corpus MCP server (stdio)");

    let service = CorpusServer::new()
        .serve(stdio())
        .await
        .context("starting corpus MCP service over stdio")?;

    service.waiting().await?;
    Ok(())
}
