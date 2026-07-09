//! MCP tool surface: `corpus_index` + `corpus_search`.
//!
//! Manual `ServerHandler` (matching praxec's rmcp 1.7 pattern): `list_tools`
//! advertises exactly two tools with hand-written JSON Schemas, and `call_tool`
//! dispatches to the [`Corpus`](crate::corpus::Corpus) engine. Each call resolves
//! its own repo/data-dir/config, so one server can index many repos.

use std::borrow::Cow;
use std::sync::Arc;

use rmcp::model::{
    CallToolRequestParams, CallToolResult, Implementation, InitializeResult, JsonObject,
    ListToolsResult, PaginatedRequestParams, ProtocolVersion, ServerCapabilities, ServerInfo, Tool,
};
use rmcp::service::{RequestContext, RoleServer};
use rmcp::ErrorData as McpError;
use rmcp::ServerHandler;
use serde_json::{json, Value};

use crate::search::SearchMode;

pub const TOOL_INDEX: &str = "corpus_index";
pub const TOOL_SEARCH: &str = "corpus_search";

/// The corpus MCP server. Stateless beyond identity — each tool call operates
/// on the `repo_path` it is given.
#[derive(Clone, Default)]
pub struct CorpusServer;

impl CorpusServer {
    pub fn new() -> Self {
        Self
    }

    /// Transport-free dispatch (used by `call_tool` and by tests) — runs a tool
    /// call and returns the structured JSON result.
    pub async fn dispatch(&self, request: CallToolRequestParams) -> Result<Value, McpError> {
        let args: Value = request
            .arguments
            .as_ref()
            .map(|m| Value::Object(m.clone()))
            .unwrap_or_else(|| json!({}));

        match request.name.as_ref() {
            TOOL_INDEX => self.handle_index(args).await,
            TOOL_SEARCH => self.handle_search(args).await,
            other => Err(McpError::invalid_params(
                format!("unknown tool '{other}'. Available: {TOOL_INDEX}, {TOOL_SEARCH}"),
                None,
            )),
        }
    }

    async fn handle_index(&self, args: Value) -> Result<Value, McpError> {
        let repo_path = args
            .get("repo_path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| McpError::invalid_params("repo_path is required".to_string(), None))?;
        let repo = std::path::PathBuf::from(repo_path);
        if !repo.is_dir() {
            return Err(McpError::invalid_params(
                format!("repo_path '{repo_path}' is not a directory"),
                None,
            ));
        }
        let include: Option<Vec<String>> = args.get("include").and_then(|v| v.as_array()).map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(String::from))
                .collect()
        });
        let embeddings = args.get("embeddings").and_then(|v| v.as_bool());

        let corpus = crate::corpus::build(&repo, embeddings)
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        let report = corpus
            .index(include)
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        serde_json::to_value(&report).map_err(|e| McpError::internal_error(e.to_string(), None))
    }

    async fn handle_search(&self, args: Value) -> Result<Value, McpError> {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| McpError::invalid_params("query is required".to_string(), None))?;
        let repo_path = args
            .get("repo_path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| McpError::invalid_params("repo_path is required".to_string(), None))?;
        let repo = std::path::PathBuf::from(repo_path);
        let k = args
            .get("k")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .unwrap_or(8)
            .max(1);
        let mode = SearchMode::parse(args.get("mode").and_then(|v| v.as_str()))
            .map_err(|e| McpError::invalid_params(e.to_string(), None))?;

        // Semantic/hybrid need the embedder; text mode does not.
        let embeddings_override = match mode {
            SearchMode::Text => Some(false),
            _ => None,
        };
        let corpus = crate::corpus::build(&repo, embeddings_override)
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        let results = corpus
            .search(query, k, mode)
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        serde_json::to_value(results).map_err(|e| McpError::internal_error(e.to_string(), None))
    }
}

/// The two tool definitions with hand-written JSON Schemas.
pub fn tool_definitions() -> Vec<Tool> {
    let index_schema: JsonObject = serde_json::from_value(json!({
        "type": "object",
        "properties": {
            "repo_path": { "type": "string", "description": "Absolute path to the repo to index." },
            "include": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Optional glob overrides (default: **/*.md, **/*.mdx, **/*.txt, **/*.adoc)."
            },
            "embeddings": {
                "type": "boolean",
                "description": "Enable semantic embedding for this run (default: off / config-driven)."
            }
        },
        "required": ["repo_path"],
        "additionalProperties": false
    }))
    .expect("static index schema is valid");

    let search_schema: JsonObject = serde_json::from_value(json!({
        "type": "object",
        "properties": {
            "query": { "type": "string", "description": "The search query." },
            "repo_path": { "type": "string", "description": "Absolute path to the indexed repo." },
            "k": { "type": "integer", "description": "Max results (default 8)." },
            "mode": {
                "type": "string",
                "enum": ["hybrid", "text", "semantic"],
                "description": "Retrieval mode (default hybrid)."
            }
        },
        "required": ["query", "repo_path"],
        "additionalProperties": false
    }))
    .expect("static search schema is valid");

    vec![
        Tool::new(
            Cow::Borrowed(TOOL_INDEX),
            Cow::Borrowed(
                "Index a repo's docs for retrieval. Incremental by content hash: \
                 unchanged files are skipped, changed/new files re-chunked, deleted \
                 files dropped. Returns { indexed, skipped_unchanged, removed, chunks, embedded }.",
            ),
            Arc::new(index_schema),
        ),
        Tool::new(
            Cow::Borrowed(TOOL_SEARCH),
            Cow::Borrowed(
                "Search indexed docs. Returns ranked [{ path, heading_path, snippet, score }]. \
                 mode = hybrid (BM25 + semantic RRF) | text | semantic.",
            ),
            Arc::new(search_schema),
        ),
    ]
}

impl ServerHandler for CorpusServer {
    fn get_info(&self) -> ServerInfo {
        let mut server_info = Implementation::new("corpus", env!("CARGO_PKG_VERSION"));
        server_info.title = Some("corpus".to_string());
        server_info.description = Some("Minimal docs-RAG MCP server".to_string());

        let mut info = InitializeResult::default();
        info.protocol_version = ProtocolVersion::default();
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info.server_info = server_info;
        info.instructions = Some(
            "corpus indexes a repo's documentation and serves hybrid retrieval.\n\
             1. corpus_index { repo_path } → build/update the index.\n\
             2. corpus_search { query, repo_path, k?, mode? } → ranked chunks.\n\
             Enable semantic search by indexing with embeddings: true (opt-in)."
                .to_string(),
        );
        info
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        Ok(ListToolsResult::with_all_items(tool_definitions()))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        self.dispatch(request).await.map(CallToolResult::structured)
    }

    fn get_tool(&self, name: &str) -> Option<Tool> {
        tool_definitions().into_iter().find(|t| t.name == name)
    }
}
