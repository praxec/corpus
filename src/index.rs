//! Tantivy BM25 text index over chunks.
//!
//! Always-on lexical retrieval. The index is persisted under
//! `<data_dir>/tantivy/`. We store only the `chunk_id` (retrieval metadata
//! lives in the chunk store); `path` is indexed untokenized so a changed or
//! deleted file's chunks can be dropped with a single `delete_term`.

use std::path::Path;

use tantivy::collector::TopDocs;
use tantivy::directory::MmapDirectory;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, Schema, Value, STORED, STRING, TEXT};
use tantivy::{doc, Index, IndexWriter, TantivyDocument, Term};

use crate::chunk::Chunk;

/// Handle to the persisted BM25 index + its field ids.
pub struct TextIndex {
    index: Index,
    path_field: Field,
    chunk_id_field: Field,
    heading_field: Field,
    body_field: Field,
}

/// One BM25 hit.
pub struct TextHit {
    pub chunk_id: String,
    pub score: f32,
}

impl TextIndex {
    /// Open (or create) the index under `<data_dir>/tantivy/`.
    pub fn open(data_dir: &Path) -> anyhow::Result<Self> {
        let dir = data_dir.join("tantivy");
        std::fs::create_dir_all(&dir)?;

        let mut builder = Schema::builder();
        // Untokenized + indexed so `delete_term(path)` drops a file's chunks.
        let path_field = builder.add_text_field("path", STRING);
        let chunk_id_field = builder.add_text_field("chunk_id", STORED);
        let heading_field = builder.add_text_field("heading", TEXT);
        let body_field = builder.add_text_field("body", TEXT);
        let schema = builder.build();

        let mmap = MmapDirectory::open(&dir)?;
        let index = Index::open_or_create(mmap, schema)?;

        Ok(Self {
            index,
            path_field,
            chunk_id_field,
            heading_field,
            body_field,
        })
    }

    /// A writer with a 50 MB budget (well above tantivy's minimum).
    pub fn writer(&self) -> anyhow::Result<IndexWriter> {
        Ok(self.index.writer(50_000_000)?)
    }

    /// Delete every chunk belonging to `relpath` (exact term match on `path`).
    pub fn delete_path(writer: &IndexWriter, path_field: Field, relpath: &str) {
        writer.delete_term(Term::from_field_text(path_field, relpath));
    }

    /// The `path` field id (for [`delete_path`](Self::delete_path)).
    pub fn path_field(&self) -> Field {
        self.path_field
    }

    /// Add a chunk's searchable document. Heading path is joined into the
    /// `heading` field so heading terms contribute to BM25.
    pub fn add_chunk(&self, writer: &IndexWriter, chunk: &Chunk) -> anyhow::Result<()> {
        let heading = chunk.heading_path.join(" ");
        writer.add_document(doc!(
            self.path_field => chunk.path.clone(),
            self.chunk_id_field => chunk.id.clone(),
            self.heading_field => heading,
            self.body_field => chunk.text.clone(),
        ))?;
        Ok(())
    }

    /// BM25 search over heading + body. Returns up to `k` hits. The query text
    /// is sanitized (tantivy query-syntax chars stripped) so arbitrary user
    /// input is treated as a bag of terms, never a parse error.
    pub fn search(&self, query: &str, k: usize) -> anyhow::Result<Vec<TextHit>> {
        let reader = self.index.reader()?;
        reader.reload()?;
        let searcher = reader.searcher();

        let parser = QueryParser::for_index(&self.index, vec![self.heading_field, self.body_field]);
        let sanitized = sanitize_query(query);
        if sanitized.trim().is_empty() {
            return Ok(Vec::new());
        }
        let parsed = parser.parse_query(&sanitized)?;

        let top = searcher.search(&parsed, &TopDocs::with_limit(k).order_by_score())?;
        let mut hits = Vec::with_capacity(top.len());
        for (score, addr) in top {
            let doc: TantivyDocument = searcher.doc(addr)?;
            if let Some(id) = doc.get_first(self.chunk_id_field).and_then(|v| v.as_str()) {
                hits.push(TextHit {
                    chunk_id: id.to_string(),
                    score,
                });
            }
        }
        Ok(hits)
    }
}

/// Replace tantivy query-syntax metacharacters with spaces so the query is a
/// plain bag of terms.
fn sanitize_query(q: &str) -> String {
    q.chars()
        .map(|c| match c {
            '+' | '-' | '&' | '|' | '!' | '(' | ')' | '{' | '}' | '[' | ']' | '^' | '"' | '~'
            | '*' | '?' | ':' | '\\' | '/' => ' ',
            other => other,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(id: &str, path: &str, heading: &[&str], text: &str) -> Chunk {
        Chunk {
            id: id.to_string(),
            path: path.to_string(),
            heading_path: heading.iter().map(|s| s.to_string()).collect(),
            text: text.to_string(),
        }
    }

    #[test]
    fn indexes_and_searches() {
        let dir = tempfile::tempdir().unwrap();
        let idx = TextIndex::open(dir.path()).unwrap();
        let mut w = idx.writer().unwrap();
        idx.add_chunk(
            &w,
            &chunk("a#0", "a.md", &["Config"], "the executor kinds are llm and tool"),
        )
        .unwrap();
        idx.add_chunk(
            &w,
            &chunk("b#0", "b.md", &["Intro"], "welcome to the project overview"),
        )
        .unwrap();
        w.commit().unwrap();

        let hits = idx.search("executor kinds", 5).unwrap();
        assert!(!hits.is_empty());
        assert_eq!(hits[0].chunk_id, "a#0");
    }

    #[test]
    fn delete_path_removes_chunks() {
        let dir = tempfile::tempdir().unwrap();
        let idx = TextIndex::open(dir.path()).unwrap();
        let mut w = idx.writer().unwrap();
        idx.add_chunk(&w, &chunk("a#0", "a.md", &[], "alpha content here"))
            .unwrap();
        w.commit().unwrap();
        assert!(!idx.search("alpha", 5).unwrap().is_empty());

        TextIndex::delete_path(&w, idx.path_field(), "a.md");
        w.commit().unwrap();
        assert!(idx.search("alpha", 5).unwrap().is_empty());
    }

    #[test]
    fn special_chars_do_not_error() {
        let dir = tempfile::tempdir().unwrap();
        let idx = TextIndex::open(dir.path()).unwrap();
        let mut w = idx.writer().unwrap();
        idx.add_chunk(&w, &chunk("a#0", "a.md", &[], "some text about rust"))
            .unwrap();
        w.commit().unwrap();
        // A query full of metacharacters must not panic/Err.
        let hits = idx.search("rust: (foo) +bar* ??", 5).unwrap();
        assert!(!hits.is_empty());
    }
}
