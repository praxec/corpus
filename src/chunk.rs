//! Markdown-aware chunking.
//!
//! Splits a document on headings using `pulldown-cmark`, carrying a
//! `heading_path` (the stack of enclosing headings, e.g.
//! `["Configuration", "Executor kinds"]`) as metadata. Sections larger than a
//! size cap are further split into windows with a small overlap so a single
//! long section still yields retrievable, bounded chunks.
//!
//! Non-markdown docs (`.txt`, `.adoc`) have no heading structure we parse; they
//! are chunked purely by the size/overlap window with an empty `heading_path`.

use pulldown_cmark::{Event, HeadingLevel, Parser, Tag, TagEnd};
use serde::{Deserialize, Serialize};

/// Target maximum chunk size in characters (soft cap; sections are split into
/// windows no larger than this).
pub const CHUNK_CHAR_CAP: usize = 1200;

/// Character overlap between adjacent windows of an over-cap section.
pub const CHUNK_OVERLAP: usize = 150;

/// One retrievable unit: text plus where it came from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Chunk {
    /// Stable id: `<relpath>#<ordinal>`.
    pub id: String,
    /// Repo-relative source path.
    pub path: String,
    /// Enclosing heading stack (outermost first). Empty for non-markdown.
    pub heading_path: Vec<String>,
    /// The chunk body.
    pub text: String,
}

/// A parsed section: its heading stack + accumulated body text.
struct Section {
    heading_path: Vec<String>,
    body: String,
}

/// Chunk a markdown document into sections split on headings, then windowed to
/// the size cap. `relpath` seeds chunk ids.
pub fn chunk_markdown(relpath: &str, source: &str) -> Vec<Chunk> {
    let sections = parse_sections(source);
    windows_to_chunks(relpath, sections)
}

/// Chunk a plain-text document by size window only (no heading structure).
pub fn chunk_plain(relpath: &str, source: &str) -> Vec<Chunk> {
    let sections = vec![Section {
        heading_path: Vec::new(),
        body: source.to_string(),
    }];
    windows_to_chunks(relpath, sections)
}

/// Dispatch on extension: markdown-aware for `.md`/`.mdx`, plain otherwise.
pub fn chunk_file(relpath: &str, source: &str) -> Vec<Chunk> {
    let is_md = relpath
        .rsplit('.')
        .next()
        .map(|e| e.eq_ignore_ascii_case("md") || e.eq_ignore_ascii_case("mdx"))
        .unwrap_or(false);
    if is_md {
        chunk_markdown(relpath, source)
    } else {
        chunk_plain(relpath, source)
    }
}

/// Walk markdown events, accumulating body text under the running heading
/// stack. A new heading closes the current section and updates the stack.
fn parse_sections(source: &str) -> Vec<Section> {
    let mut sections: Vec<Section> = Vec::new();
    let mut stack: Vec<(u8, String)> = Vec::new(); // (level, title)
    let mut cur_body = String::new();
    let mut cur_heading_path: Vec<String> = Vec::new();

    // Heading capture state.
    let mut in_heading = false;
    let mut heading_level: u8 = 0;
    let mut heading_text = String::new();

    let flush = |sections: &mut Vec<Section>, heading_path: &[String], body: &mut String| {
        let trimmed = body.trim();
        if !trimmed.is_empty() {
            sections.push(Section {
                heading_path: heading_path.to_vec(),
                body: trimmed.to_string(),
            });
        }
        body.clear();
    };

    for event in Parser::new(source) {
        match event {
            Event::Start(Tag::Heading { level, .. }) => {
                // Close the section that belongs to the *previous* heading.
                flush(&mut sections, &cur_heading_path, &mut cur_body);
                in_heading = true;
                heading_level = heading_level_num(level);
                heading_text.clear();
            }
            Event::End(TagEnd::Heading(_)) => {
                in_heading = false;
                // Pop the stack to the parent level, then push this heading.
                while let Some((lvl, _)) = stack.last() {
                    if *lvl >= heading_level {
                        stack.pop();
                    } else {
                        break;
                    }
                }
                stack.push((heading_level, heading_text.trim().to_string()));
                cur_heading_path = stack.iter().map(|(_, t)| t.clone()).collect();
            }
            Event::Text(t) | Event::Code(t) => {
                if in_heading {
                    heading_text.push_str(&t);
                } else {
                    cur_body.push_str(&t);
                }
            }
            Event::SoftBreak | Event::HardBreak if !in_heading => {
                cur_body.push('\n');
            }
            Event::End(TagEnd::Paragraph)
            | Event::End(TagEnd::Item)
            | Event::End(TagEnd::CodeBlock)
                if !in_heading =>
            {
                cur_body.push('\n');
            }
            _ => {}
        }
    }
    flush(&mut sections, &cur_heading_path, &mut cur_body);
    sections
}

fn heading_level_num(level: HeadingLevel) -> u8 {
    match level {
        HeadingLevel::H1 => 1,
        HeadingLevel::H2 => 2,
        HeadingLevel::H3 => 3,
        HeadingLevel::H4 => 4,
        HeadingLevel::H5 => 5,
        HeadingLevel::H6 => 6,
    }
}

/// Turn sections into chunks, windowing any section over the size cap. Chunk
/// ids are assigned sequentially across the whole document so they are stable
/// for a given input.
fn windows_to_chunks(relpath: &str, sections: Vec<Section>) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let mut ordinal = 0usize;
    for section in sections {
        for window in window_text(&section.body) {
            chunks.push(Chunk {
                id: format!("{relpath}#{ordinal}"),
                path: relpath.to_string(),
                heading_path: section.heading_path.clone(),
                text: window,
            });
            ordinal += 1;
        }
    }
    chunks
}

/// Split `body` into windows of at most [`CHUNK_CHAR_CAP`] characters with
/// [`CHUNK_OVERLAP`] characters of overlap. Splits on a char boundary near the
/// cap (prefers the last newline/space in the window) to avoid cutting words.
fn window_text(body: &str) -> Vec<String> {
    let chars: Vec<char> = body.chars().collect();
    if chars.len() <= CHUNK_CHAR_CAP {
        return vec![body.to_string()];
    }
    let mut out = Vec::new();
    let mut start = 0usize;
    while start < chars.len() {
        let hard_end = (start + CHUNK_CHAR_CAP).min(chars.len());
        // Prefer a break at whitespace within the last ~15% of the window.
        let mut end = hard_end;
        if hard_end < chars.len() {
            let lookback = start + (CHUNK_CHAR_CAP * 85 / 100);
            if let Some(pos) = (lookback..hard_end)
                .rev()
                .find(|&i| chars[i].is_whitespace())
            {
                end = pos + 1;
            }
        }
        let window: String = chars[start..end].iter().collect();
        let trimmed = window.trim().to_string();
        if !trimmed.is_empty() {
            out.push(trimmed);
        }
        if end >= chars.len() {
            break;
        }
        start = end.saturating_sub(CHUNK_OVERLAP);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_on_headings_and_keeps_heading_path() {
        let src = "\
# Title

Intro text.

## Configuration

Config body.

### Executor kinds

Executor body here.
";
        let chunks = chunk_markdown("docs/guide.md", src);
        // One chunk per non-empty section.
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].heading_path, vec!["Title"]);
        assert_eq!(chunks[1].heading_path, vec!["Title", "Configuration"]);
        assert_eq!(
            chunks[2].heading_path,
            vec!["Title", "Configuration", "Executor kinds"]
        );
        assert!(chunks[2].text.contains("Executor body"));
        assert_eq!(chunks[0].id, "docs/guide.md#0");
    }

    #[test]
    fn sibling_headings_pop_the_stack() {
        let src = "\
# A

body a

## B1

body b1

## B2

body b2
";
        let chunks = chunk_markdown("f.md", src);
        assert_eq!(chunks[1].heading_path, vec!["A", "B1"]);
        // B2 is a sibling of B1 — B1 must be popped.
        assert_eq!(chunks[2].heading_path, vec!["A", "B2"]);
    }

    #[test]
    fn large_section_is_windowed_with_overlap() {
        let big = "word ".repeat(1000); // ~5000 chars, single section
        let src = format!("# H\n\n{big}");
        let chunks = chunk_markdown("big.md", &src);
        assert!(chunks.len() > 1, "expected multiple windows");
        for c in &chunks {
            assert!(c.text.chars().count() <= CHUNK_CHAR_CAP);
            assert_eq!(c.heading_path, vec!["H"]);
        }
    }

    #[test]
    fn plain_text_has_empty_heading_path() {
        let chunks = chunk_plain("notes.txt", "just some plain notes");
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].heading_path.is_empty());
    }

    #[test]
    fn chunk_ids_are_stable_across_runs() {
        let src = "# H\n\nbody";
        let a = chunk_markdown("x.md", src);
        let b = chunk_markdown("x.md", src);
        assert_eq!(a, b);
    }
}
