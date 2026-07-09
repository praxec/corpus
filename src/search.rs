//! Search result types + Reciprocal Rank Fusion.
//!
//! Text search (BM25) is always available. When semantic search is enabled and
//! vectors exist, hybrid mode fuses the two ranked lists via RRF — a
//! score-agnostic fusion that just needs each list's rank order, so BM25 scores
//! and cosine similarities never have to be normalized onto a common scale.

use serde::Serialize;

/// A search result surfaced to the MCP caller.
#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    pub path: String,
    pub heading_path: Vec<String>,
    pub snippet: String,
    pub score: f32,
}

/// Search mode. `hybrid` fuses text + semantic when semantic is available,
/// else falls back to text. `text` and `semantic` force a single lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    Hybrid,
    Text,
    Semantic,
}

impl SearchMode {
    /// Parse the `mode` tool argument; defaults to [`Hybrid`](Self::Hybrid).
    pub fn parse(s: Option<&str>) -> anyhow::Result<Self> {
        match s.unwrap_or("hybrid") {
            "hybrid" => Ok(Self::Hybrid),
            "text" => Ok(Self::Text),
            "semantic" => Ok(Self::Semantic),
            other => anyhow::bail!("unknown mode '{other}' (expected hybrid | text | semantic)"),
        }
    }
}

/// RRF constant. 60 is the value from the original Cormack et al. paper and the
/// de-facto default; it damps the influence of very-high ranks.
pub const RRF_K: f32 = 60.0;

/// Reciprocal Rank Fusion over any number of ranked id-lists. Each list is
/// `chunk_id`s in descending relevance. Returns `(chunk_id, fused_score)` sorted
/// by descending fused score. A chunk's score is `Σ 1/(RRF_K + rank)` over the
/// lists it appears in (rank is 1-based).
pub fn reciprocal_rank_fusion(lists: &[Vec<String>]) -> Vec<(String, f32)> {
    use std::collections::HashMap;
    let mut scores: HashMap<String, f32> = HashMap::new();
    let mut order: Vec<String> = Vec::new();
    for list in lists {
        for (i, id) in list.iter().enumerate() {
            let rank = (i + 1) as f32;
            if !scores.contains_key(id) {
                order.push(id.clone());
            }
            *scores.entry(id.clone()).or_insert(0.0) += 1.0 / (RRF_K + rank);
        }
    }
    let mut fused: Vec<(String, f32)> = order.into_iter().map(|id| (id.clone(), scores[&id])).collect();
    fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    fused
}

/// Make a bounded snippet from chunk text: collapse whitespace and truncate on
/// a char boundary to `max` chars, appending an ellipsis if cut.
pub fn snippet(text: &str, max: usize) -> String {
    let collapsed: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() <= max {
        return collapsed;
    }
    let cut: String = collapsed.chars().take(max).collect();
    format!("{}…", cut.trim_end())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_parses_and_defaults() {
        assert_eq!(SearchMode::parse(None).unwrap(), SearchMode::Hybrid);
        assert_eq!(SearchMode::parse(Some("text")).unwrap(), SearchMode::Text);
        assert!(SearchMode::parse(Some("nonsense")).is_err());
    }

    #[test]
    fn rrf_rewards_agreement_across_lists() {
        let text = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let sem = vec!["b".to_string(), "a".to_string(), "d".to_string()];
        let fused = reciprocal_rank_fusion(&[text, sem]);
        // `b` (ranks 2,1) and `a` (ranks 1,2) appear in both and lead; a `c`/`d`
        // that appears once must rank below them.
        assert!(fused[0].0 == "a" || fused[0].0 == "b");
        let top_two: Vec<&str> = fused.iter().take(2).map(|(id, _)| id.as_str()).collect();
        assert!(top_two.contains(&"a") && top_two.contains(&"b"));
    }

    #[test]
    fn snippet_truncates_with_ellipsis() {
        let s = snippet("one two three four five", 8);
        assert!(s.ends_with('…'));
        assert!(s.chars().count() <= 9);
    }

    #[test]
    fn snippet_collapses_whitespace() {
        assert_eq!(snippet("a\n\n  b\tc", 100), "a b c");
    }
}
