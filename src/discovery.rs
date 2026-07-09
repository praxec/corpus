//! Doc-file discovery: walk the repo honoring `.gitignore` (via the `ignore`
//! crate) and the configured include globs.

use std::path::{Path, PathBuf};

use ignore::WalkBuilder;

/// A discovered documentation file: its absolute path plus the path relative
/// to the repo root (the manifest key — the freshness contract is keyed by
/// relpath so a moved repo re-uses its manifest).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocFile {
    pub abs: PathBuf,
    pub relpath: String,
}

/// Walk `repo` for documentation files matching any of `include` globs,
/// respecting `.gitignore`. Returns files sorted by relpath for determinism.
pub fn discover(repo: &Path, include: &[String]) -> anyhow::Result<Vec<DocFile>> {
    let mut builder = globset::GlobSetBuilder::new();
    for g in include {
        builder.add(globset::Glob::new(g)?);
    }
    let globset = builder.build()?;

    let mut out = Vec::new();
    let walker = WalkBuilder::new(repo)
        .hidden(false) // don't skip dotfiles wholesale; .gitignore still applies
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .parents(true)
        .build();

    for result in walker {
        let entry = match result {
            Ok(e) => e,
            Err(_) => continue,
        };
        if !entry.file_type().is_some_and(|t| t.is_file()) {
            continue;
        }
        let abs = entry.path();
        let rel = match abs.strip_prefix(repo) {
            Ok(r) => r,
            Err(_) => continue,
        };
        // Never index our own data dir.
        if rel.starts_with(".corpus") {
            continue;
        }
        if !globset.is_match(rel) {
            continue;
        }
        out.push(DocFile {
            abs: abs.to_path_buf(),
            relpath: rel.to_string_lossy().replace('\\', "/"),
        });
    }

    out.sort_by(|a, b| a.relpath.cmp(&b.relpath));
    out.dedup();
    Ok(out)
}
