//! Virtual-directory synthesis. The file table is flat: every stored
//! entry is a bag of bytes keyed by a full path like `docs/2025/plan.md`.
//! FUSE needs to present directories; we do it by prefix-matching.
//!
//! For a directory at virtual path `P`:
//!   - File children: every entry whose name equals `P/X` where X has no
//!     further `/`.
//!   - Dir children: every entry whose name starts with `P/X/` (the `/X/`
//!     implies X is a virtual directory). Deduplicated by name.
//!   - Explicit dirs: directories that were `mkdir`'d but have no files
//!     under them yet. Tracked in-memory only; lost on unmount.
//!
//! Root is represented by the empty string `""` — every stored name is
//! under root by construction.

use std::collections::{BTreeMap, HashSet};

use crate::fs::file_table::FileEntry;

#[derive(Debug, Clone)]
pub enum ChildKind {
    File(FileEntry),
    Directory,
}

#[derive(Debug, Clone)]
pub struct Child {
    pub name: String,
    pub kind: ChildKind,
}

pub struct DirView<'a> {
    dir_path: &'a str,
    entries: &'a [FileEntry],
    explicit_dirs: &'a HashSet<String>,
}

impl<'a> DirView<'a> {
    pub fn new(
        dir_path: &'a str,
        entries: &'a [FileEntry],
        explicit_dirs: &'a HashSet<String>,
    ) -> Self {
        Self {
            dir_path,
            entries,
            explicit_dirs,
        }
    }

    /// List of immediate children of `dir_path`. Files take precedence
    /// over directories if the same name appears as both — but that
    /// collision can't happen under validate_filename's rules, so the
    /// deduping is just defensive.
    pub fn children(&self) -> Vec<Child> {
        let prefix = if self.dir_path.is_empty() {
            String::new()
        } else {
            format!("{}/", self.dir_path)
        };

        // BTreeMap gives deterministic order; FUSE doesn't require it but
        // tests and users appreciate a stable listing.
        let mut out: BTreeMap<String, ChildKind> = BTreeMap::new();

        for entry in self.entries {
            if !entry.is_live() {
                continue;
            }
            let Some(rel) = entry.filename.strip_prefix(&prefix) else {
                continue;
            };
            if rel.is_empty() {
                continue;
            }
            match rel.find('/') {
                None => {
                    // Direct file child.
                    out.insert(rel.to_owned(), ChildKind::File(entry.clone()));
                }
                Some(slash) => {
                    // Nested file → virtual directory child.
                    let name = rel[..slash].to_owned();
                    out.entry(name).or_insert(ChildKind::Directory);
                }
            }
        }

        // Explicit dirs whose parent is `dir_path`.
        for dir in self.explicit_dirs {
            let Some(rel) = dir.strip_prefix(&prefix) else {
                continue;
            };
            if rel.is_empty() || rel.contains('/') {
                continue;
            }
            out.entry(rel.to_owned()).or_insert(ChildKind::Directory);
        }

        out.into_iter()
            .map(|(name, kind)| Child { name, kind })
            .collect()
    }

    /// True if any live entry lies under `dir_path` (it's a non-empty
    /// virtual directory). Explicit-dir markers don't count.
    pub fn has_descendants(dir_path: &str, entries: &[FileEntry]) -> bool {
        let prefix = format!("{dir_path}/");
        entries
            .iter()
            .any(|e| e.is_live() && e.filename.starts_with(&prefix))
    }

    /// True if `path` is exposed as a directory in this view — either
    /// because entries live under it, or because it's been explicitly
    /// `mkdir`'d.
    pub fn path_is_directory(
        path: &str,
        entries: &[FileEntry],
        explicit_dirs: &HashSet<String>,
    ) -> bool {
        if path.is_empty() {
            return true;
        }
        if explicit_dirs.contains(path) {
            return true;
        }
        Self::has_descendants(path, entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fs::file_table::FileEntryType;
    use crate::stego::integrity::NO_BLOCK;

    fn live_entry(name: &str) -> FileEntry {
        FileEntry {
            entry_type: FileEntryType::Regular,
            flags: 0,
            mode: 0o644,
            uid: 0,
            gid: 0,
            size_bytes: 1,
            created: 0,
            modified: 0,
            crc32: 0,
            block_count: 1,
            overflow_block: NO_BLOCK,
            filename: name.to_owned(),
            inline_blocks: vec![0],
        }
    }

    #[test]
    fn root_listing_groups_nested_paths_into_virtual_dirs() {
        let entries = vec![
            live_entry("notes.txt"),
            live_entry("docs/a.txt"),
            live_entry("docs/b.txt"),
            live_entry("docs/2025/plan.md"),
        ];
        let explicit: HashSet<String> = HashSet::new();
        let view = DirView::new("", &entries, &explicit);
        let names: Vec<_> = view.children().into_iter().map(|c| c.name).collect();
        assert_eq!(names, vec!["docs".to_string(), "notes.txt".to_string()]);
    }

    #[test]
    fn nested_listing_descends_one_level() {
        let entries = vec![
            live_entry("docs/a.txt"),
            live_entry("docs/b.txt"),
            live_entry("docs/2025/plan.md"),
            live_entry("docs/2025/oct/retro.md"),
            live_entry("other/c.txt"),
        ];
        let explicit: HashSet<String> = HashSet::new();

        let docs = DirView::new("docs", &entries, &explicit);
        let names: Vec<_> = docs.children().into_iter().map(|c| c.name).collect();
        assert_eq!(
            names,
            vec!["2025".to_string(), "a.txt".to_string(), "b.txt".to_string()]
        );

        let year = DirView::new("docs/2025", &entries, &explicit);
        let names: Vec<_> = year.children().into_iter().map(|c| c.name).collect();
        assert_eq!(names, vec!["oct".to_string(), "plan.md".to_string()]);
    }

    #[test]
    fn explicit_dirs_surface_even_with_no_files_under_them() {
        let entries = vec![live_entry("notes.txt")];
        let mut explicit: HashSet<String> = HashSet::new();
        explicit.insert("empty".to_string());

        let view = DirView::new("", &entries, &explicit);
        let children = view.children();
        let empty = children
            .iter()
            .find(|c| c.name == "empty")
            .expect("empty dir surfaces");
        assert!(matches!(empty.kind, ChildKind::Directory));
    }

    #[test]
    fn tombstoned_entries_do_not_appear_as_children() {
        let mut dead = live_entry("docs/gone.txt");
        dead.flags = crate::fs::file_table::FLAG_DELETED;
        let entries = vec![dead, live_entry("docs/live.txt")];
        let explicit: HashSet<String> = HashSet::new();

        let view = DirView::new("docs", &entries, &explicit);
        let names: Vec<_> = view.children().into_iter().map(|c| c.name).collect();
        assert_eq!(names, vec!["live.txt".to_string()]);
    }

    #[test]
    fn has_descendants_reports_non_empty_virtual_dirs() {
        let entries = vec![live_entry("docs/a.txt"), live_entry("root.txt")];
        assert!(DirView::has_descendants("docs", &entries));
        assert!(!DirView::has_descendants("empty", &entries));
    }

    #[test]
    fn path_is_directory_accepts_root_virtual_and_explicit() {
        let entries = vec![live_entry("docs/a.txt")];
        let mut explicit: HashSet<String> = HashSet::new();
        explicit.insert("explicit_dir".to_string());

        assert!(DirView::path_is_directory("", &entries, &explicit));
        assert!(DirView::path_is_directory("docs", &entries, &explicit));
        assert!(DirView::path_is_directory(
            "explicit_dir",
            &entries,
            &explicit
        ));
        assert!(!DirView::path_is_directory("ghost", &entries, &explicit));
    }
}
