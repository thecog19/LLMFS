//! Session-scoped inode table. Maps between virtual paths and the u64
//! inodes the kernel hands us across FUSE ops. Inodes are assigned on
//! first sight and stay stable for the life of the mount; remounting
//! resets the map (same as tmpfs).

use std::collections::HashMap;

/// The kernel reserves inode 1 for the root.
pub const ROOT_INO: u64 = 1;

#[derive(Debug)]
pub struct InodeMap {
    next_ino: u64,
    path_to_ino: HashMap<String, u64>,
    ino_to_path: HashMap<u64, String>,
}

impl Default for InodeMap {
    fn default() -> Self {
        let mut map = Self {
            next_ino: ROOT_INO + 1,
            path_to_ino: HashMap::new(),
            ino_to_path: HashMap::new(),
        };
        map.path_to_ino.insert(String::new(), ROOT_INO);
        map.ino_to_path.insert(ROOT_INO, String::new());
        map
    }
}

impl InodeMap {
    /// Returns the inode for `path`, assigning a new one if unseen.
    /// Root is the empty path and always maps to `ROOT_INO`.
    pub fn intern(&mut self, path: &str) -> u64 {
        if let Some(ino) = self.path_to_ino.get(path) {
            return *ino;
        }
        let ino = self.next_ino;
        self.next_ino = self
            .next_ino
            .checked_add(1)
            .expect("u64 inode overflow — one session can't go that long");
        self.path_to_ino.insert(path.to_owned(), ino);
        self.ino_to_path.insert(ino, path.to_owned());
        ino
    }

    pub fn path_of(&self, ino: u64) -> Option<&str> {
        self.ino_to_path.get(&ino).map(String::as_str)
    }

    pub fn ino_of(&self, path: &str) -> Option<u64> {
        self.path_to_ino.get(path).copied()
    }

    /// Drop an inode from the map. FUSE's `forget` op should trigger this,
    /// but V1 is loose about it — the map grows until unmount. Kept here
    /// so the forget op is a one-liner when we wire it up.
    #[allow(dead_code)]
    pub fn forget(&mut self, ino: u64) {
        if let Some(path) = self.ino_to_path.remove(&ino) {
            self.path_to_ino.remove(&path);
        }
    }

    /// Rewrite the path for `ino` and every descendant path that lives
    /// under `ino`'s old path. Used by rename.
    pub fn rename(&mut self, ino: u64, new_path: &str) {
        let Some(old_path) = self.ino_to_path.get(&ino).cloned() else {
            return;
        };
        if old_path == new_path {
            return;
        }
        // Collect paths to rewrite so we don't mutate while iterating.
        let old_prefix = format!("{old_path}/");
        let new_prefix = format!("{new_path}/");
        let to_update: Vec<(u64, String)> = self
            .ino_to_path
            .iter()
            .filter_map(|(ino, path)| {
                if path == &old_path {
                    Some((*ino, new_path.to_owned()))
                } else if path.starts_with(&old_prefix) {
                    Some((*ino, format!("{}{}", new_prefix, &path[old_prefix.len()..])))
                } else {
                    None
                }
            })
            .collect();
        for (ino, replacement) in to_update {
            if let Some(old) = self.ino_to_path.insert(ino, replacement.clone()) {
                self.path_to_ino.remove(&old);
            }
            self.path_to_ino.insert(replacement, ino);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_is_always_ino_1() {
        let map = InodeMap::default();
        assert_eq!(map.ino_of(""), Some(ROOT_INO));
        assert_eq!(map.path_of(ROOT_INO), Some(""));
    }

    #[test]
    fn intern_assigns_fresh_inos_and_is_idempotent() {
        let mut map = InodeMap::default();
        let a1 = map.intern("notes.txt");
        let a2 = map.intern("notes.txt");
        let b = map.intern("other.txt");
        assert_eq!(a1, a2, "same path interns to the same inode");
        assert_ne!(a1, b);
        assert_ne!(a1, ROOT_INO);
    }

    #[test]
    fn path_and_ino_round_trip() {
        let mut map = InodeMap::default();
        let ino = map.intern("docs/plan.md");
        assert_eq!(map.path_of(ino), Some("docs/plan.md"));
        assert_eq!(map.ino_of("docs/plan.md"), Some(ino));
    }

    #[test]
    fn rename_rewrites_descendants() {
        let mut map = InodeMap::default();
        let dir_ino = map.intern("old");
        let _ = map.intern("old/a.txt");
        let _ = map.intern("old/nested/b.txt");
        let outside = map.intern("outside.txt");

        map.rename(dir_ino, "new");

        assert_eq!(map.ino_of("old"), None);
        assert_eq!(map.ino_of("old/a.txt"), None);
        assert_eq!(map.ino_of("old/nested/b.txt"), None);
        assert!(map.ino_of("new").is_some());
        assert!(map.ino_of("new/a.txt").is_some());
        assert!(map.ino_of("new/nested/b.txt").is_some());
        assert_eq!(
            map.path_of(outside),
            Some("outside.txt"),
            "rename of one subtree must not disturb unrelated paths"
        );
    }
}
