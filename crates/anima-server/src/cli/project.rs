//! The project a session works on, which names its memory namespace
//! (`<namespace>/<project>`). Writes and the briefing must agree on it, so
//! `mcp` and `recall` both ask here.
//!
//! In order: `ANIMA_PROJECT`, then the git repository (its `origin` remote's
//! name, else the main checkout's folder, so every worktree of one repository
//! is one project), then the folder itself.

use std::path::{Path, PathBuf};

/// Folder names that say nothing about a project.
const GENERIC: [&str; 10] = ["src", "home", "desktop", "documents", "downloads", "tmp", "temp", "projects", "code", "workspace"];

/// The project for a session working in `dir`.
pub fn detect(dir: &Path) -> Option<String> {
    from_env().or_else(|| from_git(dir)).or_else(|| from_folder(dir))
}

/// `ANIMA_PROJECT`, when set.
pub fn from_env() -> Option<String> {
    std::env::var("ANIMA_PROJECT")
        .ok()
        .map(|n| n.trim().to_string())
        .filter(|n| !n.is_empty())
}

/// The project for the current working directory.
pub fn detect_here() -> Option<String> {
    std::env::current_dir().ok().and_then(|d| detect(&d))
}

fn usable(name: &str) -> Option<String> {
    let name = name.trim();
    (name.chars().count() >= 2 && !GENERIC.contains(&name.to_lowercase().as_str())).then(|| name.to_string())
}

fn from_folder(dir: &Path) -> Option<String> {
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).map(PathBuf::from);
    if home.as_deref() == Some(dir) || dir.parent().is_none() {
        return None;
    }
    dir.file_name().and_then(|n| n.to_str()).and_then(usable)
}

/// The `.git` directory holding the config of the repository around `dir`,
/// and the folder of its main checkout.
fn git_dirs(dir: &Path) -> Option<(PathBuf, PathBuf)> {
    for d in dir.ancestors() {
        let dot_git = d.join(".git");
        if dot_git.is_dir() {
            return Some((dot_git, d.to_path_buf()));
        }
        if dot_git.is_file() {
            // A worktree: "gitdir: <main>/.git/worktrees/<name>".
            let text = std::fs::read_to_string(&dot_git).ok()?;
            let gitdir = PathBuf::from(text.trim().strip_prefix("gitdir:")?.trim());
            let gitdir = if gitdir.is_absolute() { gitdir } else { d.join(gitdir) };
            let common = gitdir.ancestors().find(|a| a.file_name().is_some_and(|n| n == ".git"))?.to_path_buf();
            let main = common.parent()?.to_path_buf();
            return Some((common, main));
        }
    }
    None
}

fn from_git(dir: &Path) -> Option<String> {
    let (git, main) = git_dirs(dir)?;
    let config = std::fs::read_to_string(git.join("config")).unwrap_or_default();
    origin_url(&config)
        .and_then(|url| repo_name(&url))
        .and_then(|n| usable(&n))
        .or_else(|| main.file_name().and_then(|n| n.to_str()).and_then(usable))
}

/// The `url` of `[remote "origin"]`, else of the first remote.
fn origin_url(config: &str) -> Option<String> {
    let mut section = String::new();
    let mut first: Option<String> = None;
    for line in config.lines().map(str::trim) {
        if line.starts_with('[') {
            section = line.to_string();
            continue;
        }
        if !section.starts_with("[remote ") {
            continue;
        }
        if let Some(url) = line.strip_prefix("url").map(str::trim_start).and_then(|l| l.strip_prefix('=')) {
            let url = url.trim().to_string();
            if section == "[remote \"origin\"]" {
                return Some(url);
            }
            first.get_or_insert(url);
        }
    }
    first
}

/// `AnimaLab/Anima` from `git@github.com:AnimaLab/Anima.git` → `Anima`.
fn repo_name(url: &str) -> Option<String> {
    let last = url.trim_end_matches('/').rsplit(['/', ':']).next()?;
    let name = last.strip_suffix(".git").unwrap_or(last);
    (!name.is_empty()).then(|| name.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repository_names_come_from_the_remote_url() {
        assert_eq!(repo_name("git@github.com:AnimaLab/Anima.git").as_deref(), Some("Anima"));
        assert_eq!(repo_name("https://gitlab.example/pf/ai/llmaas-proxy.git").as_deref(), Some("llmaas-proxy"));
        assert_eq!(repo_name("https://example/x/repo/").as_deref(), Some("repo"));
        let config = "[core]\n\tbare = false\n[remote \"upstream\"]\n\turl = git@x:a/up.git\n[remote \"origin\"]\n\turl = git@x:a/mine.git\n";
        assert_eq!(origin_url(config).as_deref(), Some("git@x:a/mine.git"));
        assert_eq!(origin_url("[remote \"fork\"]\n url = u/f.git\n").as_deref(), Some("u/f.git"));
        assert_eq!(origin_url("[core]\n"), None);
    }

    #[test]
    fn a_worktree_belongs_to_its_repository() {
        let root = tempfile::tempdir().unwrap();
        let main = root.path().join("billing-api");
        std::fs::create_dir_all(main.join(".git/worktrees/feat-x")).unwrap();
        std::fs::create_dir_all(main.join("crates/core")).unwrap();
        let wt = root.path().join("feat-x");
        std::fs::create_dir_all(&wt).unwrap();
        std::fs::write(wt.join(".git"), format!("gitdir: {}\n", main.join(".git/worktrees/feat-x").display())).unwrap();

        // No remote: the main checkout's folder names the project, from anywhere inside.
        assert_eq!(from_git(&wt).as_deref(), Some("billing-api"));
        assert_eq!(from_git(&main.join("crates/core")).as_deref(), Some("billing-api"));

        std::fs::write(main.join(".git/config"), "[remote \"origin\"]\n\turl = git@host:team/Billing.git\n").unwrap();
        assert_eq!(from_git(&wt).as_deref(), Some("Billing"));
    }

    #[test]
    fn generic_folders_name_no_project() {
        let root = tempfile::tempdir().unwrap();
        let src = root.path().join("src");
        std::fs::create_dir_all(&src).unwrap();
        assert_eq!(from_folder(&src), None);
        assert_eq!(from_folder(root.path()).is_some(), true);
        assert_eq!(from_folder(Path::new("/")), None);
    }
}
