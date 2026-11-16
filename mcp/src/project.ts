// The project a session works on, as `anima-server mcp` detects it:
// ANIMA_PROJECT, then the git repository (origin remote name, else the main
// checkout's folder), then the folder. Writes and the briefing must agree on it.
import { existsSync, readFileSync, statSync } from "node:fs";
import { basename, dirname, isAbsolute, join, resolve } from "node:path";
import { homedir } from "node:os";

const GENERIC = new Set(["src", "home", "desktop", "documents", "downloads", "tmp", "temp", "projects", "code", "workspace"]);

function usable(name: string | undefined): string | undefined {
  const n = (name ?? "").trim();
  return n.length >= 2 && !GENERIC.has(n.toLowerCase()) ? n : undefined;
}

function gitDirs(dir: string): { git: string; main: string } | undefined {
  let d = resolve(dir);
  for (;;) {
    const dotGit = join(d, ".git");
    if (existsSync(dotGit)) {
      if (statSync(dotGit).isDirectory()) return { git: dotGit, main: d };
      const m = /^gitdir:\s*(.+)$/m.exec(readFileSync(dotGit, "utf8"));
      if (!m) return undefined;
      let gitdir = m[1].trim();
      if (!isAbsolute(gitdir)) gitdir = resolve(d, gitdir);
      let common = gitdir;
      while (basename(common) !== ".git" && dirname(common) !== common) common = dirname(common);
      return basename(common) === ".git" ? { git: common, main: dirname(common) } : undefined;
    }
    const up = dirname(d);
    if (up === d) return undefined;
    d = up;
  }
}

/** The `url` of `[remote "origin"]`, else of the first remote. */
export function originUrl(config: string): string | undefined {
  let section = "";
  let first: string | undefined;
  for (const raw of config.split("\n")) {
    const line = raw.trim();
    if (line.startsWith("[")) { section = line; continue; }
    if (!section.startsWith("[remote ")) continue;
    const m = /^url\s*=\s*(.+)$/.exec(line);
    if (!m) continue;
    if (section === '[remote "origin"]') return m[1].trim();
    first ??= m[1].trim();
  }
  return first;
}

export function repoName(url: string): string | undefined {
  const last = url.replace(/\/+$/, "").split(/[/:]/).pop() ?? "";
  const name = last.replace(/\.git$/, "");
  return name || undefined;
}

export function detectProject(dir: string = process.cwd()): string | undefined {
  const fromEnv = (process.env.ANIMA_PROJECT ?? "").trim();
  if (fromEnv) return fromEnv;
  const git = gitDirs(dir);
  if (git) {
    let config = "";
    try { config = readFileSync(join(git.git, "config"), "utf8"); } catch { /* no config */ }
    const url = originUrl(config);
    const name = (url && usable(repoName(url))) || usable(basename(git.main));
    if (name) return name;
  }
  const d = resolve(dir);
  if (d === homedir() || dirname(d) === d) return undefined;
  return usable(basename(d));
}

/** A project name as one namespace segment, as the server builds it. */
export function projectSegment(name: string): string | undefined {
  let out = "";
  for (const ch of name.trim().toLowerCase()) {
    const c = /[a-z0-9_.]/.test(ch) ? ch : "-";
    if (c === "-" && (out === "" || out.endsWith("-"))) continue;
    out += c;
  }
  out = out.replace(/^[-.]+|[-.]+$/g, "").slice(0, 64).replace(/[-.]+$/, "");
  return out || undefined;
}
