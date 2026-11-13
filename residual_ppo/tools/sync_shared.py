"""Copy the team's shared modules (and scene asset) byte-for-byte from 61d2722, and verify them later.

Identity of a shared file = its git blob ID at the source commit, NOT its on-disk SHA-256.
core.autocrlf=true makes working copies CRLF, so on-disk hashes differ from the LF bytes in git
even when the content is identical; `git hash-object` normalizes line endings first.
See docs/milestone1_plan.md, steps M1.2 and M1.3.

Two kinds of entries:
  modules  team drone_code/<name>.py      -> residual_ppo/<name>.py      (mode "copy")
  assets   team <repo-relative path>      -> same repo-relative path     (mode "copy" or "check")
A "check" entry is never written, not even with --force: it must already match the source.

Usage (normal venv is enough; standard library only):
    python tools/sync_shared.py            copy missing files; refuse if any existing file differs
    python tools/sync_shared.py --force    also overwrite "copy" files that differ, with the source bytes
    python tools/sync_shared.py --check    verify every entry; exit 1 on any mismatch

Exit codes: 0 OK, 1 mismatch / refused, 2 precondition or git error.
Read-only toward drone_team_main: only `git rev-parse`, `git cat-file`, `git remote get-url` run there.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

RESIDUAL_DIR = Path(__file__).resolve().parents[1]           # ...\drone_final_project1\residual_ppo
DEST_REPO = RESIDUAL_DIR.parent                               # ...\drone_final_project1
DEST_PREFIX = RESIDUAL_DIR.relative_to(DEST_REPO).as_posix()  # "residual_ppo"
TEAM_REPO = DEST_REPO.parent / "drone_team_main"              # read-only source worktree
SOURCE_DIR = "drone_code"
EXPECTED_COMMIT = "61d2722"
MANIFEST_PATH = RESIDUAL_DIR / "SHARED_MANIFEST.json"
IDENTITY = "git blob id (git hash-object, autocrlf-normalized)"

# Exact import closure of PathSacEnv at 61d2722 (16) + SAC-checkpoint loading (2). See CLAUDE.md.
MODULES = [
    "path_sac_env", "path_sac_policy", "path_sac_config", "path_reference",
    "hover_env", "hover_task", "hover_ppo_config", "sensor_config",
    "state_estimation", "model_contract", "controller", "sim_config",
    "drone_sim", "motor", "wind", "drone_diagnostics",
    "hover_sac_policy", "hover_sac_config",
]

# Local files of the scene's dependency closure (UsdUtils.ComputeAllDependencies on
# sim_scene/dij3d_boxes.usd at 61d2722). Repo-relative; the USD-relative reference
# ../sim_asset/iris.usd only works at these exact locations. Remote S3 assets cannot be pinned here.
ASSETS = {
    "sim_scene/dij3d_boxes.usd": "copy",   # opened by hover_env.make_hover_sim
    "sim_asset/iris.usd": "check",         # already on Claire's branch; must stay identical
}


@dataclass(frozen=True)
class Entry:
    key: str        # manifest key
    kind: str       # "module" or "asset"
    mode: str       # "copy" or "check"
    dest_rel: str   # path relative to DEST_REPO (posix)
    src_rel: str    # path inside the source commit (posix)


ENTRIES = (
    [Entry(f"{m}.py", "module", "copy", f"{DEST_PREFIX}/{m}.py", f"{SOURCE_DIR}/{m}.py") for m in MODULES]
    + [Entry(p, "asset", mode, p, p) for p, mode in ASSETS.items()]
)


class SyncError(Exception):
    """Precondition or git failure: stop without touching anything else."""


def git(repo: Path, *args: str) -> bytes:
    """Run git in `repo` and return raw stdout bytes (no decoding, no newline translation)."""
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True)
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace").strip()
        raise SyncError(f"`git {' '.join(args)}` failed in {repo}: {err}")
    return result.stdout


def git_text(repo: Path, *args: str) -> str:
    return git(repo, *args).decode("ascii").strip()


def resolve_source_commit() -> str:
    """Full SHA of EXPECTED_COMMIT; also require the team HEAD to be exactly that commit."""
    if not TEAM_REPO.is_dir():
        raise SyncError(f"team worktree not found: {TEAM_REPO}")
    full = git_text(TEAM_REPO, "rev-parse", "--verify", f"{EXPECTED_COMMIT}^{{commit}}")
    head = git_text(TEAM_REPO, "rev-parse", "HEAD")
    if head != full:
        raise SyncError(f"team HEAD is {head}, expected {full} ({EXPECTED_COMMIT}). Refusing to run.")
    return full


def source_blob_id(commit: str, entry: Entry) -> str:
    return git_text(TEAM_REPO, "rev-parse", "--verify", f"{commit}:{entry.src_rel}")


def source_bytes(commit: str, entry: Entry) -> bytes:
    # cat-file blob = the exact stored bytes; never applies textconv or eol conversion.
    return git(TEAM_REPO, "cat-file", "blob", f"{commit}:{entry.src_rel}")


def local_blob_id(entry: Entry) -> str | None:
    """Blob ID of the destination file as git would store it (autocrlf applied), or None if missing."""
    if not (DEST_REPO / entry.dest_rel).is_file():
        return None
    return git_text(DEST_REPO, "hash-object", "--", entry.dest_rel)


def short(blob: str | None) -> str:
    return blob[:12] if blob else "missing"


def load_manifest() -> dict | None:
    if not MANIFEST_PATH.is_file():
        return None
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def manifest_blob(manifest: dict, entry: Entry) -> str | None:
    if entry.kind == "module":
        return manifest.get("files", {}).get(entry.key)
    rec = manifest.get("assets", {}).get(entry.key)
    return rec.get("blob_id") if isinstance(rec, dict) else None


def sync(force: bool) -> int:
    commit = resolve_source_commit()
    print(f"source: {TEAM_REPO.name} @ {commit}")
    print(f"dest:   {DEST_REPO}")

    rows = [(e, source_blob_id(commit, e), local_blob_id(e)) for e in ENTRIES]
    check_bad = [(e, s, l) for e, s, l in rows if e.mode == "check" and l != s]
    conflicts = [(e, s, l) for e, s, l in rows if e.mode == "copy" and l is not None and l != s]

    if check_bad:
        print(f"\nREFUSING: {len(check_bad)} check-only file(s) are missing or differ (never written, even with --force):")
        for e, src, loc in check_bad:
            print(f"  {e.dest_rel:<34} local {short(loc)}  source {short(src)}")
        print("Nothing was written.")
        return 1
    if conflicts and not force:
        print(f"\nREFUSING: {len(conflicts)} existing file(s) differ from the source commit:")
        for e, src, loc in conflicts:
            print(f"  {e.dest_rel:<34} local {short(loc)}  source {short(src)}")
        print("Nothing was written. A shared file must never be edited; "
              "re-run with --force to restore the source bytes.")
        return 1

    print(f"\n{'path':<34} {'mode':<6} {'source blob':<13} result")
    for e, src, loc in rows:
        if loc == src:
            result = "unchanged (already identical)" if e.mode == "copy" else "verified identical (check only)"
        else:
            dest = DEST_REPO / e.dest_rel
            if not dest.parent.is_dir():
                raise SyncError(f"destination folder missing: {dest.parent}")
            dest.write_bytes(source_bytes(commit, e))
            after = local_blob_id(e)
            if after != src:
                raise SyncError(f"{e.dest_rel}: wrote source bytes but hash-object gives {after}, expected {src}")
            result = "written" if loc is None else f"OVERWRITTEN with --force (was {short(loc)})"
        print(f"{e.dest_rel:<34} {e.mode:<6} {short(src):<13} {result}")

    files = {e.key: s for e, s, _ in rows if e.kind == "module"}
    assets = {e.key: {"blob_id": s, "mode": e.mode} for e, s, _ in rows if e.kind == "asset"}
    old = load_manifest()
    if old and old.get("source_commit") == commit and old.get("files") == files and old.get("assets") == assets:
        print(f"\nmanifest unchanged: {MANIFEST_PATH.name} (copied_at {old.get('copied_at')})")
    else:
        manifest = {
            "source_repo": git_text(TEAM_REPO, "remote", "get-url", "origin"),
            "source_worktree": TEAM_REPO.name,
            "source_path": SOURCE_DIR,
            "source_commit": commit,
            "identity": IDENTITY,
            "copied_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "files": files,
            "assets": assets,
        }
        with open(MANIFEST_PATH, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(manifest, fh, indent=2)
            fh.write("\n")
        print(f"\nmanifest written: {MANIFEST_PATH.name} ({len(files)} files, {len(assets)} assets)")
    return 0


def check() -> int:
    commit = resolve_source_commit()
    manifest = load_manifest()
    if manifest is None:
        print(f"FAIL: manifest not found: {MANIFEST_PATH}")
        return 1
    problems = 0
    if manifest.get("source_commit") != commit:
        print(f"FAIL: manifest source_commit {manifest.get('source_commit')} != {commit}")
        problems += 1
    if manifest.get("identity") != IDENTITY:
        print(f"FAIL: manifest identity {manifest.get('identity')!r} != {IDENTITY!r}")
        problems += 1
    known_files = {e.key for e in ENTRIES if e.kind == "module"}
    known_assets = {e.key for e in ENTRIES if e.kind == "asset"}
    extra = [k for k in manifest.get("files", {}) if k not in known_files]
    extra += [k for k in manifest.get("assets", {}) if k not in known_assets]
    for k in extra:
        print(f"FAIL: manifest entry {k!r} is not a known shared file")
        problems += 1
    for k, rec in manifest.get("assets", {}).items():
        if k in known_assets and isinstance(rec, dict) and rec.get("mode") != ASSETS[k]:
            print(f"FAIL: manifest mode for {k!r} is {rec.get('mode')!r}, expected {ASSETS[k]!r}")
            problems += 1

    print(f"{'path':<34} {'manifest':<13} {'local':<13} {'source':<13} status")
    n_ok = 0
    for e in ENTRIES:
        man = manifest_blob(manifest, e)
        loc = local_blob_id(e)
        try:
            src = source_blob_id(commit, e)
        except SyncError:
            src = None
        ok = man is not None and man == loc == src
        n_ok += ok
        if ok:
            status = "OK"
        elif man is None:
            status = "FAIL (not in manifest)"
        elif loc is None:
            status = "FAIL (file missing)"
        else:
            status = "FAIL (blob mismatch)"
        print(f"{e.dest_rel:<34} {short(man):<13} {short(loc):<13} {short(src):<13} {status}")

    all_ok = problems == 0 and n_ok == len(ENTRIES)
    n_mod = sum(e.kind == "module" for e in ENTRIES)
    print(f"\n{n_ok}/{len(ENTRIES)} OK ({n_mod} modules + {len(ENTRIES) - n_mod} assets) -> {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="verify only; exit 1 on any mismatch")
    mode.add_argument("--force", action="store_true", help='overwrite "copy" files that differ from the source')
    args = parser.parse_args()
    try:
        return check() if args.check else sync(force=args.force)
    except SyncError as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
