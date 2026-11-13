"""Copy the team's shared modules into residual_ppo/ byte-for-byte, and verify them later.

Identity of a shared file = its git blob ID at the source commit, NOT its on-disk SHA-256.
core.autocrlf=true makes working copies CRLF, so on-disk hashes differ from the LF bytes in git
even when the content is identical; `git hash-object` normalizes line endings first.
See docs/milestone1_plan.md, step M1.2.

Usage (normal venv is enough; standard library only):
    python tools/sync_shared.py            copy missing files; refuse if any existing file differs
    python tools/sync_shared.py --force    also overwrite files that differ, with the source bytes
    python tools/sync_shared.py --check    verify every file; exit 1 on any mismatch

Exit codes: 0 OK, 1 mismatch / refused, 2 precondition or git error.
Read-only toward drone_team_main: only `git rev-parse` and `git cat-file` run there.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
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
FILES = [f"{m}.py" for m in MODULES]


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


def source_blob_id(commit: str, name: str) -> str:
    return git_text(TEAM_REPO, "rev-parse", "--verify", f"{commit}:{SOURCE_DIR}/{name}")


def source_bytes(commit: str, name: str) -> bytes:
    # cat-file blob = the exact stored bytes; never applies textconv or eol conversion.
    return git(TEAM_REPO, "cat-file", "blob", f"{commit}:{SOURCE_DIR}/{name}")


def local_blob_id(name: str) -> str | None:
    """Blob ID of residual_ppo/<name> as git would store it (autocrlf applied), or None if missing."""
    if not (RESIDUAL_DIR / name).is_file():
        return None
    return git_text(DEST_REPO, "hash-object", "--", f"{DEST_PREFIX}/{name}")


def short(blob: str | None) -> str:
    return blob[:12] if blob else "missing"


def load_manifest() -> dict | None:
    if not MANIFEST_PATH.is_file():
        return None
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def sync(force: bool) -> int:
    commit = resolve_source_commit()
    print(f"source: {TEAM_REPO.name} @ {commit}  ({SOURCE_DIR}/)")
    print(f"dest:   {RESIDUAL_DIR}")

    rows = [(name, source_blob_id(commit, name), local_blob_id(name)) for name in FILES]
    conflicts = [(n, s, l) for n, s, l in rows if l is not None and l != s]

    if conflicts and not force:
        print(f"\nREFUSING: {len(conflicts)} existing file(s) differ from the source commit:")
        for name, src, loc in conflicts:
            print(f"  {name:<22} local {short(loc)}  source {short(src)}")
        print("Nothing was written. A shared module must never be edited; "
              "re-run with --force to restore the source bytes.")
        return 1

    print(f"\n{'file':<22} {'source blob':<13} result")
    for name, src, loc in rows:
        if loc == src:
            result = "unchanged (already identical)"
        else:
            (RESIDUAL_DIR / name).write_bytes(source_bytes(commit, name))
            after = local_blob_id(name)
            if after != src:
                raise SyncError(f"{name}: wrote source bytes but hash-object gives {after}, expected {src}")
            result = "written" if loc is None else f"OVERWRITTEN with --force (was {short(loc)})"
        print(f"{name:<22} {short(src):<13} {result}")

    files = {name: src for name, src, _ in rows}
    old = load_manifest()
    if old and old.get("source_commit") == commit and old.get("files") == files:
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
        }
        with open(MANIFEST_PATH, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(manifest, fh, indent=2)
            fh.write("\n")
        print(f"\nmanifest written: {MANIFEST_PATH.name} ({len(files)} files)")
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

    man_files: dict = manifest.get("files", {})
    names = FILES + [n for n in man_files if n not in FILES]
    print(f"{'file':<22} {'manifest':<13} {'local':<13} {'source':<13} status")
    n_ok = 0
    for name in names:
        man = man_files.get(name)
        loc = local_blob_id(name)
        try:
            src = source_blob_id(commit, name)
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
        elif name not in FILES:
            status = "FAIL (not a shared module)"
        else:
            status = "FAIL (blob mismatch)"
        print(f"{name:<22} {short(man):<13} {short(loc):<13} {short(src):<13} {status}")

    all_ok = problems == 0 and n_ok == len(names)
    print(f"\n{n_ok}/{len(names)} OK -> {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="verify only; exit 1 on any mismatch")
    mode.add_argument("--force", action="store_true", help="overwrite files that differ from the source")
    args = parser.parse_args()
    try:
        return check() if args.check else sync(force=args.force)
    except SyncError as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
