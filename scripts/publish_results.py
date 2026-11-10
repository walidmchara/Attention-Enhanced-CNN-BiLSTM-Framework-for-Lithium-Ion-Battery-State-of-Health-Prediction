"""Publish a completed CNN-BiLSTM-Attention SOH experiment's small scientific artifacts to GitHub.

Run after training succeeds, from a clean Colab clone. Requires GITHUB_TOKEN
with Contents: Read and write scoped only to this public repository.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

BASE_FILES = {
    "metrics.json",
    "config.json",
    "feature_schema.json",
    "splits.json",
    "scaler.json",
    "test_predictions.csv",
    "training_history.csv",
}
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_TOTAL_BYTES = 20 * 1024 * 1024


def git(root: Path, args: list[str], *, env: dict[str, str] | None = None):
    result = subprocess.run(
        ["git", *args], cwd=root, env=env, text=True,
        capture_output=True,
    )
    if result.returncode:
        # Never print git's command environment or authentication headers.
        raise RuntimeError(f"git {args[0]} failed (exit {result.returncode}): "
                           f"{result.stderr[-1200:]}")
    return result.stdout.strip()


def authenticated_env(token: str) -> dict[str, str]:
    encoded = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    env = os.environ.copy()
    env["GIT_CONFIG_COUNT"] = "1"
    env["GIT_CONFIG_KEY_0"] = "http.https://github.com/.extraheader"
    env["GIT_CONFIG_VALUE_0"] = f"AUTHORIZATION: basic {encoded}"
    return env


def publish(source: Path, run_id: str, *, push: bool = True) -> Path:
    if not RUN_ID_PATTERN.fullmatch(run_id) or run_id in {".", ".."}:
        raise ValueError("Invalid run ID")
    root = Path(git(Path.cwd(), ["rev-parse", "--show-toplevel"])).resolve()
    source = source.resolve()
    if not source.is_dir() or not (source / "metrics.json").is_file():
        raise ValueError("Source must be a completed run containing metrics.json")
    if source.is_symlink():
        raise ValueError("Source directory must not be a symbolic link")
    if push and not os.environ.get("GITHUB_TOKEN"):
        raise RuntimeError("Set GITHUB_TOKEN as a Colab Secret first")
    branch = git(root, ["branch", "--show-current"])
    if not branch:
        raise RuntimeError("Checkout a normal branch before publishing")
    dest = root / "results" / run_id
    if dest.exists():
        raise FileExistsError(f"Run already published locally: {dest}")
    allowed = [p for p in source.iterdir()
               if p.is_file() and not p.is_symlink()
               and (p.name in BASE_FILES or p.suffix.lower() == ".png")]
    if not allowed:
        raise ValueError("No permitted output files found")
    dest.mkdir(parents=True)
    stored, skipped = [], []
    total = 0
    for item in sorted(allowed):
        size = item.stat().st_size
        if size > MAX_FILE_BYTES or total + size > MAX_TOTAL_BYTES:
            skipped.append({"name": item.name, "size_bytes": size, "reason": "size limit"})
            continue
        data_path = dest / item.name
        shutil.copy2(item, data_path)
        stored.append({
            "name": item.name, "size_bytes": size,
            "sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        })
        total += size
    if not any(x["name"] == "metrics.json" for x in stored):
        raise RuntimeError("metrics.json could not be included")
    manifest = {
        "run_id": run_id,
        "published_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": stored,
        "skipped_files": skipped,
        "checkpoint_note": "Full Keras .keras epoch checkpoints remain in persistent Drive/volume, outside ordinary Git history",
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    rel = dest.relative_to(root).as_posix()
    git(root, ["add", "--", rel])
    staged = git(root, ["diff", "--cached", "--name-only"]).splitlines()
    if not staged or any(not path.startswith(rel + "/") for path in staged):
        raise RuntimeError("Unexpected staged changes: publish from a clean clone")
    git(root, [
        "-c", "user.name=Colab Training",
        "-c", "user.email=colab-training@users.noreply.github.com",
        "commit", "-m", f"Record CNN-BiLSTM-Attention SOH results: {run_id}",
    ])
    if push:
        git(root, ["push", "origin", f"HEAD:{branch}"],
            env=authenticated_env(os.environ["GITHUB_TOKEN"]))
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--no-push", action="store_true", help="Only commit locally")
    args = parser.parse_args()
    dest = publish(args.source, args.run_id, push=not args.no_push)
    print(f"Recorded results: {dest.as_posix()}")
    print("Checkpoints and datasets were not committed.")


if __name__ == "__main__":
    main()
