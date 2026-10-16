"""Pure-Python Git commit helper using dulwich for environments without git.exe on PATH."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

from dulwich import porcelain
from dulwich.repo import Repo


def commit_paths(message: str, paths: Optional[List[str]] = None) -> str:
    """Stage paths (or all tracked/untracked non-ignored files) and commit to .git."""
    repo_root = Path.cwd()
    if not (repo_root / ".git").exists():
        porcelain.init(str(repo_root))
    repo = Repo(str(repo_root))
    if paths:
        porcelain.add(repo, paths=paths)
    else:
        # Stage all non-ignored files
        to_add: List[str] = []
        exclude_prefixes = (".git", ".venv", "data/raw", "data/interim", "data/processed", "__pycache__", ".pytest_cache")
        for p in repo_root.rglob("*"):
            if p.is_file():
                rel = p.relative_to(repo_root).as_posix()
                if not any(rel == pref or rel.startswith(pref + "/") for pref in exclude_prefixes):
                    if not rel.endswith((".pyc", ".pyo")):
                        to_add.append(str(p))
        if to_add:
            porcelain.add(repo, paths=to_add)

    sha = porcelain.commit(
        repo,
        message=message.encode("utf-8"),
        author=b"Antigravity Benchmark <benchmark@tsdomain.local>",
        committer=b"Antigravity Benchmark <benchmark@tsdomain.local>",
    )
    return sha.decode("ascii")


if __name__ == "__main__":
    msg = sys.argv[1] if len(sys.argv) > 1 else "Update benchmark repository"
    file_args = sys.argv[2:] if len(sys.argv) > 2 else None
    commit_sha = commit_paths(msg, file_args)
    print(f"Committed {commit_sha}: {msg}")
