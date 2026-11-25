"""Read the raw data: one pinned revision of a Hugging Face dataset, cached on this machine.

Football sentence: every run reads the same match sheets, so two runs that differ only in the
model give results that differ only because of the model.
Pipeline place: data (the first step). Nothing is fitted; the revision is fixed by hand.

Settings (environment):
- `WCSIMFW_DATA_REPO`: dataset id (default `parthkapur/wcsimfw-data`, private: log in with
  `hf auth login` or set `HF_TOKEN`).
- `WCSIMFW_DATA_REVISION`: commit to read (default `REVISION` below).
- `WCSIMFW_DATA`: local cache folder (default `~/.cache/wcsimfw`).

Layout: `raw/<source>/...`, each with `SOURCE.md` and `MANIFEST.sha256`, and
`processed/<artifact>/...` (tables made by `wcsimfw.pipelines`, with `manifest.json`). An artifact
name carries a version, so tables made by older code are never read by newer code.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "parthkapur/wcsimfw-data"
REVISION: str | None = "0296912864dd6147febcc23f367fb209324adae4"  # dataset commit all runs read


def data_dir() -> Path:
    """Local cache folder: $WCSIMFW_DATA, else ~/.cache/wcsimfw."""
    return Path(os.environ.get("WCSIMFW_DATA", Path.home() / ".cache" / "wcsimfw"))


def fetch(patterns: list[str] | None = None, revision: str | None = None) -> Path:
    """Download (or reuse) the pinned snapshot; `patterns` limits it, e.g. ["raw/eafc/*"]."""
    revision = revision or os.environ.get("WCSIMFW_DATA_REVISION") or REVISION
    if revision is None:
        raise RuntimeError(
            "no data revision: set WCSIMFW_DATA_REVISION or wcsimfw.data.REVISION to a commit sha"
        )
    return Path(
        snapshot_download(
            os.environ.get("WCSIMFW_DATA_REPO", REPO_ID), repo_type="dataset", revision=revision,
            allow_patterns=patterns, cache_dir=data_dir(),
        )
    )


def processed(artifact: str, build: Callable[..., Path]) -> Path:
    """Folder of a processed artifact: the local one, else the dataset's, else built now."""
    local = data_dir() / "processed" / artifact
    if (local / "manifest.json").exists():
        return local
    remote = fetch([f"processed/{artifact}/*"]) / "processed" / artifact
    if (remote / "manifest.json").exists():
        return remote
    build(out=local)
    return local
