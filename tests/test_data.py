from pathlib import Path

import pytest

from wcsimfw import data


def test_fetch_requires_revision(monkeypatch):
    monkeypatch.setattr(data, "REVISION", None)
    monkeypatch.delenv("WCSIMFW_DATA_REVISION", raising=False)
    with pytest.raises(RuntimeError, match="REVISION"):
        data.fetch()


def test_fetch_passes_pin(monkeypatch, tmp_path):
    seen = {}

    def fake(repo_id, **kw):
        seen.update(kw, repo_id=repo_id)
        return str(tmp_path / "snap")

    monkeypatch.setattr(data, "snapshot_download", fake)
    monkeypatch.setattr(data, "REVISION", "abc123")
    monkeypatch.setenv("WCSIMFW_DATA", str(tmp_path))
    monkeypatch.delenv("WCSIMFW_DATA_REPO", raising=False)
    monkeypatch.delenv("WCSIMFW_DATA_REVISION", raising=False)
    out = data.fetch(["raw/eafc/*"])
    assert out == tmp_path / "snap"
    assert seen == {
        "repo_id": "parthkapur/wcsimfw-data", "repo_type": "dataset", "revision": "abc123",
        "allow_patterns": ["raw/eafc/*"], "cache_dir": tmp_path,
    }
    data.fetch(revision="def456")
    assert seen["revision"] == "def456"
    monkeypatch.setenv("WCSIMFW_DATA_REPO", "someone/other-data")
    monkeypatch.setenv("WCSIMFW_DATA_REVISION", "env789")
    data.fetch()
    assert (seen["repo_id"], seen["revision"]) == ("someone/other-data", "env789")


def test_data_dir_default(monkeypatch):
    monkeypatch.delenv("WCSIMFW_DATA", raising=False)
    assert data.data_dir() == Path.home() / ".cache" / "wcsimfw"


def test_processed_prefers_local_then_dataset_then_builds(monkeypatch, tmp_path):
    monkeypatch.setenv("WCSIMFW_DATA", str(tmp_path))
    monkeypatch.setattr(data, "fetch", lambda patterns: tmp_path / "snap")
    built = []
    local, remote = tmp_path / "processed/x-v1", tmp_path / "snap/processed/x-v1"
    assert data.processed("x-v1", lambda out: built.append(out)) == local and built == [local]
    remote.mkdir(parents=True)
    (remote / "manifest.json").write_text("{}")
    assert data.processed("x-v1", built.append) == remote
    local.mkdir(parents=True)
    (local / "manifest.json").write_text("{}")
    assert data.processed("x-v1", built.append) == local and len(built) == 1
