# wcsimfw

A small, fast World Cup simulation framework. Every match model gives a scoreline probability
matrix; the tournament plays the full format many times as batched tensors, on CPU or GPU with
the same code.

```text
data → strength → match (score matrix) → tournament → evaluation
```

## Install

```bash
uv sync            # Python 3.12; PyTorch runs on CPU on macOS, CUDA wheels on Linux
uv run pytest      # tests
uv run ruff check  # lint
```

## Data

`wcsimfw.data.fetch()` downloads a pinned revision of a Hugging Face dataset into a local cache.

| Variable | Meaning | Default |
|---|---|---|
| `WCSIMFW_DATA_REPO` | dataset id | `parthkapur/wcsimfw-data` (private) |
| `WCSIMFW_DATA_REVISION` | dataset commit to read | `wcsimfw.data.REVISION` |
| `WCSIMFW_DATA` | local cache folder | `~/.cache/wcsimfw` |

The dataset is laid out as `raw/<source>/...`, each source with a `SOURCE.md` (origin, dates,
licence) and a `MANIFEST.sha256`.
