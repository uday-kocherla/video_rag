# video-rag

Retrieval over mixed, unstructured YouTube video. Two query types:

- **Moment retrieval** — "find the part where X happens" → ranked time spans, scored on temporal IoU.
- **Content QA** — "what did they say about X" → an answer with timestamp citations.

Not a transcript search engine. Speech is one of four signals; the system is built to work
on videos where the transcript is empty or useless — silent demos, slide decks, rapid-cut
edits.

## Status

Milestone **M3** — the transcript-only baseline, shipped and scored end to end.

| Pass | Does | State |
|---|---|---|
| 0 · ingest | download, shot-detect at native fps, 16 kHz audio + 2 fps proxy | ✅ |
| 1 · speech | faster-whisper ASR, then wav2vec2 forced alignment | ✅ |
| 2 · segment | L0 clips (12–20s) and L1 scenes (2–5 min) | ✅ `fixed` only |
| 3 · visual | VLM captions, SigLIP frame vectors, OCR | not built |
| 4 · index | BM25 over clip speech (SQLite FTS5) | ✅ transcript only |
| 5 · retrieve | RRF fusion + span merging | ✅ single-shot |

Still to come: the visual pass, the other three indices, the agentic loop, and an MCP server.

## Requirements

Python 3.11+, plus two system binaries that pip cannot install:

- **ffmpeg / ffprobe** — audio extraction and the proxy. Present on Kaggle and Colab.
- **deno ≥ 2.3** — YouTube's `n` challenge is JavaScript and yt-dlp needs a runtime to
  solve it. Node works too but must be ≥ 22; the system node on most boxes is older.
  `curl -fsSL https://deno.land/install.sh | sh`

```bash
python -m venv .venv && .venv/bin/pip install -e .
```

## Setup

**1. Cookies.** YouTube blocks datacenter IPs with "Sign in to confirm you're not a bot",
which applies to this box, to Kaggle and to Colab alike. Export a Netscape-format cookie
jar from a logged-in browser to `cookies.txt` (gitignored). Use a throwaway Google account
— YouTube flags accounts whose cookies drive bulk downloading.

**2. Artifact store.** A private Hugging Face dataset repo holds everything durable.

```bash
.venv/bin/hf auth login          # token lands in ~/.cache/huggingface/token
```

Then set `repo_id` in `configs/default.yaml`. On Kaggle put the token in Secrets, on Colab
in the secrets panel — never in the config file, which is tracked.

## Usage

```python
from vrag import config, db, sync
from vrag.passes import p0_ingest, p1_speech, p2_segment, p4_index
from vrag.retrieve import search as retrieve

cfg = config.load_config()
local = sync.pull(cfg["repo_id"], cfg["local_dir"])
conn = db.connect(sync.db_path(local))

p0_ingest.run_corpus(conn, ["https://youtu.be/..."], cfg, local)
p1_speech.run_corpus(conn, cfg, local)
p2_segment.run_corpus(conn, cfg)
p4_index.run_corpus(conn, cfg, local)

index = p4_index.open_index(sync.bm25_path(local))
for span in retrieve.search(conn, index, "what did they say about espresso", cfg):
    print(f"{span.video_id} [{span.t_start:.1f}-{span.t_end:.1f}] {span.score:.4f}")

sync.push(cfg["repo_id"], local, conn=conn)   # conn so the WAL is checkpointed first
```

Every pass is idempotent: it checks `stage_runs`, skips videos already done under the same
config hash, and appends. Kill a run halfway and re-run it.

## Layout

```
configs/default.yaml   every threshold, tolerance and model name
vrag/
  db.py                SQLite schema, WAL setup, stage tracking
  schema.py            dataclasses mirroring the tables
  sync.py              Hugging Face push/pull, WAL checkpoint guard
  media.py, vram.py    ffmpeg wrappers; load/run/free discipline
  models/              per-model load-unload wrappers
  passes/              p0_ingest, p1_speech, p2_segment, p4_index
  retrieve/            fusion.py (RRF + span merging), search.py
tests/                 CPU-only, no network, no model downloads
```

## Invariants

Four rules the code enforces and the tests check:

- **SQLite is the source of truth.** WAL mode, on local disk only — never a mounted
  filesystem, where broken file locking will corrupt it.
- **The index is disposable.** `index/bm25.sqlite` is a separate file from the corpus and
  must be fully rebuildable by re-running pass 4 alone.
- **One clock.** The proxy is constant frame rate, so `frame_index / fps` is exact. No
  timestamp is ever derived from a variable-frame-rate source. Shot detection runs at
  ingest, on the original file, at native frame rate — before that file is deleted.
- **Raw video never leaves ephemeral disk.** It downloads into a temp directory and is
  regenerable from its URL. Captions are not, so those are what get uploaded.

## Tests

```bash
.venv/bin/python -m pytest
```

CPU-only, no network, no model downloads. Segmentation and fusion run against synthetic
shot and sentence fixtures; the index tests use real SQLite and real FTS5.
