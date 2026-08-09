"""Dataclasses mirroring the SQLite tables in db.py.

Field names are column names — db.insert() builds its INSERT from
dataclasses.fields(), so the two must not drift. `id` is excluded from
inserts (SQLite assigns it).

Times are seconds (REAL) on the one clock established at ingest. Never
store a frame index here; store the derived time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True, slots=True)
class Video:
    table: ClassVar[str] = "videos"
    video_id: str  # YouTube ID — natural key, stable across re-ingest
    url: str
    added_at: float
    title: str | None = None
    duration_s: float | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    chapters: str | None = None  # JSON array from YouTube, or NULL; pass 2 reads it for L1
    is_vfr: int | None = None  # 1 if declared and average frame rates disagree — see media.py


@dataclass(frozen=True, slots=True)
class Shot:
    table: ClassVar[str] = "shots"
    video_id: str
    t_start: float
    t_end: float
    conf: float | None = None
    id: int | None = None


@dataclass(frozen=True, slots=True)
class Word:
    table: ClassVar[str] = "words"
    video_id: str
    t_start: float
    t_end: float
    text: str
    conf: float | None = None
    id: int | None = None


@dataclass(frozen=True, slots=True)
class Sentence:
    table: ClassVar[str] = "sentences"
    video_id: str
    t_start: float
    t_end: float
    text: str
    speaker: str | None = None
    id: int | None = None


@dataclass(frozen=True, slots=True)
class Unit:
    table: ClassVar[str] = "units"
    video_id: str
    level: int  # 0 = clip, 1 = scene
    t_start: float
    t_end: float
    strategy: str  # fixed | fixed_overlap | snapped
    parent_id: int | None = None
    flags: str | None = None  # JSON array, e.g. ["silent"] — retrieval reads this
    id: int | None = None


@dataclass(frozen=True, slots=True)
class Caption:
    table: ClassVar[str] = "captions"
    unit_id: int
    captioner: str  # producer + version, e.g. "qwen2.5-vl-3b/v1" — lets outputs coexist
    payload: str  # whole caption JSON; read fields with json_extract, never migrate
    id: int | None = None


@dataclass(frozen=True, slots=True)
class OcrSpan:
    table: ClassVar[str] = "ocr_spans"
    video_id: str
    t_start: float
    t_end: float
    text: str
    conf: float | None = None
    text_hash: str | None = None  # normalized-text hash, for dedup within a scene
    id: int | None = None
