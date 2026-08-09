"""The VRAM exclusivity invariant, in one place.

Peak memory must be one model, never a sum. Every pass that loads something
large releases it through here before loading the next thing, and checkpoints
in between — not just at the end of the pass.

torch is imported lazily throughout: this module is imported by db.py, which
has to work on a laptop with no torch installed at all.
"""

from __future__ import annotations

import gc
import logging
from contextlib import contextmanager
from typing import Any, Iterator

log = logging.getLogger(__name__)


def _torch_cuda():
    """Return torch.cuda if a GPU is actually usable, else None."""
    try:
        import torch
    except ImportError:
        return None
    return torch.cuda if torch.cuda.is_available() else None


def free() -> None:
    """Release anything unreferenced and hand the memory back to the driver.

    Call this immediately after `del`-ing a model. Python dropping the last
    reference is not enough on its own — the caching allocator holds the blocks
    until empty_cache(), so the next model sees a full GPU and OOMs.
    """
    gc.collect()
    cuda = _torch_cuda()
    if cuda is not None:
        cuda.empty_cache()


def reset_peak() -> None:
    """Start a fresh peak-memory measurement. Called at the top of every stage."""
    cuda = _torch_cuda()
    if cuda is not None:
        cuda.reset_peak_memory_stats()


def peak_mb() -> float | None:
    """Peak allocated VRAM since the last reset, or None without a GPU."""
    cuda = _torch_cuda()
    if cuda is None:
        return None
    return cuda.max_memory_allocated() / 2**20


@contextmanager
def loaded(model: Any) -> Iterator[Any]:
    """Hold a model for the duration of a block, then release it.

    Use as `with vram.loaded(load_whisper(...)) as model:` — passing the loader's
    result straight in, without binding it to a name in the caller, so that when
    this releases its reference there is genuinely no other one left.
    """
    try:
        yield model
    finally:
        del model
        free()
        log.debug("released model, peak was %s MB", peak_mb())
