from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch


@contextmanager
def nvtx_range(message: str) -> Iterator[None]:
    pushed = False
    try:
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push(str(message))
            pushed = True
    except Exception:
        pushed = False
    try:
        yield
    finally:
        if pushed:
            try:
                torch.cuda.nvtx.range_pop()
            except Exception:
                pass
