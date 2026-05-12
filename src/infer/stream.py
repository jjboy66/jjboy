from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator


@dataclass
class StreamSource:
    source_id: str
    kind: str  # image | video | rtsp
    uri: str


def iterate_frames(source: StreamSource) -> Iterator[int]:
    """A lightweight frame iterator placeholder for image/video/RTSP modes."""
    if source.kind == "image":
        yield 0
        return
    for idx in range(30):
        yield idx
