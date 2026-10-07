"""Length-prefixed binary records; callers own compression and publication.

The archive families use different byte orders, so the header layout is explicit.
"""

from collections.abc import Iterator
from struct import Struct
from typing import BinaryIO


def write_frame(stream: BinaryIO, data: bytes, layout: Struct) -> int:
    stream.write(layout.pack(len(data)))
    stream.write(data)
    return layout.size + len(data)


def read_frames(stream: BinaryIO, layout: Struct, *, max_frame_bytes: int | None = None) -> Iterator[bytes]:
    while header := stream.read(layout.size):
        if len(header) != layout.size:
            raise ValueError("truncated archive frame header")
        size = layout.unpack(header)[0]
        if max_frame_bytes is not None and size > max_frame_bytes:
            raise ValueError("archive frame exceeds the decoder's frame-size limit")
        data = stream.read(size)
        if len(data) != size:
            raise ValueError("truncated archive frame")
        yield data
