"""Display encoded storage sizes without confusing them with decoded memory."""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(slots=True)
class BlobSizes:
    stored_bytes: int = 0
    uncompressed_bytes: int | None = 0
    largest_uncompressed_bytes: int | None = 0


def report_file_sizes(path: Path, uncompressed_bytes: int, *, kind: Literal["stream", "members"] = "stream") -> None:
    """Report after publication; callers count raw bytes during encoding/writing.

    Stream sizes include framing and any still-compressed embedded components.
    ZIP member sizes include their .npy headers, not ZIP container overhead.
    Neither measures decoded Python objects or native graph allocations.
    """
    stored = path.stat().st_size
    prefix = "\n" if sys.stderr.isatty() else ""
    print(
        f"{prefix}{path.name}: stored {stored:,} bytes ({stored / 2**20:.2f} MiB); "
        f"uncompressed {kind} {uncompressed_bytes:,} bytes ({uncompressed_bytes / 2**20:.2f} MiB)",
        file=sys.stderr,
    )
