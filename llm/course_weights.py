"""
Download ND CSE 10124 lecture release weights (GitHub Release assets).

Same URL layout as the course instructions:
  https://github.com/wtheisen/nd-cse-10124-lectures/releases/download/weights-v1/<file>.pt
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

DOWNLOAD_BASE = (
    "https://github.com/wtheisen/nd-cse-10124-lectures/releases/download/weights-v1"
)

COURSE_WEIGHT_FILES = (
    "gpt2_small_converted.pt",
    "gpt_2_small_sft.pt",
    "gpt2_small_toolcall_sft.pt",
)


def default_weights_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "weights"


def course_checkpoint_url(filename: str) -> str:
    if filename not in COURSE_WEIGHT_FILES:
        raise ValueError(f"Unknown weight file {filename!r}; expected one of {COURSE_WEIGHT_FILES}")
    return f"{DOWNLOAD_BASE}/{filename}"


def ensure_course_checkpoint(
    filename: str = "gpt2_small_converted.pt",
    *,
    dest_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """
    Download the release .pt into dest_dir (default: repo /weights) if missing.
    Returns the path to the local file.
    """
    if filename not in COURSE_WEIGHT_FILES:
        raise ValueError(f"Unknown weight file {filename!r}; expected one of {COURSE_WEIGHT_FILES}")
    root = dest_dir if dest_dir is not None else default_weights_dir()
    root.mkdir(parents=True, exist_ok=True)
    dest = root / filename
    if dest.exists() and not force:
        return dest

    url = course_checkpoint_url(filename)
    print(f"[course_weights] downloading\n  {url}\n  -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    return dest
