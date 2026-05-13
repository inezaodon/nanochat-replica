"""
Download ND CSE 10124 lecture release weights (GitHub Release assets).

Same URL layout as the course instructions:
  https://github.com/wtheisen/nd-cse-10124-lectures/releases/download/weights-v1/<file>.pt
"""

from __future__ import annotations

import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

# GitHub often returns 403 for the default Python urllib User-Agent on release assets.
_USER_AGENT = (
    "nanochat-replica/1.0 (llm.course_weights; +https://github.com/inezaodon/nanochat-replica)"
)
_MIN_CHECKPOINT_BYTES = 50_000_000  # sanity: real gpt2_small_converted.pt is hundreds of MB

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


def _download_url_to_file(url: str, dest: Path, *, timeout_sec: int = 600) -> None:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": _USER_AGENT,
            "Accept": "*/*",
        },
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    for attempt in range(4):
        try:
            if partial.exists():
                partial.unlink()
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                with open(partial, "wb") as out:
                    shutil.copyfileobj(resp, out, length=1024 * 1024)
            sz = partial.stat().st_size
            if sz < _MIN_CHECKPOINT_BYTES:
                raise RuntimeError(
                    f"Downloaded file is only {sz} bytes; expected a large .pt checkpoint "
                    f"(min {_MIN_CHECKPOINT_BYTES}). The server may have returned an error page."
                )
            partial.replace(dest)
            return
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code in (429, 503) and attempt < 3:
                time.sleep(2**attempt)
                continue
            raise


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
    _download_url_to_file(url, dest)
    return dest
