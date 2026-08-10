from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class VideoRecord:
    path: str
    pen: Optional[int]
    ear_tag: Optional[int]
    date: str
    size_bytes: int
    view: str


def infer_pen(path: Path) -> tuple[Optional[int], Optional[int]]:
    text = str(path)
    if "Pen4_90807" in text:
        return 4, 90807
    if "Pen5_90808" in text:
        return 5, 90808
    if "Pen6_90809" in text:
        return 6, 90809
    return None, None


def infer_date(path: Path) -> str:
    for part in reversed(path.parts):
        digits = "".join(char for char in part if char.isdigit())
        if len(digits) >= 8 and digits[:8].startswith("20"):
            return digits[:8]
    name = path.name
    return name[:10].replace("-", "") if len(name) >= 10 else ""


def infer_view(path: Path) -> str:
    name = path.name
    if "#S#H" in name:
        return "stitched"
    if "#H" in name:
        return "fixed"
    if "#M" in name:
        return "motion"
    return "unknown"


def scan_videos(data_root: Path) -> List[VideoRecord]:
    records = []
    for path in sorted(data_root.rglob("*.mp4")):
        pen, ear_tag = infer_pen(path)
        records.append(
            VideoRecord(
                path=str(path.resolve()),
                pen=pen,
                ear_tag=ear_tag,
                date=infer_date(path),
                size_bytes=path.stat().st_size,
                view=infer_view(path),
            )
        )
    return records


def select_representative(
    records: Iterable[VideoRecord], per_pen: int = 3
) -> List[VideoRecord]:
    """Select small/median/large files across dates for each pen."""
    selected: List[VideoRecord] = []
    by_pen: dict[Optional[int], list[VideoRecord]] = {}
    for record in records:
        by_pen.setdefault(record.pen, []).append(record)
    for pen_records in by_pen.values():
        ordered = sorted(pen_records, key=lambda item: (item.date, item.size_bytes))
        if len(ordered) <= per_pen:
            selected.extend(ordered)
            continue
        indexes = {
            round(index * (len(ordered) - 1) / (per_pen - 1))
            for index in range(per_pen)
        }
        selected.extend(ordered[index] for index in sorted(indexes))
    return sorted(selected, key=lambda item: (item.pen or 0, item.date, item.path))


def write_manifest(records: Iterable[VideoRecord], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(record) for record in records]
    if output.suffix.lower() == ".json":
        output.write_text(json.dumps(rows, indent=2) + "\n")
        return
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(VideoRecord.__annotations__))
        writer.writeheader()
        writer.writerows(rows)


def stable_video_key(video: Path) -> str:
    digest = hashlib.sha1(str(video.resolve()).encode()).hexdigest()[:10]
    return f"{video.stem[:50]}-{digest}"
