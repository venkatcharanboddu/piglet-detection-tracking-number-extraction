import csv
from pathlib import Path

import numpy as np
import pytest

from piglet_identity.mark_boxes import BOX_FIELDS, export_mark_dataset


def _dataset(tmp_path: Path) -> Path:
    cv2 = pytest.importorskip("cv2")
    boxes_csv = tmp_path / "mark_boxes.csv"
    rows = []
    for video in ("video_a", "video_b"):
        folder = tmp_path / "samples" / video
        folder.mkdir(parents=True)
        for track in range(3):
            image_path = folder / f"track_{track}.jpg"
            cv2.imwrite(str(image_path), np.full((100, 200, 3), 128, dtype=np.uint8))
            rows.append(
                {
                    "image_path": str(image_path),
                    "pig_id": "11" if track % 2 else "3",
                    "x1": 50,
                    "y1": 20,
                    "x2": 90,
                    "y2": 60,
                    "status": "boxed",
                }
            )
    with boxes_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BOX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return boxes_csv


def test_export_writes_normalized_yolo_labels(tmp_path: Path) -> None:
    boxes_csv = _dataset(tmp_path)
    result = export_mark_dataset(boxes_csv, tmp_path / "out")

    assert result["classes"] == ["3", "11"]
    assert result["train_images"] + result["val_images"] == 6
    # whole videos are held out, never individual frames
    assert len(result["validation_videos"]) == 1

    label = next((tmp_path / "out" / "labels").rglob("*.txt")).read_text().split()
    class_id, cx, cy, bw, bh = label
    assert class_id in {"0", "1"}
    assert float(cx) == pytest.approx(70 / 200)
    assert float(cy) == pytest.approx(40 / 100)
    assert float(bw) == pytest.approx(40 / 200)
    assert float(bh) == pytest.approx(40 / 100)


def test_export_single_class_collapses_ids(tmp_path: Path) -> None:
    boxes_csv = _dataset(tmp_path)
    result = export_mark_dataset(boxes_csv, tmp_path / "out", single_class=True)

    assert result["classes"] == ["mark"]
    classes = {
        path.read_text().split()[0]
        for path in (tmp_path / "out" / "labels").rglob("*.txt")
    }
    assert classes == {"0"}


def test_export_requires_boxes(tmp_path: Path) -> None:
    empty = tmp_path / "mark_boxes.csv"
    with empty.open("w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=BOX_FIELDS).writeheader()
    with pytest.raises(ValueError, match="No boxed rows"):
        export_mark_dataset(empty, tmp_path / "out")
