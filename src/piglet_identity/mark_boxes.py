"""Draw one box around the painted number on each labeled crop, then export YOLO data.

The pig ID is already known for these crops, so labeling is a single drag per image.
Boxes are stored in a resumable CSV and exported as a YOLO detection dataset.
"""

from __future__ import annotations

import csv
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import List, Optional

BOX_FIELDS = ["image_path", "pig_id", "x1", "y1", "x2", "y2", "status"]
DISPLAY_MAX_SIDE = 720


def _read_boxes(boxes_csv: Path) -> dict:
    if not boxes_csv.exists():
        return {}
    with boxes_csv.open(newline="") as handle:
        return {row["image_path"]: row for row in csv.DictReader(handle)}


def _write_boxes(boxes_csv: Path, rows: List[dict]) -> None:
    boxes_csv.parent.mkdir(parents=True, exist_ok=True)
    with boxes_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BOX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _labeled_crops(labels_csv: Path) -> List[dict]:
    rows = []
    with labels_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            pig_id = (row.get("pig_id") or "").strip()
            if row.get("status") != "labeled" or not pig_id.isdigit():
                continue
            if row.get("source_type") == "reference":
                continue
            if not Path(row["image_path"]).exists():
                continue
            rows.append({"image_path": row["image_path"], "pig_id": pig_id})
    return rows


def label_mark_boxes(
    labels_csv: Path,
    boxes_csv: Path,
    limit: Optional[int] = None,
    shuffle: bool = True,
) -> dict:
    """Drag a box around the number. Enter=save, s=skip, u=undo, q=quit."""
    import cv2

    crops = _labeled_crops(labels_csv)
    if shuffle:
        random.Random(0).shuffle(crops)
    existing = _read_boxes(boxes_csv)
    pending = [row for row in crops if row["image_path"] not in existing]
    if limit is not None:
        pending = pending[:limit]
    if not pending:
        print(f"No crops left to box ({len(existing)} already done).")
        return {"boxes_csv": str(boxes_csv), "pending": 0, "boxed": 0}

    print(f"{len(pending)} crops to box ({len(existing)} already done).")
    print("Drag a box around the painted number. Enter=save  s=skip  u=undo  q=quit")

    state = {"start": None, "box": None, "drawing": False}

    def on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["start"] = (x, y)
            state["drawing"] = True
            state["box"] = None
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            state["box"] = (*state["start"], x, y)
        elif event == cv2.EVENT_LBUTTONUP and state["drawing"]:
            state["drawing"] = False
            state["box"] = (*state["start"], x, y)

    window = "Draw box around the number"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    boxed = skipped = 0
    quit_early = False
    for index, row in enumerate(pending):
        image = cv2.imread(row["image_path"])
        if image is None:
            continue
        height, width = image.shape[:2]
        scale = min(DISPLAY_MAX_SIDE / max(height, width), 4.0)
        display_size = (int(width * scale), int(height * scale))
        state["box"] = None
        while True:
            canvas = cv2.resize(image, display_size)
            if state["box"]:
                x1, y1, x2, y2 = state["box"]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                canvas,
                f"[{index + 1}/{len(pending)}] pig {row['pig_id']}",
                (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.imshow(window, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10):  # Enter
                if not state["box"]:
                    continue
                x1, y1, x2, y2 = state["box"]
                if abs(x2 - x1) < 6 or abs(y2 - y1) < 6:
                    state["box"] = None
                    continue
                existing[row["image_path"]] = {
                    "image_path": row["image_path"],
                    "pig_id": row["pig_id"],
                    "x1": round(min(x1, x2) / scale),
                    "y1": round(min(y1, y2) / scale),
                    "x2": round(max(x1, x2) / scale),
                    "y2": round(max(y1, y2) / scale),
                    "status": "boxed",
                }
                boxed += 1
                break
            if key == ord("s"):
                existing[row["image_path"]] = {
                    "image_path": row["image_path"],
                    "pig_id": row["pig_id"],
                    "x1": "",
                    "y1": "",
                    "x2": "",
                    "y2": "",
                    "status": "skip",
                }
                skipped += 1
                break
            if key == ord("u"):
                state["box"] = None
            if key == ord("q"):
                quit_early = True
                break
        _write_boxes(boxes_csv, list(existing.values()))
        if quit_early:
            break

    cv2.destroyAllWindows()
    total_boxed = sum(1 for row in existing.values() if row["status"] == "boxed")
    return {
        "boxes_csv": str(boxes_csv.resolve()),
        "boxed_this_session": boxed,
        "skipped_this_session": skipped,
        "total_boxed": total_boxed,
        "remaining": len(_labeled_crops(labels_csv)) - len(existing),
    }


def export_mark_dataset(
    boxes_csv: Path,
    output_directory: Path,
    validation_fraction: float = 0.2,
    single_class: bool = False,
    exclude_ids: Optional[List[str]] = None,
) -> dict:
    """Write a YOLO detection dataset from the boxed crops."""
    import cv2

    from .ocr import load_allowed_ids

    rows = [row for row in _read_boxes(boxes_csv).values() if row["status"] == "boxed"]
    if not rows:
        raise ValueError(f"No boxed rows in {boxes_csv}. Run: piglet-id label-marks")

    # Typos such as "1010" would otherwise become a real YOLO class.
    roster = load_allowed_ids()
    dropped = set(exclude_ids or [])
    excluded = Counter()
    kept = []
    for row in rows:
        pig_id = row["pig_id"]
        if pig_id in dropped or pig_id not in roster:
            excluded[pig_id] += 1
            continue
        kept.append(row)
    rows = kept
    if not rows:
        raise ValueError("Every boxed row was excluded; check pig IDs against the roster")

    if single_class:
        names = ["mark"]
        class_of = lambda _row: 0  # noqa: E731
    else:
        names = sorted({row["pig_id"] for row in rows}, key=int)
        index = {name: position for position, name in enumerate(names)}
        class_of = lambda row: index[row["pig_id"]]  # noqa: E731

    # Hold out whole source videos so nearby frames cannot leak into validation.
    videos = sorted({Path(row["image_path"]).parts[-2] for row in rows})
    rng = random.Random(42)
    rng.shuffle(videos)
    holdout = set(videos[: max(1, round(len(videos) * validation_fraction))])

    if output_directory.exists():
        shutil.rmtree(output_directory)
    counts = Counter()
    for row in rows:
        source = Path(row["image_path"])
        split = "val" if source.parts[-2] in holdout else "train"
        image = cv2.imread(str(source))
        if image is None:
            continue
        height, width = image.shape[:2]
        x1, y1 = int(row["x1"]), int(row["y1"])
        x2, y2 = int(row["x2"]), int(row["y2"])
        cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        bw, bh = abs(x2 - x1) / width, abs(y2 - y1) / height

        stem = f"{source.parts[-2]}__{source.stem}".replace(" ", "_")
        image_directory = output_directory / "images" / split
        label_directory = output_directory / "labels" / split
        image_directory.mkdir(parents=True, exist_ok=True)
        label_directory.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, image_directory / f"{stem}{source.suffix}")
        (label_directory / f"{stem}.txt").write_text(
            f"{class_of(row)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
        )
        counts[split] += 1

    data_yaml = output_directory / "data.yaml"
    data_yaml.write_text(
        "path: {path}\ntrain: images/train\nval: images/val\nnames:\n{names}\n".format(
            path=output_directory.resolve(),
            names="\n".join(f"  {i}: {name}" for i, name in enumerate(names)),
        )
    )
    return {
        "dataset": str(output_directory.resolve()),
        "data_yaml": str(data_yaml.resolve()),
        "train_images": counts["train"],
        "val_images": counts["val"],
        "classes": names,
        "validation_videos": sorted(holdout),
        "excluded_ids": dict(excluded),
    }


def score_mark_detector(
    weights: Path,
    labels_csv: Path,
    confidence: float = 0.25,
    imgsz: int = 320,
) -> dict:
    """Read pig IDs on labeled crops with the mark detector and score against the labels.

    Needs no boxes on the test crops: the highest-confidence detection's class is the
    predicted pig ID, so any labeled crop set works as a held-out test.
    """
    from collections import defaultdict

    from ultralytics import YOLO

    rows = _labeled_crops(labels_csv)
    if not rows:
        raise ValueError(f"No labeled crops in {labels_csv}")

    model = YOLO(str(weights))
    names = model.names
    known = {str(name) for name in names.values()}

    predictions = []
    for start in range(0, len(rows), 32):
        batch = rows[start : start + 32]
        results = model.predict(
            [row["image_path"] for row in batch],
            imgsz=imgsz,
            conf=confidence,
            verbose=False,
        )
        for row, result in zip(batch, results):
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                predictions.append((row, None, 0.0))
                continue
            best = int(boxes.conf.argmax())
            predictions.append(
                (row, str(names[int(boxes.cls[best])]), float(boxes.conf[best]))
            )

    detected = [item for item in predictions if item[1] is not None]
    correct = sum(1 for row, pig_id, _ in detected if pig_id == row["pig_id"])

    # Classes the detector was never trained on can never be right; report both views.
    in_vocabulary = [item for item in predictions if item[0]["pig_id"] in known]
    vocabulary_correct = sum(
        1 for row, pig_id, _ in in_vocabulary if pig_id == row["pig_id"]
    )

    votes: dict[str, list] = defaultdict(list)
    truth: dict[str, str] = {}
    for row, pig_id, score in predictions:
        source = Path(row["image_path"])
        track = f"{source.parts[-2]}:{source.stem.split('_f')[0]}"
        truth[track] = row["pig_id"]
        if pig_id is not None:
            votes[track].append((pig_id, score))
    track_correct = 0
    for track, entries in votes.items():
        totals: dict[str, float] = defaultdict(float)
        for pig_id, score in entries:
            totals[pig_id] += score
        if max(totals.items(), key=lambda kv: kv[1])[0] == truth[track]:
            track_correct += 1

    return {
        "weights": str(weights),
        "labels_csv": str(labels_csv),
        "images": len(rows),
        "detector_classes": sorted(known, key=int),
        "coverage": round(len(detected) / len(rows), 4),
        "image_accuracy": round(correct / len(rows), 4),
        "image_accuracy_on_detected": round(correct / max(1, len(detected)), 4),
        "images_in_detector_vocabulary": len(in_vocabulary),
        "image_accuracy_in_vocabulary": round(
            vocabulary_correct / max(1, len(in_vocabulary)), 4
        ),
        "tracks": len(votes),
        "track_accuracy": round(track_correct / max(1, len(votes)), 4),
    }


def train_mark_detector(
    data_yaml: Path,
    output_directory: Path,
    epochs: int = 80,
    imgsz: int = 320,
    base_model: str = "yolov8n.pt",
) -> dict:
    from ultralytics import YOLO

    model = YOLO(base_model)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        project=str(output_directory.parent),
        name=output_directory.name,
        exist_ok=True,
        degrees=180.0,  # marks appear at every orientation
        fliplr=0.0,  # never mirror digits
        flipud=0.0,
    )
    metrics = getattr(results, "results_dict", {}) or {}
    weights = output_directory / "weights" / "best.pt"
    return {
        "weights": str(weights),
        "epochs": epochs,
        "imgsz": imgsz,
        "metrics": {key: round(float(value), 4) for key, value in metrics.items()},
    }
