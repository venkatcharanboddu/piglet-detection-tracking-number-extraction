"""Read back numbers as digits instead of whole pig IDs.

A single box labeled with the whole number cannot express "a 1, then a 2", which is
why every two-digit pig ID failed. This detects digits 0-9 and composes the number
afterwards, constrained by the roster.

Manual digit boxes (label-digits) are the preferred training source for multi-digit marks.
Single-digit mark boxes convert automatically; geometric splits are only a fallback.
"""

from __future__ import annotations

import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, List, Optional

from .mark_boxes import DISPLAY_MAX_SIDE, _read_boxes

DIGITS = [str(d) for d in range(10)]
DIGIT_BOX_FIELDS = [
    "image_path",
    "pig_id",
    "digit",
    "digit_index",
    "x1",
    "y1",
    "x2",
    "y2",
    "status",
]


def _read_digit_rows(digit_boxes_csv: Path) -> List[dict]:
    if not digit_boxes_csv.exists():
        return []
    with digit_boxes_csv.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_digit_rows(digit_boxes_csv: Path, rows: List[dict]) -> None:
    digit_boxes_csv.parent.mkdir(parents=True, exist_ok=True)
    with digit_boxes_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DIGIT_BOX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _digit_status_by_image(rows: List[dict]) -> dict[str, str]:
    """Return image_path -> done status ('boxed' or 'skip'). Incomplete images omitted."""
    by_image: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_image[row["image_path"]].append(row)
    done = {}
    for image_path, group in by_image.items():
        if any(row.get("status") == "skip" for row in group):
            done[image_path] = "skip"
            continue
        boxed = [row for row in group if row.get("status") == "boxed"]
        if not boxed:
            continue
        pig_id = boxed[0]["pig_id"]
        indexes = {int(row["digit_index"]) for row in boxed}
        if indexes == set(range(len(pig_id))):
            done[image_path] = "boxed"
    return done


def seed_single_digit_boxes(mark_boxes_csv: Path, digit_boxes_csv: Path) -> int:
    """Copy single-digit mark boxes into the digit CSV so they need no re-labeling."""
    existing = _read_digit_rows(digit_boxes_csv)
    done = _digit_status_by_image(existing)
    added = 0
    for row in _read_boxes(mark_boxes_csv).values():
        if row.get("status") != "boxed" or len(row["pig_id"]) != 1:
            continue
        if row["image_path"] in done:
            continue
        existing.append(
            {
                "image_path": row["image_path"],
                "pig_id": row["pig_id"],
                "digit": row["pig_id"],
                "digit_index": "0",
                "x1": row["x1"],
                "y1": row["y1"],
                "x2": row["x2"],
                "y2": row["y2"],
                "status": "boxed",
            }
        )
        added += 1
    if added:
        _write_digit_rows(digit_boxes_csv, existing)
    return added


def label_digit_boxes(
    mark_boxes_csv: Path,
    digit_boxes_csv: Path,
    limit: Optional[int] = None,
    shuffle: bool = True,
    two_digit_only: bool = True,
) -> dict:
    """Draw one box per digit. Enter=save digit, u=undo, s=skip image, q=quit."""
    import cv2

    seeded = seed_single_digit_boxes(mark_boxes_csv, digit_boxes_csv)
    existing_rows = _read_digit_rows(digit_boxes_csv)
    done = _digit_status_by_image(existing_rows)

    pending = []
    for row in _read_boxes(mark_boxes_csv).values():
        if row.get("status") != "boxed":
            continue
        if two_digit_only and len(row["pig_id"]) < 2:
            continue
        if row["image_path"] in done:
            continue
        if not Path(row["image_path"]).exists():
            continue
        pending.append(row)

    if shuffle:
        random.Random(0).shuffle(pending)
    if limit is not None:
        pending = pending[:limit]
    if not pending:
        print(
            f"No crops left to digit-box "
            f"({sum(1 for s in done.values() if s == 'boxed')} already done)."
        )
        return {
            "digit_boxes_csv": str(digit_boxes_csv),
            "pending": 0,
            "seeded_single_digit": seeded,
        }

    print(
        f"{len(pending)} crops to digit-box "
        f"({sum(1 for s in done.values() if s == 'boxed')} already done, "
        f"seeded {seeded} single-digit)."
    )
    print(
        "Draw a box around EACH digit in order. "
        "Enter=save digit  u=undo  s=skip image  q=quit"
    )

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

    window = "Draw box around each digit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    boxed_images = skipped = 0
    quit_early = False

    for index, row in enumerate(pending):
        image = cv2.imread(row["image_path"])
        if image is None:
            continue
        height, width = image.shape[:2]
        mark = (
            int(row["x1"]),
            int(row["y1"]),
            int(row["x2"]),
            int(row["y2"]),
        )
        pad = max(12, int(0.25 * max(mark[2] - mark[0], mark[3] - mark[1])))
        view_x1 = max(0, mark[0] - pad)
        view_y1 = max(0, mark[1] - pad)
        view_x2 = min(width, mark[2] + pad)
        view_y2 = min(height, mark[3] + pad)
        view = image[view_y1:view_y2, view_x1:view_x2]
        view_h, view_w = view.shape[:2]
        scale = min(DISPLAY_MAX_SIDE / max(view_h, view_w), 6.0)
        display_size = (max(1, int(view_w * scale)), max(1, int(view_h * scale)))

        pig_id = row["pig_id"]
        confirmed: list[tuple[str, tuple[int, int, int, int]]] = []
        digit_index = 0
        state["box"] = None
        image_done = False

        while digit_index < len(pig_id) and not quit_early and not image_done:
            digit = pig_id[digit_index]
            while True:
                canvas = cv2.resize(view, display_size)
                # Whole-mark guide (cyan).
                mx1 = int((mark[0] - view_x1) * scale)
                my1 = int((mark[1] - view_y1) * scale)
                mx2 = int((mark[2] - view_x1) * scale)
                my2 = int((mark[3] - view_y1) * scale)
                cv2.rectangle(canvas, (mx1, my1), (mx2, my2), (255, 255, 0), 1)
                for saved_digit, (bx1, by1, bx2, by2) in confirmed:
                    cv2.rectangle(
                        canvas,
                        (int((bx1 - view_x1) * scale), int((by1 - view_y1) * scale)),
                        (int((bx2 - view_x1) * scale), int((by2 - view_y1) * scale)),
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        canvas,
                        saved_digit,
                        (
                            int((bx1 - view_x1) * scale),
                            max(14, int((by1 - view_y1) * scale) - 4),
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                if state["box"]:
                    x1, y1, x2, y2 = state["box"]
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    canvas,
                    (
                        f"[{index + 1}/{len(pending)}] pig {pig_id}  "
                        f"draw digit '{digit}' ({digit_index + 1}/{len(pig_id)})"
                    ),
                    (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow(window, canvas)
                key = cv2.waitKey(20) & 0xFF
                if key in (13, 10):  # Enter
                    if not state["box"]:
                        continue
                    x1, y1, x2, y2 = state["box"]
                    if abs(x2 - x1) < 4 or abs(y2 - y1) < 4:
                        state["box"] = None
                        continue
                    abs_box = (
                        view_x1 + int(min(x1, x2) / scale),
                        view_y1 + int(min(y1, y2) / scale),
                        view_x1 + int(max(x1, x2) / scale),
                        view_y1 + int(max(y1, y2) / scale),
                    )
                    confirmed.append((digit, abs_box))
                    state["box"] = None
                    digit_index += 1
                    break
                if key == ord("u"):
                    if state["box"]:
                        state["box"] = None
                    elif confirmed:
                        confirmed.pop()
                        digit_index = max(0, digit_index - 1)
                    break
                if key == ord("s"):
                    existing_rows = [
                        item
                        for item in existing_rows
                        if item["image_path"] != row["image_path"]
                    ]
                    existing_rows.append(
                        {
                            "image_path": row["image_path"],
                            "pig_id": pig_id,
                            "digit": "",
                            "digit_index": "",
                            "x1": "",
                            "y1": "",
                            "x2": "",
                            "y2": "",
                            "status": "skip",
                        }
                    )
                    skipped += 1
                    image_done = True
                    break
                if key == ord("q"):
                    quit_early = True
                    break

        if quit_early:
            break
        if image_done:
            _write_digit_rows(digit_boxes_csv, existing_rows)
            continue
        if len(confirmed) != len(pig_id):
            continue

        existing_rows = [
            item for item in existing_rows if item["image_path"] != row["image_path"]
        ]
        for digit_i, (digit, (x1, y1, x2, y2)) in enumerate(confirmed):
            existing_rows.append(
                {
                    "image_path": row["image_path"],
                    "pig_id": pig_id,
                    "digit": digit,
                    "digit_index": str(digit_i),
                    "x1": str(x1),
                    "y1": str(y1),
                    "x2": str(x2),
                    "y2": str(y2),
                    "status": "boxed",
                }
            )
        boxed_images += 1
        _write_digit_rows(digit_boxes_csv, existing_rows)

    cv2.destroyAllWindows()
    final_done = _digit_status_by_image(existing_rows)
    return {
        "digit_boxes_csv": str(digit_boxes_csv.resolve()),
        "boxed_this_session": boxed_images,
        "skipped_this_session": skipped,
        "total_boxed_images": sum(1 for status in final_done.values() if status == "boxed"),
        "remaining": len(pending) - boxed_images - skipped if not quit_early else None,
        "seeded_single_digit": seeded,
    }


def compose_pig_id(
    detections: Iterable[tuple[str, float, tuple[float, float, float, float]]],
    roster: set[str],
    max_digits: int = 2,
) -> tuple[Optional[str], float]:
    """Turn digit detections into a roster-valid pig ID.

    Every two-digit ID in the roster starts with '1', so reading order never has to be
    recovered from the image: if a '1' is present alongside one other digit, the number
    is '1X'.
    """
    ordered = sorted(detections, key=lambda item: item[1], reverse=True)[:max_digits]
    if not ordered:
        return None, 0.0

    digits = [digit for digit, _, _ in ordered]
    confidence = sum(score for _, score, _ in ordered) / len(ordered)

    if len(digits) == 1:
        candidate = digits[0]
        return (candidate, confidence) if candidate in roster else (None, confidence)

    for first, second in ((digits[0], digits[1]), (digits[1], digits[0])):
        if first == "1":
            candidate = first + second
            if candidate in roster:
                return candidate, confidence

    # Two digits that cannot form a valid ID: fall back to the confident single digit.
    best = digits[0]
    if best in roster:
        return best, ordered[0][1]
    return None, confidence


def _manual_digit_annotations(
    digit_boxes_csv: Optional[Path],
) -> dict[str, list[tuple[str, tuple[int, int, int, int]]]]:
    if digit_boxes_csv is None or not digit_boxes_csv.exists():
        return {}
    rows = _read_digit_rows(digit_boxes_csv)
    done = _digit_status_by_image(rows)
    by_image: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "boxed" and done.get(row["image_path"]) == "boxed":
            by_image[row["image_path"]].append(row)
    annotations = {}
    for image_path, group in by_image.items():
        group = sorted(group, key=lambda item: int(item["digit_index"]))
        annotations[image_path] = [
            (
                row["digit"],
                (int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])),
            )
            for row in group
        ]
    return annotations


def export_digit_dataset(
    boxes_csv: Path,
    output_directory: Path,
    validation_fraction: float = 0.2,
    pseudo_labels: Optional[dict[str, list[tuple[str, tuple[int, int, int, int]]]]] = None,
    digit_boxes_csv: Optional[Path] = None,
) -> dict:
    """Build a digit YOLO dataset from manual digit boxes, then fallbacks."""
    import cv2

    if digit_boxes_csv is not None:
        seed_single_digit_boxes(boxes_csv, digit_boxes_csv)

    annotations = _manual_digit_annotations(digit_boxes_csv)
    from_manual = len(annotations)

    rows = [row for row in _read_boxes(boxes_csv).values() if row["status"] == "boxed"]
    single = [row for row in rows if len(row["pig_id"]) == 1]
    for row in single:
        if row["image_path"] in annotations:
            continue
        box = (int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"]))
        annotations[row["image_path"]] = [(row["pig_id"], box)]

    from_pseudo = 0
    for image_path, entries in (pseudo_labels or {}).items():
        if image_path in annotations:
            continue
        annotations[image_path] = entries
        from_pseudo += 1

    if not annotations:
        raise ValueError(
            "No digit boxes available; run piglet-id label-digits or label-marks first"
        )

    videos = sorted({Path(path).parts[-2] for path in annotations})
    rng = random.Random(42)
    rng.shuffle(videos)
    holdout = set(videos[: max(1, round(len(videos) * validation_fraction))])

    if output_directory.exists():
        shutil.rmtree(output_directory)
    counts = Counter()
    digit_counts = Counter()
    for image_path, entries in annotations.items():
        source = Path(image_path)
        image = cv2.imread(image_path)
        if image is None:
            continue
        height, width = image.shape[:2]
        lines = []
        for digit, (x1, y1, x2, y2) in entries:
            cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
            bw, bh = abs(x2 - x1) / width, abs(y2 - y1) / height
            lines.append(f"{DIGITS.index(digit)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            digit_counts[digit] += 1
        if not lines:
            continue
        split = "val" if source.parts[-2] in holdout else "train"
        stem = f"{source.parts[-2]}__{source.stem}".replace(" ", "_")
        image_directory = output_directory / "images" / split
        label_directory = output_directory / "labels" / split
        image_directory.mkdir(parents=True, exist_ok=True)
        label_directory.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, image_directory / f"{stem}{source.suffix}")
        (label_directory / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        counts[split] += 1

    data_yaml = output_directory / "data.yaml"
    data_yaml.write_text(
        "path: {path}\ntrain: images/train\nval: images/val\nnames:\n{names}\n".format(
            path=output_directory.resolve(),
            names="\n".join(f"  {i}: '{d}'" for i, d in enumerate(DIGITS)),
        )
    )
    return {
        "dataset": str(output_directory.resolve()),
        "data_yaml": str(data_yaml.resolve()),
        "train_images": counts["train"],
        "val_images": counts["val"],
        "digit_instances": dict(sorted(digit_counts.items())),
        "from_manual_digit_boxes": from_manual,
        "from_single_digit_crops": len(single),
        "from_pseudo_labels": from_pseudo,
    }


def _intersection_over_area(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> float:
    ix1 = max(inner[0], outer[0])
    iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2])
    iy2 = min(inner[3], outer[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inner_area = max(1, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return (ix2 - ix1) * (iy2 - iy1) / inner_area


def _remaining_rectangle(
    full: tuple[int, int, int, int], taken: tuple[int, int, int, int]
) -> Optional[tuple[int, int, int, int]]:
    """Largest strip of `full` left over after removing `taken`.

    The second digit sits beside the first, but the pair may be rotated to any angle, so
    the leftover strip is looked for on all four sides rather than assuming left/right.
    """
    fx1, fy1, fx2, fy2 = full
    tx1, ty1, tx2, ty2 = (
        max(fx1, taken[0]),
        max(fy1, taken[1]),
        min(fx2, taken[2]),
        min(fy2, taken[3]),
    )
    candidates = [
        (fx1, fy1, tx1, fy2),  # left
        (tx2, fy1, fx2, fy2),  # right
        (fx1, fy1, fx2, ty1),  # above
        (fx1, ty2, fx2, fy2),  # below
    ]
    full_area = max(1, (fx2 - fx1) * (fy2 - fy1))
    best = max(candidates, key=lambda r: max(0, r[2] - r[0]) * max(0, r[3] - r[1]))
    area = max(0, best[2] - best[0]) * max(0, best[3] - best[1])
    if area < 0.15 * full_area:
        return None
    return best


def _ink_mask(patch):
    import cv2
    import numpy as np

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    mask = (
        (hsv[:, :, 0] >= 90)
        & (hsv[:, :, 0] <= 170)
        & (hsv[:, :, 1] >= 35)
        & (hsv[:, :, 2] >= 35)
    )
    return mask.astype(np.uint8)


def split_two_digit_box(
    image, full: tuple[int, int, int, int], second_digit: str
) -> Optional[list[tuple[str, tuple[int, int, int, int]]]]:
    """Cut a two-digit mark into its two digits using the shape of the ink.

    The digits sit side by side along the writing direction, which the marks do not share
    because pigs lie at any angle. The first principal axis of the ink recovers that
    direction, so splitting at the median projection separates the pair. The leading digit
    of every two-digit roster ID is '1', the thinnest digit, so the sparser half is the '1'.
    """
    import numpy as np

    x1, y1, x2, y2 = full
    patch = image[max(0, y1) : y2, max(0, x1) : x2]
    if patch.size == 0 or patch.shape[0] < 8 or patch.shape[1] < 8:
        return None

    mask = _ink_mask(patch)
    points = np.argwhere(mask > 0)  # (row, col)
    if len(points) < 40:
        return None

    centered = points - points.mean(axis=0)
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    axis = components[0]
    projection = centered @ axis

    # Cut at the sparsest point between the two strokes rather than the median, which
    # would force equal ink on both sides and erase the '1' vs other-digit difference.
    counts, edges = np.histogram(projection, bins=24)
    smoothed = np.convolve(counts, np.ones(3) / 3, mode="same")
    interior = slice(6, 18)
    valley = int(np.argmin(smoothed[interior])) + interior.start
    threshold = float((edges[valley] + edges[valley + 1]) / 2)

    halves = []
    for selector in (projection <= threshold, projection > threshold):
        selected = points[selector]
        if len(selected) < 15:
            return None
        rows, columns = selected[:, 0], selected[:, 1]
        halves.append(
            (
                len(selected),
                (
                    int(x1 + columns.min()),
                    int(y1 + rows.min()),
                    int(x1 + columns.max()) + 1,
                    int(y1 + rows.max()) + 1,
                ),
            )
        )

    if second_digit == "1":
        return [("1", halves[0][1]), ("1", halves[1][1])]

    thin, thick = sorted(halves, key=lambda item: item[0])
    return [("1", thin[1]), (second_digit, thick[1])]


def build_geometric_pseudo_labels(boxes_csv: Path) -> dict:
    """Split every boxed two-digit mark into digit boxes without a detector."""
    import cv2

    rows = [
        row
        for row in _read_boxes(boxes_csv).values()
        if row["status"] == "boxed" and len(row["pig_id"]) == 2
    ]
    accepted: dict[str, list[tuple[str, tuple[int, int, int, int]]]] = {}
    rejected = Counter()
    for row in rows:
        image = cv2.imread(row["image_path"])
        if image is None:
            rejected["unreadable"] += 1
            continue
        full = (int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"]))
        split = split_two_digit_box(image, full, row["pig_id"][1])
        if split is None:
            rejected["no_split"] += 1
            continue
        accepted[row["image_path"]] = split
    return {
        "attempted": len(rows),
        "accepted_count": len(accepted),
        "rejected": dict(rejected),
        "accepted": accepted,
    }


def build_pseudo_labels(
    weights: Path,
    boxes_csv: Path,
    confidence: float = 0.2,
    imgsz: int = 320,
) -> dict:
    """Derive per-digit boxes for two-digit crops without any new manual labeling.

    Every two-digit roster ID contains a '1', and the stage-1 detector knows '1'. The
    human-drawn box already encloses the whole number, so locating the '1' inside it
    leaves the second digit's region by subtraction.
    """
    from ultralytics import YOLO

    rows = [
        row
        for row in _read_boxes(boxes_csv).values()
        if row["status"] == "boxed" and len(row["pig_id"]) == 2
    ]
    if not rows:
        return {"accepted": {}, "attempted": 0, "accepted_count": 0}

    model = YOLO(str(weights))
    names = model.names
    accepted: dict[str, list[tuple[str, tuple[int, int, int, int]]]] = {}
    rejected = Counter()

    for start in range(0, len(rows), 32):
        batch = rows[start : start + 32]
        results = model.predict(
            [row["image_path"] for row in batch],
            imgsz=imgsz,
            conf=confidence,
            verbose=False,
        )
        for row, result in zip(batch, results):
            full = (int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"]))
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                rejected["no_detection"] += 1
                continue

            ones = []
            for index in range(len(boxes)):
                if str(names[int(boxes.cls[index])]) != "1":
                    continue
                box = tuple(int(v) for v in boxes.xyxy[index].tolist())
                if _intersection_over_area(box, full) < 0.6:
                    continue
                ones.append((float(boxes.conf[index]), box))
            if not ones:
                rejected["no_one_inside_box"] += 1
                continue

            ones.sort(reverse=True)
            first = ones[0][1]
            second_digit = row["pig_id"][1]
            if second_digit == "1":
                # '11': prefer a second detected '1', else split the box.
                if len(ones) > 1 and _intersection_over_area(ones[1][1], full) >= 0.6:
                    accepted[row["image_path"]] = [("1", first), ("1", ones[1][1])]
                    continue
            remainder = _remaining_rectangle(full, first)
            if remainder is None:
                rejected["no_remainder"] += 1
                continue
            accepted[row["image_path"]] = [("1", first), (second_digit, remainder)]

    return {
        "attempted": len(rows),
        "accepted_count": len(accepted),
        "rejected": dict(rejected),
        "accepted": accepted,
    }


def score_digit_reader(
    weights: Path,
    labels_csv: Path,
    confidence: float = 0.25,
    imgsz: int = 320,
) -> dict:
    """Score composed pig IDs from digit detections on labeled crops."""
    from collections import defaultdict

    from ultralytics import YOLO

    from .mark_boxes import _labeled_crops
    from .ocr import load_allowed_ids

    rows = _labeled_crops(labels_csv)
    roster = load_allowed_ids()
    model = YOLO(str(weights))
    names = model.names

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
            detections = []
            if boxes is not None and len(boxes) > 0:
                for index in range(len(boxes)):
                    detections.append(
                        (
                            str(names[int(boxes.cls[index])]),
                            float(boxes.conf[index]),
                            tuple(boxes.xyxy[index].tolist()),
                        )
                    )
            pig_id, score = compose_pig_id(detections, roster)
            predictions.append((row, pig_id, score))

    read = [item for item in predictions if item[1] is not None]
    correct = sum(1 for row, pig_id, _ in read if pig_id == row["pig_id"])

    votes: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    truth: dict[str, str] = {}
    for row, pig_id, score in predictions:
        source = Path(row["image_path"])
        track = f"{source.parts[-2]}:{source.stem.split('_f')[0]}"
        truth[track] = row["pig_id"]
        if pig_id is not None:
            votes[track][pig_id] += score
    track_correct = sum(
        max(scores.items(), key=lambda kv: kv[1])[0] == truth[track]
        for track, scores in votes.items()
        if scores
    )

    per_class = defaultdict(lambda: [0, 0])
    for row, pig_id, _ in predictions:
        per_class[row["pig_id"]][1] += 1
        if pig_id == row["pig_id"]:
            per_class[row["pig_id"]][0] += 1

    return {
        "weights": str(weights),
        "images": len(rows),
        "coverage": round(len(read) / max(1, len(rows)), 4),
        "image_accuracy": round(correct / max(1, len(rows)), 4),
        "image_accuracy_on_read": round(correct / max(1, len(read)), 4),
        "tracks": len(votes),
        "track_accuracy": round(track_correct / max(1, len(votes)), 4),
        "per_class_accuracy": {
            pig_id: round(hit / total, 2)
            for pig_id, (hit, total) in sorted(per_class.items(), key=lambda kv: int(kv[0]))
        },
    }
