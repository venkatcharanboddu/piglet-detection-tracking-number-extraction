"""Read pig IDs on one video's track crops and write predictions plus a review montage.

Usage: python experiments/read_video_digits.py <crop_directory> <weights> <output_prefix>
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from piglet_identity.digit_marks import compose_pig_id  # noqa: E402
from piglet_identity.ocr import load_allowed_ids  # noqa: E402


def main() -> None:
    crop_directory = Path(sys.argv[1])
    weights = Path(sys.argv[2])
    prefix = Path(sys.argv[3])

    from ultralytics import YOLO

    images = sorted(crop_directory.glob("track_*.jpg"))
    if not images:
        raise SystemExit(f"no track crops in {crop_directory}")

    roster = load_allowed_ids()
    model = YOLO(str(weights))
    names = model.names

    per_image = []
    for start in range(0, len(images), 32):
        batch = images[start : start + 32]
        results = model.predict([str(p) for p in batch], imgsz=320, conf=0.25, verbose=False)
        for path, result in zip(batch, results):
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
            per_image.append((path, pig_id, score, detections))

    votes: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    frames: dict[str, list] = defaultdict(list)
    for path, pig_id, score, detections in per_image:
        track = path.stem.split("_f")[0]
        frames[track].append((path, pig_id, score, detections))
        if pig_id is not None:
            votes[track][pig_id] += score

    rows = []
    for track, entries in sorted(frames.items()):
        scores = votes.get(track, {})
        best, total = (None, 0.0)
        if scores:
            best, total = max(scores.items(), key=lambda kv: kv[1])
        rows.append(
            {
                "track": track,
                "crops": len(entries),
                "crops_read": sum(1 for _, pid, _, _ in entries if pid is not None),
                "pig_id": best or "",
                "vote_score": round(total, 3),
                "alternatives": ";".join(
                    f"{pid}:{round(value, 2)}"
                    for pid, value in sorted(scores.items(), key=lambda kv: -kv[1])[1:3]
                ),
            }
        )

    csv_path = prefix.with_name(prefix.name + "_track_predictions.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Montage: best-read crop per track, annotated with the composed ID.
    tiles = []
    for track, entries in sorted(frames.items()):
        best_entry = max(entries, key=lambda item: item[2])
        path, pig_id, score, detections = best_entry
        image = cv2.imread(str(path))
        if image is None:
            continue
        tile = cv2.resize(image, (160, 160))
        scale_x, scale_y = 160 / image.shape[1], 160 / image.shape[0]
        for digit, conf, (x1, y1, x2, y2) in detections:
            cv2.rectangle(
                tile,
                (int(x1 * scale_x), int(y1 * scale_y)),
                (int(x2 * scale_x), int(y2 * scale_y)),
                (0, 255, 0),
                1,
            )
        row = next(item for item in rows if item["track"] == track)
        label = f"{track} -> {row['pig_id'] or '?'}"
        cv2.rectangle(tile, (0, 0), (160, 18), (0, 0, 0), -1)
        cv2.putText(tile, label, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
        tiles.append(tile)

    if tiles:
        columns = 6
        while len(tiles) % columns:
            tiles.append(np.zeros((160, 160, 3), dtype=np.uint8))
        grid = np.vstack(
            [np.hstack(tiles[i : i + columns]) for i in range(0, len(tiles), columns)]
        )
        montage_path = prefix.with_name(prefix.name + "_montage.png")
        cv2.imwrite(str(montage_path), grid)

    summary = {
        "crops": len(images),
        "crops_read": sum(1 for _, pid, _, _ in per_image if pid is not None),
        "tracks": len(frames),
        "tracks_with_id": sum(1 for row in rows if row["pig_id"]),
        "distinct_ids": sorted({row["pig_id"] for row in rows if row["pig_id"]}, key=int),
        "predictions_csv": str(csv_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
