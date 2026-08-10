"""Does classifying the detected mark crop beat YOLO's own class head?

The detector finds the mark on ~85% of unseen crops but only names the pig right
~36% of the time. This uses the detector purely as a localizer, then trains a
dedicated classifier on the tight mark crop and scores it on held-out July 16.

Run:  .venv/bin/python experiments/tight_crop_classifier.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TRAIN_CSV = ROOT / "artifacts" / "identity_labeling_v2" / "identity_labels.csv"
TEST_CSV = ROOT / "artifacts" / "identity_labeling_v2" / "holdout_2026-07-16_labels.csv"
WEIGHTS = ROOT / "artifacts" / "models" / "mark_detector" / "weights" / "best.pt"
PAD = 0.15
ROTATIONS = (0, 90, 180, 270)


def load_rows(path: Path) -> list[dict]:
    rows = []
    for row in csv.DictReader(path.open()):
        pig_id = (row.get("pig_id") or "").strip()
        if row.get("status") != "labeled" or not pig_id.isdigit():
            continue
        if row.get("source_type") == "reference" or not Path(row["image_path"]).exists():
            continue
        rows.append({"path": row["image_path"], "pig_id": pig_id})
    return rows


def crop_marks(rows: list[dict], model: YOLO) -> tuple[list[np.ndarray], list[str], list[str]]:
    """Return the tight mark crop for every row where the detector found something."""
    images, labels, tracks = [], [], []
    misses = 0
    for start in range(0, len(rows), 32):
        batch = rows[start : start + 32]
        results = model.predict(
            [row["path"] for row in batch], imgsz=320, conf=0.25, verbose=False
        )
        for row, result in zip(batch, results):
            boxes = result.boxes
            image = cv2.imread(row["path"])
            if boxes is None or len(boxes) == 0 or image is None:
                misses += 1
                continue
            best = int(boxes.conf.argmax())
            x1, y1, x2, y2 = boxes.xyxy[best].tolist()
            h, w = image.shape[:2]
            px, py = (x2 - x1) * PAD, (y2 - y1) * PAD
            crop = image[
                max(0, int(y1 - py)) : min(h, int(y2 + py)),
                max(0, int(x1 - px)) : min(w, int(x2 + px)),
            ]
            if crop.size == 0:
                misses += 1
                continue
            images.append(crop)
            labels.append(row["pig_id"])
            source = Path(row["path"])
            tracks.append(f"{source.parts[-2]}:{source.stem.split('_f')[0]}")
    print(f"  detector found marks on {len(images)}/{len(rows)} ({misses} missed)")
    return images, labels, tracks


def rotate(image: np.ndarray, angle: int) -> np.ndarray:
    return image if angle == 0 else np.ascontiguousarray(np.rot90(image, angle // 90))


class Embedder:
    def __init__(self) -> None:
        weights = ResNet18_Weights.DEFAULT
        net = resnet18(weights=weights)
        net.fc = nn.Identity()
        self.net = net.eval()
        self.transform = weights.transforms()

    def __call__(self, images: list[np.ndarray]) -> np.ndarray:
        out = []
        for start in range(0, len(images), 64):
            tensors = torch.stack(
                [
                    self.transform(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                    for img in images[start : start + 64]
                ]
            )
            with torch.no_grad():
                out.append(self.net(tensors).numpy())
        return np.concatenate(out).astype(np.float32)


def main() -> None:
    train_rows, test_rows = load_rows(TRAIN_CSV), load_rows(TEST_CSV)
    model = YOLO(str(WEIGHTS))

    print(f"cropping marks on {len(train_rows)} training images ...", flush=True)
    train_images, y_train, _ = crop_marks(train_rows, model)
    print(f"cropping marks on {len(test_rows)} held-out July 16 images ...", flush=True)
    test_images, y_test, test_tracks = crop_marks(test_rows, model)

    sizes = np.array([img.shape[0] * img.shape[1] for img in train_images])
    print(f"  tight crop area: median {np.median(sizes):.0f}px^2")

    classes = set(y_train) & set(y_test)
    keep_train = [i for i, label in enumerate(y_train) if label in classes]
    keep_test = [i for i, label in enumerate(y_test) if label in classes]
    train_images = [train_images[i] for i in keep_train]
    test_images = [test_images[i] for i in keep_test]
    y_train = np.array([y_train[i] for i in keep_train])
    y_test = np.array([y_test[i] for i in keep_test])
    test_tracks = [test_tracks[i] for i in keep_test]
    print(f"\nshared classes ({len(classes)}): {sorted(classes, key=int)}")
    print(f"train {len(y_train)}  test {len(y_test)}")
    print(f"test class counts: {dict(Counter(y_test))}\n")

    embed = Embedder()
    print("embedding tight crops with rotation augmentation ...", flush=True)
    rotated = [rotate(img, a) for img in train_images for a in ROTATIONS]
    x_train = embed(rotated)
    y_train_rot = np.repeat(y_train, len(ROTATIONS))
    x_test = embed([rotate(img, a) for img in test_images for a in ROTATIONS])

    scaler = StandardScaler().fit(x_train)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    clf.fit(scaler.transform(x_train), y_train_rot)

    probabilities = clf.predict_proba(scaler.transform(x_test))
    probabilities = probabilities.reshape(len(y_test), len(ROTATIONS), -1).max(axis=1)
    predictions = clf.classes_[probabilities.argmax(axis=1)]
    image_accuracy = float((predictions == y_test).mean())

    votes: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    truth: dict[str, str] = {}
    for track, prediction, actual, score in zip(
        test_tracks, predictions, y_test, probabilities.max(axis=1)
    ):
        votes[track][prediction] += score
        truth[track] = actual
    track_accuracy = sum(
        max(scores.items(), key=lambda kv: kv[1])[0] == truth[track]
        for track, scores in votes.items()
    ) / len(votes)

    results = {
        "tight_crop_image_accuracy": round(image_accuracy, 4),
        "tight_crop_track_accuracy": round(track_accuracy, 4),
        "test_images_with_a_detected_mark": len(y_test),
        "tracks": len(votes),
        "reference_yolo_direct_image_accuracy": 0.3579,
        "reference_whole_crop_image_accuracy": 0.1761,
    }
    print(json.dumps(results, indent=2))
    out = ROOT / "artifacts" / "experiments" / "tight_crop_classifier.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
