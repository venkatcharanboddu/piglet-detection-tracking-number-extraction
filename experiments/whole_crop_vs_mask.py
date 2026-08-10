"""Does removing the broken ink mask fix mark reading?

Trains the same classifier head on three feature variants and scores all of them
on the held-out July 16 videos (never used for training).

    A  current   existing extract_features (128px + blue/purple mask)
    B  wholecrop full crop at 224px, no mask
    C  rot_tta   full crop at 224px + rotation augmentation and test-time voting

Run:  .venv/bin/python experiments/whole_crop_vs_mask.py
"""

from __future__ import annotations

import csv
import json
import re
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from piglet_identity.identity import extract_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "artifacts" / "identity_labeling_v2" / "identity_labels.csv"
TEST_CSV = ROOT / "artifacts" / "identity_labeling_v2" / "holdout_2026-07-16_labels.csv"
ROTATIONS = (0, 90, 180, 270, 30, -30)


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_rows(path: Path) -> list[dict]:
    rows = []
    for row in csv.DictReader(path.open()):
        pig_id = (row.get("pig_id") or "").strip()
        if row.get("status") != "labeled" or not pig_id.isdigit():
            continue
        if not Path(row["image_path"]).exists():
            continue
        rows.append({"path": row["image_path"], "pig_id": pig_id})
    return rows


def track_key(path: str) -> str:
    parts = Path(path).parts
    match = re.match(r"(track_\d+)", Path(path).stem)
    return f"{parts[-2]}:{match.group(1) if match else Path(path).stem}"


def rotate(image: np.ndarray, angle: int) -> np.ndarray:
    if angle == 0:
        return image
    if angle in (90, 180, 270):
        return np.ascontiguousarray(np.rot90(image, angle // 90))
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    matrix[0, 2] += nw / 2 - w / 2
    matrix[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(image, matrix, (nw, nh), borderMode=cv2.BORDER_REPLICATE)


class Embedder:
    """ResNet18 penultimate features on the full crop, no masking."""

    def __init__(self) -> None:
        weights = ResNet18_Weights.DEFAULT
        net = resnet18(weights=weights)
        net.fc = nn.Identity()
        self.net = net.eval().to(device())
        self.transform = weights.transforms()
        self.device = device()

    def __call__(self, images: list[np.ndarray], batch_size: int = 64) -> np.ndarray:
        out = []
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            tensors = torch.stack(
                [
                    self.transform(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                    for img in batch
                ]
            ).to(self.device)
            with torch.no_grad():
                out.append(self.net(tensors).cpu().numpy())
        return np.concatenate(out).astype(np.float32)


def fit_and_score(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    test_tracks: list[str],
    groups: int = 1,
) -> dict:
    """groups>1 means x_test holds `groups` rotated views per test image."""
    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    model.fit(scaler.transform(x_train), y_train)

    probabilities = model.predict_proba(scaler.transform(x_test))
    if groups > 1:
        probabilities = probabilities.reshape(len(y_test), groups, -1).max(axis=1)
    predictions = model.classes_[probabilities.argmax(axis=1)]

    image_accuracy = float((predictions == y_test).mean())

    votes: dict[str, list] = defaultdict(list)
    truth: dict[str, str] = {}
    for track, prediction, actual, probability in zip(
        test_tracks, predictions, y_test, probabilities.max(axis=1)
    ):
        votes[track].append((prediction, probability))
        truth[track] = actual
    correct = 0
    for track, entries in votes.items():
        scores: dict[str, float] = defaultdict(float)
        for prediction, probability in entries:
            scores[prediction] += probability
        if max(scores.items(), key=lambda kv: kv[1])[0] == truth[track]:
            correct += 1
    return {
        "image_accuracy": round(image_accuracy, 4),
        "track_accuracy": round(correct / len(votes), 4),
        "tracks": len(votes),
    }


def main() -> None:
    train_rows = load_rows(TRAIN_CSV)
    test_rows = load_rows(TEST_CSV)
    classes = {row["pig_id"] for row in train_rows} & {row["pig_id"] for row in test_rows}
    train_rows = [r for r in train_rows if r["pig_id"] in classes]
    test_rows = [r for r in test_rows if r["pig_id"] in classes]

    print(f"device: {device()}")
    print(f"train images: {len(train_rows)}  test images: {len(test_rows)}")
    print(f"shared classes ({len(classes)}): {sorted(classes, key=int)}")
    print(f"test class counts: {dict(Counter(r['pig_id'] for r in test_rows))}\n")

    train_images = [cv2.imread(r["path"]) for r in train_rows]
    test_images = [cv2.imread(r["path"]) for r in test_rows]
    y_train = np.array([r["pig_id"] for r in train_rows])
    y_test = np.array([r["pig_id"] for r in test_rows])
    test_tracks = [track_key(r["path"]) for r in test_rows]

    results = {}

    print("A  current (128px + ink mask) ...", flush=True)
    xa_train = np.stack([extract_features(img) for img in train_images])
    xa_test = np.stack([extract_features(img) for img in test_images])
    results["A_current_mask"] = fit_and_score(xa_train, y_train, xa_test, y_test, test_tracks)
    print("   ", results["A_current_mask"], flush=True)

    embed = Embedder()

    print("B  whole crop 224px, no mask ...", flush=True)
    xb_train = embed(train_images)
    xb_test = embed(test_images)
    results["B_whole_crop"] = fit_and_score(xb_train, y_train, xb_test, y_test, test_tracks)
    print("   ", results["B_whole_crop"], flush=True)

    print(f"C  whole crop + rotations {ROTATIONS} ...", flush=True)
    rotated_train, rotated_labels = [], []
    for image, label in zip(train_images, y_train):
        for angle in ROTATIONS:
            rotated_train.append(rotate(image, angle))
            rotated_labels.append(label)
    xc_train = embed(rotated_train)
    yc_train = np.array(rotated_labels)

    rotated_test = [rotate(img, angle) for img in test_images for angle in ROTATIONS]
    xc_test = embed(rotated_test)
    results["C_rotation_tta"] = fit_and_score(
        xc_train, yc_train, xc_test, y_test, test_tracks, groups=len(ROTATIONS)
    )
    print("   ", results["C_rotation_tta"], flush=True)

    baseline = 1.0 / len(classes)
    results["random_baseline"] = round(baseline, 4)
    out = ROOT / "artifacts" / "experiments" / "whole_crop_vs_mask.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nrandom baseline: {baseline:.3f}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
