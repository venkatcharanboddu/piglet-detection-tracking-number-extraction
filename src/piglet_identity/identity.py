from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

_FEATURE_EXTRACTOR = None


@dataclass(frozen=True)
class IdentityPrediction:
    pig_id: str
    confidence: float
    candidate_id: Optional[str] = None


def vote_predictions(
    predictions: Iterable[IdentityPrediction],
    threshold: float = 0.65,
    min_votes: int = 3,
    include_candidates: bool = True,
) -> IdentityPrediction:
    scores: dict[str, List[float]] = defaultdict(list)
    for prediction in predictions:
        label = prediction.pig_id
        if label == "unknown" and include_candidates:
            label = prediction.candidate_id
        if label:
            scores[label].append(prediction.confidence)
    if not scores:
        return IdentityPrediction("unknown", 0.0)

    def rank_key(item: tuple[str, List[float]]) -> tuple:
        pig_id, values = item
        return (len(values), sum(values), len(pig_id), pig_id)

    pig_id, values = max(scores.items(), key=rank_key)
    confidence = float(sum(values) / len(values))
    if len(values) < min_votes or confidence < threshold:
        return IdentityPrediction("unknown", confidence, pig_id)
    return IdentityPrediction(pig_id, confidence, pig_id)


def vote_candidates(predictions: Iterable[IdentityPrediction]) -> IdentityPrediction:
    scores: dict[str, List[float]] = defaultdict(list)
    for prediction in predictions:
        candidate = prediction.candidate_id or (
            prediction.pig_id if prediction.pig_id != "unknown" else None
        )
        if candidate:
            scores[candidate].append(prediction.confidence)
    if not scores:
        return IdentityPrediction("unknown", 0.0)
    pig_id, values = max(
        scores.items(), key=lambda item: (sum(item[1]), len(item[1]), item[0])
    )
    return IdentityPrediction(pig_id, float(sum(values) / len(values)), pig_id)


def extract_features(image: np.ndarray) -> np.ndarray:
    """Pretrained visual descriptor focused on the painted-mark region."""
    import cv2
    import torch
    from PIL import Image
    from torch import nn
    from torchvision.models import ResNet18_Weights, resnet18

    resized = cv2.resize(image, (128, 128))
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    # Purple/blue paint is high-saturation relative to pink skin.
    mask = ((hsv[:, :, 1] > 65) & (hsv[:, :, 0] > 95) & (hsv[:, :, 0] < 170)).astype(
        np.uint8
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    coordinates = cv2.findNonZero(mask)
    if coordinates is not None:
        x, y, width, height = cv2.boundingRect(coordinates)
        mark = mask[y : y + height, x : x + width]
    else:
        mark = mask
    scale = min(52 / max(1, mark.shape[1]), 52 / max(1, mark.shape[0]))
    mark = cv2.resize(
        mark,
        (max(1, round(mark.shape[1] * scale)), max(1, round(mark.shape[0] * scale))),
        interpolation=cv2.INTER_NEAREST,
    )
    canvas = np.zeros((64, 64), dtype=np.uint8)
    y_offset = (64 - mark.shape[0]) // 2
    x_offset = (64 - mark.shape[1]) // 2
    canvas[y_offset : y_offset + mark.shape[0], x_offset : x_offset + mark.shape[1]] = mark
    moments = cv2.HuMoments(cv2.moments(mask)).ravel()
    moments = -np.sign(moments) * np.log10(np.abs(moments) + 1e-12)
    hist_h = cv2.calcHist([hsv], [0], mask, [16], [0, 180]).ravel()
    hist_h /= hist_h.sum() + 1e-6
    if coordinates is not None:
        margin = max(width, height) // 3
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(128, x + width + margin), min(128, y + height + margin)
        visual = resized[y1:y2, x1:x2]
    else:
        visual = resized
    global _FEATURE_EXTRACTOR
    if _FEATURE_EXTRACTOR is None:
        weights = ResNet18_Weights.DEFAULT
        network = resnet18(weights=weights)
        network.fc = nn.Identity()
        network.eval()
        _FEATURE_EXTRACTOR = (network, weights.transforms())
    network, transform = _FEATURE_EXTRACTOR
    rgb = cv2.cvtColor(visual, cv2.COLOR_BGR2RGB)
    tensor = transform(Image.fromarray(rgb)).unsqueeze(0)
    with torch.no_grad():
        embedding = network(tensor).squeeze(0).cpu().numpy()
    return np.concatenate(
        [embedding.astype(np.float32), moments.astype(np.float32), hist_h]
    ).astype(np.float32)


def load_labeled_images(labels_csv: Path) -> List[Tuple[np.ndarray, str, str]]:
    import cv2

    records = []
    with labels_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            pig_id = row.get("pig_id", "").strip()
            status = row.get("status", "labeled").strip().lower()
            if (
                not pig_id
                or not pig_id.isdigit()
                or pig_id in {"unknown", "ambiguous"}
                or status not in {"labeled", "ok"}
            ):
                continue
            image_path = Path(row["image_path"])
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            records.append((image, pig_id, row.get("source_video") or str(image_path)))
    if not records:
        raise ValueError(f"No usable labeled images in {labels_csv}")
    return records


def augment_training_image(image: np.ndarray) -> List[np.ndarray]:
    """Deterministic, label-preserving views; digit mirroring is intentionally excluded."""
    import cv2

    views = [image]
    for angle in (-30, 30, 90, 180, 270):
        height, width = image.shape[:2]
        center = (width / 2, height / 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
        new_width = int(height * sine + width * cosine)
        new_height = int(height * cosine + width * sine)
        matrix[0, 2] += new_width / 2 - center[0]
        matrix[1, 2] += new_height / 2 - center[1]
        views.append(
            cv2.warpAffine(
                image,
                matrix,
                (new_width, new_height),
                borderMode=cv2.BORDER_REFLECT_101,
            )
        )

    height, width = image.shape[:2]
    source = np.float32([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])
    offset_x, offset_y = width * 0.07, height * 0.07
    target = np.float32(
        [
            [offset_x, 0],
            [width - 1, offset_y],
            [0, height - 1 - offset_y],
            [width - 1 - offset_x, height - 1],
        ]
    )
    perspective = cv2.getPerspectiveTransform(source, target)
    views.append(
        cv2.warpPerspective(
            image, perspective, (width, height), borderMode=cv2.BORDER_REFLECT_101
        )
    )

    down_width, down_height = max(16, width // 3), max(16, height // 3)
    degraded = cv2.resize(image, (down_width, down_height), interpolation=cv2.INTER_AREA)
    degraded = cv2.resize(degraded, (width, height), interpolation=cv2.INTER_LINEAR)
    degraded = cv2.GaussianBlur(degraded, (5, 5), 0)
    views.append(cv2.convertScaleAbs(degraded, alpha=0.8, beta=18))
    return views


def _features_for_records(
    records: List[Tuple[np.ndarray, str, str]], augment: bool
) -> Tuple[np.ndarray, np.ndarray]:
    features, labels = [], []
    for image, label, _source in records:
        views = augment_training_image(image) if augment else [image]
        for view in views:
            features.append(extract_features(view))
            labels.append(label)
    return np.stack(features), np.asarray(labels)


def train_classifier(labels_csv: Path, output_model: Path) -> dict:
    import joblib
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    records = load_labeled_images(labels_csv)
    labels = np.asarray([record[1] for record in records])
    groups = np.asarray([record[2] for record in records])
    class_counts = {
        label: int(np.sum(labels == label)) for label in sorted(set(labels))
    }
    usable = {label for label, count in class_counts.items() if count >= 2}
    records = [record for record in records if record[1] in usable]
    labels = np.asarray([record[1] for record in records])
    groups = np.asarray([record[2] for record in records])
    if len(set(labels)) < 2:
        raise ValueError("Identity training needs at least two IDs with two images each")

    train_idx = np.arange(len(labels))
    test_idx = np.asarray([], dtype=int)
    source_counts = {source: int(np.sum(groups == source)) for source in set(groups)}
    all_sources_unique = max(source_counts.values()) == 1
    if all_sources_unique and min(np.sum(labels == label) for label in set(labels)) >= 2:
        train_idx, test_idx = next(
            StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42).split(
                np.zeros(len(labels)), labels
            )
        )
    elif len(set(groups)) > 1 and len(labels) >= 8:
        train_idx, test_idx = next(
            GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42).split(
                np.zeros(len(labels)), labels, groups
            )
        )
        if set(labels[train_idx]) != set(labels):
            train_idx, test_idx = np.arange(len(labels)), np.asarray([], dtype=int)

    train_records = [records[index] for index in train_idx]
    x_train, y_train = _features_for_records(train_records, augment=True)

    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
    )
    model.fit(x_train, y_train)
    metrics = {
        "training_images": int(len(train_idx)),
        "augmented_training_samples": int(len(y_train)),
        "augmentation": [
            "rotate_-30",
            "rotate_30",
            "rotate_90",
            "rotate_180",
            "rotate_270",
            "perspective",
            "blur_downscale_brightness",
        ],
        "mirroring": False,
        "class_counts": class_counts,
    }
    threshold = 0.65
    if len(test_idx):
        test_records = [records[index] for index in test_idx]
        x_test, y_test = _features_for_records(test_records, augment=False)
        probabilities = model.predict_proba(x_test)
        predictions = model.classes_[np.argmax(probabilities, axis=1)]
        confidence = np.max(probabilities, axis=1)
        correct = predictions == y_test
        candidates = np.arange(0.15, 0.91, 0.05)
        eligible = []
        for candidate in candidates:
            accepted = confidence >= candidate
            precision = float(np.mean(correct[accepted])) if np.any(accepted) else 0.0
            coverage = float(np.mean(accepted))
            if precision >= 0.8:
                eligible.append((coverage, float(candidate), precision))
        if eligible:
            _, threshold, _ = max(eligible)
        metrics.update(
            validation_images=int(len(test_idx)),
            validation_accuracy=float(accuracy_score(y_test, predictions)),
            validation_coverage_at_threshold=float(np.mean(confidence >= threshold)),
            calibrated_threshold=threshold,
        )
    x_all, y_all = _features_for_records(records, augment=True)
    model.fit(x_all, y_all)
    metrics["final_augmented_samples"] = int(len(y_all))
    artifact = {"model": model, "threshold": threshold, "metrics": metrics}
    output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_model)
    return metrics


class IdentityModel:
    """Independent identity readers plus the original mark/classifier hybrid."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        use_ocr: bool = False,
        labels_csv: Optional[Path] = None,
        method: str = "hybrid",
        trocr_model_path: Optional[Path] = None,
        digit_weights_path: Optional[Path] = None,
        digit_confidence: float = 0.25,
        digit_imgsz: int = 320,
    ):
        import joblib

        self.use_ocr = use_ocr
        self.method = "easyocr" if use_ocr and method == "hybrid" else method
        self.labels_csv = labels_csv
        self.trocr_model_path = trocr_model_path
        self.digit_weights_path = digit_weights_path
        self.digit_confidence = digit_confidence
        self.digit_imgsz = digit_imgsz
        self._trocr_reader = None
        self._digit_model = None
        self.model = None
        self.threshold = 0.45
        if model_path is not None and Path(model_path).exists():
            artifact = joblib.load(model_path)
            self.model = artifact["model"]
            self.threshold = float(artifact.get("threshold", 0.45))
        from .ocr import load_allowed_ids

        self.allowed_ids = load_allowed_ids()

    def predict_mark(self, image: np.ndarray) -> IdentityPrediction:
        from .mark_match import match_mark

        prediction = match_mark(image, self.labels_csv)
        if prediction.pig_id != "unknown" and prediction.pig_id not in self.allowed_ids:
            return IdentityPrediction("unknown", prediction.confidence, prediction.pig_id)
        return prediction

    def predict_classifier(self, image: np.ndarray) -> IdentityPrediction:
        if self.model is None:
            return IdentityPrediction("unknown", 0.0)
        probabilities = self.model.predict_proba(extract_features(image)[None, :])[0]
        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])
        candidate_id = str(self.model.classes_[index])
        pig_id = (
            candidate_id
            if confidence >= self.threshold and candidate_id in self.allowed_ids
            else "unknown"
        )
        return IdentityPrediction(pig_id, confidence, candidate_id)

    def predict_easyocr(self, image: np.ndarray) -> IdentityPrediction:
        from .ocr import read_pig_number

        return read_pig_number(image, self.allowed_ids, min_confidence=0.35)

    def predict_trocr(self, image: np.ndarray) -> IdentityPrediction:
        if self.trocr_model_path is None:
            raise RuntimeError("TrOCR mode requires --trocr-model or a trained default model")
        if self._trocr_reader is None:
            from .ocr_trocr import TrOCRPigletReader

            self._trocr_reader = TrOCRPigletReader(self.trocr_model_path, self.allowed_ids)
        return self._trocr_reader.predict(image)

    def predict_digits(self, image: np.ndarray) -> IdentityPrediction:
        """Detect individual digits with YOLO and compose a roster-valid pig ID."""
        if self.digit_weights_path is None or not Path(self.digit_weights_path).exists():
            raise RuntimeError(
                "Digits mode requires --digit-weights pointing to a trained digit detector"
            )
        if self._digit_model is None:
            from ultralytics import YOLO

            self._digit_model = YOLO(str(self.digit_weights_path))
        from .digit_marks import compose_pig_id

        result = self._digit_model.predict(
            image,
            imgsz=self.digit_imgsz,
            conf=self.digit_confidence,
            verbose=False,
        )[0]
        detections = []
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            names = self._digit_model.names
            for index in range(len(boxes)):
                detections.append(
                    (
                        str(names[int(boxes.cls[index])]),
                        float(boxes.conf[index]),
                        tuple(boxes.xyxy[index].tolist()),
                    )
                )
        pig_id, confidence = compose_pig_id(detections, self.allowed_ids)
        if pig_id is None:
            candidate = "".join(digit for digit, _, _ in detections[:2]) if detections else ""
            return IdentityPrediction("unknown", float(confidence), candidate or None)
        return IdentityPrediction(pig_id, float(confidence), pig_id)

    def predict_hybrid(self, image: np.ndarray) -> IdentityPrediction:
        mark = self.predict_mark(image)
        if mark.pig_id != "unknown":
            return mark
        classifier = self.predict_classifier(image)
        if classifier.pig_id != "unknown":
            return classifier
        return mark if mark.candidate_id else classifier

    def predict_all(self, image: np.ndarray) -> dict[str, IdentityPrediction]:
        return {
            "mark": self.predict_mark(image),
            "classifier": self.predict_classifier(image),
            "trocr": self.predict_trocr(image),
        }

    def predict(self, image: np.ndarray) -> IdentityPrediction:
        readers = {
            "hybrid": self.predict_hybrid,
            "mark": self.predict_mark,
            "classifier": self.predict_classifier,
            "easyocr": self.predict_easyocr,
            "trocr": self.predict_trocr,
            "digits": self.predict_digits,
        }
        if self.method == "compare":
            return self.predict_hybrid(image)
        try:
            reader = readers[self.method]
        except KeyError as exc:
            raise ValueError(f"Unknown identity method: {self.method}") from exc
        return reader(image)
