from __future__ import annotations

import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional, Sequence, Set

import numpy as np

from .identity import IdentityPrediction
from .ocr import load_allowed_ids

DEFAULT_BASE_MODEL = "microsoft/trocr-small-handwritten"
DIGIT_TO_ID = {str(digit): digit + 3 for digit in range(10)}
ID_TO_DIGIT = {value: key for key, value in DIGIT_TO_ID.items()}
PAD_ID, BOS_ID, EOS_ID = 0, 1, 2


def _require_transformers():
    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    except ImportError as exc:
        raise RuntimeError(
            'TrOCR is not installed. Run: python -m pip install -e ".[trocr]"'
        ) from exc
    return TrOCRProcessor, VisionEncoderDecoderModel


def _device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def extract_mark_region(image: np.ndarray, padding: float = 0.2) -> np.ndarray:
    """Crop around blue/purple paint while retaining the original RGB appearance."""
    import cv2

    if image is None or image.size == 0:
        return image
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = (
        (hsv[:, :, 0] >= 90)
        & (hsv[:, :, 0] <= 170)
        & (hsv[:, :, 1] >= 35)
        & (hsv[:, :, 2] >= 35)
    ).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    coordinates = cv2.findNonZero(mask)
    if coordinates is None:
        return image
    x, y, w, h = cv2.boundingRect(coordinates)
    pad = max(4, round(max(w, h) * padding))
    return image[max(0, y - pad) : min(height, y + h + pad), max(0, x - pad) : min(width, x + w + pad)]


def _rotate_quarter(image: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(image, quarter_turns))


def _normalize_prediction(text: str, allowed_ids: Set[str]) -> Optional[str]:
    digits = re.sub(r"[^0-9]", "", text or "").lstrip("0")
    if not digits:
        return None
    return digits if digits in allowed_ids else None


def _decode_digit_ids(token_ids: Iterable[int]) -> str:
    return "".join(ID_TO_DIGIT[token] for token in token_ids if token in ID_TO_DIGIT)


class TrOCRPigletReader:
    """Fine-tuned sequence OCR reader for one- or two-digit painted marks."""

    def __init__(self, model_directory: Path, allowed_ids: Optional[Set[str]] = None):
        import torch

        TrOCRProcessor, VisionEncoderDecoderModel = _require_transformers()
        if not model_directory.exists():
            raise RuntimeError(
                f"Missing TrOCR model at {model_directory}. Run piglet-id train-ocr first."
            )
        self.processor = TrOCRProcessor.from_pretrained(str(model_directory))
        self.model = VisionEncoderDecoderModel.from_pretrained(str(model_directory))
        self.device = _device()
        self.model.to(self.device).eval()
        self.allowed_ids = allowed_ids or load_allowed_ids()
        self._torch = torch
        self.digit_only = (model_directory / "piglet_digit_vocab.json").exists()
        metrics_path = model_directory / "piglet_ocr_metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        self.validation_exact_match = float(metrics.get("validation_exact_match", 0.0))
        self.production_ready = self.validation_exact_match >= 0.8
        if self.digit_only:
            self.model.generation_config.decoder_start_token_id = BOS_ID
            self.model.generation_config.pad_token_id = PAD_ID
            self.model.generation_config.eos_token_id = EOS_ID
            self.model.generation_config.max_length = 4

    def predict(self, image: np.ndarray, min_confidence: float = 0.45) -> IdentityPrediction:
        import cv2
        from PIL import Image

        region = extract_mark_region(image)
        if region is None or region.size == 0:
            return IdentityPrediction("unknown", 0.0)

        views = [
            Image.fromarray(cv2.cvtColor(_rotate_quarter(region, turn), cv2.COLOR_BGR2RGB))
            for turn in range(4)
        ]
        pixels = self.processor(images=views, return_tensors="pt").pixel_values.to(self.device)
        with self._torch.inference_mode():
            generated = self.model.generate(
                pixels,
                max_new_tokens=3,
                num_beams=1,
                return_dict_in_generate=True,
                output_scores=True,
            )
        texts = (
            [_decode_digit_ids(sequence.tolist()) for sequence in generated.sequences]
            if self.digit_only
            else self.processor.batch_decode(generated.sequences, skip_special_tokens=True)
        )
        transition = self.model.compute_transition_scores(
            generated.sequences, generated.scores, normalize_logits=True
        )

        ranked = []
        for text, log_scores in zip(texts, transition):
            pig_id = _normalize_prediction(text, self.allowed_ids)
            if pig_id is None:
                continue
            finite = log_scores[self._torch.isfinite(log_scores)]
            confidence = (
                float(self._torch.exp(finite.mean()).cpu()) if len(finite) else 0.0
            )
            ranked.append((confidence, pig_id))
        if not ranked:
            return IdentityPrediction("unknown", 0.0)
        confidence, pig_id = max(ranked)
        if confidence < min_confidence or not self.production_ready:
            return IdentityPrediction("unknown", confidence, pig_id)
        return IdentityPrediction(pig_id, confidence, pig_id)


def _labeled_rows(labels_csv: Path, allowed_ids: Optional[Set[str]] = None) -> list[dict]:
    allowed_ids = allowed_ids or load_allowed_ids()
    rows = []
    with labels_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            pig_id = (row.get("pig_id") or "").strip()
            status = (row.get("status") or "").strip().lower()
            path = Path(row.get("image_path") or "")
            if (
                status in {"labeled", "ok"}
                and pig_id.isdigit()
                and pig_id in allowed_ids
                and path.exists()
            ):
                rows.append({**row, "pig_id": pig_id, "image_path": str(path)})
    return rows


def _group_key(row: dict) -> str:
    filename = Path(row["image_path"]).stem
    match = re.match(r"(track_\d+)_frame_", filename)
    if match:
        return f"{row.get('source_video', '')}:{match.group(1)}"
    return str(Path(row["image_path"]).resolve())


def _split_rows(rows: Sequence[dict], validation_fraction: float) -> tuple[list[dict], list[dict]]:
    """Split whole tracks, preventing nearby frames from leaking into validation."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[_group_key(row)].append(row)
    keys_by_label: dict[str, list[str]] = defaultdict(list)
    for key, values in groups.items():
        keys_by_label[values[0]["pig_id"]].append(key)
    validation_keys = set()
    rng = random.Random(42)
    for keys in keys_by_label.values():
        keys = sorted(keys)
        rng.shuffle(keys)
        count = min(len(keys) - 1, max(1, round(len(keys) * validation_fraction)))
        if count > 0:
            validation_keys.update(keys[:count])
    train = [row for key, values in groups.items() if key not in validation_keys for row in values]
    validation = [row for key, values in groups.items() if key in validation_keys for row in values]
    return train, validation


def train_trocr(
    labels_csv: Path,
    output_directory: Path,
    base_model: str = DEFAULT_BASE_MODEL,
    epochs: int = 8,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    validation_fraction: float = 0.2,
) -> dict:
    """Fine-tune TrOCR from full sequence labels; no digit bounding boxes are required."""
    import cv2
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

    TrOCRProcessor, VisionEncoderDecoderModel = _require_transformers()
    rows = _labeled_rows(labels_csv)
    if len(rows) < 20:
        raise ValueError("TrOCR training needs at least 20 labeled images")
    train_rows, validation_rows = _split_rows(rows, validation_fraction)
    if not train_rows or not validation_rows:
        raise ValueError("Could not create non-empty train and validation splits")

    processor = TrOCRProcessor.from_pretrained(base_model)
    model = VisionEncoderDecoderModel.from_pretrained(base_model)
    model.config.decoder_start_token_id = BOS_ID
    model.config.pad_token_id = PAD_ID
    model.config.eos_token_id = EOS_ID
    model.config.max_length = 4
    model.decoder.resize_token_embeddings(13)
    model.config.decoder.vocab_size = 13
    model.config.vocab_size = 13
    model.generation_config.decoder_start_token_id = BOS_ID
    model.generation_config.pad_token_id = PAD_ID
    model.generation_config.eos_token_id = EOS_ID
    model.generation_config.max_length = 4

    class MarkDataset(Dataset):
        def __init__(self, records: Sequence[dict], augment: bool):
            self.records = records
            self.augment = augment

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            row = self.records[index]
            image = cv2.imread(row["image_path"])
            image = extract_mark_region(image)
            if self.augment:
                angle = random.uniform(-15, 15)
                height, width = image.shape[:2]
                matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
                image = cv2.warpAffine(
                    image,
                    matrix,
                    (width, height),
                    borderMode=cv2.BORDER_REFLECT_101,
                )
            rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            pixels = processor(images=rgb, return_tensors="pt").pixel_values.squeeze(0)
            tokens = [BOS_ID] + [DIGIT_TO_ID[digit] for digit in row["pig_id"]] + [EOS_ID]
            tokens = (tokens + [PAD_ID] * 4)[:4]
            labels = torch.tensor([token if token != PAD_ID else -100 for token in tokens])
            return pixels, labels, row["pig_id"]

    def collate(batch):
        pixels, labels, texts = zip(*batch)
        return torch.stack(pixels), torch.stack(labels), texts

    device = _device()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    label_counts = defaultdict(int)
    for row in train_rows:
        label_counts[row["pig_id"]] += 1
    sample_weights = [1.0 / label_counts[row["pig_id"]] for row in train_rows]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_rows), replacement=True)
    loader = DataLoader(
        MarkDataset(train_rows, augment=True),
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate,
    )
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for pixels, labels, _texts in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model(pixel_values=pixels.to(device), labels=labels.to(device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch + 1, "train_loss": sum(losses) / max(1, len(losses))})

    model.eval()
    correct = 0
    validation_loader = DataLoader(
        MarkDataset(validation_rows, augment=False),
        batch_size=batch_size,
        collate_fn=collate,
    )
    with torch.inference_mode():
        for pixels, _labels, texts in validation_loader:
            generated = model.generate(pixels.to(device), max_new_tokens=3)
            predictions = [_decode_digit_ids(sequence.tolist()) for sequence in generated]
            correct += sum(
                re.sub(r"[^0-9]", "", prediction).lstrip("0") == truth
                for prediction, truth in zip(predictions, texts)
            )

    output_directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_directory)
    processor.save_pretrained(output_directory)
    (output_directory / "piglet_digit_vocab.json").write_text(
        json.dumps(
            {
                "pad_id": PAD_ID,
                "bos_id": BOS_ID,
                "eos_id": EOS_ID,
                "digits": DIGIT_TO_ID,
            },
            indent=2,
        )
        + "\n"
    )
    exact_match = correct / len(validation_rows)
    metrics = {
        "base_model": base_model,
        "training_images": len(train_rows),
        "validation_images": len(validation_rows),
        "validation_exact_match": exact_match,
        "production_ready": exact_match >= 0.8,
        "epochs": epochs,
        "batch_size": batch_size,
        "device": str(device),
        "history": history,
    }
    (output_directory / "piglet_ocr_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics
