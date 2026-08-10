from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Set

import numpy as np

from .identity import IdentityPrediction


def load_allowed_ids(workbook: Optional[Path] = None) -> Set[str]:
    if workbook is None:
        root = Path(__file__).resolve().parents[3] / "latest-data"
        workbooks = sorted(root.glob("*.xlsx"))
        workbook = workbooks[0] if workbooks else None
    if workbook is None or not workbook.exists():
        return {str(i) for i in range(1, 18)}
    from .labeling import parse_roster

    return {
        row["pig_id"]
        for row in parse_roster(workbook)
        if row.get("present", True) and str(row.get("pig_id", "")).isdigit()
    }


def extract_mark_views(image: np.ndarray) -> list[np.ndarray]:
    """Isolate purple/blue ink and build OCR-friendly grayscale views."""
    import cv2

    if image is None or image.size == 0:
        return []
    height, width = image.shape[:2]
    if height < 8 or width < 8:
        return []

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = (
        ((hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 170) & (hsv[:, :, 1] >= 40) & (hsv[:, :, 2] >= 40))
        | ((hsv[:, :, 0] >= 100) & (hsv[:, :, 0] <= 140) & (hsv[:, :, 1] >= 25))
    ).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    coordinates = cv2.findNonZero(mask)
    if coordinates is None:
        # Fall back to full crop when paint is faint/smudged.
        region = image
    else:
        x, y, w, h = cv2.boundingRect(coordinates)
        pad = max(4, int(0.15 * max(w, h)))
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(width, x + w + pad), min(height, y + h + pad)
        region = image[y1:y2, x1:x2]
        if region.size == 0:
            region = image

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
    )
    # Keep ink-colored pixels too for smudged marks.
    region_hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    ink = (
        (region_hsv[:, :, 0] >= 90)
        & (region_hsv[:, :, 0] <= 170)
        & (region_hsv[:, :, 1] >= 35)
    ).astype(np.uint8) * 255
    combined = cv2.bitwise_or(binary, ink)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    views = []
    # Keep a small set of views: OCR is expensive and full-video runs call it often.
    for source in (gray, combined):
        scaled = cv2.resize(source, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        for angle in (0, 90, 180, 270):
            rotated = _rotate(scaled, angle)
            views.append(cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR))
    return views


def _rotate(image: np.ndarray, angle: float) -> np.ndarray:
    import cv2

    if angle % 360 == 0:
        return image
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_width = int(height * sine + width * cosine)
    new_height = int(height * cosine + width * sine)
    matrix[0, 2] += new_width / 2 - center[0]
    matrix[1, 2] += new_height / 2 - center[1]
    return cv2.warpAffine(
        image, matrix, (new_width, new_height), borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )


@lru_cache(maxsize=1)
def _reader():
    import os
    import easyocr

    model_dir = Path(__file__).resolve().parents[2] / "artifacts" / ".easyocr"
    model_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("EASYOCR_MODULE_PATH", str(model_dir))
    return easyocr.Reader(
        ["en"],
        gpu=False,
        verbose=False,
        model_storage_directory=str(model_dir),
        download_enabled=True,
    )


def _normalize_digits(text: str) -> str:
    cleaned = re.sub(r"[^0-9]", "", text or "")
    # Common OCR confusions for handwritten livestock marks.
    return cleaned.lstrip("0") or cleaned


def _best_roster_match(raw: str, allowed: Iterable[str]) -> Optional[str]:
    digits = _normalize_digits(raw)
    if not digits:
        return None
    allowed_set = set(allowed)
    if digits in allowed_set:
        return digits
    # Prefer longest exact suffix/prefix match among valid IDs.
    candidates = [
        value
        for value in allowed_set
        if digits.endswith(value) or digits.startswith(value) or value in digits
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda value: (len(value), value), reverse=True)[0]


def read_pig_number(
    image: np.ndarray,
    allowed_ids: Optional[Set[str]] = None,
    min_confidence: float = 0.25,
) -> IdentityPrediction:
    """Read a painted pig number from a pig crop; designed for smudged ink."""
    from collections import defaultdict

    allowed = allowed_ids or load_allowed_ids()
    views = extract_mark_views(image)
    if not views:
        return IdentityPrediction("unknown", 0.0)

    reader = _reader()
    votes: dict[str, list[float]] = defaultdict(list)
    for view in views:
        try:
            results = reader.readtext(
                view,
                allowlist="0123456789",
                detail=1,
                paragraph=False,
                mag_ratio=1.8,
            )
        except Exception:
            continue
        for _box, text, score in results:
            confidence = float(score)
            if confidence < min_confidence:
                continue
            digits = _normalize_digits(str(text))
            if not digits:
                continue
            matches = [value for value in allowed if value == digits or value in digits]
            for matched in matches:
                votes[matched].append(confidence + 0.12 * max(0, len(matched) - 1))

    if not votes:
        return IdentityPrediction("unknown", 0.0)

    ranked = sorted(
        votes.items(),
        key=lambda item: (sum(item[1]) / len(item[1]), len(item[0]), len(item[1])),
        reverse=True,
    )
    best_id, scores = ranked[0]
    best_score = float(sum(scores) / len(scores))
    # Strip length bonus for reported confidence.
    best_score = min(1.0, max(0.0, best_score - 0.15 * max(0, len(best_id) - 1)))
    if best_score < min_confidence:
        return IdentityPrediction("unknown", best_score, best_id)
    return IdentityPrediction(best_id, best_score, best_id)
