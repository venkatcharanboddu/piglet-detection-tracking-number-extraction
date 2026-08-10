from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .identity import IdentityPrediction


def _ink_mask(image: np.ndarray) -> np.ndarray:
    import cv2

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = (
        ((hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 170) & (hsv[:, :, 1] >= 40) & (hsv[:, :, 2] >= 40))
    ).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


def _normalized_mark(image: np.ndarray, size: int = 64) -> Optional[np.ndarray]:
    import cv2

    mask = _ink_mask(image)
    coordinates = cv2.findNonZero(mask)
    if coordinates is None:
        return None
    x, y, w, h = cv2.boundingRect(coordinates)
    if w < 3 or h < 3:
        return None
    mark = mask[y : y + h, x : x + w]
    scale = min((size - 8) / w, (size - 8) / h)
    resized = cv2.resize(
        mark,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_NEAREST,
    )
    canvas = np.zeros((size, size), dtype=np.uint8)
    y0 = (size - resized.shape[0]) // 2
    x0 = (size - resized.shape[1]) // 2
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return canvas


def _rotations(mask: np.ndarray) -> List[np.ndarray]:
    import cv2

    views = [mask]
    for angle in (90, 180, 270, -25, 25):
        matrix = cv2.getRotationMatrix2D((mask.shape[1] / 2, mask.shape[0] / 2), angle, 1.0)
        views.append(
            cv2.warpAffine(
                mask,
                matrix,
                (mask.shape[1], mask.shape[0]),
                flags=cv2.INTER_NEAREST,
                borderValue=0,
            )
        )
    return views


@lru_cache(maxsize=1)
def load_mark_templates(
    labels_csv: str = "",
) -> Dict[str, List[np.ndarray]]:
    import csv
    import cv2

    path = Path(labels_csv) if labels_csv else (
        Path(__file__).resolve().parents[2]
        / "artifacts"
        / "identity_labeling_v2"
        / "identity_labels.csv"
    )
    templates: Dict[str, List[np.ndarray]] = {}
    if not path.exists():
        return templates
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            pig_id = (row.get("pig_id") or "").strip()
            status = (row.get("status") or "").strip().lower()
            if status not in {"labeled", "ok"} or not pig_id.isdigit():
                continue
            image = cv2.imread(row["image_path"])
            if image is None:
                continue
            mark = _normalized_mark(image)
            if mark is None:
                continue
            templates.setdefault(pig_id, []).extend(_rotations(mark))
    return templates


def _ink_bbox_aspect(image: np.ndarray) -> float:
    import cv2

    mask = _ink_mask(image)
    coordinates = cv2.findNonZero(mask)
    if coordinates is None:
        return 1.0
    _x, _y, width, height = cv2.boundingRect(coordinates)
    return width / max(height, 1)


def _pick_id_with_two_digit_bias(
    scores_by_id: Dict[str, float], aspect: float, min_score: float
) -> Tuple[str, float]:
    """Reduce 12?1/2 splits when the ink blob is wide (two digits)."""
    if not scores_by_id:
        return "unknown", 0.0
    ordered = sorted(scores_by_id.items(), key=lambda item: item[1], reverse=True)
    best_id, best_score = ordered[0]
    if best_score < min_score:
        return "unknown", best_score

    wide_mark = aspect >= 1.28
    if wide_mark and len(best_id) == 1:
        two_digit = [
            (pig_id, score)
            for pig_id, score in ordered
            if len(pig_id) == 2 and score >= best_score - 0.12
        ]
        if two_digit:
            pig_id, score = max(two_digit, key=lambda item: (item[1], len(item[0])))
            if score >= min_score:
                return pig_id, score

    if len(best_id) == 1 and wide_mark and best_score < 0.55:
        return "unknown", best_score

    if len(ordered) > 1:
        second_id, second_score = ordered[1]
        margin = best_score - second_score
        if margin < 0.06 and len(best_id) != len(second_id):
            longer = best_id if len(best_id) >= len(second_id) else second_id
            longer_score = scores_by_id[longer]
            if wide_mark and len(longer) == 2 and longer_score >= min_score:
                return longer, longer_score
            return "unknown", best_score

    return best_id, best_score


def match_mark(
    image: np.ndarray,
    labels_csv: Optional[Path] = None,
    min_score: float = 0.45,
) -> IdentityPrediction:
    """Fast ink-shape matching against labeled marking photos."""
    import cv2

    aspect = _ink_bbox_aspect(image)
    query = _normalized_mark(image)
    if query is None:
        return IdentityPrediction("unknown", 0.0)
    templates = load_mark_templates(str(labels_csv) if labels_csv else "")
    if not templates:
        return IdentityPrediction("unknown", 0.0)

    scores_by_id: Dict[str, float] = {}
    query_views = _rotations(query)
    for pig_id, gallery in templates.items():
        peak = 0.0
        for template in gallery:
            for view in query_views:
                peak = max(
                    peak, float(cv2.matchTemplate(view, template, cv2.TM_CCOEFF_NORMED).max())
                )
        scores_by_id[pig_id] = peak

    best_id, best_score = _pick_id_with_two_digit_bias(scores_by_id, aspect, min_score)
    if best_id == "unknown":
        candidate = max(scores_by_id, key=scores_by_id.get) if scores_by_id else None
        return IdentityPrediction("unknown", best_score, candidate)
    return IdentityPrediction(best_id, best_score, best_id)
