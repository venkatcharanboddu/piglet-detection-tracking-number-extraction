from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class CropFilterConfig:
    min_quality: float = 55.0
    min_ink_fraction: float = 0.0008
    reject_edge_boxes: bool = True
    edge_margin_fraction: float = 0.02
    min_aspect: float = 0.35
    max_aspect: float = 2.8
    min_gray_std: float = 12.0
    # Soft multi-pig gate used by the pipeline (None = use pipeline default / off)
    multi_pig_overlap: Optional[float] = 0.22


def sharpness_score(crop: np.ndarray) -> float:
    import cv2

    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    area_factor = min(1.0, crop.shape[0] * crop.shape[1] / 40000)
    return lap * area_factor


def ink_fraction(crop: np.ndarray) -> float:
    import cv2

    if crop is None or crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = (
        (hsv[:, :, 0] >= 90)
        & (hsv[:, :, 0] <= 170)
        & (hsv[:, :, 1] >= 35)
        & (hsv[:, :, 2] >= 35)
    )
    return float(mask.sum()) / max(1, mask.size)


def ranking_score(crop: np.ndarray) -> float:
    """Sharpness with a bonus for visible mark ink (prefer readable backs)."""
    sharp = sharpness_score(crop)
    if sharp <= 0:
        return 0.0
    return sharp * (1.0 + 40.0 * ink_fraction(crop))


def box_touches_frame_edge(
    box: Tuple[int, int, int, int],
    width: int,
    height: int,
    margin_fraction: float,
) -> bool:
    x1, y1, x2, y2 = box
    mx = int(width * margin_fraction)
    my = int(height * margin_fraction)
    return x1 <= mx or y1 <= my or x2 >= width - mx or y2 >= height - my


def box_aspect_ratio(box: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    return width / height


def reject_reason(
    crop: np.ndarray,
    box: Tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
    config: CropFilterConfig,
) -> Optional[str]:
    import cv2

    if crop is None or crop.size == 0:
        return "empty"
    quality = sharpness_score(crop)
    if quality < config.min_quality:
        return "blurry"
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if float(np.std(gray)) < config.min_gray_std:
        return "flat"
    aspect = box_aspect_ratio(box)
    if aspect < config.min_aspect or aspect > config.max_aspect:
        return "bad_aspect"
    if config.reject_edge_boxes and box_touches_frame_edge(
        box, frame_width, frame_height, config.edge_margin_fraction
    ):
        return "edge_partial"
    if ink_fraction(crop) < config.min_ink_fraction:
        return "no_ink"
    return None


def box_overlap_fraction(
    box_a: Tuple[int, int, int, int],
    box_b: Tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / min(area_a, area_b)


def box_overlaps_other_pig(
    box: Tuple[int, int, int, int],
    track_id: int,
    others: list[tuple[int, Tuple[int, int, int, int]]],
    min_overlap: float = 0.22,
) -> bool:
    for other_id, other_box in others:
        if other_id == track_id:
            continue
        if box_overlap_fraction(box, other_box) >= min_overlap:
            return True
    return False
