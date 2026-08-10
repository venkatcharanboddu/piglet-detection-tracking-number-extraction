import numpy as np

from piglet_identity.crop_quality import (
    CropFilterConfig,
    box_overlap_fraction,
    box_overlaps_other_pig,
    reject_reason,
    sharpness_score,
)


def test_box_overlap_detects_two_pigs_in_frame() -> None:
    a = (10, 10, 100, 100)
    b = (80, 20, 160, 110)
    assert box_overlap_fraction(a, b) > 0.2
    others = [(1, a), (2, b)]
    assert box_overlaps_other_pig(a, 1, others)
    assert box_overlaps_other_pig(b, 2, others)


def test_reject_blurry_and_no_ink() -> None:
    flat = np.full((80, 80, 3), 128, dtype=np.uint8)
    cfg = CropFilterConfig(min_quality=1000.0, min_ink_fraction=0.5)
    box = (0, 0, 80, 80)
    assert reject_reason(flat, box, 640, 480, cfg) == "blurry"


def test_sharpness_positive_on_noise() -> None:
    noisy = np.random.randint(0, 255, (120, 120, 3), dtype=np.uint8)
    assert sharpness_score(noisy) > 0.0
