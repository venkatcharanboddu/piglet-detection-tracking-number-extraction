from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Point = Tuple[int, int]


@dataclass(frozen=True)
class FeedingROI:
    video_key: str
    zones: Dict[str, List[Point]]
    frame_width: int
    frame_height: int

    def validate(self) -> None:
        if not self.zones:
            raise ValueError("At least one feeding zone is required")
        for name, points in self.zones.items():
            if len(points) < 3:
                raise ValueError(f"Zone {name!r} must contain at least three points")
            if any(x < 0 or y < 0 for x, y in points):
                raise ValueError(f"Zone {name!r} has negative coordinates")

    def contains(self, point: Point) -> Optional[str]:
        for name, polygon in self.zones.items():
            if point_in_polygon(point, polygon):
                return name
        return None


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Boundary-inclusive ray casting, independent of OpenCV."""
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        if _on_segment(point, previous, current):
            return True
        crosses = (y1 > y) != (y2 > y)
        if crosses:
            x_intersection = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < x_intersection:
                inside = not inside
        previous = current
    return inside


def _on_segment(point: Point, start: Point, end: Point) -> bool:
    x, y = point
    x1, y1 = start
    x2, y2 = end
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if cross != 0:
        return False
    return min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)


def save_roi(roi: FeedingROI, output: Path) -> None:
    roi.validate()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "video_key": roi.video_key,
        "frame_width": roi.frame_width,
        "frame_height": roi.frame_height,
        "zones": {name: [list(point) for point in points] for name, points in roi.zones.items()},
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")


def load_roi(path: Path) -> FeedingROI:
    payload = json.loads(path.read_text())
    roi = FeedingROI(
        video_key=payload["video_key"],
        frame_width=int(payload["frame_width"]),
        frame_height=int(payload["frame_height"]),
        zones={
            name: [(int(point[0]), int(point[1])) for point in points]
            for name, points in payload["zones"].items()
        },
    )
    roi.validate()
    return roi


def mark_roi(video: Path, output: Path, zone_names: Iterable[str]) -> FeedingROI:
    """Interactive OpenCV polygon editor: click vertices, Enter accepts, R resets."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install project dependencies before using mark-roi") from exc
    from .inventory import stable_video_key

    capture = cv2.VideoCapture(str(video))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read first frame from {video}")

    zones: Dict[str, List[Point]] = {}
    window = "Feeding ROI"
    for name in zone_names:
        points: List[Point] = []

        def click(event: int, x: int, y: int, _flags: int, _param: object) -> None:
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))

        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, click)
        while True:
            preview = frame.copy()
            if points:
                import numpy as np

                cv2.polylines(preview, [np.asarray(points)], False, (0, 255, 255), 3)
            cv2.putText(
                preview,
                f"{name}: click >=3 points; Enter=accept, R=reset, Esc=cancel",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.imshow(window, preview)
            key = cv2.waitKey(20) & 0xFF
            if key in (10, 13) and len(points) >= 3:
                zones[name] = points.copy()
                break
            if key in (ord("r"), ord("R")):
                points.clear()
            if key == 27:
                cv2.destroyAllWindows()
                raise RuntimeError("ROI marking cancelled")
    cv2.destroyAllWindows()
    height, width = frame.shape[:2]
    roi = FeedingROI(stable_video_key(video), zones, width, height)
    save_roi(roi, output)
    return roi
