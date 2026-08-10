from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class FeedingEvent:
    track_id: int
    zone: str
    start_frame: int
    end_frame: int
    duration_frames: int

    def to_record(self, fps: float) -> dict:
        record = asdict(self)
        record.update(
            start_seconds=self.start_frame / fps,
            end_seconds=self.end_frame / fps,
            duration_seconds=self.duration_frames / fps,
        )
        return record


@dataclass
class _ActiveEvent:
    zone: str
    start_frame: int
    last_inside_frame: int
    last_seen_frame: int


class FeedingEventMachine:
    """Track feeding-zone occupancy while tolerating brief detector gaps."""

    def __init__(self, min_frames: int = 20, max_gap_frames: int = 5):
        if min_frames < 1 or max_gap_frames < 0:
            raise ValueError("Invalid event thresholds")
        self.min_frames = min_frames
        self.max_gap_frames = max_gap_frames
        self._active: Dict[int, _ActiveEvent] = {}
        self.events: List[FeedingEvent] = []

    def update(self, frame_index: int, observations: Dict[int, Optional[str]]) -> None:
        observed_ids = set(observations)
        for track_id, zone in observations.items():
            active = self._active.get(track_id)
            if zone is not None:
                if active and active.zone == zone:
                    active.last_inside_frame = frame_index
                    active.last_seen_frame = frame_index
                else:
                    if active:
                        self._finish(track_id, active.last_inside_frame)
                    self._active[track_id] = _ActiveEvent(
                        zone=zone,
                        start_frame=frame_index,
                        last_inside_frame=frame_index,
                        last_seen_frame=frame_index,
                    )
            elif active:
                active.last_seen_frame = frame_index
                if frame_index - active.last_inside_frame > self.max_gap_frames:
                    self._finish(track_id, active.last_inside_frame)

        for track_id, active in list(self._active.items()):
            if track_id not in observed_ids and frame_index - active.last_seen_frame > self.max_gap_frames:
                self._finish(track_id, active.last_inside_frame)

    def _finish(self, track_id: int, end_frame: int) -> None:
        active = self._active.pop(track_id)
        duration = end_frame - active.start_frame + 1
        if duration >= self.min_frames:
            self.events.append(
                FeedingEvent(
                    track_id=track_id,
                    zone=active.zone,
                    start_frame=active.start_frame,
                    end_frame=end_frame,
                    duration_frames=duration,
                )
            )

    def flush(self, final_frame: int) -> List[FeedingEvent]:
        for track_id, active in list(self._active.items()):
            self._finish(track_id, min(final_frame, active.last_inside_frame))
        return self.events


def events_to_records(events: Iterable[FeedingEvent], fps: float) -> List[dict]:
    if fps <= 0:
        raise ValueError("FPS must be positive")
    return [event.to_record(fps) for event in events]
