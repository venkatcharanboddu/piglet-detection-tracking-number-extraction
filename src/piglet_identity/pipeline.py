from __future__ import annotations

import csv
import heapq
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .crop_quality import (
    CropFilterConfig,
    box_overlaps_other_pig,
    ink_fraction,
    ranking_score,
    reject_reason,
    sharpness_score,
)
from .events import FeedingEventMachine, events_to_records
from .identity import IdentityModel, IdentityPrediction, vote_candidates, vote_predictions
from .roi import FeedingROI


@dataclass
class PipelineConfig:
    confidence: float = 0.35
    iou: float = 0.7
    imgsz: int = 960
    min_feed_seconds: float = 2.0
    max_gap_seconds: float = 0.5
    crop_interval: int = 15
    best_crops_per_track: int = 6
    max_box_area_fraction: float = 0.18
    max_frames: Optional[int] = None
    use_ocr: bool = False
    ocr_on_feeding_tracks: bool = False
    identity_method: str = "hybrid"
    trocr_model_path: Optional[Path] = None
    digit_weights_path: Optional[Path] = None
    identity_during_video: bool = False
    write_annotated_video: bool = True
    progress_every: int = 200
    label_sample_only: bool = False
    min_crop_quality: float = 0.0
    strict_identity_crops: bool = True
    # training | strict | loose — sample-crops defaults to training
    crop_profile: str = "strict"
    min_ink_fraction: Optional[float] = None
    track_vote_threshold: Optional[float] = None
    track_min_votes: Optional[int] = None


def _crop_filter_config(config: PipelineConfig) -> CropFilterConfig:
    profile = config.crop_profile
    if not config.strict_identity_crops:
        profile = "loose"

    min_quality = config.min_crop_quality
    if profile == "training":
        if min_quality <= 0:
            min_quality = 35.0
        return CropFilterConfig(
            min_quality=min_quality,
            min_ink_fraction=0.0,  # prefer ink via ranking_score, do not hard-reject
            reject_edge_boxes=False,
            min_aspect=0.25,
            max_aspect=3.5,
            min_gray_std=8.0,
            multi_pig_overlap=0.45,
        )
    if profile == "loose":
        if min_quality <= 0:
            min_quality = 0.0
        return CropFilterConfig(
            min_quality=min_quality,
            min_ink_fraction=0.0,
            reject_edge_boxes=False,
            multi_pig_overlap=None,
        )
    # strict (analyze default)
    if min_quality <= 0:
        min_quality = 55.0
    ink = config.min_ink_fraction
    if ink is None:
        ink = 0.0008
    return CropFilterConfig(
        min_quality=min_quality,
        min_ink_fraction=ink,
        reject_edge_boxes=True,
        multi_pig_overlap=0.22,
    )


def _track_vote_settings(
    config: PipelineConfig, method: Optional[str] = None
) -> tuple[float, int, bool]:
    """Per-method track voting: threshold, min_votes, include_candidate_ids."""
    key = method or config.identity_method
    if config.track_vote_threshold is not None and config.track_min_votes is not None:
        include = key not in {"trocr", "easyocr"}
        return config.track_vote_threshold, config.track_min_votes, include
    if key == "classifier":
        return 0.38, 1, True
    if key == "mark":
        return 0.45, 2, True
    if key == "digits":
        return 0.35, 2, True
    if key in {"trocr", "easyocr"}:
        return 0.48, 2, False
    return 0.48, 2, True


def _scaled_roi(roi: FeedingROI, width: int, height: int) -> FeedingROI:
    if (width, height) == (roi.frame_width, roi.frame_height):
        return roi
    x_scale, y_scale = width / roi.frame_width, height / roi.frame_height
    return FeedingROI(
        video_key=roi.video_key,
        frame_width=width,
        frame_height=height,
        zones={
            name: [(round(x * x_scale), round(y * y_scale)) for x, y in points]
            for name, points in roi.zones.items()
        },
    )


def _zone_for_box(box: Tuple[int, int, int, int], roi: FeedingROI) -> Optional[str]:
    """Prefer the forward/upper body point (notebook-style), then fall back to votes."""
    x1, y1, x2, y2 = box
    preferred = (int((x1 + x2) / 2), int(y1 + 0.25 * (y2 - y1)))
    zone = roi.contains(preferred)
    if zone is not None:
        return zone
    xs = [x1, (x1 + x2) // 2, x2]
    ys = [y1, (y1 + y2) // 2, y2]
    votes = Counter(roi.contains((x, y)) for x in xs for y in ys)
    votes.pop(None, None)
    return votes.most_common(1)[0][0] if votes else None


def _draw_roi(frame: np.ndarray, roi: FeedingROI) -> None:
    import cv2

    for name, points in roi.zones.items():
        polygon = np.asarray(points, dtype=np.int32)
        cv2.polylines(frame, [polygon], True, (0, 255, 255), 3)
        cv2.putText(
            frame,
            name,
            tuple(polygon[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )


def _write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_pipeline(
    video: Path,
    detector_weights: Path,
    roi: FeedingROI,
    output_directory: Path,
    config: PipelineConfig,
    identity_model_path: Optional[Path] = None,
) -> dict:
    import cv2
    from ultralytics import YOLO

    output_directory.mkdir(parents=True, exist_ok=True)
    if config.label_sample_only:
        crop_directory = output_directory
        # Re-sample should not keep stale track_*.jpg from an older run
        for stale in crop_directory.glob("track_*.jpg"):
            stale.unlink()
    else:
        crop_directory = output_directory / "identity_crops" / video.stem
    crop_directory.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid or unreadable video: {video}")
    scaled_roi = _scaled_roi(roi, width, height)

    event_machine = None
    if not config.label_sample_only:
        event_machine = FeedingEventMachine(
            min_frames=max(1, round(config.min_feed_seconds * fps)),
            max_gap_frames=max(0, round(config.max_gap_seconds * fps)),
        )
    detector = YOLO(str(detector_weights))
    identity_model = None
    if not config.label_sample_only:
        identity_model = IdentityModel(
            identity_model_path,
            use_ocr=False,
            method=config.identity_method,
            trocr_model_path=config.trocr_model_path,
            digit_weights_path=config.digit_weights_path,
        )
    candidates: Dict[int, List[tuple]] = defaultdict(list)
    live_ids: Dict[int, IdentityPrediction] = {}
    frame_index = -1
    processed_frames = 0
    detection_count = 0
    track_ids_seen = set()
    crop_filter_stats: Counter = Counter()
    crop_filter_cfg = _crop_filter_config(config)
    writer = None
    if config.write_annotated_video:
        writer = cv2.VideoWriter(
            str(output_directory / f"{video.stem}_annotated.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("Could not open annotated-video writer")

    limit = config.max_frames if config.max_frames is not None else frame_count
    if config.progress_every > 0:
        print(
            f"Processing up to {limit} frames (YOLO imgsz={config.imgsz}, CPU)...",
            flush=True,
        )

    results = detector.track(
        source=str(video),
        stream=True,
        persist=True,
        tracker="botsort.yaml",
        conf=config.confidence,
        iou=config.iou,
        imgsz=config.imgsz,
        verbose=False,
    )
    try:
        for frame_index, result in enumerate(results):
            if config.max_frames is not None and frame_index >= config.max_frames:
                break
            clean_frame = result.orig_img
            annotated_frame = clean_frame.copy() if writer is not None else None
            observations: Dict[int, Optional[str]] = {}
            frame_piglets: List[tuple[int, Tuple[int, int, int, int], float]] = []
            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                coordinates = boxes.xyxy.cpu().numpy().astype(int)
                track_ids = boxes.id.cpu().numpy().astype(int)
                confidences = boxes.conf.cpu().numpy()
                detection_count += len(track_ids)
                for box, track_id, confidence in zip(coordinates, track_ids, confidences):
                    x1, y1, x2, y2 = (
                        max(0, box[0]),
                        max(0, box[1]),
                        min(width, box[2]),
                        min(height, box[3]),
                    )
                    if x2 <= x1 or y2 <= y1:
                        continue
                    is_piglet = (
                        (x2 - x1) * (y2 - y1) / float(width * height)
                        <= config.max_box_area_fraction
                    )
                    if is_piglet:
                        track_ids_seen.add(int(track_id))
                        observations[int(track_id)] = _zone_for_box(
                            (x1, y1, x2, y2), scaled_roi
                        )
                        frame_piglets.append(
                            (int(track_id), (x1, y1, x2, y2), float(confidence))
                        )
                    if writer is not None and annotated_frame is not None:
                        color = (0, 200, 0) if is_piglet else (128, 128, 128)
                        label = f"track {track_id} {confidence:.2f}"
                        if not is_piglet:
                            label = f"ignored-large {track_id} {confidence:.2f}"
                        live = live_ids.get(int(track_id))
                        if live is not None and is_piglet:
                            shown = live.pig_id if live.pig_id != "unknown" else live.candidate_id
                            if shown:
                                label = f"#{shown} t{track_id} {confidence:.2f}"
                                if live.pig_id != "unknown":
                                    color = (0, 165, 255)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            annotated_frame,
                            label,
                            (x1, max(20, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            color,
                            2,
                        )
                if frame_index % config.crop_interval == 0 and frame_piglets:
                    piglet_boxes = [(tid, b) for tid, b, _ in frame_piglets]
                    multi_overlap = crop_filter_cfg.multi_pig_overlap
                    for track_id, (x1, y1, x2, y2), _confidence in frame_piglets:
                        if multi_overlap is not None and box_overlaps_other_pig(
                            (x1, y1, x2, y2),
                            track_id,
                            piglet_boxes,
                            min_overlap=multi_overlap,
                        ):
                            crop_filter_stats["multi_pig"] += 1
                            continue
                        crop = clean_frame[y1:y2, x1:x2].copy()
                        reason = reject_reason(
                            crop,
                            (x1, y1, x2, y2),
                            width,
                            height,
                            crop_filter_cfg,
                        )
                        if reason:
                            crop_filter_stats[reason] += 1
                            continue
                        score = ranking_score(crop)
                        item = (score, frame_index, crop)
                        heap = candidates[int(track_id)]
                        if len(heap) < config.best_crops_per_track:
                            heapq.heappush(heap, item)
                        elif score > heap[0][0]:
                            heapq.heapreplace(heap, item)
                        if config.identity_during_video and identity_model is not None:
                            prediction = identity_model.predict(crop)
                            previous = live_ids.get(int(track_id))
                            if previous is None or prediction.confidence >= previous.confidence:
                                live_ids[int(track_id)] = prediction
            if event_machine is not None:
                event_machine.update(frame_index, observations)
            if writer is not None and annotated_frame is not None:
                _draw_roi(annotated_frame, scaled_roi)
                writer.write(annotated_frame)
            processed_frames += 1
            if (
                config.progress_every > 0
                and processed_frames % config.progress_every == 0
            ):
                print(f"  frame {processed_frames}/{limit}", flush=True)
    finally:
        if writer is not None:
            writer.release()

    crop_manifest = []
    skipped_blurry = 0
    if config.label_sample_only:
        if config.progress_every > 0:
            print(
                f"Writing up to {config.best_crops_per_track} ranked crop(s) per track...",
                flush=True,
            )
        for track_id, heap in sorted(candidates.items()):
            if not heap:
                skipped_blurry += 1
                continue
            ranked = sorted(heap, key=lambda item: item[0], reverse=True)
            wrote = 0
            for score, candidate_frame, crop in ranked:
                sharp = sharpness_score(crop)
                if config.min_crop_quality > 0 and sharp < config.min_crop_quality:
                    continue
                if config.best_crops_per_track == 1:
                    crop_path = crop_directory / f"track_{track_id}.jpg"
                else:
                    crop_path = (
                        crop_directory / f"track_{track_id}_f{candidate_frame}.jpg"
                    )
                cv2.imwrite(str(crop_path), crop)
                crop_manifest.append(
                    {
                        "image_path": str(crop_path.resolve()),
                        "track_id": track_id,
                        "frame": candidate_frame,
                        "quality": score,
                        "sharpness": sharp,
                        "ink_fraction": ink_fraction(crop),
                        "source_video": str(video.resolve()),
                        "status": "unlabeled",
                    }
                )
                wrote += 1
            if wrote == 0:
                skipped_blurry += 1
        events = []
        track_predictions = {}
        track_candidates = {}
        comparison_track_predictions = {}
    else:
        if config.progress_every > 0:
            print("Running identity on best crops per track...", flush=True)

        events = event_machine.flush(max(0, frame_index))
        track_predictions: Dict[int, IdentityPrediction] = {}
        track_candidates: Dict[int, IdentityPrediction] = {}
        comparison_track_predictions: Dict[str, Dict[int, IdentityPrediction]] = {
            method: {} for method in ("mark", "classifier", "trocr")
        }
        for track_id, heap in sorted(candidates.items()):
            predictions = []
            comparison_crop_predictions: Dict[str, List[IdentityPrediction]] = defaultdict(list)
            for quality, candidate_frame, crop in sorted(heap, reverse=True):
                crop_path = crop_directory / f"track_{track_id}_frame_{candidate_frame}.jpg"
                cv2.imwrite(str(crop_path), crop)
                method_results = (
                    identity_model.predict_all(crop)
                    if identity_model and config.identity_method == "compare"
                    else {}
                )
                if method_results:
                    mark = method_results["mark"]
                    classifier = method_results["classifier"]
                    prediction = mark if mark.pig_id != "unknown" else classifier
                    if prediction.pig_id == "unknown" and not prediction.candidate_id:
                        prediction = mark
                    for method, method_prediction in method_results.items():
                        comparison_crop_predictions[method].append(method_prediction)
                else:
                    prediction = identity_model.predict(crop) if identity_model else None
                if prediction:
                    predictions.append(prediction)
                crop_row = {
                    "image_path": str(crop_path.resolve()),
                    "track_id": track_id,
                    "frame": candidate_frame,
                    "quality": quality,
                    "pig_id": prediction.pig_id if prediction else "",
                    "confidence": prediction.confidence if prediction else "",
                    "candidate_pig_id": prediction.candidate_id if prediction else "",
                    "source_video": str(video.resolve()),
                }
                for method in ("mark", "classifier", "trocr"):
                    result = method_results.get(method)
                    crop_row[f"{method}_pig_id"] = result.pig_id if result else ""
                    crop_row[f"{method}_confidence"] = result.confidence if result else ""
                    crop_row[f"{method}_candidate_pig_id"] = result.candidate_id if result else ""
                crop_manifest.append(crop_row)
            vote_threshold, vote_min, include_cands = _track_vote_settings(config)
            track_predictions[track_id] = (
                vote_predictions(
                    predictions,
                    threshold=vote_threshold,
                    min_votes=vote_min,
                    include_candidates=include_cands,
                )
                if predictions
                else IdentityPrediction("unknown", 0.0)
            )
            track_candidates[track_id] = (
                vote_candidates(predictions) if predictions else IdentityPrediction("unknown", 0.0)
            )
            for method, method_predictions in comparison_crop_predictions.items():
                m_th, m_votes, m_cand = _track_vote_settings(config, method)
                comparison_track_predictions[method][track_id] = vote_predictions(
                    method_predictions,
                    threshold=m_th,
                    min_votes=m_votes,
                    include_candidates=m_cand,
                )
            live_ids[int(track_id)] = track_predictions[track_id]

    if (
        not config.label_sample_only
        and (config.ocr_on_feeding_tracks or config.use_ocr)
        and config.identity_method == "hybrid"
    ):
        if config.progress_every > 0:
            print("OCR refine on feeding tracks (slow)...", flush=True)
        from .ocr import read_pig_number

        feeding_track_ids = {int(event.track_id) for event in events}
        refine_ids = feeding_track_ids if config.ocr_on_feeding_tracks else set(candidates)
        for track_id in refine_ids:
            heap = candidates.get(track_id) or []
            if not heap:
                continue
            ocr_votes = []
            for quality, candidate_frame, crop in sorted(heap, reverse=True)[:3]:
                ocr_votes.append(read_pig_number(crop))
            ocr_vote = vote_predictions(ocr_votes, threshold=0.35, min_votes=1)
            if ocr_vote.pig_id != "unknown":
                track_predictions[track_id] = ocr_vote
                track_candidates[track_id] = ocr_vote
            else:
                candidate = vote_candidates(ocr_votes)
                if candidate.pig_id != "unknown":
                    track_candidates[track_id] = candidate

    event_rows = []
    if config.label_sample_only:
        manifest_name = "label_crops.csv"
        manifest_fields = [
            "image_path",
            "track_id",
            "frame",
            "quality",
            "sharpness",
            "ink_fraction",
            "source_video",
            "status",
        ]
    else:
        manifest_name = f"{video.stem}_crop_manifest.csv"
        manifest_fields = [
            "image_path",
            "track_id",
            "frame",
            "quality",
            "pig_id",
            "confidence",
            "candidate_pig_id",
            "mark_pig_id",
            "mark_confidence",
            "mark_candidate_pig_id",
            "classifier_pig_id",
            "classifier_confidence",
            "classifier_candidate_pig_id",
            "trocr_pig_id",
            "trocr_confidence",
            "trocr_candidate_pig_id",
            "source_video",
        ]
    for row in events_to_records(events, fps):
        prediction = track_predictions.get(
            int(row["track_id"]), IdentityPrediction("unknown", 0.0)
        )
        candidate = track_candidates.get(
            int(row["track_id"]), IdentityPrediction("unknown", 0.0)
        )
        row.update(
            pen=_pen_from_path(video),
            source_video=str(video.resolve()),
            pig_id=prediction.pig_id,
            identity_confidence=prediction.confidence,
            candidate_pig_id=candidate.pig_id,
            candidate_confidence=candidate.confidence,
        )
        for method in ("mark", "classifier", "trocr"):
            method_prediction = comparison_track_predictions[method].get(
                int(row["track_id"]), IdentityPrediction("unknown", 0.0)
            )
            row[f"{method}_pig_id"] = (
                method_prediction.pig_id if config.identity_method == "compare" else ""
            )
            row[f"{method}_identity_confidence"] = (
                method_prediction.confidence if config.identity_method == "compare" else ""
            )
        event_rows.append(row)
    event_fields = [
        "pen",
        "source_video",
        "pig_id",
        "identity_confidence",
        "candidate_pig_id",
        "candidate_confidence",
        "mark_pig_id",
        "mark_identity_confidence",
        "classifier_pig_id",
        "classifier_identity_confidence",
        "trocr_pig_id",
        "trocr_identity_confidence",
        "track_id",
        "zone",
        "start_frame",
        "end_frame",
        "duration_frames",
        "start_seconds",
        "end_seconds",
        "duration_seconds",
    ]
    if not config.label_sample_only:
        _write_csv(output_directory / f"{video.stem}_feeding_events.csv", event_rows, event_fields)
    _write_csv(output_directory / manifest_name, crop_manifest, manifest_fields)
    summary = {
        "video": str(video.resolve()),
        "fps": fps,
        "frames_reported": frame_count,
        "frames_processed": processed_frames,
        "detections": detection_count,
        "tracks": len(track_ids_seen),
        "identity_crop_rows": len(crop_manifest),
        "label_crops": len(crop_manifest) if config.label_sample_only else 0,
        "feeding_events": len(event_rows),
        "identified_events": sum(row.get("pig_id") != "unknown" for row in event_rows),
        "label_sample_only": config.label_sample_only,
        "skipped_blurry_tracks": skipped_blurry if config.label_sample_only else 0,
        "strict_identity_crops": config.strict_identity_crops,
        "crop_profile": config.crop_profile,
        "crops_per_track": config.best_crops_per_track,
        "crop_filter_rejected": dict(crop_filter_stats),
    }
    summary_name = (
        "summary.json" if config.label_sample_only else f"{video.stem}_summary.json"
    )
    (output_directory / summary_name).write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _pen_from_path(video: Path) -> str:
    text = str(video)
    if "Pen4_" in text:
        return "4"
    if "Pen5_" in text:
        return "5"
    if "Pen6_" in text:
        return "6"
    return ""
