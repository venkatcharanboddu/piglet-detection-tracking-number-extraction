from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List


def create_detection_review(
    videos: Iterable[Path],
    weights: Iterable[Path],
    output_directory: Path,
    frames_per_video: int = 5,
) -> dict:
    import cv2
    from ultralytics import YOLO

    output_directory.mkdir(parents=True, exist_ok=True)
    models = [(path.stem + "-" + path.parent.parent.name, YOLO(str(path))) for path in weights]
    rows: List[dict] = []
    for video in videos:
        capture = cv2.VideoCapture(str(video))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            rows.append({"video": str(video), "status": "unreadable"})
            capture.release()
            continue
        indexes = sorted(
            {round(i * (frame_count - 1) / max(1, frames_per_video - 1)) for i in range(frames_per_video)}
        )
        for frame_index in indexes:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            for model_name, model in models:
                result = model.predict(frame, conf=0.25, iou=0.7, verbose=False)[0]
                detected = len(result.boxes) if result.boxes is not None else 0
                mean_confidence = (
                    float(result.boxes.conf.mean().cpu()) if detected else 0.0
                )
                image_name = f"{video.stem}_{frame_index}_{model_name}.jpg"
                cv2.imwrite(str(output_directory / image_name), result.plot())
                rows.append(
                    {
                        "video": str(video.resolve()),
                        "frame": frame_index,
                        "model": model_name,
                        "detected_count": detected,
                        "mean_confidence": mean_confidence,
                        "review_image": str((output_directory / image_name).resolve()),
                        "actual_count": "",
                        "false_positives": "",
                        "missed_pigs": "",
                        "status": "pending_review",
                    }
                )
        capture.release()
    manifest = output_directory / "detection_review.csv"
    _write_rows(manifest, rows)
    return {"review_rows": len(rows), "manifest": str(manifest)}


def score_detection_review(review_csv: Path) -> dict:
    rows = list(csv.DictReader(review_csv.open(newline="")))
    by_model: dict[str, dict] = {}
    for row in rows:
        if not row.get("actual_count"):
            continue
        model = row["model"]
        metrics = by_model.setdefault(
            model, {"true_positive": 0, "false_positive": 0, "false_negative": 0, "frames": 0}
        )
        detected = int(row["detected_count"])
        false_positive = int(row.get("false_positives") or 0)
        missed = int(row.get("missed_pigs") or 0)
        metrics["true_positive"] += max(0, detected - false_positive)
        metrics["false_positive"] += false_positive
        metrics["false_negative"] += missed
        metrics["frames"] += 1
    for metrics in by_model.values():
        tp, fp, fn = (
            metrics["true_positive"],
            metrics["false_positive"],
            metrics["false_negative"],
        )
        metrics["precision"] = tp / (tp + fp) if tp + fp else 0.0
        metrics["recall"] = tp / (tp + fn) if tp + fn else 0.0
    return by_model


def evaluate_events(predictions_csv: Path, ground_truth_csv: Path, min_iou: float = 0.5) -> dict:
    predicted = list(csv.DictReader(predictions_csv.open(newline="")))
    truth = list(csv.DictReader(ground_truth_csv.open(newline="")))
    used = set()
    matched = 0
    identity_correct = 0
    for prediction in predicted:
        best_index, best_iou = None, 0.0
        for index, target in enumerate(truth):
            if index in used or prediction.get("zone") != target.get("zone"):
                continue
            iou = _temporal_iou(prediction, target)
            if iou > best_iou:
                best_index, best_iou = index, iou
        if best_index is not None and best_iou >= min_iou:
            used.add(best_index)
            matched += 1
            identity_correct += prediction.get("pig_id") == truth[best_index].get("pig_id")
    metrics = {
        "predicted_events": len(predicted),
        "ground_truth_events": len(truth),
        "matched_events": matched,
        "event_precision": matched / len(predicted) if predicted else 0.0,
        "event_recall": matched / len(truth) if truth else 0.0,
        "identity_accuracy_on_matched": identity_correct / matched if matched else 0.0,
        "end_to_end_correct_events": identity_correct,
    }
    return metrics


def write_metrics(metrics: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n")


def _temporal_iou(first: dict, second: dict) -> float:
    first_start, first_end = int(first["start_frame"]), int(first["end_frame"])
    second_start, second_end = int(second["start_frame"]), int(second["end_frame"])
    intersection = max(0, min(first_end, second_end) - max(first_start, second_start) + 1)
    union = max(first_end, second_end) - min(first_start, second_start) + 1
    return intersection / union


def _write_rows(path: Path, rows: List[dict]) -> None:
    fields = list(rows[0]) if rows else [
        "video",
        "frame",
        "model",
        "detected_count",
        "mean_confidence",
        "review_image",
        "actual_count",
        "false_positives",
        "missed_pigs",
        "status",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
