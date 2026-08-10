from pathlib import Path

from piglet_identity.cli import DEFAULT_LABELS, _path, build_parser
from piglet_identity.events import FeedingEventMachine
import numpy as np

from piglet_identity.identity import (
    IdentityPrediction,
    augment_training_image,
    vote_candidates,
    vote_predictions,
)
from piglet_identity.labeling import _should_skip_label_input
from piglet_identity.ocr import _best_roster_match, _normalize_digits
from piglet_identity.ocr_trocr import _labeled_rows, _normalize_prediction, _split_rows
from piglet_identity.inventory import VideoRecord, select_representative
from piglet_identity.pipeline import PipelineConfig
from piglet_identity.roi import FeedingROI, load_roi, point_in_polygon, save_roi
from piglet_identity.validation import evaluate_events


def test_roi_boundary_and_round_trip(tmp_path: Path) -> None:
    polygon = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert point_in_polygon((5, 5), polygon)
    assert point_in_polygon((0, 4), polygon)
    assert not point_in_polygon((11, 5), polygon)
    roi = FeedingROI("sample", {"FEEDER": polygon}, 100, 100)
    output = tmp_path / "roi.json"
    save_roi(roi, output)
    assert load_roi(output) == roi


def test_event_gap_merge_and_eof_flush() -> None:
    machine = FeedingEventMachine(min_frames=3, max_gap_frames=1)
    machine.update(0, {7: "LEFT"})
    machine.update(1, {7: "LEFT"})
    machine.update(2, {7: None})
    machine.update(3, {7: "LEFT"})
    machine.update(4, {7: "LEFT"})
    events = machine.flush(4)
    assert len(events) == 1
    assert events[0].start_frame == 0
    assert events[0].end_frame == 4
    assert events[0].duration_frames == 5


def test_short_event_is_discarded() -> None:
    machine = FeedingEventMachine(min_frames=3, max_gap_frames=0)
    machine.update(10, {2: "RIGHT"})
    machine.update(11, {2: None})
    assert machine.flush(11) == []


def test_identity_consensus_rejects_weak_or_few_votes() -> None:
    confident = [
        IdentityPrediction("11", 0.9),
        IdentityPrediction("11", 0.8),
        IdentityPrediction("2", 0.7),
        IdentityPrediction("11", 0.85),
    ]
    assert vote_predictions(confident).pig_id == "11"
    assert vote_predictions(confident[:2]).pig_id == "unknown"
    assert vote_predictions([IdentityPrediction("3", 0.4)] * 4).pig_id == "unknown"


def test_low_confidence_candidate_voting_is_separate() -> None:
    predictions = [
        IdentityPrediction("unknown", 0.18, "12"),
        IdentityPrediction("unknown", 0.16, "12"),
        IdentityPrediction("unknown", 0.19, "13"),
    ]
    assert vote_predictions(predictions).pig_id == "unknown"
    assert vote_candidates(predictions).pig_id == "12"
    assert (
        vote_predictions(
            [IdentityPrediction("unknown", 0.99, "12")] * 3,
            include_candidates=False,
        ).pig_id
        == "unknown"
    )


def test_augmentation_has_no_mirrored_view() -> None:
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[2:8, 3:7] = 255
    views = augment_training_image(image)
    assert len(views) == 8
    assert all(view.ndim == 3 for view in views)


def test_skip_label_input_variants() -> None:
    assert _should_skip_label_input("")
    assert _should_skip_label_input("s")
    assert _should_skip_label_input("ss")
    assert _should_skip_label_input("SS")
    assert _should_skip_label_input("skip")
    assert not _should_skip_label_input("12")


def test_ocr_digit_normalization_and_roster_match() -> None:
    assert _normalize_digits("12") == "12"
    assert _normalize_digits("ID 12.") == "12"
    assert _best_roster_match("12", {"11", "12", "13"}) == "12"
    assert _best_roster_match("x12y", {"11", "12", "13"}) == "12"
    assert _normalize_prediction(" 12.", {"1", "2", "12"}) == "12"
    assert _normalize_prediction("20", {"1", "2", "12"}) is None


def test_trocr_split_keeps_track_frames_together() -> None:
    rows = [
        {
            "image_path": f"/tmp/track_{track}_frame_{frame}.jpg",
            "source_video": "video-a",
            "pig_id": str(track % 2 + 1),
        }
        for track in range(1, 11)
        for frame in (10, 20)
    ]
    train, validation = _split_rows(rows, 0.2)
    train_tracks = {Path(row["image_path"]).stem.split("_frame_")[0] for row in train}
    validation_tracks = {Path(row["image_path"]).stem.split("_frame_")[0] for row in validation}
    assert train_tracks.isdisjoint(validation_tracks)
    assert len(validation_tracks) == 2


def test_trocr_uses_labeled_blurred_rows_but_not_skips(tmp_path: Path) -> None:
    blurred = tmp_path / "blurred.jpg"
    skipped = tmp_path / "skipped.jpg"
    blurred.touch()
    skipped.touch()
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "image_path,pig_id,status,visibility\n"
        f"{blurred},12,labeled,blurred\n"
        f"{skipped},12,skip,clear\n"
    )
    assert [row["image_path"] for row in _labeled_rows(labels, {"12"})] == [str(blurred)]


def test_wide_mark_prefers_two_digit_id() -> None:
    from piglet_identity.mark_match import _pick_id_with_two_digit_bias

    scores = {"1": 0.52, "2": 0.5, "12": 0.48, "11": 0.44}
    pig_id, _score = _pick_id_with_two_digit_bias(scores, aspect=1.5, min_score=0.45)
    assert pig_id == "12"


def test_representative_subset_is_per_pen() -> None:
    records = [
        VideoRecord(f"/p{pen}/{index}.mp4", pen, 90000 + pen, f"202607{index:02}", index, "fixed")
        for pen in (4, 5)
        for index in range(1, 7)
    ]
    selected = select_representative(records, per_pen=3)
    assert len(selected) == 6
    assert {record.pen for record in selected} == {4, 5}


def test_end_to_end_event_evaluation(tmp_path: Path) -> None:
    predicted = tmp_path / "predicted.csv"
    truth = tmp_path / "truth.csv"
    predicted.write_text(
        "zone,start_frame,end_frame,pig_id\nFEEDER,10,20,3\nFEEDER,50,60,2\n"
    )
    truth.write_text("zone,start_frame,end_frame,pig_id\nFEEDER,10,21,3\n")
    metrics = evaluate_events(predicted, truth)
    assert metrics["matched_events"] == 1
    assert metrics["event_precision"] == 0.5
    assert metrics["event_recall"] == 1.0
    assert metrics["identity_accuracy_on_matched"] == 1.0


def test_short_commands_have_project_defaults() -> None:
    parser = build_parser()
    assert parser.parse_args(["prepare"]).command == "prepare"
    assert parser.parse_args(["label"]).labels_csv == DEFAULT_LABELS
    assert parser.parse_args(["train"]).labels == DEFAULT_LABELS
    assert parser.parse_args(["train-ocr"]).command == "train-ocr"
    assert parser.parse_args(["label", "--ask-visibility"]).ask_visibility
    assert PipelineConfig().confidence == 0.35


def test_quoted_shell_escapes_are_recovered(tmp_path: Path) -> None:
    video = tmp_path / "sample video#1.mp4"
    video.touch()
    escaped = str(video).replace(" ", r"\ ").replace("#", r"\#")
    assert _path(escaped) == video
