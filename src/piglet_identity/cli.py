from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import List, Optional

from .identity import train_classifier
from .inventory import scan_videos, select_representative, write_manifest
from .digit_marks import (
    build_geometric_pseudo_labels,
    export_digit_dataset,
    label_digit_boxes,
    score_digit_reader,
)
from .mark_boxes import (
    export_mark_dataset,
    label_mark_boxes,
    score_mark_detector,
    train_mark_detector,
)
from .labeling import (
    export_manifest_labels_to_training,
    interactive_label,
    interactive_label_manifest,
    prepare_labeling_package,
    reset_label_rows,
    score_manifest_labels,
)
from .pipeline import PipelineConfig, run_pipeline
from .roi import load_roi, mark_roi
from .validation import (
    create_detection_review,
    evaluate_events,
    score_detection_review,
    write_metrics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT.parent / "latest-data"
DEFAULT_LABELS = PROJECT_ROOT / "artifacts" / "identity_labeling_v2" / "identity_labels.csv"
DEFAULT_MODEL = PROJECT_ROOT / "artifacts" / "models" / "mark_classifier_latest_data.joblib"
DEFAULT_WEIGHTS = PROJECT_ROOT.parent / "runs" / "detect" / "train-3" / "weights" / "best.pt"
DEFAULT_MARKING_PHOTOS = DATA_ROOT / "Pigs marking (1)"
DEFAULT_TROCR_MODEL = PROJECT_ROOT / "artifacts" / "models" / "trocr_piglet_digits"
DEFAULT_LABEL_SAMPLES = PROJECT_ROOT / "artifacts" / "label_samples"
DEFAULT_MARK_BOXES = PROJECT_ROOT / "artifacts" / "mark_boxes" / "mark_boxes.csv"
DEFAULT_MARK_DATASET = PROJECT_ROOT / "artifacts" / "mark_boxes" / "dataset"
DEFAULT_MARK_MODEL = PROJECT_ROOT / "artifacts" / "models" / "mark_detector"
DEFAULT_DIGIT_BOXES = PROJECT_ROOT / "artifacts" / "digit_boxes" / "digit_boxes.csv"
DEFAULT_DIGIT_DATASET = PROJECT_ROOT / "artifacts" / "digit_dataset"
DEFAULT_DIGIT_MODEL = PROJECT_ROOT / "artifacts" / "models" / "digit_detector"


def _default_digit_weights() -> Path:
    candidates = [
        DEFAULT_DIGIT_MODEL / "weights" / "best.pt",
        PROJECT_ROOT / "runs" / "detect" / "artifacts" / "models" / "digit_detector" / "weights" / "best.pt",
        PROJECT_ROOT
        / "runs"
        / "detect"
        / "artifacts"
        / "models"
        / "digit_detector_stage1"
        / "weights"
        / "best.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _default_workbook() -> Path:
    workbooks = sorted(DATA_ROOT.glob("*.xlsx"))
    if len(workbooks) != 1:
        raise RuntimeError(
            f"Expected one XLSX workbook in {DATA_ROOT}, found {len(workbooks)}. "
            "Pass --workbook explicitly."
        )
    return workbooks[0]


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.exists() and "\\" in value:
        unescaped = shlex.split(value)
        if len(unescaped) == 1:
            candidate = Path(unescaped[0]).expanduser()
            if candidate.exists():
                path = candidate
    return path.resolve()


def _latest_data_path(value: str) -> Path:
    path = _path(value)
    try:
        path.relative_to(DATA_ROOT.resolve())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Video must be inside {DATA_ROOT}") from exc
    return path


def _profile(value: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    if not value or any(char not in allowed for char in value):
        raise argparse.ArgumentTypeError("Profile may contain letters, numbers, '-' and '_' only")
    return value


def _default_profile_for_video(video: Path) -> str:
    text = str(video)
    if "Pen5_" in text:
        return "pen5_90808_reference"
    if "Pen4_" in text:
        return "pen4_90807_reference"
    if "Pen6_" in text:
        return "pen5_90808_reference"
    return "pen4_verified"


def _resolve_roi_profile(video: Path, profile: Optional[str]) -> tuple[Path, str]:
    candidates: List[str] = []
    if profile:
        candidates.append(profile)
    candidates.append(_default_profile_for_video(video))
    for fallback in ("pen4_verified", "pen4_90807_reference", "pen5_90808_reference"):
        if fallback not in candidates:
            candidates.append(fallback)
    for name in candidates:
        path = PROJECT_ROOT / "config" / f"{name}.json"
        if path.exists():
            return path, name
    raise RuntimeError(
        f"No ROI profile found (tried: {', '.join(candidates)}). "
        f"Run: piglet-id mark \"{video}\" my-profile-name"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="piglet-id")
    subcommands = parser.add_subparsers(dest="command", required=True)

    inventory = subcommands.add_parser("inventory", help="Inventory videos and select subset")
    inventory.add_argument("data_root", type=_path)
    inventory.add_argument("--output", type=_path, required=True)
    inventory.add_argument("--subset-output", type=_path)
    inventory.add_argument("--per-pen", type=int, default=3)

    roi = subcommands.add_parser("mark-roi", help="Interactively mark feeding polygons")
    roi.add_argument("video", type=_path)
    roi.add_argument("--output", type=_path, required=True)
    roi.add_argument("--zones", nargs="+", default=["FEEDER"])

    run = subcommands.add_parser("run", help="Run feeding and identity pipeline")
    run.add_argument("video", type=_path)
    run.add_argument("--weights", type=_path, required=True)
    run.add_argument("--roi", type=_path, required=True)
    run.add_argument("--output", type=_path, required=True)
    run.add_argument("--identity-model", type=_path)
    run.add_argument("--confidence", type=float, default=0.35)
    run.add_argument("--min-feed-seconds", type=float, default=2.0)
    run.add_argument("--max-gap-seconds", type=float, default=0.5)
    run.add_argument("--max-frames", type=int)

    package = subcommands.add_parser(
        "prepare-labeling", help="Create roster-linked identity label manifest"
    )
    package.add_argument(
        "--workbook",
        type=_path,
        help=f"Roster XLSX (default: sole *.xlsx under {DATA_ROOT})",
    )
    package.add_argument(
        "--marking-photos",
        type=_path,
        default=DEFAULT_MARKING_PHOTOS,
        help="Reference marking photos directory",
    )
    package.add_argument(
        "--crops",
        type=_path,
        help="Optional analyze identity_crops folder to add for manual labeling",
    )
    package.add_argument(
        "--crops-root",
        type=_path,
        help="Merge track_*.jpg from each video folder produced by sample-crops",
    )
    package.add_argument(
        "--output",
        type=_path,
        default=DEFAULT_LABELS.parent,
        help="Directory for identity_labels.csv (default: artifacts/identity_labeling_v2)",
    )

    label = subcommands.add_parser("label", help="Interactively fill identity labels")
    label.add_argument("labels_csv", type=_path, nargs="?", default=DEFAULT_LABELS)
    label.add_argument(
        "--ask-visibility",
        action="store_true",
        help="Ask clear/partial/blurred/hidden after each ID (off by default)",
    )

    reset_labels = subcommands.add_parser(
        "reset-labels",
        help="Set track_crop rows back to unlabeled (prepare keeps skip/labeled as-is)",
    )
    reset_labels.add_argument("labels_csv", type=_path, nargs="?", default=DEFAULT_LABELS)
    reset_labels.add_argument(
        "--source-video",
        metavar="STEM",
        help="Only rows whose source_video matches sample-crops folder name (e.g. 2026-07-10__10_00_07#M)",
    )
    reset_labels.add_argument(
        "--from-status",
        default="skip",
        choices=("skip", "labeled"),
        help="Which rows to reset (default: skip)",
    )

    label_marks = subcommands.add_parser(
        "label-marks",
        help="Drag one box around the painted number on each already-labeled crop",
    )
    label_marks.add_argument("--labels", type=_path, default=DEFAULT_LABELS)
    label_marks.add_argument("--boxes", type=_path, default=DEFAULT_MARK_BOXES)
    label_marks.add_argument(
        "--limit",
        type=int,
        help="Stop after N new crops this session (resumable)",
    )
    label_marks.add_argument(
        "--in-order",
        action="store_true",
        help="Label in CSV order instead of a shuffled mix of videos",
    )

    export_marks = subcommands.add_parser(
        "export-marks", help="Write a YOLO detection dataset from boxed marks"
    )
    export_marks.add_argument("--boxes", type=_path, default=DEFAULT_MARK_BOXES)
    export_marks.add_argument("--output", type=_path, default=DEFAULT_MARK_DATASET)
    export_marks.add_argument("--validation-fraction", type=float, default=0.2)
    export_marks.add_argument(
        "--single-class",
        action="store_true",
        help="One 'mark' class instead of one class per pig ID",
    )
    export_marks.add_argument(
        "--exclude-id",
        action="append",
        default=[],
        metavar="PIG_ID",
        help="Drop a pig ID from the dataset (repeatable); IDs missing from the roster are dropped anyway",
    )

    train_marks = subcommands.add_parser(
        "train-marks", help="Fine-tune YOLO to find (and optionally read) back marks"
    )
    train_marks.add_argument(
        "--data", type=_path, default=DEFAULT_MARK_DATASET / "data.yaml"
    )
    train_marks.add_argument("--output", type=_path, default=DEFAULT_MARK_MODEL)
    train_marks.add_argument("--epochs", type=int, default=80)
    train_marks.add_argument("--imgsz", type=int, default=320)
    train_marks.add_argument("--base-model", default="yolov8n.pt")

    score_marks = subcommands.add_parser(
        "score-marks",
        help="Read pig IDs on labeled crops with the mark detector and score them",
    )
    score_marks.add_argument(
        "--weights", type=_path, default=DEFAULT_MARK_MODEL / "weights" / "best.pt"
    )
    score_marks.add_argument(
        "--labels",
        type=_path,
        default=DEFAULT_LABELS.parent / "holdout_2026-07-16_labels.csv",
        help="Labeled crops to score (default: the held-out July 16 set)",
    )
    score_marks.add_argument("--confidence", type=float, default=0.25)
    score_marks.add_argument("--imgsz", type=int, default=320)

    label_digits = subcommands.add_parser(
        "label-digits",
        help="Draw one box around each digit on two-digit mark crops (resumable)",
    )
    label_digits.add_argument("--boxes", type=_path, default=DEFAULT_MARK_BOXES)
    label_digits.add_argument("--digit-boxes", type=_path, default=DEFAULT_DIGIT_BOXES)
    label_digits.add_argument(
        "--limit",
        type=int,
        help="Stop after N new crops this session (resumable)",
    )
    label_digits.add_argument(
        "--in-order",
        action="store_true",
        help="Label in CSV order instead of a shuffled mix",
    )
    label_digits.add_argument(
        "--include-single",
        action="store_true",
        help="Also show single-digit crops (normally auto-seeded from mark boxes)",
    )

    export_digits = subcommands.add_parser(
        "export-digits",
        help="Write a YOLO dataset of individual digits (0-9)",
    )
    export_digits.add_argument("--boxes", type=_path, default=DEFAULT_MARK_BOXES)
    export_digits.add_argument("--digit-boxes", type=_path, default=DEFAULT_DIGIT_BOXES)
    export_digits.add_argument("--output", type=_path, default=DEFAULT_DIGIT_DATASET)
    export_digits.add_argument("--validation-fraction", type=float, default=0.2)
    export_digits.add_argument(
        "--single-digit-only",
        action="store_true",
        help="Do not use geometric splits for two-digit crops missing manual digit boxes",
    )
    export_digits.add_argument(
        "--use-geometric",
        action="store_true",
        help="Fill missing two-digit crops with geometric digit splits",
    )

    score_digits = subcommands.add_parser(
        "score-digits",
        help="Compose pig IDs from digit detections and score them on labeled crops",
    )
    score_digits.add_argument(
        "--weights",
        type=_path,
        default=DEFAULT_DIGIT_MODEL / "weights" / "best.pt",
    )
    score_digits.add_argument(
        "--labels",
        type=_path,
        default=DEFAULT_LABELS.parent / "holdout_2026-07-16_labels.csv",
    )
    score_digits.add_argument("--confidence", type=float, default=0.25)
    score_digits.add_argument("--imgsz", type=int, default=320)

    label_manifest = subcommands.add_parser(
        "label-manifest",
        help="Add manual_pig_id on analyze crop_manifest.csv rows (compare to model columns)",
    )
    label_manifest.add_argument(
        "manifest_csv",
        type=_path,
        help="Path to *_crop_manifest.csv from analyze",
    )

    score_manifest = subcommands.add_parser(
        "score-manifest",
        help="Accuracy of pig_id / classifier / mark / trocr vs manual_pig_id in manifest",
    )
    score_manifest.add_argument("manifest_csv", type=_path)

    import_manifest = subcommands.add_parser(
        "import-manifest-labels",
        help="Copy manual_pig_id rows from manifest into identity_labels.csv for train",
    )
    import_manifest.add_argument("manifest_csv", type=_path)
    import_manifest.add_argument("--labels", type=_path, default=DEFAULT_LABELS)

    train = subcommands.add_parser("train-identity", help="Train back-mark classifier")
    train.add_argument("labels_csv", type=_path)
    train.add_argument("--output", type=_path, required=True)

    benchmark = subcommands.add_parser(
        "benchmark-detection", help="Generate checkpoint review images"
    )
    benchmark.add_argument("--videos", nargs="+", type=_path, required=True)
    benchmark.add_argument("--weights", nargs="+", type=_path, required=True)
    benchmark.add_argument("--output", type=_path, required=True)
    benchmark.add_argument("--frames-per-video", type=int, default=5)

    score = subcommands.add_parser("score-detection", help="Score reviewed detections")
    score.add_argument("review_csv", type=_path)
    score.add_argument("--output", type=_path, required=True)

    evaluate = subcommands.add_parser("evaluate-events", help="Score end-to-end event CSV")
    evaluate.add_argument("predictions_csv", type=_path)
    evaluate.add_argument("ground_truth_csv", type=_path)
    evaluate.add_argument("--output", type=_path, required=True)

    prepare = subcommands.add_parser(
        "prepare", help="Prepare latest-data marking photos with saved-label preservation"
    )
    prepare.add_argument(
        "--crops",
        type=_path,
        help="Optional analyze identity_crops folder to merge into the label CSV",
    )
    prepare.add_argument(
        "--crops-root",
        type=_path,
        help="Merge one-crop-per-track folders from sample-crops (e.g. artifacts/label_samples)",
    )

    quick_train = subcommands.add_parser("train", help="Train from saved latest-data labels")
    quick_train.add_argument("--labels", type=_path, default=DEFAULT_LABELS)

    train_ocr = subcommands.add_parser(
        "train-ocr", help="Fine-tune small TrOCR on labeled one/two-digit piglet marks"
    )
    train_ocr.add_argument("--labels", type=_path, default=DEFAULT_LABELS)
    train_ocr.add_argument("--output", type=_path, default=DEFAULT_TROCR_MODEL)
    train_ocr.add_argument("--epochs", type=int, default=8)
    train_ocr.add_argument("--batch-size", type=int, default=4)
    train_ocr.add_argument("--learning-rate", type=float, default=5e-5)

    quick_mark = subcommands.add_parser("mark", help="Save ROI for a camera profile")
    quick_mark.add_argument("video", type=_latest_data_path)
    quick_mark.add_argument("profile", type=_profile)

    analyze = subcommands.add_parser("analyze", help="Analyze one latest-data video")
    analyze.add_argument("video", type=_latest_data_path)
    analyze.add_argument("profile", type=_profile)
    analyze.add_argument("--max-frames", type=int)
    analyze.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="Minimum YOLO pig detection confidence (default: 0.35)",
    )
    analyze.add_argument(
        "--fast",
        action="store_true",
        help="640px YOLO, no annotated video (much faster on CPU)",
    )
    analyze.add_argument(
        "--no-video",
        action="store_true",
        help="Skip annotated MP4; still write CSVs and crops",
    )
    analyze.add_argument(
        "--ocr",
        action="store_true",
        help="Deprecated alias for --identity-method easyocr",
    )
    analyze.add_argument(
        "--identity-method",
        choices=("hybrid", "mark", "classifier", "trocr", "easyocr", "compare", "digits"),
        default="hybrid",
        help=(
            "Identity reader: existing hybrid (default), digits (YOLO digit detector), "
            "one independent reader, or compare mark/classifier/TrOCR in separate CSV columns"
        ),
    )
    analyze.add_argument(
        "--trocr-model",
        type=_path,
        default=DEFAULT_TROCR_MODEL,
        help="Fine-tuned TrOCR directory used by trocr/compare modes",
    )
    analyze.add_argument(
        "--digit-weights",
        type=_path,
        default=None,
        help="YOLO digit detector weights for --identity-method digits",
    )
    analyze.add_argument("--imgsz", type=int, help="YOLO input size (default 960, fast uses 640)")
    analyze.add_argument(
        "--track-min-votes",
        type=int,
        help="Frames/crops that must agree on an ID (classifier default: 1)",
    )
    analyze.add_argument(
        "--track-vote-threshold",
        type=float,
        help="Average confidence to accept track ID (classifier default: 0.38)",
    )
    analyze.add_argument(
        "--relaxed-crops",
        action="store_true",
        help="Disable ink/edge/multi-pig crop filters (sharpness-only if --min-crop-quality set)",
    )
    analyze.add_argument(
        "--min-crop-quality",
        type=float,
        default=0.0,
        help="Minimum Laplacian sharpness score; strict mode defaults to 55 when unset",
    )

    sample = subcommands.add_parser(
        "sample-crops",
        help="Best back crop(s) per track for labeling (whole video, no identity/feeding)",
    )
    sample.add_argument("video", type=_latest_data_path)
    sample.add_argument(
        "profile",
        type=_profile,
        nargs="?",
        default=None,
        help="Optional ROI profile; sample-crops ignores feeder zone and auto-picks pen reference if omitted",
    )
    sample.add_argument("--max-frames", type=int)
    sample.add_argument("--confidence", type=float, default=0.35)
    sample.add_argument("--fast", action="store_true")
    sample.add_argument("--imgsz", type=int)
    sample.add_argument(
        "--crops-per-track",
        type=int,
        default=3,
        help="Keep top N ranked crops per track (default 3; more training candidates)",
    )
    sample.add_argument(
        "--crop-interval",
        type=int,
        default=10,
        help="Sample every N frames (default 10)",
    )
    sample.add_argument(
        "--crop-profile",
        choices=("training", "strict", "loose"),
        default="training",
        help=(
            "training (default): milder filters, ink boosts rank; "
            "strict: ink/edge/multi-pig hard gates; loose: sharpness only"
        ),
    )
    sample.add_argument(
        "--min-crop-quality",
        type=float,
        default=35.0,
        help="Minimum sharpness floor for kept crops (default 35; was 90)",
    )
    sample.add_argument(
        "--relaxed-crops",
        action="store_true",
        help="Alias for --crop-profile loose",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "inventory":
        records = scan_videos(args.data_root)
        write_manifest(records, args.output)
        subset = select_representative(records, args.per_pen)
        if args.subset_output:
            write_manifest(subset, args.subset_output)
        result = {"videos": len(records), "representative_subset": len(subset)}
    elif args.command == "mark-roi":
        result = vars(mark_roi(args.video, args.output, args.zones))
    elif args.command == "run":
        result = run_pipeline(
            args.video,
            args.weights,
            load_roi(args.roi),
            args.output,
            PipelineConfig(
                confidence=args.confidence,
                min_feed_seconds=args.min_feed_seconds,
                max_gap_seconds=args.max_gap_seconds,
                max_frames=args.max_frames,
            ),
            args.identity_model,
        )
    elif args.command == "prepare-labeling":
        workbook = args.workbook or _default_workbook()
        result = prepare_labeling_package(
            workbook, args.marking_photos, args.output, args.crops, args.crops_root
        )
    elif args.command == "label":
        interactive_label(args.labels_csv, ask_visibility=args.ask_visibility)
        result = {"labels": str(args.labels_csv)}
    elif args.command == "reset-labels":
        result = reset_label_rows(
            args.labels_csv,
            source_video=args.source_video,
            from_status=args.from_status,
        )
    elif args.command == "label-marks":
        result = label_mark_boxes(
            args.labels,
            args.boxes,
            limit=args.limit,
            shuffle=not args.in_order,
        )
    elif args.command == "export-marks":
        result = export_mark_dataset(
            args.boxes,
            args.output,
            validation_fraction=args.validation_fraction,
            single_class=args.single_class,
            exclude_ids=args.exclude_id,
        )
    elif args.command == "train-marks":
        result = train_mark_detector(
            args.data,
            args.output,
            epochs=args.epochs,
            imgsz=args.imgsz,
            base_model=args.base_model,
        )
    elif args.command == "score-marks":
        result = score_mark_detector(
            args.weights,
            args.labels,
            confidence=args.confidence,
            imgsz=args.imgsz,
        )
    elif args.command == "label-digits":
        result = label_digit_boxes(
            args.boxes,
            args.digit_boxes,
            limit=args.limit,
            shuffle=not args.in_order,
            two_digit_only=not args.include_single,
        )
    elif args.command == "export-digits":
        pseudo = None
        if args.use_geometric and not args.single_digit_only:
            proposals = build_geometric_pseudo_labels(args.boxes)
            pseudo = proposals.pop("accepted")
            print(json.dumps(proposals, indent=2))
        result = export_digit_dataset(
            args.boxes,
            args.output,
            validation_fraction=args.validation_fraction,
            pseudo_labels=pseudo,
            digit_boxes_csv=args.digit_boxes,
        )
    elif args.command == "score-digits":
        result = score_digit_reader(
            args.weights,
            args.labels,
            confidence=args.confidence,
            imgsz=args.imgsz,
        )
    elif args.command == "label-manifest":
        result = interactive_label_manifest(args.manifest_csv)
    elif args.command == "score-manifest":
        result = score_manifest_labels(args.manifest_csv)
    elif args.command == "import-manifest-labels":
        result = export_manifest_labels_to_training(args.manifest_csv, args.labels)
    elif args.command == "train-identity":
        result = train_classifier(args.labels_csv, args.output)
    elif args.command == "benchmark-detection":
        result = create_detection_review(
            args.videos, args.weights, args.output, args.frames_per_video
        )
    elif args.command == "score-detection":
        result = score_detection_review(args.review_csv)
        write_metrics(result, args.output)
    elif args.command == "evaluate-events":
        result = evaluate_events(args.predictions_csv, args.ground_truth_csv)
        write_metrics(result, args.output)
    elif args.command == "prepare":
        result = prepare_labeling_package(
            _default_workbook(),
            DEFAULT_MARKING_PHOTOS,
            DEFAULT_LABELS.parent,
            args.crops,
            args.crops_root,
        )
    elif args.command == "train":
        result = train_classifier(args.labels, DEFAULT_MODEL)
        result["model"] = str(DEFAULT_MODEL)
    elif args.command == "train-ocr":
        from .ocr_trocr import train_trocr

        result = train_trocr(
            args.labels,
            args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
        result["model"] = str(args.output)
    elif args.command == "mark":
        roi_path = PROJECT_ROOT / "config" / f"{args.profile}.json"
        result = vars(mark_roi(args.video, roi_path, ["FEEDER"]))
    elif args.command == "analyze":
        roi_path = PROJECT_ROOT / "config" / f"{args.profile}.json"
        if not roi_path.exists():
            raise RuntimeError(f"Missing ROI profile. Run: piglet-id mark VIDEO {args.profile}")
        model_path = DEFAULT_MODEL if DEFAULT_MODEL.exists() else None
        output = PROJECT_ROOT / "artifacts" / "results" / args.video.stem
        imgsz = args.imgsz or (640 if args.fast else 960)
        identity_method = "easyocr" if args.ocr else args.identity_method
        digit_weights = args.digit_weights or (
            _default_digit_weights() if identity_method == "digits" else None
        )
        if identity_method == "digits" and (digit_weights is None or not digit_weights.exists()):
            raise RuntimeError(
                "Missing digit detector weights. Train with export-digits/train-marks, "
                "or pass --digit-weights PATH"
            )
        vote_kw = {}
        if args.track_min_votes is not None:
            vote_kw["track_min_votes"] = args.track_min_votes
        if args.track_vote_threshold is not None:
            vote_kw["track_vote_threshold"] = args.track_vote_threshold
        result = run_pipeline(
            args.video,
            DEFAULT_WEIGHTS,
            load_roi(roi_path),
            output,
            PipelineConfig(
                max_frames=args.max_frames,
                confidence=args.confidence,
                imgsz=imgsz,
                identity_method=identity_method,
                trocr_model_path=args.trocr_model,
                digit_weights_path=digit_weights,
                identity_during_video=False,
                write_annotated_video=not (args.fast or args.no_video),
                strict_identity_crops=not args.relaxed_crops,
                crop_profile="loose" if args.relaxed_crops else "strict",
                min_crop_quality=args.min_crop_quality,
                **vote_kw,
            ),
            model_path,
        )
    elif args.command == "sample-crops":
        roi_path, profile_used = _resolve_roi_profile(args.video, args.profile)
        if args.profile and profile_used != args.profile:
            raise RuntimeError(
                f"Missing ROI profile {args.profile!r}. "
                f"Existing profiles include pen5_90808_reference, pen4-20260716-1430, etc. "
                f"Or omit profile to use {profile_used!r}."
            )
        output = DEFAULT_LABEL_SAMPLES / args.video.stem
        imgsz = args.imgsz or (640 if args.fast else 960)
        crop_profile = "loose" if args.relaxed_crops else args.crop_profile
        result = run_pipeline(
            args.video,
            DEFAULT_WEIGHTS,
            load_roi(roi_path),
            output,
            PipelineConfig(
                max_frames=args.max_frames,
                confidence=args.confidence,
                imgsz=imgsz,
                label_sample_only=True,
                write_annotated_video=False,
                min_crop_quality=args.min_crop_quality,
                best_crops_per_track=max(1, args.crops_per_track),
                crop_interval=max(1, args.crop_interval),
                crop_profile=crop_profile,
                strict_identity_crops=crop_profile == "strict",
            ),
            None,
        )
        result["output"] = str(output)
        result["roi_profile"] = profile_used
        result["crop_profile"] = crop_profile
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
