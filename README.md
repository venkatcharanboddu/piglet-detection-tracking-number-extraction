# Piglet feeding identity

**Full CLI reference:** [docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md) — every command, parameter, and default path.

This project replaces the stateful notebook pipeline with reproducible commands. It:

1. inventories customer videos and selects a validation subset;
2. saves feeder polygons per camera/video;
3. detects and tracks pigs, records feeding events, and saves best back-mark crops;
4. prepares a roster-linked manual labeling file;
5. trains a constrained mark classifier and combines several frame predictions per track;
6. exports event CSVs, review videos, and measurable evaluation results.

The original notebook and model runs in the parent directory are not modified.

## Short workflow

These commands use the fixed `latest-data` paths and preserve saved annotations:

```bash
piglet-id prepare
piglet-id label
piglet-id train
piglet-id mark "/path/inside/latest-data/video.mp4" pen4-view1
piglet-id analyze "/path/inside/latest-data/video.mp4" pen4-view1
```

Use a new profile name, such as `pen4-view2`, whenever the camera or feeder view changes. `analyze`
uses the notebook detector (`train-3` / YOLOv8s), BoT-SORT, `imgsz=1280`, and writes under
`artifacts/results/`.

Number reading order:
1. fast ink-shape matching against `Pigs marking (1)` labels;
2. augmented classifier fallback;
3. EasyOCR only on feeding-track best crops at the end (slow, used sparingly).

Annotated boxes show `#number` when a mark is read during the run.

### Commands by situation

First setup (run once):

```bash
piglet-id prepare
piglet-id label
piglet-id train
```

`prepare` reads only `latest-data/Pigs marking (1)` and preserves existing annotations. `label`
skips images already labeled, so it does not ask for them again.

`train` creates rotated (-30�, +30�, 90�, 180�, 270�), perspective-distorted,
blurred/downscaled, and lighting-adjusted copies in memory. It does not write duplicate images and
does not mirror digits. Validation images remain unaugmented and are not mixed into training.

For a new video with an already configured camera view:

```bash
piglet-id analyze "/path/inside/latest-data/new-video.mp4" pen4-view1
```

`analyze` now requires at least **0.35 YOLO confidence** by default before a detection can become a
track or identity crop. This reduces empty-feeder/non-pig crops. Override it only when testing,
for example `--confidence 0.45` for stricter filtering. A higher threshold can also miss difficult
or partly hidden piglets, so validate it on the annotated 30-minute video.

On CPU, a full ~30 minute video can take a long time. Use progress lines in the terminal,
`--max-frames 1000` for a quick check, or `--fast` (640px YOLO, no review MP4):

```bash
piglet-id analyze "VIDEO" pen4-view1 --fast --max-frames 3000
```

The old `--ocr` flag is retained as an alias for `--identity-method easyocr`.

## TrOCR: train and compare identity approaches

The optional TrOCR reader is a small sequence recognizer fine-tuned on the existing numeric
`pig_id` labels. Label the **whole visible number** (`7`, `10`, `12`, etc.); individual digit boxes
or a fake piglet `0` class are not needed. The label `10` teaches the model that `0` is a valid
character. Every numeric row with `status=labeled` is used regardless of the visibility field;
`unknown`, `ambiguous`, and `skip` rows are excluded. Visibility is therefore optional, but do not
assign a numeric ID when the number is not actually readable—the OCR model would learn background
or blur as that number.

Install the optional dependency and train once:

```bash
python -m pip install -e ".[trocr]"
piglet-id train-ocr
```

The default model is written to `artifacts/models/trocr_piglet_digits/`. Training splits complete
tracks rather than random nearby frames, which reduces train/validation leakage. The current crop
labels come mainly from one video, so the reported validation score is not evidence of accuracy on
new dates/views; label and hold out at least one additional video before production use.
Read `piglet_ocr_metrics.json` before using the model. TrOCR predictions are kept as candidates
unless held-out exact-match accuracy reaches 80%, preventing a confidently wrong experimental
model from becoming a confirmed feeding identity.

Choose one identity method without changing tracking:

```bash
piglet-id analyze "VIDEO" PROFILE --identity-method mark
piglet-id analyze "VIDEO" PROFILE --identity-method classifier
piglet-id analyze "VIDEO" PROFILE --identity-method trocr
piglet-id analyze "VIDEO" PROFILE --identity-method easyocr
piglet-id analyze "VIDEO" PROFILE --identity-method hybrid
```

`hybrid` is the existing mark-match-then-classifier behavior. OCR runs on the saved best crops
after YOLO/BoT-SORT tracking, so it does not reduce detector frame rate. TrOCR tries the isolated
paint region at four quarter-turn rotations and rejects outputs not present in the roster.

For a fair single-pass comparison, use:

```bash
piglet-id analyze "VIDEO" PROFILE --identity-method compare --no-video
```

This runs mark matching, classifier, and TrOCR on the same crops and adds separate
`mark_*`, `classifier_*`, and `trocr_*` columns to both the crop manifest and feeding-event CSV.
The normal `pig_id` columns retain the existing hybrid result. EasyOCR is excluded from `compare`
because it is much slower; run `--identity-method easyocr` separately if required.

For a changed camera or feeder view, mark a new profile once and then reuse it:

```bash
piglet-id mark "/path/inside/latest-data/new-view-video.mp4" pen4-view2
piglet-id analyze "/path/inside/latest-data/new-view-video.mp4" pen4-view2
```

When new marking photos are added to `latest-data/Pigs marking (1)`:

```bash
piglet-id prepare
piglet-id label
piglet-id train
```

To add **video crops** from an analyze run into the same label file (then fix wrong IDs):

```bash
piglet-id prepare --crops "artifacts/results/VIDEO_STEM/identity_crops"
```

For **much smaller** labeling sets (recommended), export **one best back crop per tracker ID**
anywhere in the video — feeding area not required:

```bash
piglet-id sample-crops "/path/inside/latest-data/your-video.mp4" pen4-20260716-1430 \
  --confidence 0.35 --fast
```

Output: `artifacts/label_samples/<video-stem>/track_<id>_f<frame>.jpg` plus `label_crops.csv`.
Default **training** profile keeps up to **3** ranked crops per track (sharpness + ink bonus),
with milder filters so you get more labeling candidates (`--min-crop-quality` default `35`).
Use `--crop-profile strict` for the old hard ink/edge/multi-pig gates, or `--crops-per-track 5`
for even more. Check `summary.json` → `crop_filter_rejected` and `label_crops`. Still skip
unreadable images in `label` — quantity is for candidates; only clear digits go into training.

After running `sample-crops` on several videos:

```bash
piglet-id prepare --crops-root artifacts/label_samples
piglet-id label
```

Use quotes around paths with `#` or spaces; do not backslash-escape inside quotes.
Only label crops where the painted number is clearly visible; set others to `skip`.
Then `piglet-id label`, `piglet-id train`, and re-run `piglet-id analyze`.

`piglet-id label` does not ask for visibility by default and permanently skips rows marked `s`,
`ss`, Enter, or `skip`. Re-running `prepare` fixes accidental `ss` typed as a pig ID.

Only the newly added photos require labels. Retraining is not needed for ordinary new videos when
the marking-photo set has not changed.

Identity output contains both `pig_id` (confirmed above the calibrated threshold) and
`candidate_pig_id` (experimental best guess even when confidence is low). Use candidates only for
visual review, not final feeding reports.

## Why this is not plain OCR

The marks are hand-painted, curved, blurred, and sometimes stylized. Full-frame OCR would
produce plausible but wrong numbers. This pipeline classifies tight dorsal crops and returns
`unknown` unless several good frames agree above a confidence threshold.

## Install

Python 3.9+ is supported.

```bash
cd piglet-detection-main
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## 1. Inventory and validation subset

```bash
piglet-id inventory ../latest-data \
  --output artifacts/video_manifest.csv \
  --subset-output artifacts/representative_subset.csv \
  --per-pen 3
```

The default subset contains three date/size-stratified files per pen. Review
`representative_subset.csv` before expensive processing.

## 2. Mark a feeding ROI

Use a narrow polygon around the physical feeding-contact area. Make one JSON for each changed
camera view. Left-click at least three points, Enter accepts a polygon, R resets, and Esc cancels.

```bash
piglet-id mark-roi "/path/to/video.mp4" \
  --zones LEFT_FEEDER RIGHT_FEEDER \
  --output config/pen5_view1.json
```

Unlike the old notebook, the production run always loads a saved ROI; coordinates are scaled when
video resolution changes.

## 3. Benchmark the existing detectors

```bash
piglet-id benchmark-detection \
  --videos "/path/to/pen4.mp4" "/path/to/pen5.mp4" \
  --weights ../runs/detect/train-2/weights/best.pt ../runs/detect/train-3/weights/best.pt \
  --output artifacts/detection_review
```

Inspect the generated images and fill `actual_count`, `false_positives`, and `missed_pigs` in
`detection_review.csv`, then:

```bash
piglet-id score-detection artifacts/detection_review/detection_review.csv \
  --output artifacts/detection_metrics.json
```

Fine-tune YOLO only when the reviewed precision/recall shows it is necessary.

## 4. Detect, track, and extract feeding events

Start with a short smoke run:

```bash
piglet-id run "/path/to/video.mp4" \
  --weights ../runs/detect/train-3/weights/best.pt \
  --roi config/pen5_view1.json \
  --output artifacts/smoke \
  --max-frames 500
```

Outputs include:

- `*_feeding_events.csv` with frame and real-time timestamps;
- `*_annotated.mp4`;
- `identity_crops/<video>/` containing only the best sharp crops per track;
- `*_crop_manifest.csv` and `*_summary.json`.

Track IDs are temporary tracker identifiers, not pig numbers.

## 5. Manual identity labeling

```bash
piglet-id prepare-labeling \
  --workbook "../latest-data/Piglet numbers_1st run_02.07.-23.07. (1).xlsx" \
  --marking-photos "../latest-data/Pigs marking (1)" \
  --output artifacts/identity_labeling
```

Follow `artifacts/identity_labeling/LABELING.md`. You can edit the CSV in a spreadsheet or run:

```bash
piglet-id label artifacts/identity_labeling/identity_labels.csv
```

By default, only images in `latest-data/Pigs marking (1)` are included. Re-running
`prepare-labeling` preserves labels already saved in the same output CSV, so each image is labeled
only once. Generated video crops are included only when `--crops <directory>` is explicitly passed.

Images without a readable painted mark must be `unknown`, `ambiguous`, or `skip`. Labels should
cover multiple dates/videos for each ID. Do not report model accuracy until held-out video sources
are labeled.

## 6. Train identity and rerun

```bash
piglet-id train-identity artifacts/identity_labeling/identity_labels.csv \
  --output artifacts/models/mark_classifier.joblib

piglet-id run "/path/to/video.mp4" \
  --weights ../runs/detect/train-3/weights/best.pt \
  --roi config/pen5_view1.json \
  --identity-model artifacts/models/mark_classifier.joblib \
  --output artifacts/identified
```

The classifier rejects weak single-frame guesses and uses track-level consensus.

## 7. End-to-end evaluation

Manually create a ground-truth event CSV with `zone`, `start_frame`, `end_frame`, and `pig_id`, then:

```bash
piglet-id evaluate-events artifacts/identified/video_feeding_events.csv \
  artifacts/ground_truth_events.csv --output artifacts/event_metrics.json
```

Expand to all 150 videos only after both pens meet agreed thresholds for event recall, identity
accuracy, and non-`unknown` coverage. Missing Pen 6 videos and July 20�21 recordings cannot be
recovered by code and should be requested from the customer if required.
