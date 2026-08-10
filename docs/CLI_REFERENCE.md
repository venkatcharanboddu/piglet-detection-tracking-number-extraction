# `piglet-id` command reference

Run from the project root with the virtualenv active:

```bash
cd piglet-detection-main
source .venv/bin/activate
piglet-id --help
piglet-id <command> --help
```

All commands print a JSON summary to stdout when they finish.

---

## Default paths

These paths are used when flags are omitted (relative to repo layout).

| Resource | Default path |
|----------|----------------|
| Customer videos | `../latest-data/` (required for `analyze`, `sample-crops`, `mark`) |
| YOLO weights | `../runs/detect/train-3/weights/best.pt` |
| Identity labels CSV | `artifacts/identity_labeling_v2/identity_labels.csv` |
| Trained classifier | `artifacts/models/mark_classifier_latest_data.joblib` |
| TrOCR model dir | `artifacts/models/trocr_piglet_digits/` |
| Marking photos | `../latest-data/Pigs marking (1)/` |
| Roster workbook | Sole `*.xlsx` in `../latest-data/` |
| ROI profiles | `config/<profile>.json` |
| `analyze` output | `artifacts/results/<video-stem>/` |
| `sample-crops` output | `artifacts/label_samples/<video-stem>/` |

---

## Recommended workflows

### One-time setup

```bash
piglet-id prepare
piglet-id label
piglet-id train
piglet-id mark "/path/inside/latest-data/video.mp4" my-pen-profile
```

### Better training data (overhead video crops)

```bash
piglet-id sample-crops "/path/inside/latest-data/video.mp4" [profile] --confidence 0.40 --fast
piglet-id prepare --crops-root artifacts/label_samples
piglet-id label
piglet-id train
```

### Read back numbers (digit detection)

Whole-crop classification does not work: the number is a small, arbitrarily rotated part of the crop. Detect individual digits `0-9`, then compose a roster-valid pig ID.

```bash
piglet-id label-marks --limit 300        # one drag per crop (whole mark), resumable
piglet-id label-digits --limit 100       # one box per digit on two-digit marks
piglet-id export-digits                  # prefers manual digit boxes
piglet-id train-marks --data artifacts/digit_dataset/data.yaml \
  --output artifacts/models/digit_detector --epochs 60
piglet-id score-digits
```

Whole-ID mark detection (`export-marks` / `train-marks` / `score-marks`) is still available for comparison.

### Evaluate models on a manifest

```bash
piglet-id analyze "/path/inside/latest-data/video.mp4" my-pen-profile --identity-method compare --no-video
piglet-id label-manifest artifacts/results/<video-stem>/<video-stem>_crop_manifest.csv
piglet-id score-manifest artifacts/results/<video-stem>/<video-stem>_crop_manifest.csv
piglet-id import-manifest-labels artifacts/results/<video-stem>/<video-stem>_crop_manifest.csv
piglet-id train
```

**Video paths:** must live under `latest-data/`. Quote paths that contain `#` or spaces; do not backslash-escape inside quotes.

---

## Commands (alphabetical)

### `analyze`

Full pipeline on one video: YOLO + BoT-SORT tracking, feeder ROI events, identity crops, optional annotated MP4.

**Usage**

```text
piglet-id analyze VIDEO PROFILE [options]
```

| Argument / option | Type | Default | Description |
|-------------------|------|---------|-------------|
| `VIDEO` | path | � | Video file under `latest-data/` |
| `PROFILE` | string | � | ROI profile name (`config/<PROFILE>.json` must exist) |
| `--max-frames` | int | all frames | Stop after N frames (quick tests) |
| `--confidence` | float | `0.35` | Minimum YOLO detection confidence |
| `--fast` | flag | off | YOLO `imgsz=640`, skip annotated video |
| `--no-video` | flag | off | Skip annotated MP4; still write CSVs and crops |
| `--ocr` | flag | off | Deprecated alias for `--identity-method easyocr` |
| `--identity-method` | choice | `hybrid` | See [Identity methods](#identity-methods); use `digits` for the YOLO digit detector |
| `--trocr-model` | path | `artifacts/models/trocr_piglet_digits` | TrOCR checkpoint directory |
| `--digit-weights` | path | auto | Digit YOLO `best.pt` for `--identity-method digits` |
| `--imgsz` | int | `960` (`640` if `--fast`) | YOLO input size |
| `--track-min-votes` | int | method default | Crops that must agree for track-level ID |
| `--track-vote-threshold` | float | method default | Mean confidence to accept track ID |
| `--relaxed-crops` | flag | off | Disable ink / edge / multi-pig filters |
| `--min-crop-quality` | float | `0` | Minimum sharpness; in strict mode effective floor is ~55 if unset |

**Strict identity crops (default):** rejects blurry crops, crops without visible ink, bad aspect ratio, frame-edge partial boxes, and boxes overlapping another pig in the same frame. Summary: `<output>/<video-stem>_summary.json` ? `crop_filter_rejected`.

**Outputs (under `artifacts/results/<video-stem>/`)**

- `<video-stem>_feeding_events.csv` � feeding intervals with pig IDs when voted from crops
- `<video-stem>_crop_manifest.csv` � per-crop model predictions
- `identity_crops/<video-stem>/` � saved crop images
- `<video-stem>_annotated.mp4` � unless `--fast` or `--no-video`
- `<video-stem>_summary.json`

---

### `benchmark-detection`

Sample frames from videos and write review images for manual detection QA.

**Usage**

```text
piglet-id benchmark-detection --videos PATH [PATH ...] --weights PATH [PATH ...] --output DIR [options]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--videos` | paths | required | One or more video files |
| `--weights` | paths | required | YOLO weight files (paired with videos) |
| `--output` | path | required | Output directory for review assets |
| `--frames-per-video` | int | `5` | Frames sampled per video |

---

### `evaluate-events`

Compare predicted feeding events CSV to ground truth.

**Usage**

```text
piglet-id evaluate-events PREDICTIONS_CSV GROUND_TRUTH_CSV --output METRICS_PATH
```

| Argument / option | Description |
|-------------------|-------------|
| `PREDICTIONS_CSV` | Predicted events file |
| `GROUND_TRUTH_CSV` | Labeled ground truth |
| `--output` | JSON metrics output path |

---

### `import-manifest-labels`

Copy rows with `manual_status=labeled` and numeric `manual_pig_id` from a crop manifest into `identity_labels.csv` for retraining.

**Usage**

```text
piglet-id import-manifest-labels MANIFEST_CSV [--labels PATH]
```

| Option | Default | Description |
|--------|---------|-------------|
| `MANIFEST_CSV` | � | `*_crop_manifest.csv` from `analyze` |
| `--labels` | `artifacts/identity_labeling_v2/identity_labels.csv` | Target labels file (merge/update by `image_path`) |

---

### `inventory`

Scan a data directory for videos and optionally write a representative subset manifest.

**Usage**

```text
piglet-id inventory DATA_ROOT --output MANIFEST [--subset-output PATH] [--per-pen N]
```

| Option | Default | Description |
|--------|---------|-------------|
| `DATA_ROOT` | � | Root folder to scan |
| `--output` | required | Full inventory manifest path |
| `--subset-output` | � | Optional subset manifest |
| `--per-pen` | `3` | Videos per pen in representative subset |

---

### `label`

Interactive labeling for `identity_labels.csv` (marking photos and merged crops).

**Usage**

```text
piglet-id label [LABELS_CSV] [--ask-visibility]
```

| Option | Default | Description |
|--------|---------|-------------|
| `LABELS_CSV` | `artifacts/identity_labeling_v2/identity_labels.csv` | CSV to edit in place |
| `--ask-visibility` | off | After each ID, prompt `clear` / `partial` / `blurred` / `hidden` |

**Interactive keys**

| Input | Effect |
|-------|--------|
| Digit(s) | Set `pig_id`, `status=labeled` |
| Enter, `s`, `ss`, `skip`, `sk`, `x` | Skip row (`status=skip`) |
| `q` | Quit and save progress |

Rows already `labeled` or `skip` are not shown again.

### `reset-labels`

`prepare` **never** changes rows that are already `skip` or `labeled`. Re-running `prepare` after `sample-crops` only **adds new** image paths. To label Pen6 (or any video) again after skipping everything:

```text
piglet-id reset-labels [--source-video STEM] [--from-status skip|labeled] [LABELS_CSV]
```

| Option | Default | Description |
|--------|---------|-------------|
| `LABELS_CSV` | default identity labels path | CSV to update |
| `--source-video` | all track crops | Folder name under `label_samples/` (e.g. `2026-07-10__10_00_07#M`) |
| `--from-status` | `skip` | Rows to reopen as `unlabeled` (reference photos are never reset) |

---

### `label-marks`

Draw one box around the painted number on each crop you already gave a pig ID. The ID is reused, so it is a single drag per image. Progress is saved after every image and the session is resumable.

**Usage**

```text
piglet-id label-marks [--labels PATH] [--boxes PATH] [--limit N] [--in-order]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--labels` | default identity labels CSV | Source of labeled crops (reference photos are excluded) |
| `--boxes` | `artifacts/mark_boxes/mark_boxes.csv` | Where boxes are stored |
| `--limit` | all | Stop after N new crops this session |
| `--in-order` | off | Label in CSV order instead of a shuffled mix of videos |

**Keys:** drag to draw, `Enter` save, `s` skip (mark not readable), `u` undo the box, `q` quit.

---

### `export-marks`

Write a YOLO detection dataset from the boxed crops. Whole source videos are held out for validation so nearby frames cannot leak.

**Usage**

```text
piglet-id export-marks [--boxes PATH] [--output DIR] [--validation-fraction F] [--single-class]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--boxes` | `artifacts/mark_boxes/mark_boxes.csv` | Boxed crops |
| `--output` | `artifacts/mark_boxes/dataset` | Dataset directory (recreated each run) |
| `--validation-fraction` | `0.2` | Fraction of source videos held out |
| `--single-class` | off | One `mark` class instead of one class per pig ID |

---

### `train-marks`

Fine-tune YOLO on the exported dataset. Rotation augmentation is enabled (`degrees=180`) and mirroring is disabled, since mirrored digits are wrong.

**Usage**

```text
piglet-id train-marks [--data data.yaml] [--output DIR] [--epochs N] [--imgsz N] [--base-model NAME]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--data` | `artifacts/mark_boxes/dataset/data.yaml` | Dataset config from `export-marks` |
| `--output` | `artifacts/models/mark_detector` | Run directory; weights at `weights/best.pt` |
| `--epochs` | `80` | Training epochs |
| `--imgsz` | `320` | Input size (crops are small) |
| `--base-model` | `yolov8n.pt` | Starting checkpoint |

---

### `score-marks`

Run the mark detector on labeled crops and score the pig IDs it reads. Needs no boxes on the test crops, so any labeled crop set works as a held-out test.

**Usage**

```text
piglet-id score-marks [--weights PATH] [--labels PATH] [--confidence F] [--imgsz N]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--weights` | `artifacts/models/mark_detector/weights/best.pt` | Detector to score |
| `--labels` | `holdout_2026-07-16_labels.csv` | Labeled crops to score against |
| `--confidence` | `0.25` | Minimum detection confidence |
| `--imgsz` | `320` | Inference size (match training) |

**Reported:** `coverage` (fraction of crops where a mark was found), `image_accuracy`, `image_accuracy_on_detected`, `image_accuracy_in_vocabulary` (excluding pig IDs the detector was never trained on), and `track_accuracy` from confidence-weighted voting.

---

### `label-digits`

Draw one box around each digit on two-digit mark crops. Single-digit mark boxes are auto-seeded, so you only review multi-digit marks. Resumable; saves after each image.

**Usage**

```text
piglet-id label-digits [--boxes PATH] [--digit-boxes PATH] [--limit N] [--in-order] [--include-single]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--boxes` | `artifacts/mark_boxes/mark_boxes.csv` | Whole-mark boxes (source list + zoom guide) |
| `--digit-boxes` | `artifacts/digit_boxes/digit_boxes.csv` | Per-digit boxes CSV |
| `--limit` | none | Stop after N new crops this session |
| `--in-order` | off | Label in CSV order instead of shuffled |
| `--include-single` | off | Also show single-digit crops |

**Keys:** drag to draw, `Enter` save current digit and advance, `u` undo, `s` skip image, `q` quit.

For pig `14` you draw `'1'` then `'4'`. Cyan outline is the old whole-mark box; green boxes are digits already saved on this crop.

---

### `export-digits`

Write a YOLO dataset of individual digits (`0`–`9`). Prefers manual digit boxes from `label-digits`. Single-digit mark boxes fill gaps. Optional geometric splits only with `--use-geometric`.

**Usage**

```text
piglet-id export-digits [--boxes PATH] [--digit-boxes PATH] [--output DIR] [--validation-fraction F] [--use-geometric]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--boxes` | `artifacts/mark_boxes/mark_boxes.csv` | Whole-mark boxes |
| `--digit-boxes` | `artifacts/digit_boxes/digit_boxes.csv` | Manual per-digit boxes |
| `--output` | `artifacts/digit_dataset` | Dataset directory (recreated each run) |
| `--validation-fraction` | `0.2` | Fraction of source videos held out |
| `--use-geometric` | off | Fill missing two-digit crops with auto splits |
| `--single-digit-only` | off | Ignore geometric fill even if requested |

Then train with:

```bash
piglet-id train-marks --data artifacts/digit_dataset/data.yaml \
  --output artifacts/models/digit_detector --epochs 60
```

---

### `score-digits`

Detect digits on labeled crops, compose roster-valid pig IDs, and score them.

**Usage**

```text
piglet-id score-digits [--weights PATH] [--labels PATH] [--confidence F] [--imgsz N]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--weights` | `artifacts/models/digit_detector/weights/best.pt` | Digit detector |
| `--labels` | `holdout_2026-07-16_labels.csv` | Labeled crops to score against |
| `--confidence` | `0.25` | Minimum digit confidence |
| `--imgsz` | `320` | Inference size |

**Reported:** `coverage`, `image_accuracy`, `image_accuracy_on_read`, `track_accuracy`, and `per_class_accuracy`.

---

### `label-manifest`

Interactive ground truth on an analyze crop manifest (`manual_pig_id`, `manual_status`).

**Usage**

```text
piglet-id label-manifest MANIFEST_CSV
```

Shows model columns (`pig_id`, `classifier_pig_id`, etc.) as hints. Same skip keys as `label` (`Enter`, `s`, `ss`, �). Rows already `labeled` or `skip` are skipped.

---

### `mark`

Quick alias: interactively draw feeder ROI and save `config/<PROFILE>.json`.

**Usage**

```text
piglet-id mark VIDEO PROFILE
```

| Argument | Description |
|----------|-------------|
| `VIDEO` | Under `latest-data/` |
| `PROFILE` | Alphanumeric, `-`, `_` only |

Equivalent to `mark-roi` with output `config/<PROFILE>.json` and zone `FEEDER`.

---

### `mark-roi`

Interactively mark feeding polygons on a video.

**Usage**

```text
piglet-id mark-roi VIDEO --output ROI_JSON [--zones NAME [NAME ...]]
```

| Option | Default | Description |
|--------|---------|-------------|
| `VIDEO` | � | Any readable video path |
| `--output` | required | ROI JSON path |
| `--zones` | `FEEDER` | Polygon names to draw |

---

### `prepare`

Convenience wrapper: build/update label package from latest-data marking photos + optional crops, preserving existing labels.

**Usage**

```text
piglet-id prepare [--crops PATH] [--crops-root PATH]
```

| Option | Description |
|--------|-------------|
| `--crops` | Single `identity_crops` folder from `analyze` to merge |
| `--crops-root` | Parent of per-video folders from `sample-crops` (e.g. `artifacts/label_samples`) |

Uses default workbook, marking photos, and writes under `artifacts/identity_labeling_v2/`.

---

### `prepare-labeling`

Same as `prepare` with explicit paths for workbook, photos, and output directory.

**Usage**

```text
piglet-id prepare-labeling [--workbook PATH] [--marking-photos PATH] [--crops PATH] [--crops-root PATH] [--output DIR]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--workbook` | sole `*.xlsx` in `latest-data` | Pen roster spreadsheet |
| `--marking-photos` | `latest-data/Pigs marking (1)` | Reference back-mark photos |
| `--crops` | � | Optional analyze crop folder |
| `--crops-root` | � | Merge all `track_*.jpg` from subfolders |
| `--output` | `artifacts/identity_labeling_v2` | Directory for `identity_labels.csv` |

---

### `run`

Lower-level pipeline entry (custom weights, ROI JSON, output dir). Does not use `latest-data` video restriction.

**Usage**

```text
piglet-id run VIDEO --weights PATH --roi PATH --output DIR [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--weights` | required | YOLO `.pt` file |
| `--roi` | required | Feeder ROI JSON |
| `--output` | required | Output directory |
| `--identity-model` | � | Optional `.joblib` classifier |
| `--confidence` | `0.35` | YOLO confidence |
| `--min-feed-seconds` | `2.0` | Minimum feeding duration |
| `--max-gap-seconds` | `0.5` | Gap inside one feeding event |
| `--max-frames` | � | Frame limit |

---

### `sample-crops`

Best back crop(s) per track over the whole video (no feeding events, no identity inference). Used to build training/eval crops.

**Usage**

```text
piglet-id sample-crops VIDEO [PROFILE] [options]
```

| Argument / option | Type | Default | Description |
|-------------------|------|---------|-------------|
| `VIDEO` | path | � | Under `latest-data/` |
| `PROFILE` | string | optional | ROI profile; if omitted, auto-resolves pen reference (feeder zone not used for sampling) |
| `--max-frames` | int | all | Frame limit |
| `--confidence` | float | `0.35` | YOLO confidence |
| `--fast` | flag | off | `imgsz=640` |
| `--imgsz` | int | `960` / `640` | YOLO size |
| `--crops-per-track` | int | `3` | Top N ranked crops per track |
| `--crop-interval` | int | `10` | Sample every N frames |
| `--crop-profile` | choice | `training` | `training` (mild filters, ink boosts rank), `strict`, or `loose` |
| `--min-crop-quality` | float | `35` | Minimum sharpness for kept crops |
| `--relaxed-crops` | flag | off | Alias for `--crop-profile loose` |

**Outputs:** `artifacts/label_samples/<video-stem>/track_<id>_f<frame>.jpg` (or `track_<id>.jpg` if `--crops-per-track 1`), `label_crops.csv`, `summary.json`.

---

### `score-detection`

Score a manual detection review CSV from `benchmark-detection`.

**Usage**

```text
piglet-id score-detection REVIEW_CSV --output METRICS_PATH
```

---

### `score-manifest`

Compare `manual_pig_id` to model columns for rows with `manual_status=labeled` and numeric IDs.

**Usage**

```text
piglet-id score-manifest MANIFEST_CSV
```

**Metrics (when columns exist):** `manual_labeled_rows`, `{column}_accuracy`, `{column}_coverage` for `pig_id`, `candidate_pig_id`, `classifier_pig_id`, `mark_pig_id`, `trocr_pig_id`.

---

### `train`

Train the ResNet18+SVM mark classifier from default labels path; writes default model path.

**Usage**

```text
piglet-id train [--labels PATH]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--labels` | `artifacts/identity_labeling_v2/identity_labels.csv` | Training CSV (`status=labeled`, numeric `pig_id`) |

**Output model:** `artifacts/models/mark_classifier_latest_data.joblib`

---

### `train-identity`

Same training as `train` but explicit input/output paths.

**Usage**

```text
piglet-id train-identity LABELS_CSV --output MODEL.joblib
```

---

### `train-ocr`

Fine-tune TrOCR on numeric labels (optional extras: `pip install -e ".[trocr]"`).

**Usage**

```text
piglet-id train-ocr [--labels PATH] [--output DIR] [--epochs N] [--batch-size N] [--learning-rate LR]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--labels` | default identity labels CSV | Rows with numeric `pig_id`, `status=labeled` |
| `--output` | `artifacts/models/trocr_piglet_digits` | Saved model directory |
| `--epochs` | `8` | Training epochs |
| `--batch-size` | `4` | Batch size |
| `--learning-rate` | `5e-5` | Adam learning rate |

---

## Identity methods (`analyze`)

| Method | Behavior |
|--------|----------|
| `hybrid` | Mark template match first; classifier fallback |
| `digits` | YOLO digit detector + roster composition (auto-finds trained `best.pt`) |
| `mark` | Ink-shape matching vs marking photos only |
| `classifier` | Trained `.joblib` classifier only |
| `trocr` | Fine-tuned TrOCR (needs `--trocr-model`) |
| `easyocr` | EasyOCR digits (slow; `--ocr` alias) |
| `compare` | Writes separate `mark_`, `classifier_`, `trocr_` columns on crops and events |

Track-level voting defaults differ by method (classifier uses lower vote bar). Override with `--track-min-votes` and `--track-vote-threshold`.

---

## Label CSV columns

**`identity_labels.csv`:** `image_path`, `pig_id`, `pen`, `source_video`, `visibility`, `status`, `source_type`, �

**`*_crop_manifest.csv`:** `image_path`, `track_id`, `frame`, `quality`, model prediction columns, plus after `label-manifest`: `manual_pig_id`, `manual_status`.

---

## Optional dependencies

| Feature | Install |
|---------|---------|
| TrOCR | `pip install -e ".[trocr]"` (may need `transformers<5`, `protobuf`) |
| EasyOCR | Used when `--identity-method easyocr` |

---

## Getting help

```bash
piglet-id --help
piglet-id analyze --help
piglet-id sample-crops --help
```
