# Piglet detection, tracking & number extraction

End-to-end pipeline for overhead barn videos: **detect pigs → track them → detect feeding at a marked feeder ROI → read back-numbers** and export feeding events, crops, and annotated review videos.

**CLI reference:** [docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md)  
**Data transfer notes:** [data/README.md](data/README.md)

> **Large files are not in git.** Raw videos, model weights, and full `artifacts/results/` (annotated MP4s, often hundreds of GB) stay local. Compact metrics are in [`artifacts/results_summary/`](artifacts/results_summary/). Label CSVs use **project-relative paths** (no personal machine paths).

---

## Recent developments

| Area | What changed |
|------|----------------|
| **Identity method** | Settled on **digit YOLO (`0–9`) + roster compose** (`--identity-method digits`) after comparing classifier / EasyOCR / TrOCR / whole-ID YOLO |
| **Batch analysis** | `experiments/batch_analyze_pen.py` supports flat pen folders and dated layouts; `--device mps\|cuda\|cpu`, `--results-subdir`, `--skip-done` |
| **Pen 4 / 5 / 6 runs** | Full July batches analyzed (Mac MPS); summaries published under `artifacts/results_summary/` |
| **ROI profiles** | Added `config/pen4-20260711.json`, `config/pen5-20260711.json`, plus earlier pen profiles |
| **Uni server (`ml`)** | Code on GitLab; videos via `rsync`; CUDA needs **PyTorch cu124**; pin **GPU 0** with `CUDA_VISIBLE_DEVICES=0`; videos must live under `~/latest-data/` as real paths/hardlinks (not only symlinks) |
| **Mac vs server speed** | With annotated MP4 writing, Mac (MPS + SSD) often beats the GPU node wall-clock because decode / BoT-SORT / encode stay CPU- and disk-heavy |
| **Repo hygiene** | Scrubbed absolute local paths from mark/digit/identity CSVs and result summaries; `data/*.mp4` gitignored |

---

## Pipeline

1. **Detect** — YOLOv8s pig detector (`train-3` / `best.pt`)
2. **Track** — BoT-SORT
3. **Feeding ROI** — interactive polygon per camera (`piglet-id mark` → `config/<profile>.json`)
4. **Crops** — best back views per track (sharpness + ink ranking, quality filters)
5. **Identity** — digit YOLO + roster compose (`--identity-method digits`)
6. **Outputs** — `feeding_events.csv`, crop manifest, `summary.json`, optional annotated MP4

Devices: `--device auto|mps|cuda|cpu`.

---

## Quick start

```bash
cd piglet-detection-tracking-number-extraction   # clone root
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .

# Videos must be under ../latest-data/ (required by analyze / mark)
# Detector weights: ../runs/detect/train-3/weights/best.pt
# Digit weights:    artifacts/models/digit_detector/weights/best.pt

piglet-id mark "/path/inside/latest-data/video.mp4" my-pen-profile
piglet-id analyze "/path/inside/latest-data/video.mp4" my-pen-profile \
  --identity-method digits --device mps
```

Batch a pen folder:

```bash
python experiments/batch_analyze_pen.py \
  "/path/to/latest-data/Pen5_11_July" \
  --profile pen5-20260711 \
  --device mps \
  --results-subdir pen_5_analysis \
  --piglet-id "$(pwd)/.venv/bin/piglet-id" \
  --skip-done
```

Server tip (after CUDA works):

```bash
export CUDA_VISIBLE_DEVICES=0
export YOLO_CONFIG_DIR=/tmp/Ultralytics
```

---

## Results so far

### Number-reading methods (July 16 holdout)

| Method | Approx. result |
|--------|----------------|
| Whole-crop classifier / ink mask | ~16–18% |
| EasyOCR / TrOCR on full crops | very poor |
| Tight-crop classifier | ~38% |
| YOLO whole pig-ID | ~38% image / ~41% track |
| **Digit YOLO + roster compose (current)** | **~39% image / ~43% track** |

Two-digit IDs (`10`–`14`, `17`) are still the hardest cases.

### Feeding-event ID assignment (batch runs)

Rate = `identified_events / feeding_events` (an ID was assigned).  
**Not** ground-truth accuracy.

| Batch | Videos | Feeding events | Identified | ID assignment rate |
|-------|--------|----------------|------------|--------------------|
| **Pen 6** (`pen_6_analysis`) | 33 | 1863 | 817 | **43.9%** overall |
| Pen 6 — 2026-07-11 (best day) | 7 | 357 | 261 | **73.1%** |
| Pen 6 — 2026-07-13 (worst day) | 7 | 576 | 106 | **18.4%** |
| **Pen 5** (July 11 + some July 16) | 18 | 756 | 201 | **26.6%** |
| Pen 5 — 2026-07-11 only | 15 | 651 | 171 | **26.3%** |
| **Pen 4** July 11 | 14 | 1123 | 261 | **23.2%** |

Details: [`artifacts/results_summary/`](artifacts/results_summary/).

---

## Future work

1. **Better digit reading** — more `label-digits` on two-digit crops → retrain → `score-digits`; reject weak votes as `unknown`
2. **Ground-truth scoring** — evaluate pen runs against manual labels (not only “ID assigned”)
3. **Faster runs with video kept** — NVENC encode, local scratch disk, batched YOLO, optional GPU decode
4. **Cross-day robustness** — investigate Pen 6 drop on 2026-07-13 (lighting / ink / occlusion)
5. **Tracking metrics** — separate detection/track quality from identity accuracy
6. **Deployment pack** — weights + one-command profiles per pen; document scratch paths on `ml`

---

## Repository layout

```text
src/piglet_identity/     # CLI + pipeline
config/                  # ROI profiles (pen4/5/6-*.json)
experiments/             # batch_analyze_pen.py
docs/CLI_REFERENCE.md
artifacts/
  results_summary/       # compact metrics (in git)
  digit_boxes/           # digit label CSV (relative paths)
  mark_boxes/            # mark box CSV (relative paths)
  identity_labeling_v2/  # identity / holdout labels
  results/               # full runs (gitignored)
  models/                # weights (gitignored)
data/                    # optional local videos (*.mp4 gitignored)
```

---

## License / data

Customer barn videos and trained weights stay private / local. This repository contains code, configs, label CSVs (relative paths), and summarized metrics only.
