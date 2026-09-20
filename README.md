# Piglet detection, tracking & number extraction

End-to-end pipeline for overhead barn videos: **detect pigs → track them → detect feeding at a marked feeder ROI → read back-numbers** and export feeding events, crops, and annotated review videos.

**CLI reference:** [docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md)

> **Note on large files:** Raw videos, model weights, and full `artifacts/results/` (annotated MP4s, ~hundreds of GB) are **not** in this repository. Compact run summaries live in [`artifacts/results_summary/`](artifacts/results_summary/).

---

## Pipeline

1. **Detect** — YOLOv8s pig detector (`train-3` / `best.pt`)
2. **Track** — BoT-SORT
3. **Feeding ROI** — interactive polygon per camera profile (`piglet-id mark` → `config/<profile>.json`)
4. **Crops** — best back views per track (sharpness + ink ranking, quality filters)
5. **Identity** — digit YOLO (`0–9`) + roster compose (`--identity-method digits`)
6. **Outputs** — `feeding_events.csv`, crop manifest, `summary.json`, optional annotated MP4

Devices: `--device auto|mps|cuda|cpu` (Apple MPS on Mac; CUDA on NVIDIA servers).

---

## Quick start

```bash
cd piglet-detection-main   # or your clone root
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .

# Place customer videos under ../latest-data/ (required by analyze/mark)
# Place detector weights at ../runs/detect/train-3/weights/best.pt
# Place digit weights at artifacts/models/digit_detector/weights/best.pt

piglet-id mark "/path/inside/latest-data/video.mp4" my-pen-profile
piglet-id analyze "/path/inside/latest-data/video.mp4" my-pen-profile \
  --identity-method digits --device mps
```

Batch a whole pen folder:

```bash
python experiments/batch_analyze_pen.py \
  "/path/to/latest-data/Pen5_11_July" \
  --profile pen5-20260711 \
  --device mps \
  --results-subdir pen_5_analysis \
  --piglet-id "$(pwd)/.venv/bin/piglet-id" \
  --skip-done
```

---

## What we achieved

### System delivered
- Reproducible **CLI** (`piglet-id`) replacing notebook-only workflows
- Interactive **feeder ROI** profiles for Pens 4 / 5 / 6 (multiple dates)
- Full **analyze** stack: detection, tracking, feeding events, identity crops, review video
- **Digit-based ID reading** (preferred over whole-crop classifier / EasyOCR / TrOCR for this data)
- **Batch runner** for flat or dated pen folders; MPS / CUDA / CPU
- Labeling tools: `label-digits`, `export-digits`, `score-digits`, mark-box tooling
- Ran analysis locally (Mac MPS) and on Uni Rostock `ml` (RTX 2080 Ti / CUDA)

### Number-reading methods tried (July 16 holdout)

| Method | Approx. result |
|--------|----------------|
| Whole-crop classifier / ink mask | ~16–18% |
| EasyOCR / TrOCR on full crops | very poor |
| Tight-crop classifier | ~38% |
| YOLO whole pig-ID | ~38% image / ~41% track |
| **Digit YOLO + roster compose (current)** | **~39% image / ~43% track** |

Two-digit IDs (e.g. `10`–`14`, `17`) remain the hardest cases.

### Feeding-event ID assignment on batch runs

These rates are **identified_events / feeding_events** (pipeline assigned an ID).  
They are **not** ground-truth accuracy.

| Batch | Videos | Feeding events | Identified | ID assignment rate |
|-------|--------|----------------|------------|--------------------|
| **Pen 6** (`pen_6_analysis`) | 33 | 1863 | 817 | **43.9%** overall |
| Pen 6 — 2026-07-11 (best day) | 7 | 357 | 261 | **73.1%** |
| Pen 6 — 2026-07-13 (worst day) | 7 | 576 | 106 | **18.4%** |
| **Pen 5** July 11 + some July 16 | 18 | 756 | 201 | **26.6%** |
| Pen 5 — 2026-07-11 only | 15 | 651 | 171 | **26.3%** |
| **Pen 4** July 11 | 14 | 1123 | 261 | **23.2%** |

Per-video CSVs/JSONs: [`artifacts/results_summary/`](artifacts/results_summary/).

### Ops / deployment notes
- Uni Rostock GitLab used for **code**; videos transferred with **rsync** (GitLab ~200 MiB pack limit)
- Annotated MP4 encode + NFS home storage made the GPU server often **slower wall-clock** than a Mac with MPS + SSD
- CUDA on the server needed a **cu124** PyTorch build matching the driver; pin GPU 0 with `CUDA_VISIBLE_DEVICES=0`
- `analyze` requires videos under `../latest-data/` (symlink alone is not enough after path resolve — use hardlinks or real paths)

---

## Future work

1. **Improve digit reading** — finish/expand `label-digits` on two-digit crops; retrain; `score-digits`; reject low-confidence votes as `unknown` (precision over forced IDs)
2. **Ground-truth evaluation** — score pen-level runs against manual labels (not only “ID was assigned”)
3. **Speed with video kept** — NVENC encode, write results to local scratch, batched YOLO frames, optional GPU decode
4. **Robustness across pens/days** — day-to-day drop (e.g. Pen 6 on 2026-07-13) needs lighting / ink / occlusion analysis
5. **Tracking quality metrics** — separate identity accuracy from detection/track quality
6. **Deployment** — document server scratch paths, weight packaging, and a one-command batch profile per pen

---

## Repository layout

```text
src/piglet_identity/     # CLI + pipeline
config/                  # ROI profiles (pen4/5/6-*.json)
experiments/             # batch_analyze_pen.py
docs/CLI_REFERENCE.md
artifacts/
  results_summary/       # compact metrics (in git)
  digit_boxes/           # digit label CSV
  mark_boxes/            # mark box CSV
  identity_labeling_v2/  # identity labels / holdout
  results/               # full runs (gitignored, local only)
  models/                # weights (gitignored)
data/                    # optional local videos (*.mp4 gitignored)
```

---

## License / data

Customer barn videos and trained weights stay private / local. This repo contains code, configs, label CSVs, and summarized metrics only.
