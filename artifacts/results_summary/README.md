# Results summary (GitHub-safe)

Full analyze outputs (annotated MP4s, crop dumps) are **not** in git — they are tens to hundreds of GB.

This folder keeps:

| Path | Contents |
|------|----------|
| `pen_*_analysis_summaries.csv` | Per-video feeding / identified event counts |
| `pen_*/summaries/*.json` | Copied `*_summary.json` from each run |
| `pen_*/sample_csvs/` | A few example `*_feeding_events.csv` files |
| `batch_analyze_*.csv` | Batch runner logs (ok / failed / timing) |

## How to read the rates

`identified_events / feeding_events` is the **fraction of feeding events that received a predicted pig ID**.

It is **not** ground-truth accuracy. Number-reading accuracy on the labeled holdout set is reported separately in the main README (~39% image / ~43% track for digit YOLO + roster compose).
