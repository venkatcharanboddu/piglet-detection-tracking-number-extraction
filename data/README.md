# Data (local / server transfer only)

Video files under `data/` are **not** pushed to Uni Rostock GitLab (200?MiB pack limit; each clip is ~274?MB).

## Local layout

- `Pen5_11_July/` — Pen 5, 2026-07-11 morning/afternoon clips (`#M.mp4`)

## Transfer to the university server

From your Mac (replace `USER` and `HOST`):

```bash
cd /Users/vineelap/Documents/piglets_detection/piglet-detection-main

# code via GitLab
git push origin main

# videos via scp/rsync (not GitLab)
rsync -avh --progress data/Pen5_11_July/ USER@HOST:~/piglet_detection_tracking_number_extraction/data/Pen5_11_July/
```

Or with `scp`:

```bash
scp -r data/Pen5_11_July USER@HOST:~/piglet_detection_tracking_number_extraction/data/
```
