# Data (local / server transfer only)

Video files under `data/` are **not** pushed to GitLab (pack size limits; each clip is ~274?MB).

## Local layout

- `Pen5_11_July/` — Pen 5, 2026-07-11 morning/afternoon clips (`#M.mp4`)

## Transfer to the university server

From the project root on your laptop (replace `USER` and `HOST`):

```bash
cd /path/to/piglet-detection-main

# code via Git
git push origin main

# videos via rsync (not Git)
rsync -avh --progress data/Pen5_11_July/ \
  USER@HOST:~/piglet_detection_tracking_number_extraction/data/Pen5_11_July/
```

Or with `scp`:

```bash
scp -r data/Pen5_11_July \
  USER@HOST:~/piglet_detection_tracking_number_extraction/data/
```
