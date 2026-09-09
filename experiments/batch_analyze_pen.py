#!/usr/bin/env python3
"""Batch-run piglet-id analyze over a pen folder.

Expected layout (pen root as input):
  Pen6_90809/
    20260710/
      2026-07-10__08_00_06#M.mp4
      ...
    20260711/
      ...

Does not modify the piglet-id package. Results are written by analyze to:
  piglet-detection-main/artifacts/results/<video-stem>/

Example (one day, Mac GPU, CSV + annotated video):
  python experiments/batch_analyze_pen.py \\
    "/Users/vineelap/Documents/piglets_detection/latest-data/Pen5_90808" \\
    --profile pen5_90808_reference \\
    --day 20260716 \\
    --device mps \\
    --skip-done
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path


def find_videos(pen_root: Path, day: str | None = None) -> list[Path]:
    """Return mp4s under pen_root/<YYYYMMDD>/... or directly in pen_root."""
    videos: list[Path] = []
    for day_dir in sorted(p for p in pen_root.iterdir() if p.is_dir()):
        if not day_dir.name.isdigit() or len(day_dir.name) != 8:
            continue
        if day is not None and day_dir.name != day:
            continue
        videos.extend(sorted(day_dir.glob("*.mp4")))
        videos.extend(sorted(day_dir.glob("*.MP4")))

    # Flat layout: videos directly in the pen folder (e.g. Pen4_11_july/)
    if not videos or day is None:
        root_videos = sorted(pen_root.glob("*.mp4")) + sorted(pen_root.glob("*.MP4"))
        if day is not None:
            # Accept 20260711 or 2026-07-11 in the filename
            day_dashed = f"{day[:4]}-{day[4:6]}-{day[6:8]}"
            root_videos = [
                path
                for path in root_videos
                if day in path.name or day_dashed in path.name
            ]
        if not videos:
            videos = root_videos
        else:
            videos.extend(root_videos)

    seen = set()
    ordered = []
    for path in videos:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            ordered.append(path)
    return ordered


def already_done(video: Path, results_root: Path) -> bool:
    summary = results_root / video.stem / f"{video.stem}_summary.json"
    return summary.exists()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Loop piglet-id analyze over all dated videos in a pen folder"
    )
    parser.add_argument(
        "pen_root",
        type=Path,
        help="Pen folder, e.g. .../latest-data/Pen6_90809",
    )
    parser.add_argument(
        "--profile",
        required=True,
        help="ROI profile name (must exist as config/<profile>.json), e.g. pen6-20260710-0830",
    )
    parser.add_argument(
        "--identity-method",
        default="digits",
        help="Passed to analyze (default: digits)",
    )
    parser.add_argument(
        "--piglet-id",
        default="piglet-id",
        help="piglet-id executable (default: piglet-id on PATH)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="piglet-detection-main root (for locating results)",
    )
    parser.add_argument(
        "--results-subdir",
        help="Save under artifacts/results/<subdir>/<video-stem>/ (e.g. pen_5_analysis)",
    )
    parser.add_argument(
        "--day",
        help="Only process one date folder YYYYMMDD, e.g. 20260716",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cuda", "cpu"),
        default="auto",
        help="Inference device passed to analyze (default: auto → MPS on Mac)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Pass --fast to analyze (640px YOLO; also skips annotated video)",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip annotated MP4 (CSVs/crops only). Omit this flag to save the video.",
    )
    parser.add_argument(
        "--skip-done",
        action="store_true",
        help="Skip videos that already have a summary.json under artifacts/results",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Optional --max-frames for quick tests",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only, do not run",
    )
    args = parser.parse_args()

    pen_root = args.pen_root.expanduser().resolve()
    if not pen_root.is_dir():
        print(f"Not a directory: {pen_root}", file=sys.stderr)
        return 1

    profile_json = args.project_root / "config" / f"{args.profile}.json"
    if not profile_json.exists():
        print(
            f"Missing ROI profile: {profile_json}\n"
            f"Create it first, e.g.:\n"
            f'  piglet-id mark "<one-video>" {args.profile}',
            file=sys.stderr,
        )
        return 1

    if args.day is not None and (not args.day.isdigit() or len(args.day) != 8):
        print(f"--day must be YYYYMMDD, got {args.day!r}", file=sys.stderr)
        return 1

    videos = find_videos(pen_root, day=args.day)
    if not videos:
        scope = f"day {args.day}" if args.day else "date folders"
        print(f"No .mp4 files found under {scope} in {pen_root}", file=sys.stderr)
        return 1

    results_root = args.project_root / "artifacts" / "results"
    if args.results_subdir:
        results_root = results_root / args.results_subdir
    day_tag = f"_{args.day}" if args.day else ""
    sub_tag = f"_{args.results_subdir}" if args.results_subdir else ""
    log_path = (
        args.project_root
        / "artifacts"
        / "results"
        / f"batch_analyze_{pen_root.name}{day_tag}{sub_tag}.csv"
    )
    results_root.mkdir(parents=True, exist_ok=True)

    print(f"Pen root : {pen_root}")
    print(f"Profile  : {args.profile}")
    print(f"Day      : {args.day or 'all'}")
    print(f"Device   : {args.device}")
    print(f"Videos   : {len(videos)}")
    print(f"Results  : {results_root}/<video-stem>/")
    print(f"Log CSV  : {log_path}")
    print()

    rows = []
    for index, video in enumerate(videos, start=1):
        rel = video.relative_to(pen_root)
        if args.skip_done and already_done(video, results_root):
            print(f"[{index}/{len(videos)}] SKIP (already done) {rel}")
            rows.append(
                {
                    "index": index,
                    "video": str(video),
                    "status": "skipped",
                    "seconds": 0,
                    "returncode": "",
                }
            )
            continue

        cmd = [
            args.piglet_id,
            "analyze",
            str(video),
            args.profile,
            "--identity-method",
            args.identity_method,
            "--device",
            args.device,
            "--output",
            str(results_root / video.stem),
        ]
        if args.fast:
            cmd.append("--fast")
        if args.no_video:
            cmd.append("--no-video")
        if args.max_frames is not None:
            cmd.extend(["--max-frames", str(args.max_frames)])

        print(f"[{index}/{len(videos)}] RUN  {rel}")
        print("  " + " ".join(cmd))
        if args.dry_run:
            rows.append(
                {
                    "index": index,
                    "video": str(video),
                    "status": "dry_run",
                    "seconds": 0,
                    "returncode": "",
                }
            )
            continue

        started = time.time()
        completed = subprocess.run(cmd, cwd=str(args.project_root))
        elapsed = round(time.time() - started, 1)
        status = "ok" if completed.returncode == 0 else "failed"
        print(f"  -> {status} in {elapsed}s (code={completed.returncode})")
        rows.append(
            {
                "index": index,
                "video": str(video),
                "status": status,
                "seconds": elapsed,
                "returncode": completed.returncode,
            }
        )
        if completed.returncode != 0:
            print("  Continuing with next video...", file=sys.stderr)

    with log_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["index", "video", "status", "seconds", "returncode"]
        )
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(1 for row in rows if row["status"] == "ok")
    skipped = sum(1 for row in rows if row["status"] == "skipped")
    failed = sum(1 for row in rows if row["status"] == "failed")
    print()
    print(f"Done. ok={ok} skipped={skipped} failed={failed} log={log_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
