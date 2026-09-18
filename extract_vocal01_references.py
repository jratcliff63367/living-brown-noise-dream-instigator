#!/usr/bin/env python3
"""
Extract representative vocal reference clips from:

    ./source-audio/vocal01.mp3

and write them to:

    ./ceremonies/indian/vocal-reference/

The timestamps were selected after reviewing the source for a mix of:
- strong sustained vocal tone
- expressive melodic phrasing
- ornamentation / register variation
- breath-heavy / airy performance sections
- contrasting performance character across the full recording

Requirements:
- ffmpeg available on PATH

Usage:
    python extract_vocal01_references.py

Optional:
    python extract_vocal01_references.py --overwrite
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_FILE = SCRIPT_DIR / "source-audio" / "vocal01.mp3"
OUTPUT_DIR = SCRIPT_DIR / "ceremonies" / "indian" / "vocal-reference"
MANIFEST_FILE = OUTPUT_DIR / "vocal01-reference-manifest.csv"


# A deliberately mixed set of representative sections.
# Times are in seconds from the beginning of vocal01.mp3.
#
# Most are 20 seconds long because that is a useful conditioning-reference size,
# while still leaving room under ElevenLabs' 30-second reference-slice limit.
CLIPS = [
    {
        "name": "vocal01-ref-01-breath-texture",
        "start": 150.0,   # 02:30
        "duration": 20.0,
        "notes": "Breath-forward / airy vocal texture; useful for breath-work character.",
    },
    {
        "name": "vocal01-ref-02-strong-resonant",
        "start": 390.0,   # 06:30
        "duration": 20.0,
        "notes": "Strong resonant vocal passage with fuller projection.",
    },
    {
        "name": "vocal01-ref-03-rich-phrasing",
        "start": 830.0,   # 13:50
        "duration": 20.0,
        "notes": "Rich sustained phrasing and expressive melodic shape.",
    },
    {
        "name": "vocal01-ref-04-breath-contrast",
        "start": 850.0,   # 14:10
        "duration": 20.0,
        "notes": "Contrasting breathier passage immediately following a stronger vocal section.",
    },
    {
        "name": "vocal01-ref-05-expressive-vocal",
        "start": 1270.0,  # 21:10
        "duration": 20.0,
        "notes": "Expressive full-voice passage with useful tonal richness.",
    },
    {
        "name": "vocal01-ref-06-breathwork",
        "start": 1290.0,  # 21:30
        "duration": 20.0,
        "notes": "Breath-work / airy articulation emphasized.",
    },
    {
        "name": "vocal01-ref-07-dynamic-phrasing",
        "start": 1720.0,  # 28:40
        "duration": 20.0,
        "notes": "Dynamic phrase development with strong vocal presence.",
    },
    {
        "name": "vocal01-ref-08-full-register",
        "start": 2160.0,  # 36:00
        "duration": 20.0,
        "notes": "Strong, full-register vocal reference with substantial harmonic body.",
    },
    {
        "name": "vocal01-ref-09-midlate-expression",
        "start": 2600.0,  # 43:20
        "duration": 20.0,
        "notes": "Later expressive passage chosen for a different performance contour.",
    },
    {
        "name": "vocal01-ref-10-air-and-breath",
        "start": 2800.0,  # 46:40
        "duration": 20.0,
        "notes": "Airier high-frequency vocal/breath texture.",
    },
    {
        "name": "vocal01-ref-11-strong-late-vocal",
        "start": 3040.0,  # 50:40
        "duration": 20.0,
        "notes": "Strong later vocal passage with clear projection and phrasing.",
    },
    {
        "name": "vocal01-ref-12-late-breathwork",
        "start": 3500.0,  # 58:20
        "duration": 20.0,
        "notes": "Late breath-oriented section for additional breath-work conditioning variety.",
    },
    {
        "name": "vocal01-ref-13-closing-register",
        "start": 3930.0,  # 65:30
        "duration": 20.0,
        "notes": "Near-closing strong vocal passage, useful for register and timbre variety.",
    },
]


def seconds_to_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes:02d}:{secs:05.2f}"


def extract_clip(source: Path, output: Path, start: float, duration: float) -> None:
    # Re-encode rather than stream-copy so cuts are sample-accurate enough for
    # reference conditioning and are not constrained to MP3 frame boundaries.
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-ss", f"{start:.3f}",
        "-i", str(source),
        "-t", f"{duration:.3f}",
        "-vn",
        "-acodec", "libmp3lame",
        "-q:a", "0",
        str(output),
    ]
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract curated reference clips from source-audio/vocal01.mp3"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite reference clips that already exist.",
    )
    args = parser.parse_args()

    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Source file not found:\n  {SOURCE_FILE}\n\n"
            "Expected vocal01.mp3 inside a sub-folder named 'source-audio' "
            "next to this script."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    generated = 0
    skipped = 0

    print(f"Source: {SOURCE_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Reference clips: {len(CLIPS)}")
    print()

    for index, clip in enumerate(CLIPS, start=1):
        output = OUTPUT_DIR / f"{clip['name']}.mp3"

        if output.exists() and not args.overwrite:
            print(f"[skip {index:02d}/{len(CLIPS)}] {output.name}")
            skipped += 1
        else:
            print(
                f"[cut  {index:02d}/{len(CLIPS)}] "
                f"{seconds_to_timestamp(clip['start'])} "
                f"+ {clip['duration']:.1f}s -> {output.name}"
            )
            extract_clip(
                SOURCE_FILE,
                output,
                clip["start"],
                clip["duration"],
            )
            generated += 1

        rows.append({
            "index": index,
            "file": output.name,
            "source": SOURCE_FILE.name,
            "start_seconds": f"{clip['start']:.3f}",
            "start_timestamp": seconds_to_timestamp(clip["start"]),
            "duration_seconds": f"{clip['duration']:.3f}",
            "notes": clip["notes"],
        })

    with MANIFEST_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "file",
                "source",
                "start_seconds",
                "start_timestamp",
                "duration_seconds",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("Done.")
    print(f"Generated: {generated}")
    print(f"Skipped:   {skipped}")
    print(f"Manifest:  {MANIFEST_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
