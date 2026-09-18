#!/usr/bin/env python3
"""
Extract curated vocal-reference clips from:

    ./source-audio/vocal03.mp3

and write them to:

    ./ceremonies/indian/vocal-reference/

This revision fixes the filename-prefix issue and avoids sections that contain
spoken meditation guidance. The chosen clips are intended to emphasize sung
vocal performance, including breath-work, sustained tone, ornamentation,
register variation, and expressive phrase shape.

Requirements:
- ffmpeg available on PATH

Usage:
    python extract_vocal03_references.py

Optional:
    python extract_vocal03_references.py --overwrite
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_FILE = SCRIPT_DIR / "source-audio" / "vocal03.mp3"
OUTPUT_DIR = SCRIPT_DIR / "ceremonies" / "indian" / "vocal-reference"
MANIFEST_FILE = OUTPUT_DIR / "vocal03-reference-manifest.csv"


# Curated sung sections only. Spoken meditation-guidance passages were excluded.
# Most clips are 20 seconds, which is a useful size for ElevenLabs reference
# conditioning while remaining comfortably below the 30-second slice limit.
CLIPS = [
    {
        "name": "vocal03-ref-01-resonant-sustain",
        "start": 92.0,
        "duration": 20.0,
        "notes": "Sung passage with sustained resonance and natural human tone.",
    },
    {
        "name": "vocal03-ref-02-expressive-phrase",
        "start": 178.0,
        "duration": 20.0,
        "notes": "Expressive sung phrasing with useful melodic contour and articulation.",
    },
    {
        "name": "vocal03-ref-03-ornamented-vocal",
        "start": 275.0,
        "duration": 20.0,
        "notes": "Ornamented sung section with pitch movement and expressive vocal detail.",
    },
    {
        "name": "vocal03-ref-04-breathwork-air",
        "start": 447.0,
        "duration": 20.0,
        "notes": "Breath-forward sung texture; useful for airy breath-work character.",
    },
    {
        "name": "vocal03-ref-05-rich-midrange",
        "start": 548.0,
        "duration": 20.0,
        "notes": "Rich midrange singing with clear harmonic body and controlled dynamics.",
    },
    {
        "name": "vocal03-ref-06-breath-release",
        "start": 625.0,
        "duration": 20.0,
        "notes": "Sung phrase with audible breath support, release, and changing vocal density.",
    },
    {
        "name": "vocal03-ref-07-full-dynamic-vocal",
        "start": 805.0,
        "duration": 20.0,
        "notes": "Strong sung passage with fuller projection and dynamic phrase shape.",
    },
    {
        "name": "vocal03-ref-08-register-variation",
        "start": 897.0,
        "duration": 20.0,
        "notes": "Sung passage with register movement and contrasting vocal color.",
    },
    {
        "name": "vocal03-ref-09-expressive-late-middle",
        "start": 990.0,
        "duration": 20.0,
        "notes": "Expressive sung section with natural timing and phrase irregularity.",
    },
    {
        "name": "vocal03-ref-10-breath-rich-vocal",
        "start": 1162.0,
        "duration": 20.0,
        "notes": "Breath-rich singing emphasizing air, transition, and organic vocal texture.",
    },
    {
        "name": "vocal03-ref-11-late-resonance",
        "start": 1340.0,
        "duration": 20.0,
        "notes": "Later sung passage with strong resonance and a different performance contour.",
    },
    {
        "name": "vocal03-ref-12-soft-breathwork",
        "start": 1437.0,
        "duration": 20.0,
        "notes": "Soft sung breath-work section; useful for subtle airy conditioning.",
    },
    {
        "name": "vocal03-ref-13-closing-full-voice",
        "start": 1532.0,
        "duration": 20.0,
        "notes": "Closing-region sung passage with fuller vocal body and clear projection.",
    },
]


def seconds_to_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes:02d}:{secs:05.2f}"


def extract_clip(source: Path, output: Path, start: float, duration: float) -> None:
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
        description="Extract curated sung vocal references from source-audio/vocal03.mp3"
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
            "Expected vocal03.mp3 inside a sub-folder named 'source-audio' "
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
