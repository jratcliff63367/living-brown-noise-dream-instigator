#!/usr/bin/env python3
"""
Extract representative vocal reference clips from:

    ./source-audio/vocal02.mp3

and write them to:

    ./ceremonies/indian/vocal-reference/

The timestamps were selected after reviewing the source for:
- strong natural vocal tone
- expressive phrasing / ornamentation
- register and intensity variation
- breath-forward / airy passages
- sections useful as ElevenLabs style references

Important:
The later portion of vocal02.mp3 substantially repeats earlier material, so the
selected references are taken from the unique earlier portion to avoid adding
near-duplicate reference clips.

Requirements:
- ffmpeg available on PATH

Usage:
    python extract_vocal02_references.py

Optional:
    python extract_vocal02_references.py --overwrite
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_FILE = SCRIPT_DIR / "source-audio" / "vocal02.mp3"
OUTPUT_DIR = SCRIPT_DIR / "ceremonies" / "indian" / "vocal-reference"
MANIFEST_FILE = OUTPUT_DIR / "vocal02-reference-manifest.csv"


CLIPS = [
    {
        "name": "vocal02-ref-01-natural-phrasing",
        "start": 180.0,   # 03:00
        "duration": 20.0,
        "notes": "Natural human phrasing with a clear, expressive vocal line.",
    },
    {
        "name": "vocal02-ref-02-full-resonant",
        "start": 240.0,   # 04:00
        "duration": 20.0,
        "notes": "Fuller resonant singing with strong harmonic body.",
    },
    {
        "name": "vocal02-ref-03-ornamented-vocal",
        "start": 460.0,   # 07:40
        "duration": 20.0,
        "notes": "Ornamented passage with useful pitch movement and articulation.",
    },
    {
        "name": "vocal02-ref-04-breath-and-air",
        "start": 470.0,   # 07:50
        "duration": 20.0,
        "notes": "Breath-forward texture with airy transitions and audible breath character.",
    },
    {
        "name": "vocal02-ref-05-dynamic-register",
        "start": 750.0,   # 12:30
        "duration": 20.0,
        "notes": "Strong dynamic phrasing with register movement and clear human vocal presence.",
    },
    {
        "name": "vocal02-ref-06-sustained-richness",
        "start": 880.0,   # 14:40
        "duration": 20.0,
        "notes": "Rich sustained tone with a stable, natural vocal core.",
    },
    {
        "name": "vocal02-ref-07-airy-breathwork",
        "start": 980.0,   # 16:20
        "duration": 20.0,
        "notes": "Airier passage emphasizing breath, release, and soft vocal texture.",
    },
    {
        "name": "vocal02-ref-08-breathwork-transition",
        "start": 990.0,   # 16:30
        "duration": 20.0,
        "notes": "Breath-work and vocal transition behavior; useful for conditioning natural breath detail.",
    },
    {
        "name": "vocal02-ref-09-expressive-midsection",
        "start": 1200.0,  # 20:00
        "duration": 20.0,
        "notes": "Expressive midsection phrasing with a different melodic contour.",
    },
    {
        "name": "vocal02-ref-10-breath-rich",
        "start": 1500.0,  # 25:00
        "duration": 20.0,
        "notes": "Breath-rich vocal section with softer articulation and organic timing.",
    },
    {
        "name": "vocal02-ref-11-strong-late-phrase",
        "start": 1770.0,  # 29:30
        "duration": 20.0,
        "notes": "Strong later phrase with pronounced vocal body and expressive motion.",
    },
    {
        "name": "vocal02-ref-12-resonant-and-airy",
        "start": 1990.0,  # 33:10
        "duration": 20.0,
        "notes": "Useful mixture of resonant singing and airier vocal transition.",
    },
    {
        "name": "vocal02-ref-13-breath-texture",
        "start": 2010.0,  # 33:30
        "duration": 20.0,
        "notes": "Distinct breath-texture reference with higher airy/noisy vocal content.",
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
        description="Extract curated reference clips from source-audio/vocal02.mp3"
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
            "Expected vocal02.mp3 inside a sub-folder named 'source-audio' "
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
