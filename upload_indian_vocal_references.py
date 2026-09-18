#!/usr/bin/env python3
"""
Upload Indian vocal reference MP3s to ElevenLabs Music and cache song_ids.

Scans:
    ./ceremonies/indian/vocal-reference/*.mp3

Reads API key from:
    ./eleven-labs.txt

Writes/updates:
    ./ceremonies/indian/vocal-reference/reference-song-ids.json

Behavior:
- Computes SHA-256 for each MP3.
- If an identical file hash is already in the JSON registry and has a song_id,
  the file is NOT uploaded again.
- If a filename exists in the registry but the file contents changed, it is
  uploaded again and the registry is updated.
- The JSON registry is written after every successful upload so progress is
  preserved if the script is interrupted.
- Uploads use ElevenLabs POST /v1/music/upload.
- No composition plan, transcript, or waveform is requested, keeping the upload
  as simple as possible.

Requirements:
    pip install requests

Usage:
    python upload_indian_vocal_references.py

Optional:
    python upload_indian_vocal_references.py --force
        Re-upload every MP3 regardless of cache.

    python upload_indian_vocal_references.py --limit 5
        Process only the first 5 MP3s found.

    python upload_indian_vocal_references.py --pause 2.0
        Wait 2 seconds between uploads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import requests


API_URL = "https://api.elevenlabs.io/v1/music/upload"
REQUEST_TIMEOUT = 600
MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 5.0

SCRIPT_DIR = Path(__file__).resolve().parent
API_KEY_PATH = SCRIPT_DIR / "eleven-labs.txt"
REFERENCE_DIR = SCRIPT_DIR / "ceremonies" / "indian" / "vocal-reference"
REGISTRY_PATH = REFERENCE_DIR / "reference-song-ids.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_api_key(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"API key file not found:\n  {path}\n\n"
            "Create eleven-labs.txt next to this script and put only your API key in it."
        )

    key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"API key file is empty: {path}")

    return key


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_registry(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": 1,
            "reference_directory": str(REFERENCE_DIR.relative_to(SCRIPT_DIR)),
            "files": {},
        }

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Could not parse registry JSON: {path}") from exc

    if not isinstance(data, dict):
        raise RuntimeError(f"Registry must contain a JSON object: {path}")

    data.setdefault("schema_version", 1)
    data.setdefault("reference_directory", str(REFERENCE_DIR.relative_to(SCRIPT_DIR)))
    data.setdefault("files", {})

    if not isinstance(data["files"], dict):
        raise RuntimeError(f"'files' in registry must be a JSON object: {path}")

    return data


def save_registry(path: Path, registry: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic-ish replace: write temp first, then replace.
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(registry, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temp_path.replace(path)


def find_existing_by_hash(registry: Dict[str, Any], sha256: str) -> Optional[Dict[str, Any]]:
    for entry in registry.get("files", {}).values():
        if (
            isinstance(entry, dict)
            and entry.get("sha256") == sha256
            and entry.get("song_id")
        ):
            return entry
    return None


def upload_music(
    session: requests.Session,
    api_key: str,
    path: Path,
) -> str:
    headers = {
        "xi-api-key": api_key,
    }

    # ElevenLabs accepts a multipart file. These optional flags keep the
    # request lightweight; we only need song_id.
    form_data = {
        "extract_composition_plan": "false",
        "with_timestamps": "false",
        "with_waveform_visual": "false",
    }

    mime_type = mimetypes.guess_type(path.name)[0] or "audio/mpeg"
    last_exception: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with path.open("rb") as audio_file:
                files = {
                    "file": (path.name, audio_file, mime_type),
                }

                response = session.post(
                    API_URL,
                    headers=headers,
                    data=form_data,
                    files=files,
                    timeout=REQUEST_TIMEOUT,
                )

            if response.status_code == 200:
                payload = response.json()
                song_id = payload.get("song_id")
                if not song_id:
                    raise RuntimeError(
                        f"Upload succeeded but response did not contain song_id:\n{payload}"
                    )
                return str(song_id)

            try:
                error_body = json.dumps(response.json(), indent=2)
            except Exception:
                error_body = response.text[:2000]

            raise RuntimeError(
                f"ElevenLabs returned HTTP {response.status_code} for {path.name}:\n"
                f"{error_body}"
            )

        except Exception as exc:
            last_exception = exc

            if attempt >= MAX_RETRIES:
                break

            sleep_for = RETRY_BACKOFF_SECONDS * attempt
            print(f"    Attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            print(f"    Retrying in {sleep_for:.1f} seconds...")
            time.sleep(sleep_for)

    raise RuntimeError(f"Failed to upload {path.name}") from last_exception


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload Indian vocal reference MP3s to ElevenLabs and cache song_ids."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-upload all MP3 files even if they are already cached.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N MP3 files found.",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=1.0,
        help="Seconds to wait between successful uploads (default: 1.0).",
    )
    args = parser.parse_args()

    api_key = read_api_key(API_KEY_PATH)

    if not REFERENCE_DIR.exists():
        raise FileNotFoundError(
            f"Reference directory not found:\n  {REFERENCE_DIR}"
        )

    mp3_files = sorted(
        p for p in REFERENCE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() == ".mp3"
    )

    if args.limit is not None:
        mp3_files = mp3_files[: max(0, args.limit)]

    if not mp3_files:
        print(f"No MP3 files found in:\n  {REFERENCE_DIR}")
        return 0

    registry = load_registry(REGISTRY_PATH)
    registry["last_scan_utc"] = utc_now_iso()

    session = requests.Session()

    uploaded = 0
    skipped = 0
    failed = 0

    print(f"Reference folder: {REFERENCE_DIR}")
    print(f"Registry:         {REGISTRY_PATH}")
    print(f"MP3 files:        {len(mp3_files)}")
    print()

    for index, path in enumerate(mp3_files, start=1):
        stat = path.stat()
        sha256 = sha256_file(path)

        existing_for_name = registry["files"].get(path.name)
        existing_for_hash = find_existing_by_hash(registry, sha256)

        print(f"[{index:02d}/{len(mp3_files)}] {path.name}")
        print(f"    size={stat.st_size:,} bytes")
        print(f"    sha256={sha256}")

        if not args.force:
            if (
                isinstance(existing_for_name, dict)
                and existing_for_name.get("sha256") == sha256
                and existing_for_name.get("song_id")
            ):
                skipped += 1
                print(f"    SKIP: already uploaded as song_id={existing_for_name['song_id']}")
                print()
                continue

            # If the exact audio already exists under another filename, reuse
            # that song_id instead of paying to upload duplicate audio.
            if existing_for_hash is not None:
                reused_song_id = existing_for_hash["song_id"]
                registry["files"][path.name] = {
                    "filename": path.name,
                    "sha256": sha256,
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "song_id": reused_song_id,
                    "uploaded_utc": existing_for_hash.get("uploaded_utc"),
                    "reused_from_identical_hash": True,
                }
                save_registry(REGISTRY_PATH, registry)
                skipped += 1
                print(f"    REUSE: identical audio already uploaded as song_id={reused_song_id}")
                print()
                continue

        try:
            song_id = upload_music(session, api_key, path)

            registry["files"][path.name] = {
                "filename": path.name,
                "sha256": sha256,
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "song_id": song_id,
                "uploaded_utc": utc_now_iso(),
                "reused_from_identical_hash": False,
            }
            registry["last_successful_upload_utc"] = utc_now_iso()
            save_registry(REGISTRY_PATH, registry)

            uploaded += 1
            print(f"    UPLOADED: song_id={song_id}")
            print()

            time.sleep(max(0.0, args.pause))

        except Exception as exc:
            failed += 1
            print(f"    ERROR: {exc}", file=sys.stderr)
            print(file=sys.stderr)

    registry["last_scan_utc"] = utc_now_iso()
    registry["last_run_summary"] = {
        "uploaded": uploaded,
        "skipped_or_reused": skipped,
        "failed": failed,
        "files_seen": len(mp3_files),
    }
    save_registry(REGISTRY_PATH, registry)

    print("Finished.")
    print(f"Uploaded: {uploaded}")
    print(f"Skipped/reused: {skipped}")
    print(f"Failed: {failed}")
    print(f"Registry: {REGISTRY_PATH}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
