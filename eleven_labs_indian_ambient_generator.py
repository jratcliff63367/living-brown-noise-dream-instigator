#!/usr/bin/env python3
"""
Generate Indian-theme ambient/effect assets with the ElevenLabs Sound Effects API.

Behavior
--------
- Reads the API key from ./eleven-labs.txt (same folder as this script)
- Generates 50 assets total:
    * 10 short "event" effects  (~20%)
    * 20 medium "activity" ambients
    * 20 longer "space" ambients
- Writes output under:
    ./ceremonies/indian/ambients/
        event/
        activity/
        space/
- Saves a manifest JSON file so you can see exactly what was generated
- Skips files that already exist unless you pass --overwrite

Notes
-----
- In your current engine, "effect" vs "ambient" is fundamentally about duration,
  so the script expresses that distinction mainly through clip length.
- The prompts are authored so the source audio is mostly "clean" and not
  artificially "far away"; your 3D audio system can do the spatial placement.
- For "space" assets, the environment itself is part of the sound, so those
  prompts include a stronger sense of place / acoustic environment.

Requirements
------------
pip install requests

Usage
-----
python eleven_labs_indian_ambient_generator.py
python eleven_labs_indian_ambient_generator.py --overwrite
python eleven_labs_indian_ambient_generator.py --limit 5
python eleven_labs_indian_ambient_generator.py --only event
python eleven_labs_indian_ambient_generator.py --only activity
python eleven_labs_indian_ambient_generator.py --only space
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import requests

API_URL = "https://api.elevenlabs.io/v1/sound-generation"
OUTPUT_FORMAT = "mp3_44100_128"
REQUEST_TIMEOUT = 240
MAX_RETRIES = 5
RETRY_BACKOFF_SECONDS = 4.0

SCRIPT_DIR = Path(__file__).resolve().parent
API_KEY_PATH = SCRIPT_DIR / "eleven-labs.txt"
OUTPUT_ROOT = SCRIPT_DIR / "ceremonies" / "indian" / "ambients"
MANIFEST_PATH = OUTPUT_ROOT / "generation_manifest.json"


@dataclass
class AssetSpec:
    category: str       # event / activity / space
    slug: str
    duration_seconds: float
    loop: bool
    prompt: str


def read_api_key(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"API key file not found: {path}\n"
            f"Create a text file named 'eleven-labs.txt' next to this script."
        )
    key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"API key file is empty: {path}")
    return key


def build_asset_specs() -> List[AssetSpec]:
    specs: List[AssetSpec] = [
        # SHORT EVENT EFFECTS (10 total / 20%)
        AssetSpec("event", "single_temple_bell", 3.0, False,
                  "Single resonant Indian temple bell strike, clean source, warm metallic bloom, natural decay, no crowd, no narration."),
        AssetSpec("event", "manjira_chime_pair", 2.5, False,
                  "Two delicate manjira hand-cymbal chimes, bright and devotional, crisp attack, natural ring, clean recording."),
        AssetSpec("event", "gentle_conch_call", 4.0, False,
                  "Short gentle conch shell call used in Hindu ritual, smooth breath-driven tone, ceremonial, not harsh, clean recording."),
        AssetSpec("event", "female_om_exhale", 4.5, False,
                  "Brief solo female meditative om-like exhale, breathy and intimate, devotional, no instruments, no words beyond the chant sound."),
        AssetSpec("event", "female_shanti_fragment", 5.0, False,
                  "Short solo female devotional chant fragment in an Indian meditative style, intimate, fluid, no accompaniment, no spoken narration."),
        AssetSpec("event", "hare_krishna_call_fragment", 5.5, False,
                  "Brief female kirtan-style call fragment, warm devotional energy, light melodic lift, clean recording, minimal accompaniment."),
        AssetSpec("event", "harmonium_flourish", 4.5, False,
                  "Short harmonium devotional flourish, warm reeds, gentle and organic, clean source, no crowd noise."),
        AssetSpec("event", "soft_tabla_finger_flourish", 4.0, False,
                  "Brief soft tabla finger flourish, intimate hand percussion, subtle and organic, clean recording."),
        AssetSpec("event", "tiny_puja_bells", 3.5, False,
                  "Small cluster of tiny puja bells, delicate sparkling ritual bell texture, clean recording, natural decay."),
        AssetSpec("event", "mantra_murmur_tease", 6.0, False,
                  "Very short suggestion of a few worshippers murmuring a devotional mantra, subtle and organic, no dominant crowd chatter."),

        # MEDIUM ACTIVITY AMBIENTS (20 total)
        AssetSpec("activity", "female_kirtan_harmonium_01", 14.0, True,
                  "Small intimate kirtan activity with solo female chant and harmonium, devotional Indian atmosphere, gentle pulse, clean audio."),
        AssetSpec("activity", "female_kirtan_harmonium_02", 16.0, True,
                  "Solo female Indian devotional chant over harmonium drone, expressive and meditative, lightly rhythmic, clean recording."),
        AssetSpec("activity", "female_om_drone", 15.0, True,
                  "Solo female sustained om-like meditation chant with subtle breath and evolving tone, Indian spiritual mood, no nature sounds."),
        AssetSpec("activity", "female_shanti_shruti_box", 16.0, True,
                  "Solo female meditative chant with shruti-box-like drone, calm devotional Indian feel, intimate and clean."),
        AssetSpec("activity", "tanpura_female_vocal", 18.0, True,
                  "Female devotional vocalising over tanpura drone, fluid melodic movement, meditative Indian ambience, no narration."),
        AssetSpec("activity", "kirtan_with_soft_claps", 16.0, True,
                  "Small devotional kirtan with female lead voice, soft hand claps, harmonium support, warm and organic, clean audio."),
        AssetSpec("activity", "bansuri_and_humming", 15.0, True,
                  "Gentle Indian meditation texture with soft bansuri flute and female humming-like vocalisation, calm and devotional."),
        AssetSpec("activity", "harmonium_drone_room", 14.0, True,
                  "Warm harmonium drone with subtle devotional singing nearby, small prayer-room feel, meditative and clean."),
        AssetSpec("activity", "handpan_indian_mood", 18.0, True,
                  "Handpan in a contemplative Indian spiritual mood with soft female vocalisations, meditative and clean, no birds."),
        AssetSpec("activity", "singing_bowls_indian_prayer", 16.0, True,
                  "Gentle singing bowls with subtle Indian female devotional vocal phrases, calm and spacious, clean recording."),
        AssetSpec("activity", "shruti_box_female_vocables", 18.0, True,
                  "Solo female nonverbal devotional vocables over a shruti-box style drone, expressive and hypnotic, Indian meditative mood."),
        AssetSpec("activity", "tabla_and_humming", 14.0, True,
                  "Soft tabla pulse with gentle female humming in an Indian devotional style, restrained and meditative."),
        AssetSpec("activity", "devotional_hum_tanpura", 17.0, True,
                  "Layered devotional humming over tanpura drone, intimate Indian prayer atmosphere, no spoken words."),
        AssetSpec("activity", "morning_temple_chanting", 20.0, True,
                  "Morning temple activity with soft chanting, occasional bell accents, devotional Indian atmosphere, calm and clean."),
        AssetSpec("activity", "riverbank_mantra_activity", 18.0, True,
                  "Meditative mantra activity by an Indian riverside shrine, devotional singing and subtle ritual energy, no obvious wildlife foreground."),
        AssetSpec("activity", "small_group_kirtan", 16.0, True,
                  "Small group kirtan with a female lead and a few soft supporting voices, harmonium and light percussion, intimate and clean."),
        AssetSpec("activity", "evening_aarti_activity", 17.0, True,
                  "Evening aarti devotional activity, female chant, light bells, harmonium and ritual warmth, meditative rather than celebratory."),
        AssetSpec("activity", "ashram_corridor_song", 15.0, True,
                  "Subtle devotional song activity from an ashram corridor, female-led Indian meditative singing, clean and organic."),
        AssetSpec("activity", "prayer_room_harmonium", 14.0, True,
                  "Quiet prayer-room activity with harmonium, low devotional singing, and occasional small bells, intimate Indian spiritual mood."),
        AssetSpec("activity", "meditation_hall_vocal_texture", 19.0, True,
                  "Meditation-hall activity with female Indian chant textures, evolving breathy phrases, calm and immersive, no crowd chatter."),

        # LONGER SPACE AMBIENTS (20 total)
        AssetSpec("space", "temple_hall_dawn", 22.0, True,
                  "Large Indian temple hall at dawn, gentle devotional presence, soft room tone, occasional subtle chant traces, immersive spiritual space."),
        AssetSpec("space", "stone_shrine_courtyard", 20.0, True,
                  "Indian stone shrine courtyard ambience, air and architectural resonance, subtle ritual life, quiet contemplative space."),
        AssetSpec("space", "river_ghat_sunrise", 24.0, True,
                  "Sacred Indian river ghat at sunrise, meditative atmosphere, soft ritual presence, water and devotional spatial character."),
        AssetSpec("space", "ashram_hall_noon", 21.0, True,
                  "Quiet Indian ashram meditation hall, spacious interior tone, restrained devotional presence, calm and immersive."),
        AssetSpec("space", "monsoon_temple_veranda", 24.0, True,
                  "Temple veranda during soft monsoon weather, Indian spiritual setting, rain texture and sheltered sacred calm."),
        AssetSpec("space", "market_side_street", 20.0, True,
                  "Indian market side-street ambience, subdued bazaar activity, cultural texture, not chaotic, useful as a meditative background color."),
        AssetSpec("space", "prayer_room_shrine_space", 21.0, True,
                  "Small Indian shrine room ambience with soft resonant space, devotional stillness, occasional subtle ritual detail."),
        AssetSpec("space", "courtyard_with_bells", 22.0, True,
                  "Indian courtyard sacred space with gentle bell presence and airy resonance, calm and contemplative."),
        AssetSpec("space", "village_temple_evening", 24.0, True,
                  "Rural Indian village temple in the evening, peaceful devotional atmosphere, restrained spatial life, immersive and calm."),
        AssetSpec("space", "festival_ground_resting", 23.0, True,
                  "Indian spiritual festival ground during a quiet resting moment, soft environmental presence, meditative rather than busy."),
        AssetSpec("space", "incense_room", 20.0, True,
                  "Incense-filled meditation room in an Indian spiritual setting, warm enclosed acoustics, subtle devotional calm."),
        AssetSpec("space", "cave_temple", 24.0, True,
                  "Indian cave temple ambience, deep resonant stone acoustics, sacred stillness, subtle ritual traces."),
        AssetSpec("space", "hillside_ashram", 23.0, True,
                  "Hillside Indian ashram space, meditative calm, open air and architecture together, spiritual environmental tone."),
        AssetSpec("space", "sacred_garden", 22.0, True,
                  "Sacred garden near an Indian temple, soft natural texture and contemplative spiritual atmosphere, balanced and calm."),
        AssetSpec("space", "rain_on_temple_roof", 24.0, True,
                  "Rain on an Indian temple roof, sheltered sacred ambience, softly resonant and deeply calming."),
        AssetSpec("space", "riverside_evening_prayer_space", 24.0, True,
                  "Indian riverside evening prayer space, gentle spatial devotional presence, calm water and architectural resonance."),
        AssetSpec("space", "meditation_hall_after_chant", 20.0, True,
                  "Indian meditation hall just after chanting, lingering resonance and stillness, spacious and contemplative."),
        AssetSpec("space", "palace_courtyard_devotional", 22.0, True,
                  "Historic Indian palace courtyard used for devotional practice, airy acoustic character, calm ceremonial atmosphere."),
        AssetSpec("space", "banyan_tree_shrine", 21.0, True,
                  "Small Indian shrine beneath a banyan tree, contemplative environmental texture, quiet sacred presence."),
        AssetSpec("space", "temple_steps_night", 23.0, True,
                  "Temple steps at night in an Indian spiritual setting, subtle room-and-outdoor blend, restrained devotional atmosphere."),
    ]
    assert len(specs) == 50, f"Expected 50 specs, got {len(specs)}"
    assert sum(1 for s in specs if s.category == "event") == 10
    return specs


def ensure_output_dirs(root: Path) -> None:
    (root / "event").mkdir(parents=True, exist_ok=True)
    (root / "activity").mkdir(parents=True, exist_ok=True)
    (root / "space").mkdir(parents=True, exist_ok=True)


def sanitize_response_error(response: requests.Response) -> str:
    try:
        payload = response.json()
        return json.dumps(payload, indent=2)
    except Exception:
        return response.text[:1000] or f"HTTP {response.status_code}"


def generate_sound_effect(session: requests.Session, api_key: str, spec: AssetSpec, output_path: Path) -> None:
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {
        "text": spec.prompt,
        "duration_seconds": spec.duration_seconds,
        "loop": spec.loop,
    }
    params = {"output_format": OUTPUT_FORMAT}

    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.post(
                API_URL,
                headers=headers,
                params=params,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            if response.status_code == 200:
                output_path.write_bytes(response.content)
                return
            raise RuntimeError(
                f"ElevenLabs API error for {spec.slug} "
                f"(attempt {attempt}/{MAX_RETRIES}):\n{sanitize_response_error(response)}"
            )
        except Exception as exc:
            last_exc = exc
            if attempt >= MAX_RETRIES:
                break
            sleep_for = RETRY_BACKOFF_SECONDS * attempt
            print(f"[retry] {spec.slug}: {exc}")
            print(f"        sleeping {sleep_for:.1f}s before retry...")
            time.sleep(sleep_for)
    raise RuntimeError(f"Failed generating {spec.slug}") from last_exc


def load_existing_manifest(path: Path) -> dict:
    if not path.exists():
        return {"generated": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"generated": []}


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Indian-themed ambient/effect assets with ElevenLabs.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate files even if they already exist.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N matching assets.")
    parser.add_argument("--only", choices=["event", "activity", "space"], default=None, help="Generate only one category.")
    parser.add_argument("--pause", type=float, default=1.25, help="Seconds to pause between successful requests.")
    args = parser.parse_args()

    api_key = read_api_key(API_KEY_PATH)
    specs = build_asset_specs()

    if args.only:
        specs = [s for s in specs if s.category == args.only]
    if args.limit is not None:
        specs = specs[:args.limit]

    ensure_output_dirs(OUTPUT_ROOT)
    manifest = load_existing_manifest(MANIFEST_PATH)

    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Assets queued: {len(specs)}")
    if args.only:
        print(f"Category filter: {args.only}")
    if args.limit is not None:
        print(f"Limit: {args.limit}")
    print("")

    session = requests.Session()
    generated_count = 0
    skipped_count = 0

    for index, spec in enumerate(specs, start=1):
        out_dir = OUTPUT_ROOT / spec.category
        filename = f"{index:02d}-{spec.slug}.mp3"
        output_path = out_dir / filename

        if output_path.exists() and not args.overwrite:
            skipped_count += 1
            print(f"[skip {index:02d}/{len(specs)}] {output_path.name}")
            continue

        print(f"[gen  {index:02d}/{len(specs)}] {spec.category}/{output_path.name}")
        print(f"       duration={spec.duration_seconds}s loop={spec.loop}")
        print(f"       prompt={spec.prompt}")

        generate_sound_effect(session, api_key, spec, output_path)

        manifest["generated"].append(
            {
                "category": spec.category,
                "slug": spec.slug,
                "duration_seconds": spec.duration_seconds,
                "loop": spec.loop,
                "prompt": spec.prompt,
                "file": str(output_path.relative_to(SCRIPT_DIR)),
                "timestamp_unix": time.time(),
            }
        )
        save_manifest(MANIFEST_PATH, manifest)

        generated_count += 1
        print(f"       saved -> {output_path}")
        print("")
        time.sleep(max(0.0, args.pause))

    print("Done.")
    print(f"Generated: {generated_count}")
    print(f"Skipped:   {skipped_count}")
    print(f"Manifest:  {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
