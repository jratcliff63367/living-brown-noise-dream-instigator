#!/usr/bin/env python3
"""
Generate a small exploratory library of female vocal chant samples with ElevenLabs Music.

- Reads the API key from eleven-labs.txt beside this script.
- Generates 20 substantially different 30-second female vocal chant samples.
- Saves outputs into ./eleven-labs-samples/
- Writes a manifest.txt beside the generated MP3s so each file can be traced to its prompt.

Keep eleven-labs.txt out of source control.
"""

from pathlib import Path
import time
import requests


MODEL_ID = "music_v2_5"
OUTPUT_FORMAT = "mp3_48000_192"
MUSIC_LENGTH_MS = 30_000

PROMPTS = [
    # 01 — Yesterday's core reference space: breathy, intimate, meditative
    (
        "Intimate female meditative chant, non-lexical invented syllables only. "
        "Slow expressive phrases, long sustained vowels, fluid pitch glides, variable vibrato, "
        "breathy and resonant tones, audible expressive inhalations and exhalations, humming and nasal textures. "
        "Hypnotic, calming, spacious, emotionally rich. No real language, no mantra repetition, no instruments."
    ),

    # 02 — Ethereal / Indian-influenced suspended melody
    (
        "Ethereal solo female vocal, Indian-influenced, slow and hypnotic. "
        "Long floating melodic phrases, breath-soft singing, graceful pitch bends, delicate ornamentation, "
        "fluid melisma, soft layered harmonies, spacious reverb, dreamy suspended atmosphere. "
        "No drums, no pop beat, no obvious song structure."
    ),

    # 03 — Rhythmic vocal percussion / Sheila-Chandra-adjacent space without imitation
    (
        "Solo female experimental a cappella trance vocal using invented syllables. "
        "Rapid cascading rhythmic patterns, precise syllabic percussion, pitched chant, glottal ornaments, "
        "sudden register changes, pitch bends, complex irregular cycles, virtuosic breath control. "
        "Voice acts as melody, rhythm, and texture. No instruments, no real language."
    ),

    # 04 — Overtone-rich / throat-derived
    (
        "Female overtone chant, low resonant chest voice with bright upper harmonics floating above it. "
        "Slow sustained tones, controlled growls, multiphonic textures, subtle throat-singing influence, "
        "deep meditative pacing, cavernous resonance, hypnotic and dark. "
        "No lyrics, no instruments, no percussion."
    ),

    # 05 — Whisper / breath-as-performance
    (
        "Female breath-chant performance where breathing is musical material. "
        "Whispered invented syllables, sighs, airy consonants, soft gasps, long exhaled tones, fragile humming, "
        "irregular breath gestures and intimate close-mic texture. Sparse, meditative, eerie but beautiful. "
        "No real words, no instruments."
    ),

    # 06 — Luminous choir-of-one
    (
        "Single female voice transformed into a luminous self-harmonized vocal cloud. "
        "Slow non-lexical chant, layered fifths and octaves, swelling choir blooms, pure sustained vowels, "
        "gentle harmonic motion, radiant cathedral-like resonance. Calm, sacred, floating. "
        "No lyrics, no instruments, no rhythmic pulse."
    ),

    # 07 — Dark contralto ritual
    (
        "Deep female contralto ritual chant, non-lexical syllables only. "
        "Low chest-register drones, slow descending glides, rough-edged resonance, sparse whispered accents, "
        "occasional sudden upper-register cries, dark ceremonial atmosphere, very slow pacing. "
        "No instruments, no percussion, no recognizable language."
    ),

    # 08 — High crystalline / almost glass-like
    (
        "High-register female vocal meditation, crystalline and weightless. "
        "Pure sustained vowels, tiny pitch inflections, delicate harmonics, floating falsetto, glassy overtone shimmer, "
        "long silences, minimal phrasing, extremely spacious and serene. "
        "No words, no instruments, no beat."
    ),

    # 09 — Impossible vocal organism
    (
        "Experimental female-derived vocal organism, beautiful but physically impossible. "
        "Voice splits and recombines, low and high registers sound simultaneously, formants drift independently of pitch, "
        "breath morphs into tone, impossible sustained phrases, evolving harmonics, non-lexical and hypnotic. "
        "No instruments, no beat."
    ),

    # 10 — Melismatic devotional space without actual language
    (
        "Solo female devotional-style chant with invented syllables only. "
        "Highly expressive melisma, ornamented pitch turns, elastic timing, long graceful phrases, warm vibrato, "
        "occasional soft humming, emotionally intense but still meditative. "
        "No real language, no instruments, no percussion."
    ),

    # 11 — Drone-mantra feel without mantra repetition
    (
        "Female drone chant built from long evolving vowel tones, no repeated mantra and no real words. "
        "Very slow harmonic movement, subtle pulse from breath and resonance, low-mid register, rich formants, "
        "gentle overtone beating, deeply hypnotic and sleep-friendly. "
        "No instruments, no percussion."
    ),

    # 12 — Fragmented / pointillistic
    (
        "Fragmented female a cappella meditation using tiny vocal gestures instead of long phrases. "
        "Soft clicks, hums, brief vowel fragments, whispered syllables, little glides, isolated breaths, "
        "widely spaced in silence, unpredictable but calm, intimate and organic. "
        "No language, no instruments."
    ),

    # 13 — Flowing continuous stream
    (
        "Continuous female trance vocal with almost no pauses. "
        "Long seamless stream of invented syllables, legato pitch glides, subtle register shifts, smooth melisma, "
        "irregular internal phrasing, sustained breath illusion, hypnotic and immersive. "
        "No lyrics, no instruments, no beat."
    ),

    # 14 — Primal / earthy
    (
        "Primal female chant, earthy and human, non-lexical only. "
        "Chest voice, raw breath, soft ululation, resonant calls, gentle vocal fry, irregular pulses, "
        "ancient ritual character without aggression, warm natural acoustics. "
        "No instruments, no recognizable words."
    ),

    # 15 — Balinese-inspired interlocking vocal texture
    (
        "Female a cappella trance texture inspired by interlocking vocal rhythms. "
        "Invented syllables, fast alternating pulses, layered call-and-response from one voice, syncopated accents, "
        "brief melodic surges, complex but controlled, hypnotic rather than theatrical. "
        "No instruments, no real language."
    ),

    # 16 — Soft lament / lonely suspension
    (
        "Lonely female vocal meditation, sparse and emotionally suspended. "
        "Slow falling phrases, breathy sustained vowels, gentle sob-like ornaments, delicate pitch bends, "
        "long spaces between phrases, distant layered echoes, haunting but soothing. "
        "No real words, no drums, no instruments."
    ),

    # 17 — Nasal / reed-like timbre
    (
        "Female non-lexical chant emphasizing nasal resonance and reed-like tone. "
        "Focused bright formants, narrow sustained notes, sliding intervals, humming transitions, "
        "subtle ornament, strange organic timbre, meditative and highly distinctive. "
        "No lyrics, no instruments."
    ),

    # 18 — Pulsed breath + tone
    (
        "Female meditative vocal built from alternating breath pulses and pitched tones. "
        "Irregular inhaled and exhaled gestures, soft aspirated attacks, warm sustained notes, occasional overtone bloom, "
        "slow evolving pulse with no fixed meter, trance-like and intimate. "
        "No words, no instruments."
    ),

    # 19 — Spectral / morphing timbre
    (
        "Spectral female vocal meditation, non-lexical and surreal. "
        "One sustained voice slowly changes timbre from breathy to resonant to metallic to pure, "
        "formants and harmonics continuously morph, pitch moves only slightly, long evolving tones, very spacious. "
        "No instruments, no beat."
    ),

    # 20 — Expressive wide-register journey
    (
        "Solo female experimental chant exploring an extreme vocal range. "
        "Deep contralto drones rise into clear high head voice, then fall through glides, melisma, hums and breathy transitions. "
        "Irregular pacing, non-repeating phrases, expressive dynamics, meditative but adventurous. "
        "No real language, no instruments."
    ),
]


def load_api_key(script_dir: Path) -> str:
    key_file = script_dir / "eleven-labs.txt"
    if not key_file.exists():
        raise FileNotFoundError(
            f"Missing API key file: {key_file}\n"
            "Create eleven-labs.txt beside this script and put only your ElevenLabs API key in it."
        )

    api_key = key_file.read_text(encoding="utf-8").strip()
    if not api_key:
        raise RuntimeError(f"API key file is empty: {key_file}")

    return api_key


def generate_one(api_key: str, prompt: str) -> bytes:
    url = "https://api.elevenlabs.io/v1/music"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
    }
    params = {
        "output_format": OUTPUT_FORMAT,
    }
    payload = {
        "prompt": prompt,
        "music_length_ms": MUSIC_LENGTH_MS,
        "model_id": MODEL_ID,
        "force_instrumental": False,
    }

    response = requests.post(
        url,
        headers=headers,
        params=params,
        json=payload,
        timeout=600,
    )

    if not response.ok:
        raise RuntimeError(
            f"ElevenLabs returned HTTP {response.status_code}\n{response.text}"
        )

    return response.content


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    api_key = load_api_key(script_dir)

    output_dir = script_dir / "eleven-labs-samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_lines = [
        "ElevenLabs female chant exploration",
        f"Model: {MODEL_ID}",
        f"Length per sample: {MUSIC_LENGTH_MS / 1000:.0f} seconds",
        f"Count: {len(PROMPTS)}",
        "",
    ]

    print(f"Generating {len(PROMPTS)} samples into:")
    print(output_dir)
    print()

    for index, prompt in enumerate(PROMPTS, start=1):
        filename = f"female-chant-{index:02d}.mp3"
        output_path = output_dir / filename

        print(f"[{index:02d}/{len(PROMPTS)}] Generating {filename}...")
        print(f"  {prompt}")

        try:
            audio_bytes = generate_one(api_key, prompt)
            output_path.write_bytes(audio_bytes)

            print(f"  Saved {len(audio_bytes):,} bytes")
            manifest_lines.extend([
                f"{index:02d}. {filename}",
                prompt,
                "",
            ])

        except Exception as exc:
            print(f"  ERROR: {exc}")
            manifest_lines.extend([
                f"{index:02d}. {filename}",
                prompt,
                f"ERROR: {exc}",
                "",
            ])

        # Be polite to the API and avoid hammering it in a tight loop.
        if index < len(PROMPTS):
            time.sleep(1.0)

    manifest_path = output_dir / "manifest.txt"
    manifest_path.write_text("\n".join(manifest_lines), encoding="utf-8")

    print()
    print(f"Finished. Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
