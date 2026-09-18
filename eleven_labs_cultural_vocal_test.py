#!/usr/bin/env python3
"""
Generate a culturally broad exploratory library of SOLO female vocal samples
with ElevenLabs Music, with prompts deliberately engineered to force large
differences in vocal mechanics, register, rhythm, articulation, phrasing, and timbre.

- Reads API key from eleven-labs.txt beside this script.
- Generates 20 x 30-second samples.
- Saves outputs into ./eleven-labs-cultural-samples/
- Writes manifest.txt with exact prompts.

Keep eleven-labs.txt out of source control.
"""

from pathlib import Path
import time
import requests


MODEL_ID = "music_v2_5"
OUTPUT_FORMAT = "mp3_48000_192"
MUSIC_LENGTH_MS = 30_000

PROMPTS = [
    # 01 — South Asian: melismatic, ornamented, flexible pitch
    (
        "SOLO FEMALE ONLY. South Asian-inspired non-lexical vocal improvisation. "
        "Mid-to-high register, highly ornamented melisma, fast grace-note turns, fluid slides, microtonal inflection, "
        "elastic free rhythm, clear focused tone, little breathiness. Expressive and intricate, not ambient. "
        "No choir, no instruments, no lyrics."
    ),

    # 02 — West African: chesty, pulsed, ululating
    (
        "SOLO FEMALE ONLY. West African-inspired vocal trance using invented vocables. "
        "Low-mid chest voice, strong rhythmic pulse, repeated evolving syllables, syncopated accents, ululation, "
        "short call-like phrases, dry intimate sound, earthy and energetic. Avoid ethereal breathy singing. "
        "No choir, no instruments, no lyrics."
    ),

    # 03 — Indigenous North American-inspired: open-throated, direct, sustained
    (
        "SOLO FEMALE ONLY. Indigenous North American-inspired vocable chant, respectful and non-ceremonial. "
        "Open-throated direct tone, strong sustained calls, narrow pitch set, clear repeated vocables with subtle variation, "
        "steady natural pulse, very little reverb, no whispery or ethereal delivery. "
        "No drum, no choir, no instruments, no real words."
    ),

    # 04 — Celtic / Gaelic air: pure, high, sparse
    (
        "SOLO FEMALE ONLY. Celtic/Gaelic-inspired wordless air. "
        "High pure head voice, very sparse phrasing, long held notes, gentle modal melody, small grace-note ornaments, "
        "free rhythm, clear bell-like tone, misty space but not breathy ambient singing. "
        "No choir, no instruments, no lyrics."
    ),

    # 05 — Ethereal Celtic / Enya-adjacent atmosphere but not a choir
    (
        "SOLO FEMALE ONLY. Ethereal Celtic-inspired vocal meditation with a smooth pure tone. "
        "Long legato phrases, soft modal melody, lush reverb, serene floating delivery, sustained vowels, "
        "minimal ornament, no rhythmic pulse. Dreamlike and luminous, but one clearly identifiable singer only. "
        "No choir, no instruments, no lyrics."
    ),

    # 06 — Balkan: bright, penetrating, raw, ornamented
    (
        "SOLO FEMALE ONLY. Balkan-inspired wordless lament. "
        "Bright penetrating chest-to-head mix, raw focused resonance, forceful ornamented cries, tight vibrato, "
        "small dissonant modal turns, abrupt glides, emotionally intense and direct. Avoid airy ethereal tone. "
        "No choir, no instruments, no lyrics."
    ),

    # 07 — Arabic/Middle Eastern: microtonal, highly ornamented, solo taqsim-like
    (
        "SOLO FEMALE ONLY. Middle Eastern-inspired non-lexical vocal improvisation. "
        "Warm resonant tone, intricate microtonal bends, rapid ornamental turns, long melismatic arcs, dramatic pauses, "
        "free rhythm, strong pitch focus, very little breath noise. "
        "No choir, no instruments, no lyrics."
    ),

    # 08 — Persian: delicate but precise, high ornament
    (
        "SOLO FEMALE ONLY. Persian-inspired non-lexical vocal improvisation. "
        "Fine-grained ornamentation, quick delicate note turns, airy-but-focused high register, precise pitch bends, "
        "short poetic-feeling phrases separated by silence, subtle vibrato, intimate dry recording. "
        "No choir, no instruments, no real words."
    ),

    # 09 — Ethiopian / Horn of Africa: nasal, modal, distinctive contour
    (
        "SOLO FEMALE ONLY. Ethiopian/Horn-of-Africa-inspired vocal meditation. "
        "Bright nasal resonance, distinctive pentatonic/modal contour, wavering sustained notes, unexpected melodic leaps, "
        "firm mid register, very clear tone, restrained rhythm, little reverb. Avoid generic ethereal chanting. "
        "No choir, no instruments, no lyrics."
    ),

    # 10 — Nordic kulning-inspired: piercing, high, distant-call mechanics
    (
        "SOLO FEMALE ONLY. Nordic folk-call / kulning-inspired vocal. "
        "Very high powerful ringing head voice, piercing sustained calls, stark wide intervals, almost no vibrato, "
        "long-held tones with abrupt starts and stops, outdoor-call intensity, not breathy and not soft. "
        "No choir, no instruments, no words."
    ),

    # 11 — Japanese traditional-inspired: restrained, sparse, narrow motion
    (
        "SOLO FEMALE ONLY. Japanese traditional-inspired wordless vocal meditation. "
        "Extremely sparse, restrained phrasing, narrow pitch motion, controlled breath, subtle pitch slides, "
        "focused straight tone, long silences, dry intimate acoustic. No lush reverb, no melismatic runs. "
        "No choir, no instruments, no lyrics."
    ),

    # 12 — Korean traditional-inspired: strong bends, dramatic vibrato
    (
        "SOLO FEMALE ONLY. Korean traditional-inspired wordless vocal. "
        "Powerful sustained mid-register tone, deep expressive pitch bends, broad dramatic vibrato, strong dynamic swells, "
        "slow emotional phrases with sudden ornaments, earthy and intense rather than airy. "
        "No choir, no instruments, no lyrics."
    ),

    # 13 — Central Asian: low, overtone-rich, drone-based
    (
        "SOLO FEMALE ONLY. Central Asian-inspired overtone vocal meditation. "
        "Very low chest register, sustained drone tones, overtone-rich resonance, guttural edge, slow pitch glides, "
        "minimal melody, long uninterrupted tones, dry spacious acoustic. Avoid breathy high female chant. "
        "No choir, no instruments, no lyrics."
    ),

    # 14 — Polynesian-inspired: open vowels, grounded, broad arcs
    (
        "SOLO FEMALE ONLY. Polynesian-inspired wordless chant. "
        "Open resonant vowels, grounded chest voice, broad melodic arcs, gentle but definite pulse, strong sustained calls, "
        "warm direct tone, minimal ornament, little reverb. Avoid whispering and ethereal ambience. "
        "No choir, no percussion, no instruments, no real words."
    ),

    # 15 — Afro-diasporic spiritual/moan-derived: soulful slides
    (
        "SOLO FEMALE ONLY. Afro-diasporic spiritual-inspired wordless vocal meditation. "
        "Deep soulful chest voice, expressive moans, blue-note-like bends, long slides, breath-supported sustained phrases, "
        "raw emotional dynamics, intimate close recording. One singer only, no stacked harmonies. "
        "No choir, no instruments, no lyrics."
    ),

    # 16 — Balinese / Southeast Asian rhythmic articulation: fast, bright, interlocking-feeling
    (
        "SOLO FEMALE ONLY. Southeast Asian / Balinese-inspired rhythmic trance vocal with invented syllables. "
        "Fast bright articulated pulses, clipped consonants, interlocking-feeling repeated cells, syncopated accents, "
        "sudden high melodic bursts, little sustain, dry clear recording. Avoid slow breathy chanting. "
        "No choir, no instruments, no lyrics."
    ),

    # 17 — Mediterranean / Sardinian pastoral: dry, open-throated, modal
    (
        "SOLO FEMALE ONLY. Mediterranean pastoral wordless chant. "
        "Dry open-throated tone, mid-low register, modal melody, sustained calls, slight roughness, "
        "free rhythm, long natural pauses, almost no reverb, ancient pastoral character. "
        "No choir, no instruments, no lyrics."
    ),

    # 18 — Slavic folk: bright, direct, forceful but solo
    (
        "SOLO FEMALE ONLY. Slavic folk-inspired wordless vocal. "
        "Bright forward placement, strong direct tone, modal leaps, sustained calls, tight ornament, "
        "firm rhythmic phrasing, minimal breathiness, slightly raw folk color. "
        "No choir, no instruments, no lyrics."
    ),

    # 19 — Experimental physical-vocal space: percussive / impossible
    (
        "SOLO FEMALE ONLY. Experimental non-lexical vocal performance radically unlike ambient chant. "
        "Percussive syllables, clicks, tongue attacks, glottal pops, sudden octave jumps, vocal fry, ululation, "
        "alternating low growls and clear high notes, irregular asymmetrical rhythm, abrupt silences. "
        "No choir, no instruments, no lyrics."
    ),

    # 20 — Sheila-Chandra-adjacent rhythmic virtuosity without imitation
    (
        "SOLO FEMALE ONLY. Virtuosic a cappella trance vocal using invented syllables. "
        "Extremely precise rapid rhythmic articulation, cascading syllabic patterns, pitched vocal percussion, "
        "complex irregular cycles, sudden register changes, sharp consonants, breath used rhythmically, no sustained ambient pads. "
        "No choir, no instruments, no real language."
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

    output_dir = script_dir / "eleven-labs-cultural-samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_lines = [
        "ElevenLabs culturally differentiated SOLO female vocal exploration",
        f"Model: {MODEL_ID}",
        f"Length per sample: {MUSIC_LENGTH_MS / 1000:.0f} seconds",
        f"Count: {len(PROMPTS)}",
        "",
    ]

    print(f"Generating {len(PROMPTS)} samples into:")
    print(output_dir)
    print()

    for index, prompt in enumerate(PROMPTS, start=1):
        filename = f"cultural-female-chant-{index:02d}.mp3"
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

        if index < len(PROMPTS):
            time.sleep(1.0)

    manifest_path = output_dir / "manifest.txt"
    manifest_path.write_text("\n".join(manifest_lines), encoding="utf-8")

    print()
    print(f"Finished. Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
