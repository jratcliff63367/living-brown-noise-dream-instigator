#!/usr/bin/env python3
"""
Generate Indian solo-female vocal ceremony source clips with ElevenLabs Music v2.5.

The script uses:
    ./eleven-labs.txt
    ./ceremonies/indian/vocal-reference/reference-song-ids.json

and writes generated vocals to:
    ./ceremonies/indian/vocals/

The important configuration value is NUM_CLIPS_TO_GENERATE near the top.
Start with 1 while tuning. When satisfied, change it to 50.

Design goals
------------
- Solo female voice only.
- Strongly natural / human vocal production.
- No instruments.
- Mostly non-lexical chanting/vocalisation.
- A minority of explicit mantra material such as Om / Om Shanti / Hare Krishna.
- Wide stylistic variety inside a coherent Indian devotional / meditative family.
- Every generation is conditioned on one previously uploaded vocal reference.
- References and vocal archetypes are selected from persistent "random number
  pools" (shuffle bags), so every item is used before the pool refills.
- Generated durations vary from 2 to 5 minutes.
- Every generation is logged with the exact reference, archetype, plan,
  duration, output filename, and API song-id if returned.

Requirements
------------
    pip install requests

Usage
-----
    python generate_indian_vocals.py

Notes
-----
ElevenLabs Music v2/v2.5 audio-reference conditioning is supplied through
composition-plan chunks. The first chunk is conditioned on the uploaded
reference; that first chunk influences the later chunks in the same generation.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests


# ============================================================================
# USER CONFIGURATION
# ============================================================================

# TEXT FIELD SAFETY:
# Composition-plan `text` is treated as lyrics. This script NEVER puts
# descriptive prompt prose there; all such guidance goes in style fields.


# Start with 1 while we tune the recipe. Change to 50 for the production run.
NUM_CLIPS_TO_GENERATE = 10

# Requested final clip length.
MIN_DURATION_SECONDS = 120
MAX_DURATION_SECONDS = 300

# Conditioning reference slice. Our extracted references are ~20 seconds long;
# 18 seconds gives a little safety margin around MP3 duration rounding.
REFERENCE_START_MS = 0
REFERENCE_END_MS = 18_000

# "low", "medium", "high", or "xhigh".
# High is deliberate: the reference is primarily there to carry natural human
# vocal physiology/performance character that text prompts alone did poorly.
# Reference-conditioning strength is deliberately varied to broaden the bank.
# This is a shuffle bag: every 20 generations contain exactly
# 7 medium, 10 high, and 3 xhigh strengths, in shuffled order.
CONDITION_STRENGTH_POOL_TEMPLATE = (
    ["medium"] * 7
    + ["high"] * 10
    + ["xhigh"] * 3
)

MODEL_ID = "music_v2_5"
OUTPUT_FORMAT = "mp3_48000_192"

# Pause between paid generations.
PAUSE_BETWEEN_GENERATIONS_SECONDS = 2.0

# API robustness.
REQUEST_TIMEOUT_SECONDS = 1200
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 8.0


# ============================================================================
# PATHS
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
API_KEY_PATH = SCRIPT_DIR / "eleven-labs.txt"

STYLE_DIR = SCRIPT_DIR / "ceremonies" / "indian"
REFERENCE_DIR = STYLE_DIR / "vocal-reference"
REFERENCE_REGISTRY_PATH = REFERENCE_DIR / "reference-song-ids.json"

OUTPUT_DIR = STYLE_DIR / "vocals"
MANIFEST_PATH = OUTPUT_DIR / "generation-manifest.json"
POOL_STATE_PATH = OUTPUT_DIR / "generation-pool-state.json"

API_URL = "https://api.elevenlabs.io/v1/music"


# ============================================================================
# VOCAL DESIGN
# ============================================================================

# Constant anchor applied to every generation. This is intentionally specific:
# earlier text-only generations tended to fall into a polished/synthetic
# "ethereal AI meditation voice" attractor.
HUMAN_ANCHOR_STYLES = [
    "one exceptional solo female vocalist",
    "unmistakably natural acoustic human voice",
    "rich human vocal formants and harmonic complexity",
    "organic breath support and natural inhalations",
    "subtle human pitch and timing micro-variation",
    "natural vibrato rather than synthetic modulation",
    "physical throat mouth and chest resonance",
    "expressive human dynamics",
    "Indian devotional and classical vocal character",
    "high fidelity intimate vocal recording",
]

GLOBAL_NEGATIVE_STYLES = [
    "male voice",
    "choir",
    "duet",
    "multiple singers",
    "instruments",
    "percussion",
    "harmonium",
    "tanpura",
    "drums",
    "synthesizers",
    "vocoder",
    "autotune",
    "robotic voice",
    "synthetic voice",
    "computerized vocal tone",
    "artificial vocal doubling",
    "vocal pad",
    "pop production",
    "spoken meditation guidance",
    "spoken word",
    "narration",
    "speech",
    "non-mantra English lyrics",
    "descriptive lyrics",
    "instructional words",
]


# Each archetype is intentionally mechanically different. The persistent pool
# ensures all archetypes are used before repeating.
ARCHETYPES: Dict[str, Dict[str, Any]] = {
    "flowing_melodic": {
        "styles": [
            "flowing legato non-lexical vocalisation",
            "long melodic arcs",
            "warm mid register",
            "graceful Indian melisma",
            "fluid pitch bends",
            "moderate natural vibrato",
        ],
        "direction": (
            "Sing continuously in invented non-lexical syllables and open vowels. "
            "Favor long flowing phrases, graceful ornaments, and natural breathing."
        ),
        "duration_bias": "long",
        "lexical_mode": "nonlexical",
    },

    "classical_alap": {
        "styles": [
            "Indian classical alap inspired solo vocal improvisation",
            "free rhythm",
            "slow exploratory melodic development",
            "microtonal pitch bends",
            "ornamented sustained notes",
            "unhurried raga-like phrase shaping",
        ],
        "direction": (
            "Use only non-lexical vocalisation. Explore the melodic space slowly and freely "
            "with long evolving phrases, meend-like glides, ornamented sustains, and no fixed pulse."
        ),
        "duration_bias": "long",
        "lexical_mode": "nonlexical",
    },

    "bhajan_devotional": {
        "styles": [
            "solo female bhajan inspired devotional singing",
            "warm direct melody",
            "simple emotionally sincere phrase shapes",
            "gentle ornamentation",
            "clear natural diction-like articulation without real lyrics",
            "human devotional warmth",
        ],
        "direction": (
            "Use non-lexical devotional vocalisation with simpler memorable phrase shapes, "
            "warm emotional delivery, and less ornament density than classical improvisation."
        ),
        "duration_bias": "medium",
        "lexical_mode": "nonlexical",
    },

    "kirtan_pulsed": {
        "styles": [
            "solo female kirtan inspired chant",
            "clear repeating pulse",
            "short call-like melodic phrases",
            "rhythmic repetition with evolving variation",
            "bright devotional energy",
            "a cappella vocal pulse",
        ],
        "direction": (
            "Use invented syllables only. Favor short repeated chant-like phrases with a gentle "
            "internal pulse, subtle variation, and human rhythmic lift."
        ),
        "duration_bias": "short",
        "lexical_mode": "nonlexical",
    },

    "breathwork": {
        "styles": [
            "breath-work integrated into singing",
            "audible inhalations and exhalations",
            "sigh-like releases",
            "airy transitions into full resonant tone",
            "intimate close human performance",
            "slow spacious phrasing",
        ],
        "direction": (
            "Use non-lexical sung sounds with breath as an important expressive part "
            "of the performance: audible inhalations, soft exhalations, sigh-like "
            "releases, breath-to-tone transitions, and occasional humming."
        ),
        "duration_bias": "medium",
        "lexical_mode": "nonlexical",
    },

    "drone_chant": {
        "styles": [
            "narrow-range drone chant",
            "very long sustained vowels",
            "slow pitch drift around a tonal center",
            "rich human overtone resonance",
            "minimal melodic movement",
            "deep meditative continuity",
        ],
        "direction": (
            "Use non-lexical sustained vowels and humming around a narrow tonal center. "
            "Let individual tones last unusually long with subtle natural movement."
        ),
        "duration_bias": "long",
        "lexical_mode": "nonlexical",
    },

    "low_resonant": {
        "styles": [
            "low female contralto register",
            "deep chest resonance",
            "slow sustained tones",
            "rich overtone body",
            "grounded devotional presence",
            "minimal ornamentation",
        ],
        "direction": (
            "Use invented non-lexical syllables and sustained vowels. Stay mostly in "
            "a low, resonant female register with deep chest support and slow phrases."
        ),
        "duration_bias": "medium",
        "lexical_mode": "nonlexical",
    },

    "high_clear": {
        "styles": [
            "clear high female head voice",
            "pure ringing acoustic tone",
            "light natural ornament",
            "wide open vowels",
            "controlled breath support",
            "sparse luminous phrasing",
        ],
        "direction": (
            "Use non-lexical sung vowels and invented syllables. Favor a clear high "
            "head voice with spacious phrases, clean attacks, and natural breath."
        ),
        "duration_bias": "medium",
        "lexical_mode": "nonlexical",
    },

    "intricate_melisma": {
        "styles": [
            "intricate Indian classical inspired melisma",
            "rapid grace-note turns",
            "microtonal pitch inflection",
            "fluid ornamentation",
            "virtuosic natural breath control",
            "free rhythmic phrasing",
        ],
        "direction": (
            "Perform elaborate non-lexical melodic improvisation with intricate "
            "ornaments, pitch bends, grace-note turns, and highly human phrasing."
        ),
        "duration_bias": "medium",
        "lexical_mode": "nonlexical",
    },

    "rhythmic_syllables": {
        "styles": [
            "rhythmically articulated invented syllables",
            "precise consonant attacks",
            "voice functioning as rhythm and melody",
            "irregular evolving phrase lengths",
            "rapid register changes",
            "virtuosic a cappella delivery",
        ],
        "direction": (
            "Use invented syllables only. Alternate pitched chant with rhythmically "
            "articulated vocal patterns, crisp consonants, short bursts, breath, and "
            "occasional sustained tones. Keep it human and organic."
        ),
        "duration_bias": "short",
        "lexical_mode": "nonlexical",
    },

    "sparse_devotional": {
        "styles": [
            "very sparse devotional vocalisation",
            "long silences between phrases",
            "slow sustained notes",
            "subtle natural vibrato",
            "soft breath-supported entrances",
            "intimate contemplative singing",
        ],
        "direction": (
            "Use non-lexical vocal sounds. Sing only occasional slow phrases with "
            "meaningful silence, long sustains, gentle entrances, and natural breath."
        ),
        "duration_bias": "long",
        "lexical_mode": "nonlexical",
    },

    "intense_devotional": {
        "styles": [
            "emotionally intense devotional female singing",
            "strong dynamic swells",
            "rich chest to head register transitions",
            "passionate natural ornamentation",
            "powerful but controlled breath support",
            "human expressive irregularity",
        ],
        "direction": (
            "Use non-lexical devotional vocalisation with strong emotional dynamics, "
            "full resonant phrases, register changes, and expressive ornamentation. "
            "Remain meditative rather than pop or theatrical."
        ),
        "duration_bias": "medium",
        "lexical_mode": "nonlexical",
    },

    "humming_and_nasal": {
        "styles": [
            "humming and nasal resonance",
            "closed-mouth tone opening into vowels",
            "warm human resonance",
            "slow pitch glides",
            "breath-rich transitions",
            "gentle devotional improvisation",
        ],
        "direction": (
            "Use humming, nasal resonances, open vowels, and invented syllables. "
            "Move naturally between closed-mouth humming and sung tone with audible "
            "human breath and slow pitch glides."
        ),
        "duration_bias": "medium",
        "lexical_mode": "nonlexical",
    },

    "wide_register_journey": {
        "styles": [
            "wide female vocal register",
            "contrasting chest and head voice",
            "large but natural pitch intervals",
            "changing phrase density",
            "expressive Indian ornamentation",
            "organic breath-driven development",
        ],
        "direction": (
            "Use non-lexical vocalisation while exploring a wide natural female "
            "register. Move between resonant low phrases, clear high phrases, "
            "ornamented passages, breath, and sustained tones."
        ),
        "duration_bias": "long",
        "lexical_mode": "nonlexical",
    },

    "intimate_private": {
        "styles": [
            "very intimate solo female devotional singing",
            "soft close-microphone human voice",
            "restrained dynamics",
            "audible breath",
            "small fragile phrases",
            "minimal projection",
        ],
        "direction": (
            "Use non-lexical vocalisation with very intimate, private-feeling delivery. "
            "Keep phrases small, soft, breath-supported, and physically human."
        ),
        "duration_bias": "medium",
        "lexical_mode": "nonlexical",
    },

    "ecstatic_devotional": {
        "styles": [
            "ecstatic Indian devotional solo singing",
            "powerful natural projection",
            "rising emotional intensity",
            "strong resonant peaks",
            "rapid ornamental flourishes",
            "dramatic but still devotional human phrasing",
        ],
        "direction": (
            "Use non-lexical devotional vocalisation that gradually rises into stronger "
            "emotional peaks, then relaxes again. Keep the voice natural and unaccompanied."
        ),
        "duration_bias": "medium",
        "lexical_mode": "nonlexical",
    },

    "call_and_response_solo": {
        "styles": [
            "solo singer creating self-contained call and response phrasing",
            "contrasting question and answer melodic shapes",
            "short phrase pairs",
            "clear pauses between calls and answers",
            "organic devotional pulse",
            "single female voice only",
        ],
        "direction": (
            "Use invented syllables. Shape phrases as alternating call-and-response pairs "
            "performed by the same solo singer, with clear contrast between each pair."
        ),
        "duration_bias": "short",
        "lexical_mode": "nonlexical",
    },

    "lament_like": {
        "styles": [
            "Indian devotional lament inspired vocalising",
            "plaintive human tone",
            "slow descending phrase shapes",
            "expressive pitch bends",
            "restrained sorrowful intensity",
            "natural breath and vibrato",
        ],
        "direction": (
            "Use non-lexical devotional singing with plaintive, descending phrase shapes, "
            "expressive bends, and a restrained lament-like emotional color."
        ),
        "duration_bias": "medium",
        "lexical_mode": "nonlexical",
    },

    # Mantra archetypes remain a minority.
    "mantra_om": {
        "styles": [
            "solo female devotional Om chant",
            "natural sustained human resonance",
            "slow evolving repetitions",
            "rich chest and head harmonics",
            "audible organic breath",
            "unaccompanied intimate performance",
        ],
        "direction": (
            "Chant only the mantra Om in varied natural human phrases. Some Om "
            "tones may be long and resonant, others softer or breathier."
        ),
        "duration_bias": "short",
        "lexical_mode": "om",
    },

    "mantra_om_shanti": {
        "styles": [
            "solo female Om Shanti chant",
            "warm Indian devotional delivery",
            "natural phrase variation",
            "rich resonant human tone",
            "gentle breath-work",
            "unaccompanied meditative singing",
        ],
        "direction": (
            "Chant Om Shanti as the principal mantra, with natural variation in "
            "melody, breath, register, pauses, and duration."
        ),
        "duration_bias": "short",
        "lexical_mode": "om_shanti",
    },

    "mantra_hare_krishna": {
        "styles": [
            "solo female Hare Krishna devotional chant",
            "intimate kirtan inspired vocal character",
            "natural human phrasing",
            "melodic variation",
            "expressive breath support",
            "completely a cappella",
        ],
        "direction": (
            "Sing a gentle solo female Hare Krishna mantra performance without "
            "instruments or supporting singers. Vary the melody and phrasing naturally."
        ),
        "duration_bias": "short",
        "lexical_mode": "hare_krishna",
    },
}


# This determines how often each archetype appears in the archetype pool.
# Non-lexical material deliberately dominates. The mantra archetypes appear once
# each per pool cycle, while core non-lexical archetypes get extra entries.
ARCHETYPE_POOL_TEMPLATE = [
    "flowing_melodic",
    "flowing_melodic",
    "classical_alap",
    "classical_alap",
    "bhajan_devotional",
    "kirtan_pulsed",
    "breathwork",
    "breathwork",
    "drone_chant",
    "low_resonant",
    "high_clear",
    "intricate_melisma",
    "intricate_melisma",
    "rhythmic_syllables",
    "sparse_devotional",
    "sparse_devotional",
    "intense_devotional",
    "humming_and_nasal",
    "wide_register_journey",
    "wide_register_journey",
    "intimate_private",
    "ecstatic_devotional",
    "call_and_response_solo",
    "lament_like",
    "mantra_om",
    "mantra_om_shanti",
    "mantra_hare_krishna",
]


# Smaller modifiers add variation without changing the broad family.
PHRASE_DENSITY_MODIFIERS = [
    "very spacious phrasing with generous pauses",
    "moderately continuous phrasing with natural breathing gaps",
    "slow phrases that gradually become more active, then relax again",
    "alternating sparse passages and denser expressive passages",
]

ORNAMENT_MODIFIERS = [
    "restrained ornamentation",
    "moderate fluid ornamentation",
    "frequent graceful pitch bends and turns",
    "ornament density that rises and falls organically",
]

DYNAMIC_MODIFIERS = [
    "mostly intimate dynamics with occasional fuller swells",
    "broad natural dynamic range from near-whispered tone to resonant projection",
    "gentle dynamics with a few emotionally stronger phrases",
    "slow waves of intensity without sudden theatrical changes",
]

BREATH_MODIFIERS = [
    "natural breathing should remain audible",
    "include occasional expressive inhalations and sigh-like releases",
    "breath should support the phrasing without becoming exaggerated",
    "allow some phrases to emerge directly from audible breath",
]


# ============================================================================
# HELPERS
# ============================================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_api_key() -> str:
    if not API_KEY_PATH.exists():
        raise FileNotFoundError(
            f"Missing API key file:\n  {API_KEY_PATH}\n\n"
            "Create eleven-labs.txt next to this script and put only the API key in it."
        )
    key = API_KEY_PATH.read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError(f"API key file is empty: {API_KEY_PATH}")
    return key


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(path)


def stable_reference_key(filename: str, song_id: str) -> str:
    return f"{filename}|{song_id}"


def load_reference_entries() -> List[Dict[str, Any]]:
    if not REFERENCE_REGISTRY_PATH.exists():
        raise FileNotFoundError(
            f"Reference registry not found:\n  {REFERENCE_REGISTRY_PATH}\n\n"
            "Run upload_indian_vocal_references.py first."
        )

    registry = load_json(REFERENCE_REGISTRY_PATH, {})
    files = registry.get("files", {})
    references: List[Dict[str, Any]] = []

    for filename, entry in files.items():
        if not isinstance(entry, dict):
            continue
        song_id = entry.get("song_id")
        if not song_id:
            continue
        references.append({
            "filename": filename,
            "song_id": str(song_id),
            "sha256": entry.get("sha256"),
        })

    if not references:
        raise RuntimeError(
            f"No usable song_id entries found in:\n  {REFERENCE_REGISTRY_PATH}"
        )

    references.sort(key=lambda x: x["filename"].lower())
    return references


def load_pool_state() -> Dict[str, Any]:
    return load_json(
        POOL_STATE_PATH,
        {
            "reference_pool": [],
            "archetype_pool": [],
            "condition_strength_pool": [],
            "last_reference_key": None,
            "last_archetype": None,
            "last_condition_strength": None,
        },
    )


def refill_reference_pool(
    state: Dict[str, Any],
    references: List[Dict[str, Any]],
) -> None:
    keys = [stable_reference_key(r["filename"], r["song_id"]) for r in references]
    random.shuffle(keys)

    # Avoid an immediate repeat across a pool boundary if possible.
    last_key = state.get("last_reference_key")
    if len(keys) > 1 and keys[0] == last_key:
        keys[0], keys[1] = keys[1], keys[0]

    state["reference_pool"] = keys


def refill_archetype_pool(state: Dict[str, Any]) -> None:
    pool = list(ARCHETYPE_POOL_TEMPLATE)
    random.shuffle(pool)

    last = state.get("last_archetype")
    if len(pool) > 1 and pool[0] == last:
        pool[0], pool[1] = pool[1], pool[0]

    state["archetype_pool"] = pool


def refill_condition_strength_pool(state: Dict[str, Any]) -> None:
    pool = list(CONDITION_STRENGTH_POOL_TEMPLATE)
    random.shuffle(pool)

    # Avoid the same value straddling a pool boundary when possible.
    last = state.get("last_condition_strength")
    if len(pool) > 1 and pool[0] == last:
        swap_index = next(
            (i for i, value in enumerate(pool[1:], start=1) if value != last),
            None,
        )
        if swap_index is not None:
            pool[0], pool[swap_index] = pool[swap_index], pool[0]

    state["condition_strength_pool"] = pool


def choose_reference(
    state: Dict[str, Any],
    references: List[Dict[str, Any]],
) -> Dict[str, Any]:
    valid_by_key = {
        stable_reference_key(r["filename"], r["song_id"]): r
        for r in references
    }

    # Remove stale pool entries if the registry changed.
    state["reference_pool"] = [
        key for key in state.get("reference_pool", [])
        if key in valid_by_key
    ]

    if not state["reference_pool"]:
        refill_reference_pool(state, references)

    key = state["reference_pool"].pop(0)
    state["last_reference_key"] = key
    return valid_by_key[key]


def choose_archetype(state: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    state["archetype_pool"] = [
        name for name in state.get("archetype_pool", [])
        if name in ARCHETYPES
    ]

    if not state["archetype_pool"]:
        refill_archetype_pool(state)

    name = state["archetype_pool"].pop(0)
    state["last_archetype"] = name
    return name, ARCHETYPES[name]


def choose_condition_strength(state: Dict[str, Any]) -> str:
    valid = {"medium", "high", "xhigh"}
    state["condition_strength_pool"] = [
        value for value in state.get("condition_strength_pool", [])
        if value in valid
    ]

    if not state["condition_strength_pool"]:
        refill_condition_strength_pool(state)

    value = state["condition_strength_pool"].pop(0)
    state["last_condition_strength"] = value
    return value


def choose_duration_seconds(duration_bias: str) -> int:
    # Still always within the requested 2-5 minute range.
    if duration_bias == "short":
        return random.randint(120, 210)
    if duration_bias == "long":
        return random.randint(210, 300)
    return random.randint(150, 270)


def split_into_chunk_durations(total_seconds: int) -> List[int]:
    """
    Split total duration into 60-120 second sections, respecting Music v2.5's
    maximum 120-second chunk duration.
    """
    remaining = total_seconds
    durations: List[int] = []

    while remaining > 120:
        # Keep enough for the final chunk to be >= 45s where possible.
        max_this = min(120, remaining - 45)
        min_this = min(90, max_this)
        if max_this <= 60:
            break
        this_chunk = random.randint(max(60, min_this), max_this)
        durations.append(this_chunk)
        remaining -= this_chunk

    if remaining > 0:
        if remaining < 45 and durations:
            durations[-1] += remaining
        else:
            durations.append(remaining)

    # Safety check.
    assert all(3 <= d <= 120 for d in durations), durations
    assert sum(durations) == total_seconds, (durations, total_seconds)
    return durations


def choose_chunk_styles(archetype: Dict[str, Any]) -> List[str]:
    """
    Put ALL descriptive/instructional language in positive_styles, never in the
    chunk text field. ElevenLabs treats chunk text as lyrics / singable content.
    """
    styles = (
        list(HUMAN_ANCHOR_STYLES)
        + list(archetype["styles"])
        + [
            random.choice(PHRASE_DENSITY_MODIFIERS),
            random.choice(ORNAMENT_MODIFIERS),
            random.choice(DYNAMIC_MODIFIERS),
            random.choice(BREATH_MODIFIERS),
            "solo female singing only",
            "a cappella",
            "no spoken voice",
            "no narration",
            "no meditation guidance",
        ]
    )

    if archetype["lexical_mode"] != "nonlexical":
        styles += [
            "clearly sing the exact mantra words provided in the lyrics",
            "repeat the written mantra literally and recognizably",
            "do not replace the mantra with invented syllables",
        ]

    return styles


NONLEXICAL_SYLLABLES = [
    "ah", "aa", "ha", "na", "ni", "ne", "no",
    "ra", "ri", "re", "ya", "yi", "la", "li",
    "sa", "si", "ma", "mi", "ta", "ti", "da",
    "di", "ee", "oo", "ae", "ai",
]

BREATH_SOUNDS = [
    "(aah)",
    "(ooh)",
    "(hmmm)",
    "(haa)",
    "(mmm)",
]


def make_nonlexical_line(min_tokens: int = 4, max_tokens: int = 9) -> str:
    """
    Generate intentionally meaningless, singable syllables. Keep them short and
    vowel-rich so the model has material to sing without receiving English prose
    that it can accidentally turn into lyrics.
    """
    count = random.randint(min_tokens, max_tokens)
    tokens = [random.choice(NONLEXICAL_SYLLABLES) for _ in range(count)]

    # Occasionally stretch a vowel visually, which tends to invite sustained tone.
    if random.random() < 0.35:
        i = random.randrange(len(tokens))
        stretch = {
            "ah": "aaah",
            "aa": "aaaa",
            "ee": "eeee",
            "oo": "oooo",
            "ha": "haaa",
        }
        tokens[i] = stretch.get(tokens[i], tokens[i])

    return " ".join(tokens)


def make_nonlexical_text(section_index: int, archetype_name: str) -> str:
    """
    Build only actual singable content. No descriptive English instructions are
    ever placed here.
    """
    lines = [f"[Vocalise {section_index}]"]

    # Different archetypes get slightly different lyric textures without putting
    # any instructions into the lyric field.
    if archetype_name == "breathwork":
        line_count = random.randint(5, 8)
        for _ in range(line_count):
            if random.random() < 0.45:
                lines.append(random.choice(BREATH_SOUNDS))
            lines.append(make_nonlexical_line(2, 6))

    elif archetype_name == "rhythmic_syllables":
        for _ in range(random.randint(8, 12)):
            lines.append(make_nonlexical_line(5, 11))

    elif archetype_name == "sparse_devotional":
        for _ in range(random.randint(4, 6)):
            lines.append(make_nonlexical_line(2, 5))
            if random.random() < 0.4:
                lines.append(random.choice(BREATH_SOUNDS))

    elif archetype_name == "humming_and_nasal":
        for _ in range(random.randint(5, 8)):
            if random.random() < 0.55:
                lines.append("(hmmm)")
            lines.append(make_nonlexical_line(2, 6))

    else:
        for _ in range(random.randint(6, 10)):
            lines.append(make_nonlexical_line(3, 8))

    return "\n".join(lines)


def make_mantra_text(section_index: int, lexical_mode: str) -> str:
    """
    Only actual mantra lyrics go in the text field.

    We deliberately provide many literal repetitions. A very short lyric block
    spread over a 60-120 second music chunk can invite the model to improvise
    around it instead of audibly repeating the requested mantra.
    """
    lines = [f"[Chant {section_index}]"]

    if lexical_mode == "om":
        phrase = random.choice([
            "Om",
            "Ommmm",
        ])
        lines.extend([phrase] * random.randint(10, 16))

    elif lexical_mode == "om_shanti":
        phrase = "Om Shanti"
        lines.extend([phrase] * random.randint(10, 16))

    elif lexical_mode == "hare_krishna":
        mantra_cycle = [
            "Hare Krishna",
            "Hare Krishna",
            "Krishna Krishna",
            "Hare Hare",
            "Hare Rama",
            "Hare Rama",
            "Rama Rama",
            "Hare Hare",
        ]
        repeats = random.randint(2, 3)
        for _ in range(repeats):
            lines.extend(mantra_cycle)

    return "\n".join(lines)


def section_text(
    index: int,
    lexical_mode: str,
    archetype_name: str,
) -> str:
    """
    IMPORTANT: ElevenLabs treats this field as lyrics / singable content.

    Therefore this function returns ONLY:
      - a short section label
      - invented non-lexical syllables / humming / breath sounds
      - or the intended mantra itself

    Never put prompt prose or English instructions in this field.
    """
    if lexical_mode == "nonlexical":
        return make_nonlexical_text(index, archetype_name)

    return make_mantra_text(index, lexical_mode)


def build_composition_plan(
    reference: Dict[str, Any],
    archetype_name: str,
    archetype: Dict[str, Any],
    total_seconds: int,
    condition_strength: str,
) -> Dict[str, Any]:
    chunk_durations = split_into_chunk_durations(total_seconds)

    negative_styles = list(GLOBAL_NEGATIVE_STYLES)
    chunks: List[Dict[str, Any]] = []

    for i, duration_seconds in enumerate(chunk_durations, start=1):
        # Re-pick a few modifiers per chunk so a long performance can evolve
        # while remaining inside the same broad vocal archetype.
        positive_styles = choose_chunk_styles(archetype)

        chunk: Dict[str, Any] = {
            "text": section_text(
                i,
                archetype["lexical_mode"],
                archetype_name,
            ),
            "duration_ms": duration_seconds * 1000,
            "positive_styles": positive_styles,
            "negative_styles": negative_styles,
            "context_adherence": "high",
        }

        # Conditioning the first chunk influences the remainder of the song.
        if i == 1:
            chunk["conditioning_ref"] = {
                "song_id": reference["song_id"],
                "range": {
                    "start_ms": REFERENCE_START_MS,
                    "end_ms": REFERENCE_END_MS,
                },
            }
            chunk["condition_strength"] = condition_strength

        chunks.append(chunk)

    return {"chunks": chunks}


def next_output_index() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(r"^indian-vocal-(\d+)-")
    highest = 0
    for path in OUTPUT_DIR.glob("*.mp3"):
        match = pattern.match(path.name)
        if match:
            highest = max(highest, int(match.group(1)))
    return highest + 1


def safe_archetype_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def generation_fingerprint(plan: Dict[str, Any]) -> str:
    raw = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def compose_music(
    session: requests.Session,
    api_key: str,
    plan: Dict[str, Any],
) -> Tuple[bytes, str | None]:
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }

    params = {
        "output_format": OUTPUT_FORMAT,
    }

    payload = {
        "composition_plan": plan,
        "model_id": MODEL_ID,
        "store_for_inpainting": False,
    }

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.post(
                API_URL,
                headers=headers,
                params=params,
                json=payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

            if response.status_code == 200:
                return response.content, response.headers.get("song-id")

            try:
                error_body = json.dumps(response.json(), indent=2)
            except Exception:
                error_body = response.text[:3000]

            # Retrying an unchanged validation/auth payload is pointless and can
            # make debugging slower. Fail immediately on normal client errors.
            if 400 <= response.status_code < 500:
                raise ValueError(
                    f"ElevenLabs returned HTTP {response.status_code}:\n{error_body}"
                )

            raise RuntimeError(
                f"ElevenLabs returned HTTP {response.status_code}:\n{error_body}"
            )

        except ValueError:
            raise
        except Exception as exc:
            last_error = exc
            if attempt >= MAX_RETRIES:
                break

            wait = RETRY_BACKOFF_SECONDS * attempt
            print(f"    attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            print(f"    retrying in {wait:.1f}s...")
            time.sleep(wait)

    raise RuntimeError("Music generation failed after retries") from last_error


def load_manifest() -> Dict[str, Any]:
    data = load_json(
        MANIFEST_PATH,
        {
            "schema_version": 1,
            "style": "indian",
            "model_id": MODEL_ID,
            "generations": [],
        },
    )
    data.setdefault("generations", [])
    return data


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    if NUM_CLIPS_TO_GENERATE < 1:
        raise ValueError("NUM_CLIPS_TO_GENERATE must be at least 1.")

    api_key = load_api_key()
    references = load_reference_entries()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    state = load_pool_state()
    manifest = load_manifest()
    session = requests.Session()

    print("Indian solo-female vocal generator")
    print(f"Clips this run:       {NUM_CLIPS_TO_GENERATE}")
    print(f"Reference voices:     {len(references)}")
    print(f"Duration range:       {MIN_DURATION_SECONDS}-{MAX_DURATION_SECONDS}s")
    print("Condition strengths:  medium/high/xhigh weighted 35%/50%/15%")
    print(f"Output:               {OUTPUT_DIR}")
    print()

    output_index = next_output_index()

    for run_index in range(NUM_CLIPS_TO_GENERATE):
        reference = choose_reference(state, references)
        archetype_name, archetype = choose_archetype(state)
        condition_strength = choose_condition_strength(state)
        duration_seconds = choose_duration_seconds(archetype["duration_bias"])

        # Clamp in case configuration is changed later.
        duration_seconds = max(
            MIN_DURATION_SECONDS,
            min(MAX_DURATION_SECONDS, duration_seconds),
        )

        plan = build_composition_plan(
            reference,
            archetype_name,
            archetype,
            duration_seconds,
            condition_strength,
        )

        fingerprint = generation_fingerprint(plan)
        filename = (
            f"indian-vocal-{output_index:03d}-"
            f"{safe_archetype_slug(archetype_name)}-{fingerprint}.mp3"
        )
        output_path = OUTPUT_DIR / filename

        print(f"[{run_index + 1:02d}/{NUM_CLIPS_TO_GENERATE}] {filename}")
        print(f"    archetype: {archetype_name}")
        print(f"    duration:  {duration_seconds}s")
        print(f"    reference: {reference['filename']}")
        print(f"    song_id:   {reference['song_id']}")
        print(f"    conditioning: {condition_strength}")
        print(f"    chunks:    {[c['duration_ms'] // 1000 for c in plan['chunks']]}")
        print("    generating paid audio...")

        # Save pool state BEFORE the paid call. If the call crashes after being
        # accepted remotely, rerunning will not blindly reuse exactly the same
        # reference/archetype combination.
        state["updated_utc"] = utc_now_iso()
        save_json_atomic(POOL_STATE_PATH, state)

        audio_bytes, generated_song_id = compose_music(
            session,
            api_key,
            plan,
        )

        output_path.write_bytes(audio_bytes)

        generation_record = {
            "created_utc": utc_now_iso(),
            "filename": filename,
            "duration_seconds_requested": duration_seconds,
            "archetype": archetype_name,
            "lexical_mode": archetype["lexical_mode"],
            "reference_filename": reference["filename"],
            "reference_song_id": reference["song_id"],
            "reference_sha256": reference.get("sha256"),
            "condition_strength": condition_strength,
            "reference_range_ms": {
                "start_ms": REFERENCE_START_MS,
                "end_ms": REFERENCE_END_MS,
            },
            "model_id": MODEL_ID,
            "output_format": OUTPUT_FORMAT,
            "generated_song_id": generated_song_id,
            "plan_fingerprint": fingerprint,
            "composition_plan": plan,
            "file_size_bytes": len(audio_bytes),
        }

        manifest["generations"].append(generation_record)
        manifest["last_updated_utc"] = utc_now_iso()
        save_json_atomic(MANIFEST_PATH, manifest)

        print(f"    saved: {output_path}")
        if generated_song_id:
            print(f"    generated song-id: {generated_song_id}")
        print()

        output_index += 1

        if run_index + 1 < NUM_CLIPS_TO_GENERATE:
            time.sleep(PAUSE_BETWEEN_GENERATIONS_SECONDS)

    print("Done.")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Pool state: {POOL_STATE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
