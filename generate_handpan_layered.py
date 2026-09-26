#!/usr/bin/env python3
"""Generate original, active melodic handpan music WITHOUT audio references.
Place beside eleven-labs.txt in the project root.
Install: python -m pip install requests mutagen
Run: python generate_handpan_layered.py
Defaults to ONE audition sample. Use --count 10 for a batch after auditioning.
Each run adds new clips. Ten prompt recipes cycle through a persistent shuffle bag.
Output prefix: handpan-freeflow-. Existing samples and manifests are preserved.
Optional: --dry-run to inspect prompts without an API key or paid calls.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
import random
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

NUM_CLIPS_TO_GENERATE = 1
FILE_PREFIX = "handpan-freeflow"
MIN_DURATION_SECONDS = 120
MAX_DURATION_SECONDS = 300
MODEL_ID = 'music_v2_5'
OUTPUT_FORMAT = 'mp3_48000_192'
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / 'ceremonies/indian/instruments/handpan'
API_URL = 'https://api.elevenlabs.io/v1/music'

# Activity and melodic intent are shared; meter, motifs and roles vary.
RECIPES = {
    'solo-sunlit-runs': ['Solo handpan, 124 BPM in 4/4, bright major pentatonic melody',
        'Continuous nimble alternating hands, rolling sixteenth-note runs around a memorable rising hook; low ding on downbeats, descending replies, fluid turnarounds.'],
    'solo-dorian-groove': ['Solo handpan, 128 BPM in 4/4, warm Dorian mode',
        'A catchy syncopated two-bar melodic riff, active offbeat answers and flowing connecting notes; confident elastic groove and purposeful melodic development.'],
    'solo-triplet-current': ['Solo handpan, lilting 6/8 at dotted-quarter 100 BPM',
        'Unbroken rolling triplets, a singing upper melody over alternating low anchors; broad arching phrases with nimble six-note flourishes and graceful returns.'],
    'solo-circular-seven': ['Solo handpan, flowing 7/8 grouped 2+2+3, eighth-note pulse 250 per minute',
        'An effortless circular dance with a clear recurring melodic hook; rolling two-hand figures fill the cycle, vary the melody while maintaining its recognizable shape.'],
    'solo-falling-cascades': ['Solo handpan, 132 BPM in 4/4, sweet minor pentatonic tuning',
        'Quick descending melodic cascades answered by rising sequences; rich open ringing tones over a recurring low-note pattern, short deft fills joining longer phrases.'],
    'duet-interlocking-eighths': ['Two handpans in compatible major pentatonic tuning, 126 BPM in 4/4',
        'Interlocking eighth-note parts form a continuous sixteenth-note stream; one warm low-mid ostinato and one memorable high melody, precisely shared pulse and complementary accents.'],
    'duet-lilting-melody': ['Two consonantly tuned handpans, rolling 12/8 at dotted-quarter 108 BPM',
        'One player sustains an active triplet accompaniment while the second develops a lyrical melodic theme; overlap phrases and trade roles without dropping the rhythmic flow.'],
    'duet-syncopated-conversation': ['Two handpans in compatible Dorian tuning, 130 BPM in 4/4',
        'A buoyant syncopated bass motif supports nimble melodic questions and answers; short overlapping replies, continuous accompaniment and satisfying phrase resolutions.'],
    'trio-braided-melody': ['Three harmonically matched handpans, 122 BPM in 4/4',
        'Low rhythmic anchor, mid-register rolling pattern and a clear high-register melody braid together; introduce countermelodies in complementary spaces, tightly synchronized and tonally rich.'],
    'trio-wavelike-six': ['Three handpans in compatible bright modal tuning, 6/8 at dotted-quarter 104 BPM',
        'A continuously rolling ensemble, low tones mark the dance, mid tones ripple and high tones carry a memorable evolving tune; pass the tune between players without breaks.'],
}
ANCHOR = [
    'Engaging melodic acoustic handpan performance, handpans only, no voices',
    'Nimble confident playing with sustained rhythmic momentum and frequent clear note attacks',
    'An absorbing musical groove with memorable motifs and purposeful phrase development',
    'Full warm resonant tuned steel, overlapping consonant tones, clear ringing melody and rounded attacks',
    'Relaxation through fluid musical movement; lively hands and flowing notes at a comfortable consistent listening level',
    'Continuous performance with smooth phrase transitions, stable tempo and no mid-track stops',
    'Natural acoustic room resonance, consistent recording perspective',
]
NEGATIVE = ['vocals', 'speech', 'chanting', 'instruments other than handpans',
    'drum kit', 'backing band', 'synth pads', 'ambient drone', 'sparse isolated plinks',
    'slow meandering improvisation', 'long pauses', 'aimless noodling',
    'stop and restart', 'false endings', 'tempo collapse',
    'harsh metallic clangs', 'piercing peaks', 'dissonant tuning', 'unsynchronized ensemble']


def now():
    return datetime.now(timezone.utc).isoformat()


def load(path, default):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else default


def save(path, data):
    """Checkpoint atomically, allowing transient Windows sharing locks to clear."""
    fd, name = tempfile.mkstemp(prefix=path.stem + '-', suffix='.tmp', dir=path.parent)
    temp = Path(name)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    # Close our own handle before replacing. Indexers, editors and antivirus
    # may briefly hold either file open without Windows delete sharing.
    for attempt in range(21):
        try:
            temp.replace(path)
            return
        except OSError as exc:
            if getattr(exc, 'winerror', None) not in (5, 32, 33) and not isinstance(exc, PermissionError):
                raise
            if attempt == 20:
                raise RuntimeError(
                    f'Windows kept the registry locked: {path}. '
                    f'The new data is preserved in {temp}. Close programs holding '
                    'the registry open, then replace the registry with this temporary '
                    'file before rerunning.'
                ) from exc
            if attempt == 0:
                print(f'Waiting for Windows to release {path.name} ...', flush=True)
            time.sleep(0.5)


def draw(state, name, choices):
    pool = [v for v in state.get(name, []) if v in choices]
    if not pool:
        pool = list(choices)
        random.shuffle(pool)
        if len(pool) > 1 and pool[0] == state.get('last_' + name):
            pool[0], pool[1] = pool[1], pool[0]
    value = pool.pop(0)
    state[name] = pool
    state['last_' + name] = value
    return value


def plan_for(recipe, duration):
    count = (duration + 119) // 120
    durations = [duration // count + (i < duration % count) for i in range(count)]
    chunks = []
    for chunk_index, seconds in enumerate(durations):
        chunks.append({'text': '', 'duration_ms': seconds * 1000,
                       'positive_styles': ANCHOR + RECIPES[recipe] + ([
                           'Continue the same performance seamlessly: same pulse, tuning, players and recording; no new introduction or break'
                       ] if chunk_index else []),
                       'negative_styles': NEGATIVE, 'context_adherence': 'high'})
    return {'chunks': chunks}


def generate(session, key, plan):
    response = session.post(API_URL, headers={'xi-api-key': key, 'Accept': 'audio/mpeg'},
                            params={'output_format': OUTPUT_FORMAT},
                            json={'model_id': MODEL_ID, 'composition_plan': plan, 'store_for_inpainting': False},
                            timeout=(30, 1200), allow_redirects=False)
    if response.status_code != 200:
        raise RuntimeError(f'HTTP {response.status_code}: {response.text[:2000].replace(key, "[REDACTED]")}')
    if not response.content:
        raise RuntimeError('API returned empty audio.')
    return response.content, response.headers.get('song-id')


def run(args):
    import requests
    from mutagen.mp3 import MP3
    state_path = OUTPUT_DIR / 'handpan-freeflow-pool-state.json'
    manifest_path = OUTPUT_DIR / 'handpan-freeflow-manifest.json'
    state = load(state_path, {})
    manifest = load(manifest_path, {'schema_version': 1, 'style': 'original-active-handpan', 'instrument': 'handpan', 'generations': []})
    key = '' if args.dry_run else (ROOT / 'eleven-labs.txt').read_text(encoding='utf-8-sig').strip()
    if not args.dry_run and not key:
        raise ValueError('eleven-labs.txt is empty.')
    indices = [int(m.group(1)) for p in OUTPUT_DIR.iterdir() if (m := re.match(r'^' + re.escape(FILE_PREFIX) + r'-(\d+)-', p.name))]
    index = max(indices, default=0) + 1
    print(f'{args.count} handpan samples; no references; output: {OUTPUT_DIR}')
    with requests.Session() as session:
        for i in range(args.count):
            recipe = draw(state, 'prompt_pool', list(RECIPES))
            duration = random.randint(MIN_DURATION_SECONDS, MAX_DURATION_SECONDS)
            plan = plan_for(recipe, duration)
            fingerprint = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:12]
            filename = f'{FILE_PREFIX}-{index:03d}-{recipe}-{fingerprint}.mp3'
            print(f'[{i+1}/{args.count}] {filename}\n  Original composition; {duration}s\n  ' + ' '.join(RECIPES[recipe]), flush=True)
            if args.dry_run:
                index += 1
                continue
            record = {'created_utc': now(), 'filename': filename, 'status': 'pending',
                      'uses_references': False, 'archetype': recipe,
                      'duration_seconds_requested': duration,
                      'model_id': MODEL_ID, 'output_format': OUTPUT_FORMAT, 'composition_plan': plan}
            # Save exact request before the paid call. Do not auto-retry ambiguous failures.
            manifest['generations'].append(record)
            save(manifest_path, manifest)
            save(state_path, state)
            try:
                audio, song_id = generate(session, key, plan)
                target = OUTPUT_DIR / filename
                partial = target.with_suffix('.mp3.part')
                partial.write_bytes(audio)
                actual_duration = MP3(io.BytesIO(audio)).info.length
                partial.replace(target)
                record.update(status='completed', generated_song_id=song_id,
                              duration_seconds_actual=actual_duration, file_size_bytes=len(audio))
                save(manifest_path, manifest)
                print(f'  Saved ({actual_duration:.1f}s).', flush=True)
            except Exception as exc:
                record.update(status='failed_or_uncertain', error=str(exc).replace(key, '[REDACTED]'))
                save(manifest_path, manifest)
                raise RuntimeError(f'Generation stopped: {record["error"]}. Completed samples are preserved; check the manifest before rerunning.') from None
            index += 1
            if i + 1 < args.count:
                time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=NUM_CLIPS_TO_GENERATE)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.count < 1:
        parser.error('--count must be positive')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lock = OUTPUT_DIR / 'handpan-freeflow-generation.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f'Another generator may be running. If not, remove stale lock: {lock}') from None
    try:
        os.close(fd)
        run(args)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted. Completed files remain saved; inspect any pending manifest entry.')
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
