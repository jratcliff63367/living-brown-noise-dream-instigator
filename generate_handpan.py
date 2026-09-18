#!/usr/bin/env python3
"""Generate 20 reference-conditioned solo handpan samples.
Place beside eleven-labs.txt in the project root.
Install: python -m pip install requests mutagen
Run: python generate_handpan.py
Optional: --count 1 for audition, --dry-run to inspect plans without API calls.
Each run adds COUNT new clips. Shuffle bags persist between runs.
Uses the composition-plan format of generate_indian_vocals.py.
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

NUM_CLIPS_TO_GENERATE = 20
MIN_DURATION_SECONDS = 120
MAX_DURATION_SECONDS = 300
MODEL_ID = 'music_v2_5'
OUTPUT_FORMAT = 'mp3_48000_192'
REFERENCE_SLICE_SECONDS = 18
ROOT = Path(__file__).resolve().parent
REFERENCE_DIR = ROOT / 'ceremonies/indian/instrument-reference/handpan'
OUTPUT_DIR = ROOT / 'ceremonies/indian/instruments/handpan'
API_URL = 'https://api.elevenlabs.io/v1/music'

# Five playing approaches x four musical developments = twenty distinct recipes.
# Every recipe is used once per shuffle-bag cycle, including across runs.
APPROACHES = {
    'spacious': 'Sparse isolated notes and short phrases, generous breathing space, let metal resonance decay naturally; retain the reference tonal character.',
    'gentle-pulse': 'A gentle steady underlying pulse derived from the reference, economical repeating figures, soft relaxed accents, no driving beat.',
    'flowing': 'Flowing melodic figures shaped by the reference rhythmic feel, relaxed alternating hands, smooth connected phrasing without a build-up.',
    'syncopated': 'Subtle offbeat accents within the reference basic tempo, spacious light syncopation, understated groove without showy percussion.',
    'rocking': 'Gently rocking paired-note patterns, soft alternating emphasis, preserve the reference general pacing and acoustic playing character.',
}
DEVELOPMENTS = {
    'low-anchor': 'Return periodically to the low central ding as an anchor; answer with a few mid-register notes and slowly vary the answers.',
    'upper-replies': 'Short warm mid-register phrases answered by delicate upper-register notes; vary the spaces and melodic replies gradually.',
    'evolving-motif': 'Introduce a small memorable melodic motif; change one note or accent at a time across the performance while retaining its identity.',
    'resonant-dialogue': 'Alternate rounded open ringing tones with gently damped responses; develop a quiet question-and-answer conversation on one instrument.',
}
RECIPES = {f'{a}-{b}': [x, y] for a, x in APPROACHES.items() for b, y in DEVELOPMENTS.items()}
ANCHOR = [
    'Solo acoustic handpan, one human player, instrumental only',
    'Natural tuned steel resonance, rounded fingertip attacks, delicate human timing variation',
    'Calm background accompaniment with restrained dynamics and a stable tonal center',
    'Clean intimate recording, natural note decays, consistent recording perspective',
    'Maintain one coherent playing style throughout; no dramatic intro, climax, or finale',
]
NEGATIVE = ['vocals', 'singing', 'chanting', 'speech', 'humming', 'choir',
            'other instruments', 'drum kit', 'tabla', 'harmonium', 'singing bowls',
            'synthesizers', 'ambient pads', 'backing track', 'orchestra',
            'aggressive slaps', 'virtuosic fast fills', 'cinematic build-up',
            'artificial stereo movement', 'heavy reverb', 'audience noise']


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


def references():
    from mutagen.mp3 import MP3
    registry = load(REFERENCE_DIR / 'reference-song-ids.json', {})
    result = {}
    for filename, entry in sorted(registry.get('files', {}).items()):
        if not isinstance(entry, dict) or not entry.get('song_id'):
            continue
        path = REFERENCE_DIR / filename
        if path.parent.resolve() != REFERENCE_DIR.resolve():
            raise ValueError(f'Invalid reference filename: {filename}')
        if entry.get('sha256') and hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError(f'{filename} changed since upload. Rerun the reference uploader first.')
        duration = MP3(path).info.length
        end_ms = min(REFERENCE_SLICE_SECONDS * 1000, int(duration * 1000) - 500)
        if end_ms < 3000:
            raise ValueError(f'Reference too short: {filename}')
        song_id = str(entry['song_id'])
        # Deduplicate aliases of the same uploaded reference.
        result.setdefault(song_id, {'filename': filename, 'song_id': song_id,
                                   'sha256': entry.get('sha256'), 'end_ms': end_ms})
    if not result:
        raise ValueError('No uploaded reference song IDs found.')
    return result


def plan_for(ref, recipe, duration):
    count = (duration + 119) // 120
    durations = [duration // count + (i < duration % count) for i in range(count)]
    chunks = []
    for seconds in durations:
        chunks.append({'text': '', 'duration_ms': seconds * 1000,
                       'positive_styles': ANCHOR + RECIPES[recipe],
                       'negative_styles': NEGATIVE, 'context_adherence': 'high'})
    chunks[0]['conditioning_ref'] = {'song_id': ref['song_id'], 'range': {'start_ms': 0, 'end_ms': ref['end_ms']}}
    chunks[0]['condition_strength'] = 'high'
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
    refs = references()  # Validate every reference before any paid calls.
    state_path = OUTPUT_DIR / 'generation-pool-state.json'
    manifest_path = OUTPUT_DIR / 'generation-manifest.json'
    state = load(state_path, {})
    manifest = load(manifest_path, {'schema_version': 1, 'style': 'indian', 'instrument': 'handpan', 'generations': []})
    key = '' if args.dry_run else (ROOT / 'eleven-labs.txt').read_text(encoding='utf-8-sig').strip()
    if not args.dry_run and not key:
        raise ValueError('eleven-labs.txt is empty.')
    indices = [int(m.group(1)) for p in OUTPUT_DIR.iterdir() if (m := re.match(r'^handpan-(\d+)-', p.name))]
    index = max(indices, default=0) + 1
    print(f'{args.count} handpan samples; {len(refs)} references; output: {OUTPUT_DIR}')
    with requests.Session() as session:
        for i in range(args.count):
            ref = refs[draw(state, 'reference_pool', list(refs))]
            recipe = draw(state, 'prompt_pool', list(RECIPES))
            duration = random.randint(MIN_DURATION_SECONDS, MAX_DURATION_SECONDS)
            plan = plan_for(ref, recipe, duration)
            fingerprint = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:12]
            filename = f'handpan-{index:03d}-{recipe}-{fingerprint}.mp3'
            print(f'[{i+1}/{args.count}] {filename}\n  Reference: {ref["filename"]}; {duration}s\n  ' + ' '.join(RECIPES[recipe]), flush=True)
            if args.dry_run:
                index += 1
                continue
            record = {'created_utc': now(), 'filename': filename, 'status': 'pending',
                      'reference_filename': ref['filename'], 'reference_song_id': ref['song_id'],
                      'reference_sha256': ref['sha256'], 'archetype': recipe,
                      'duration_seconds_requested': duration, 'condition_strength': 'high',
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
    lock = OUTPUT_DIR / 'generation.lock'
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
