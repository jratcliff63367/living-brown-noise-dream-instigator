#!/usr/bin/env python3
"""Generate up to 20 reference-conditioned throat-singing samples, 120-300 seconds each.
Place beside eleven-labs.txt in the project root.
Install: python -m pip install requests mutagen
Run: python generate_throat_singing.py
Optional: --count 1 for audition, --dry-run to inspect plans without API calls.
COUNT is the target total, capped at 20. Reruns finish the batch rather than adding 20 more.
Shuffle bags persist between runs. Pending/uncertain requests require manifest review.
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
REFERENCE_DIR = ROOT / 'ceremonies/indian/instrument-reference/throat-singing'
OUTPUT_DIR = ROOT / 'ceremonies/indian/instruments/throat-singing'
API_URL = 'https://api.elevenlabs.io/v1/music'

# Five vocal treatments x four developments = twenty distinct prompt recipes.
# Use each reference before recycling the reference bag; likewise for recipes.
APPROACHES = {
    'low-resonance': 'A deep low-register human throat-singing drone with softly audible upper partials; prioritize rounded chest resonance and relaxed sustained delivery.',
    'overtone-focus': 'An authentic low throat-singing fundamental with one gently emphasized overtone that slowly changes through mouth shaping; keep the overtone soft, never whistle-like or piercing.',
    'breath-phrases': 'Long relaxed throat-sung phrases with small natural breaths and soft reentries; preserve the living texture of a human performer rather than an endless synthetic tone.',
    'vowel-shading': 'Gradual rounded vowel and mouth-shape changes within a steady low throat-singing tone; reveal different harmonics slowly without articulated words or a melodic lead vocal.',
    'resonant-answers': 'A low throat-sung phrase answered by a slightly different comfortable low pitch; leisurely irregular spacing, audible vocal resonance connecting the phrases.',
}
DEVELOPMENTS = {
    'steady-anchor': 'Stay near one comfortable low tonal anchor throughout, varying resonance and breath subtly; an even, grounded and restful mood.',
    'slow-tonal-arc': 'Move gradually through a small set of compatible low pitches and return naturally to the opening tonal center; no dramatic rise or climax.',
    'harmonic-colors': 'Keep the fundamental mostly stable while changing the prominence of neighboring overtones over long phrases; restrained and warm throughout.',
    'returning-phrase': 'Revisit a simple low vocal gesture with small changes in length and harmonic emphasis; organic pacing without a rigid repeating beat.',
}
RECIPES = {f'{a}-{b}': [x, y] for a, x in APPROACHES.items() for b, y in DEVELOPMENTS.items()}
ANCHOR = [
    'Reference-led human throat singing: preserve the actual vocal technique, register, roughness and overtone relationship heard in the supplied recording',
    'The throat-singing voice must remain identifiable as a real human performance, with a low fundamental and simultaneous resonant upper harmonics; do not replace it with a synthesizer or generic ambient drone',
    'A calm, hypnotic standalone passage suitable for quiet rest, intimate and restrained rather than theatrical',
    'Gentle musical accompaniment is welcome when supported by the reference: compatible sustained drones, sparse soft instrumental tones or very understated percussion, always subordinate to the throat-singing voice',
    'Keep accompaniment harmonically compatible, sparse and stable; do not add a new ensemble merely for variety',
    'Natural breaths and short phrase gaps are welcome; maintain continuity without a long internal silence or apparent ending',
    'One uninterrupted performance across all composition sections, consistent room perspective and balanced dynamics, with no section restart or dramatic finale',
    'Use sustained nonlexical vocal tones, not spoken sentences, invented lyrics or conventional pop singing',
]
NEGATIVE = [
    'recognizable English speech', 'spoken narration', 'invented lyrical syllables',
    'pop lead vocals', 'operatic belting', 'dramatic choir', 'shouting',
    'synthetic replacement for the human voice', 'robotic voice', 'vocoder',
    'piercing whistle tones', 'harsh screaming', 'exaggerated growling',
    'loud percussion', 'driving beat', 'busy accompaniment', 'dissonant backing',
    'dramatic crescendos', 'cinematic build-up', 'sudden volume changes',
    'internal ending', 'long internal silence', 'section breaks', 'fade out and restart',
    'artificial stereo movement', 'audience noise',
]


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
    manifest = load(manifest_path, {'schema_version': 1, 'style': 'indian', 'instrument': 'throat-singing', 'generations': []})
    records = manifest['generations']
    unresolved = [r for r in records if r.get('status') in ('pending', 'failed_or_uncertain')]
    if unresolved:
        raise RuntimeError(
            'A previous request has an uncertain outcome. Review generation-manifest.json '
            'and any .mp3.part files before retrying. Recover the audio and mark completed, '
            'or mark abandoned only after confirming it cannot be recovered.'
        )
    completed = [r for r in records if r.get('status') == 'completed']
    missing = [r['filename'] for r in completed if not (OUTPUT_DIR / r['filename']).is_file()]
    if missing:
        raise RuntimeError('Completed files are missing; restore them before continuing: ' + ', '.join(missing))
    remaining = max(0, args.count - len(completed))
    if not remaining:
        print(f'Target already met: {len(completed)} completed throat-singing samples. No generation requested.')
        return
    key = '' if args.dry_run else (ROOT / 'eleven-labs.txt').read_text(encoding='utf-8-sig').strip()
    if not args.dry_run and not key:
        raise ValueError('eleven-labs.txt is empty.')
    indices = [int(m.group(1)) for p in OUTPUT_DIR.iterdir() if (m := re.match(r'^throat-singing-(\d+)-', p.name))]
    index = max(indices, default=0) + 1
    print(f'{remaining} new throat-singing samples to reach {args.count} total; {len(refs)} references; output: {OUTPUT_DIR}')
    with requests.Session() as session:
        for i in range(remaining):
            ref = refs[draw(state, 'reference_pool', list(refs))]
            recipe = draw(state, 'prompt_pool', list(RECIPES))
            duration = random.randint(MIN_DURATION_SECONDS, MAX_DURATION_SECONDS)
            plan = plan_for(ref, recipe, duration)
            fingerprint = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:12]
            filename = f'throat-singing-{index:03d}-{recipe}-{fingerprint}.mp3'
            print(f'[{i+1}/{remaining}] {filename}\n  Reference: {ref["filename"]}; {duration}s\n  ' + ' '.join(RECIPES[recipe]), flush=True)
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
            if i + 1 < remaining:
                time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=NUM_CLIPS_TO_GENERATE, help='Target total completed samples, 1-20 (default: 20)')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.count <= 20:
        parser.error('--count must be between 1 and 20')
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
