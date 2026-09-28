#!/usr/bin/env python3
"""Force-replace ONLY the three Mantra10 vocals, 2-5 minutes apiece.
Place in the project root beside eleven-labs.txt and run normally in VS Code.
Dependencies: requests, mutagen. Optional --dry-run makes no API requests.
EVERY normal invocation generates all three again, even after a partial failure.
No automatic paid retries. No backups of previous audio or manifests are made.
Request journals and incomplete new audio live under mantra-patch-runs/.
Source upload IDs are not changed; generated song IDs update both original
manifests and this run's journal. Existing filename-based curation stays intact.
Low conditioning encourages original music; it cannot guarantee no claims.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parent
REFERENCE_DIR = ROOT / 'ceremonies/indian/vocal-reference'
OUTPUT_DIR = ROOT / 'ceremonies/indian/vocals'
REFERENCE_NAME = 'mantra10-ref.mp3'
MODEL_ID = 'music_v2_5'
OUTPUT_FORMAT = 'mp3_48000_192'
CONDITION_STRENGTH = 'low'
REFERENCE_SECONDS = 12
MIN_SECONDS, MAX_SECONDS = 120, 300
BASE_MANIFEST = 'mantra-vocal-generation-manifest.json'
VAR_MANIFEST = 'mantra-vocal-variations-manifest.json'
JOBS = (
    ('mantra10-vocal', BASE_MANIFEST, REFERENCE_NAME,
     'An intimate male bass and baritone trio; a newly composed flowing modal melody '
     'with gently overlapping phrases, warm consonant harmony and subtle plucked lyre accompaniment',
     'Lux serena, cor quietum.\nPax profunda, nox benigna.'),
    ('mantra10-var01-call-response', VAR_MANIFEST, 'mantra10-var01-call-response',
     'A warm low male cantor answered by a small male choir; original short rising calls '
     'and longer descending responses, breathing room between exchanges, quiet rounded harp notes',
     'Sub stellis quies manet.\nIn silentio pax crescit.'),
    ('mantra10-var02-sustained-ensemble', VAR_MANIFEST, 'mantra10-var02-sustained-ensemble',
     'A low male chamber choir with staggered entrances and slowly moving consonant inner voices; '
     'new broad melodic arcs, soft cello beneath the voices, gently changing harmonies rather than a static drone',
     'Aura lenis, somnus venit.\nCor in pace requiescit.'),
)
POSITIVE = [
    'Original relaxing contemplative male vocal music, rich low register and warm natural resonance',
    'Use the reference only as a loose suggestion of vocal timbre and reflective atmosphere',
    'Compose an independent melody, harmonic progression, cadence pattern and phrase structure',
    'Sing only the supplied Latin words with gentle repetition and melodic development',
    'Beautiful restrained performance, flowing musical interest, soft dynamics and spacious warm reverb',
    'Voices remain central; accompaniment is delicate and blends comfortably with a bed of brown noise',
]
NEGATIVE = ['reference melody quotation', 'reference lyrics', 'English lyrics', 'spoken narration',
            'female solo lead', 'dramatic climax', 'loud percussion', 'harsh overtones',
            'static featureless drone', 'mid-performance stop and restart']

def now():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    if not path.exists():
        return {'schema_version': 1, 'files': {}}
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(data, dict) or not isinstance(data.get('files'), dict):
        raise ValueError(f'Invalid registry structure: {path}')
    return data


def save(path, data):
    data['updated_utc'] = now()
    fd, name = tempfile.mkstemp(prefix=path.stem + '-', suffix='.tmp', dir=path.parent)
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    for attempt in range(21):
        try:
            os.replace(name, path)
            return
        except OSError as exc:
            if (isinstance(exc, PermissionError) or getattr(exc, 'winerror', None) in (5, 32, 33)) and attempt < 20:
                time.sleep(0.5)
                continue
            raise RuntimeError(f'Could not save {path}. Latest registry is preserved in {name}; recover it before rerunning.') from exc


@contextmanager
def exclusive_lock(folder):
    # OS-owned lock: automatically released even if VS Code stops the process.
    # Keep the lock file in place so all processes lock the same file.
    with (folder / 'mantra-generation.lock').open('a+b') as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError('Another mantra generator is running. Close it before starting another.') from None
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)



def plan_for(song_id, reference_ms, seconds, index):
    count = (seconds + 119) // 120
    lengths = [seconds // count + (i < seconds % count) for i in range(count)]
    width = min(REFERENCE_SECONDS * 1000, reference_ms - 500)
    if width < 3000:
        raise ValueError('Reference is too short for conditioning')
    start = int((reference_ms - 500 - width) * (0.2, 0.5, 0.8)[index])
    chunks = []
    for i, length in enumerate(lengths):
        instructions = POSITIVE + [JOBS[index][3]]
        if i:
            instructions += ['Continue the newly composed music seamlessly, with the same voices and acoustic space']
        if i == count - 1:
            instructions += ['Conclude gently with a resolved phrase and a natural reverberant tail']
        chunks.append(dict(text=JOBS[index][4], duration_ms=length * 1000,
                           positive_styles=instructions, negative_styles=NEGATIVE,
                           context_adherence='high'))
    chunks[0].update(condition_strength=CONDITION_STRENGTH,
        conditioning_ref={'song_id': song_id, 'range': {'start_ms': start, 'end_ms': start + width}})
    return {'chunks': chunks}


def audio_duration(path, requested):
    from mutagen.mp3 import MP3
    actual = float(MP3(path).info.length)
    if not math.isfinite(actual) or not requested * .9 <= actual <= requested * 1.1:
        raise ValueError(f'Requested {requested}s, received {actual:.2f}s; new audio retained for review')
    return actual


def run(args):
    import requests
    from mutagen.mp3 import MP3
    registry = load(REFERENCE_DIR / 'mantra-reference-song-ids.json')
    ref = registry['files'].get(REFERENCE_NAME, {})
    if not ref.get('song_id') or ref.get('status') not in (None, 'uploaded', 'completed'):
        raise ValueError('No successful Mantra10 reference upload in registry')
    source = REFERENCE_DIR / REFERENCE_NAME
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if ref.get('sha256') and ref['sha256'] != digest:
        raise ValueError('Local Mantra10 reference differs from the uploaded reference')
    duration = float(MP3(source).info.length)
    if not math.isfinite(duration) or duration < 4:
        raise ValueError('Invalid reference duration')
    rng = random.SystemRandom()
    plans = [(rng.randint(MIN_SECONDS, MAX_SECONDS), rng.randrange(2147483648)) for _ in JOBS]
    for i, (job, (seconds, seed)) in enumerate(zip(JOBS, plans)):
        print(f'FORCE [{i+1}/3] {job[0]}.mp3: {seconds}s, low reference conditioning', flush=True)
        plan_for(ref['song_id'], int(duration * 1000), seconds, i)
    if args.dry_run:
        print('Dry run: no writes or paid requests. A normal run replaces all three.')
        return 0
    key = (ROOT / 'eleven-labs.txt').read_text(encoding='utf-8-sig').strip()
    if not key:
        raise ValueError('eleven-labs.txt is empty')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(OUTPUT_DIR), requests.Session() as session:
        manifests = {name: load(OUTPUT_DIR / name) for name in (BASE_MANIFEST, VAR_MANIFEST)}
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
        run_folder = ROOT / 'mantra-patch-runs' / run_id
        run_folder.mkdir(parents=True)
        journal = {'schema_version': 1, 'run_id': run_id, 'files': {}}
        journal_path = run_folder / 'patch-manifest.json'
        save(journal_path, journal)
        print(f'Request journal: {run_folder}', flush=True)
        for i, (job, (seconds, seed)) in enumerate(zip(JOBS, plans)):
            stem, manifest_name, entry_key, _, _ = job
            target = OUTPUT_DIR / (stem + '.mp3')
            partial = run_folder / (stem + '.new.mp3.part')
            record = dict(status='pending', filename=target.name,
                reference_filename=REFERENCE_NAME, reference_song_id=ref['song_id'],
                reference_sha256=digest, duration_seconds_requested=seconds,
                composition_plan=plan_for(ref['song_id'], int(duration * 1000), seconds, i),
                model_id=MODEL_ID, output_format=OUTPUT_FORMAT, seed=seed,
                created_utc=now(), patch_run_id=run_id)
            if i:
                record['variation'] = stem.removeprefix('mantra10-')
            journal['files'][stem] = record
            save(journal_path, journal)
            print(f'Generating {target.name} ({seconds}s)...', flush=True)
            try:
                with session.post('https://api.elevenlabs.io/v1/music',
                    headers={'xi-api-key': key, 'Accept': 'audio/mpeg'},
                    params={'output_format': OUTPUT_FORMAT},
                    json={'model_id': MODEL_ID, 'composition_plan': record['composition_plan'],
                          'seed': seed, 'store_for_inpainting': True},
                    timeout=(30, 1200), allow_redirects=False, stream=True) as response:
                    record['http_status'] = response.status_code
                    record['generated_song_id'] = response.headers.get('song-id')
                    record['request_id'] = response.headers.get('request-id')
                    save(journal_path, journal)
                    if response.status_code != 200:
                        record['status'] = 'rejected' if 400 <= response.status_code < 500 else 'uncertain'
                        raise RuntimeError(f'HTTP {response.status_code}: {response.text[:2000]}')
                    with partial.open('xb') as stream:
                        for chunk in response.iter_content(1024 * 1024):
                            if chunk:
                                stream.write(chunk)
                        stream.flush()
                        os.fsync(stream.fileno())
                record['duration_seconds_actual'] = audio_duration(partial, seconds)
                if not record['generated_song_id']:
                    raise RuntimeError('API returned audio but no song-id. Audio preserved; existing track not replaced')
                record['status'] = 'validated'
                save(journal_path, journal)
                os.replace(partial, target)
                record.update(status='completed', completed_utc=now())
                save(journal_path, journal)
                manifest = manifests[manifest_name]
                previous = manifest['files'].get(entry_key)
                if previous:
                    manifest.setdefault('history', []).append({'key': entry_key, 'record': previous})
                manifest['files'][entry_key] = dict(record)
                save(OUTPUT_DIR / manifest_name, manifest)
                print(f"Saved {target.name}: song_id={record['generated_song_id']}", flush=True)
            except BaseException as exc:
                if record['status'] == 'pending':
                    record['status'] = 'uncertain'
                record['error'] = str(exc).replace(key, '[REDACTED]')
                save(journal_path, journal)
                raise RuntimeError(f"{stem}: {record['error']}. Stopped; new audio (if not yet installed) and request metadata are preserved in {run_folder}. "
                                   'Another normal run will generate ALL THREE again.') from None
    print('Finished: all three replaced; generated song IDs updated in the original manifests.')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
