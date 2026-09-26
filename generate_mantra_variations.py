#!/usr/bin/env python3
"""Generate TWO additional variations per proven mantra reference (2-5 minutes each).
Place beside eleven-labs.txt in the project root. Run normally in VS Code.
Install: python -m pip install requests mutagen
Uses ceremonies/indian/vocal-reference/mantra-reference-song-ids.json.
Requires the original mantra-vocal-generation-manifest.json; only completed originals qualify.
Outputs mantraNN-var01-call-response.mp3 and mantraNN-var02-sustained-ensemble.mp3.
Existing originals are preserved; separate variation manifest tracks resume state.
Reruns resume; completed tracks are never regenerated, even if moved to stash.
Optional: --limit 1, --dry-run, --retry-uncertain mantra08-var01-call-response
No shuffle bag: each eligible reference is visited once, in filename order.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent
REFERENCE_DIR = ROOT / 'ceremonies/indian/vocal-reference'
OUTPUT_DIR = ROOT / 'ceremonies/indian/vocals'
REGISTRY_NAME = 'mantra-reference-song-ids.json'
MANIFEST_NAME = 'mantra-vocal-variations-manifest.json'
BASE_MANIFEST_NAME = 'mantra-vocal-generation-manifest.json'
MODEL_ID = 'music_v2_5'
OUTPUT_FORMAT = 'mp3_48000_192'
MIN_SECONDS = 120
MAX_SECONDS = 300
REFERENCE_SECONDS = 24
CONDITION_STRENGTH = 'high'
FILENAMES = tuple(f'mantra{i:02d}-ref.mp3' for i in range(1, 14))
POSITIVE = [
    'A cohesive relaxing meditative vocal chant or mantra performance',
    'Let the reference guide the vocal register, vocal texture and chanting tradition',
    'Retain the reference overall atmosphere while following the requested new phrase arrangement',
    'Allow instrumental accompaniment in the style of the reference when present; keep the voices central',
    'Natural resonant voices, comfortable balanced dynamics and immersive room ambience',
    'Sustained meditative repetition with gentle organic variation and smooth phrase transitions',
    'Non-English chanting or mantra vocalization appropriate to the reference',
]
NEGATIVE = ['English lyrics', 'English speech', 'spoken narration', 'abrupt stylistic changes',
            'sudden loud accents', 'dramatic climax', 'mid-performance stop and restart']

VARIATIONS = {
    'var01-call-response': [
        'A lead chanter sings short meditative phrases answered by a small vocal group',
        'Give the responses a distinct answering melodic contour; use a recurring question-and-answer structure with natural breathing spaces',
        'Preserve the reference vocal character and accompanying instruments while making the lead-and-response arrangement clearly audible',
    ],
    'var02-sustained-ensemble': [
        'A small vocal ensemble sustains long overlapping chant tones with staggered breaths',
        'A warm low vocal foundation supports slowly moving upper chant phrases, with consonant unison and octave blending',
        'Create a continuous spacious vocal texture and longer melodic arcs; retain the reference tradition and appropriate instrumental accompaniment',
    ],
}
JOB_IDS = tuple(name.replace('-ref.mp3', '-' + variant) for name in FILENAMES for variant in VARIATIONS)


def copyright_rejected(record):
    return (record.get('status') == 'copyright_rejected'
            or 'copyrighted_material_detected' in str(record.get('error', '')))


def reference_range(duration, variant):
    # Stay inside the recording; deliberately choose later passages, not the opening.
    usable_ms = int(duration * 1000) - 500
    width = min(REFERENCE_SECONDS * 1000, usable_ms // 2)
    if width < 3000:
        raise ValueError('Reference too short for later conditioning passages')
    latest_start = usable_ms - width
    start = int(latest_start * .55) if variant == 'var01-call-response' else latest_start
    return {'start_ms': start, 'end_ms': start + width}


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


def plan_for(song_id, slice_range, seconds, variant):
    count = (seconds + 119) // 120
    lengths = [seconds // count + (i < seconds % count) for i in range(count)]
    chunks = []
    for i, length in enumerate(lengths):
        chunks.append({'text': '[Meditative chant]', 'duration_ms': length * 1000,
                       'positive_styles': POSITIVE + VARIATIONS[variant] + (['Continue the same voices, accompaniment and chant seamlessly'] if i else []),
                       'negative_styles': NEGATIVE, 'context_adherence': 'high'})
    chunks[0].update(conditioning_ref={'song_id': song_id, 'range': slice_range},
                     condition_strength=CONDITION_STRENGTH)
    return {'chunks': chunks}


def audio_duration(source, requested):
    from mutagen.mp3 import MP3
    seconds = float(MP3(source).info.length)
    if not math.isfinite(seconds) or not max(10, requested * .8) <= seconds <= requested * 1.2:
        raise ValueError(f'Requested {requested}s, received {seconds:.2f}s; audio preserved for review')
    return seconds


def run(args):
    import requests
    from mutagen.mp3 import MP3
    registry_path = REFERENCE_DIR / REGISTRY_NAME
    if not registry_path.exists():
        raise ValueError(f'Upload references first: missing {registry_path}')
    registry = load(registry_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(OUTPUT_DIR):
        manifest_path = OUTPUT_DIR / MANIFEST_NAME
        manifest = load(manifest_path)
        base_path = OUTPUT_DIR / BASE_MANIFEST_NAME
        if not base_path.exists():
            raise ValueError(f'Missing original generation manifest: {base_path}. Needed to exclude unsuccessful references.')
        base = load(base_path)
        jobs = []
        for name in FILENAMES:
            ref = registry['files'].get(name, {})
            if not isinstance(ref, dict) or not ref.get('song_id') or ref.get('status') not in (None, 'uploaded', 'completed'):
                print(f'SKIP {name}: no successful upload recorded')
                continue
            original = base['files'].get(name, {})
            if copyright_rejected(ref) or copyright_rejected(original) or original.get('status') != 'completed':
                print(f'SKIP {name}: original generation did not complete successfully')
                continue
            # A new rejection in either variation excludes this reference on reruns.
            previous_variations = [r for r in manifest['files'].values()
                                   if r.get('reference_filename') == name]
            if any(copyright_rejected(r) for r in previous_variations):
                print(f'SKIP {name}: variation generation previously rejected for copyright')
                continue
            for variant in VARIATIONS:
                job_id = name.replace('-ref.mp3', '-' + variant)
                record = manifest['files'].get(job_id, {})
                if record.get('status') == 'completed':
                    print(f'SKIP {name}: already completed (active or stashed)')
                    continue
                filename = job_id + '.mp3'
                target = OUTPUT_DIR / filename
                partial = target.with_suffix('.mp3.part')
                stashed = ROOT / 'stash/indian/vocals' / filename
                if record and record.get('duration_seconds_requested'):
                    recovered = False
                    for candidate in (target, stashed, partial):
                        if not candidate.exists():
                            continue
                        try:
                            duration = audio_duration(candidate, record['duration_seconds_requested'])
                        except Exception as exc:
                            print(f'REVIEW {candidate.name}: {exc}')
                            continue
                        if not args.dry_run:
                            if candidate == partial:
                                if target.exists() or stashed.exists():
                                    continue
                                partial.replace(target)
                            record.update(status='completed', duration_seconds_actual=duration, recovered_utc=now())
                            save(manifest_path, manifest)
                        print(f'RECOVERED {name}: completed audio found; no request needed')
                        recovered = True
                        break
                    if recovered:
                        continue
                if copyright_rejected(record):
                    print(f'SKIP {name}: previous generation rejected for copyright')
                    continue
                if record.get('status') in ('pending', 'uncertain') and job_id not in args.retry_uncertain:
                    print(f'SKIP {name}: uncertain prior request; review saved audio before --retry-uncertain {job_id}')
                    continue
                if target.exists() or stashed.exists():
                    print(f'SKIP {name}: existing output needs review; never overwriting it')
                    continue
                source = REFERENCE_DIR / name
                if not source.exists():
                    print(f'SKIP {name}: local reference removed')
                    continue
                raw = source.read_bytes()
                if not ref.get('sha256') or hashlib.sha256(raw).hexdigest() != ref['sha256']:
                    raise ValueError(f'{name}: reference differs from uploaded file or lacks its hash; rerun uploader')
                duration = float(MP3(io.BytesIO(raw)).info.length)
                slice_range = reference_range(duration, variant)
                if record.get('composition_plan') and record.get('reference_sha256') == ref['sha256'] and record.get('reference_song_id') == ref['song_id']:
                    plan = record['composition_plan']
                    seconds = record['duration_seconds_requested']
                else:
                    seconds = random.randint(MIN_SECONDS, MAX_SECONDS)
                    plan = plan_for(ref['song_id'], slice_range, seconds, variant)
                jobs.append((name, ref, filename, seconds, plan, job_id, variant))
        if args.limit:
            jobs = jobs[:args.limit]
        print(f'\n{len(jobs)} new vocal requests; {sum(j[3] for j in jobs)/60:.1f} minutes requested. Output: {OUTPUT_DIR}', flush=True)
        if not jobs:
            return 0
        if args.dry_run:
            for name, ref, filename, seconds, plan, job_id, variant in jobs:
                print(f'{name} -> {filename} ({seconds}s)\n{json.dumps(plan, indent=2)}')
            return 0
        key = (ROOT / 'eleven-labs.txt').read_text(encoding='utf-8-sig').strip()
        if not key:
            raise ValueError('eleven-labs.txt is empty')
        with requests.Session() as session:
            for index, (name, ref, filename, seconds, plan, job_id, variant) in enumerate(jobs, 1):
                target = OUTPUT_DIR / filename
                partial = target.with_suffix('.mp3.part')
                if partial.exists():
                    # Explicit retries retain any earlier incomplete audio.
                    archive = partial.with_name(partial.name + f'.preserved-{time.time_ns()}')
                    partial.rename(archive)
                previous = manifest['files'].get(job_id)
                if previous:
                    manifest.setdefault('history', []).append({'reference': name, 'record': previous})
                record = {'status': 'pending', 'filename': filename, 'reference_filename': name, 'variation': variant,
                          'reference_song_id': ref['song_id'], 'reference_sha256': ref['sha256'],
                          'duration_seconds_requested': seconds, 'composition_plan': plan,
                          'model_id': MODEL_ID, 'output_format': OUTPUT_FORMAT, 'created_utc': now()}
                manifest['files'][job_id] = record
                save(manifest_path, manifest)
                print(f'[{index}/{len(jobs)}] {filename}: {seconds}s, reference {name}', flush=True)
                try:
                    response = session.post('https://api.elevenlabs.io/v1/music',
                        headers={'xi-api-key': key, 'Accept': 'audio/mpeg'},
                        params={'output_format': OUTPUT_FORMAT},
                        json={'model_id': MODEL_ID, 'composition_plan': plan, 'store_for_inpainting': False},
                        timeout=(30, 1200), allow_redirects=False)
                    if response.status_code != 200:
                        message = response.text[:2000].replace(key, '[REDACTED]')
                        record.update(status=('copyright_rejected' if 'copyrighted_material_detected' in message
                            else 'rejected' if 400 <= response.status_code < 500 else 'uncertain'),
                            http_status=response.status_code, error=message)
                        save(manifest_path, manifest)
                        print(f'HTTP {response.status_code}: {message}\nStopped; completed tracks remain saved.')
                        return 1
                    with partial.open('xb') as stream:
                        stream.write(response.content)
                        stream.flush()
                        os.fsync(stream.fileno())
                    actual = audio_duration(partial, seconds)
                    partial.replace(target)
                    record.update(status='completed', duration_seconds_actual=actual,
                                  generated_song_id=response.headers.get('song-id'), completed_utc=now())
                    save(manifest_path, manifest)
                    print(f'  Saved {actual:.1f}s', flush=True)
                except Exception as exc:
                    record.update(status='uncertain', error=str(exc).replace(key, '[REDACTED]'))
                    save(manifest_path, manifest)
                    raise RuntimeError(f'{name}: {record["error"]}. Stopped; paid audio is preserved. Rerun to recover completed audio.') from None
        print('Finished. Eligible variation jobs are complete.')
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--limit', type=int, help='Maximum new requests this run (e.g. 1 for audition)')
    parser.add_argument('--retry-uncertain', action='append', default=[], choices=JOB_IDS, metavar='VARIATION_ID')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    try:
        return run(args)
    except KeyboardInterrupt:
        print('\nInterrupted. Completed audio is preserved. Rerun to recover; uncertain requests are not automatically retried.')
        return 130
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
