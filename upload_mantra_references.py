#!/usr/bin/env python3
"""Upload only mantra01-ref.mp3 through mantra13-ref.mp3 to ElevenLabs Music.
Place this script beside eleven-labs.txt in your project root and run it.
Install dependencies in that Python environment: python -m pip install requests mutagen
Optional: --dry-run (validate without uploads); --retry-uncertain mantra04-ref.mp3
Song IDs are checkpointed in vocal-reference/mantra-reference-song-ids.json.
Missing files and unchanged copyright-rejected references are skipped.
Uncertain prior uploads are skipped unless explicitly retried.
This script uploads references only; it does not generate music.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from datetime import datetime, timezone
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parent
REFERENCE_DIR = ROOT / 'ceremonies/indian/vocal-reference'
REGISTRY_NAME = 'mantra-reference-song-ids.json'
FILENAMES = tuple(f'mantra{i:02d}-ref.mp3' for i in range(1, 14))
UPLOAD_URL = 'https://api.elevenlabs.io/v1/music/upload'


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
    with (folder / 'mantra-upload.lock').open('a+b') as stream:
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
            raise RuntimeError('Another mantra uploader is running. Close it before starting another.') from None
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def run(args):
    import requests
    from mutagen.mp3 import MP3
    folder = REFERENCE_DIR
    if not folder.is_dir():
        raise ValueError(f'Reference folder does not exist: {folder}')
    with exclusive_lock(folder):
        path = folder / REGISTRY_NAME
        registry = load(path)
        # Read existing IDs without changing the older vocal registry.
        old = load(folder / 'reference-song-ids.json')
        prepared = []
        errors = []
        missing = 0
        for filename in FILENAMES:
            if not (folder / filename).exists():
                print(f"SKIP {filename}: file missing (removed or not supplied)", flush=True)
                missing += 1
                continue
            try:
                raw = (folder / filename).read_bytes()
                duration = float(MP3(io.BytesIO(raw)).info.length)
                # Avoid the exact 10-second boundary that previously failed server validation.
                if not math.isfinite(duration) or not 10.1 <= duration <= 600:
                    raise ValueError(f'duration {duration:.2f}s; use a clip safely over 10 seconds and at most 600 seconds')
                digest = hashlib.sha256(raw).hexdigest()
                prepared.append((filename, raw, duration, digest))
            except Exception as exc:
                errors.append(f'{filename}: {exc}')
        if errors:
            raise ValueError('No uploads started. Fix these reference files:\n' + '\n'.join(errors))
        key = '' if args.dry_run else (ROOT / 'eleven-labs.txt').read_text(encoding='utf-8-sig').strip()
        if not args.dry_run and not key:
            raise ValueError('eleven-labs.txt is empty.')
        print(f'Reference folder: {folder}\nRegistry: {path}\nFiles: {len(prepared)}', flush=True)
        uploaded = skipped = failed = blocked = uncertain = 0
        with requests.Session() as session:
            for filename, raw, duration, digest in prepared:
                current = registry['files'].get(filename, {})
                if not isinstance(current, dict):
                    raise ValueError(f'Invalid registry entry: {filename}')
                candidates = (current, old['files'].get(filename, {}))
                reuse = next((r for r in candidates if isinstance(r, dict) and r.get('song_id') and r.get('sha256') == digest), None)
                if reuse:
                    print(f'SKIP {filename}: {reuse["song_id"]}', flush=True)
                    if not args.dry_run and reuse is not current:
                        registry['files'][filename] = dict(reuse)
                        save(path, registry)
                    skipped += 1
                    continue
                # Recognize both this version's status and the previous version's
                # saved JSON error string. Never recharge an unchanged rejection.
                copyright_rejected = (current.get('status') == 'copyright_rejected'
                    or 'copyrighted_material_detected' in str(current.get('error', '')))
                if current.get('sha256') == digest and copyright_rejected:
                    print(f'SKIP {filename}: previously rejected for copyright; no upload attempted.', flush=True)
                    blocked += 1
                    continue
                if current.get('status') in ('uploading', 'uncertain') and filename not in args.retry_uncertain:
                    print(f'SKIP {filename}: previous upload outcome uncertain. Review the registry before using --retry-uncertain {filename}.', flush=True)
                    uncertain += 1
                    continue
                print(f'UPLOAD {filename}: {duration:.2f}s, {len(raw):,} bytes' + (' [dry run]' if args.dry_run else ''), flush=True)
                if args.dry_run:
                    continue
                if current:
                    registry.setdefault('history', []).append({'filename': filename, 'record': current})
                record = {'filename': filename, 'sha256': digest, 'size_bytes': len(raw),
                          'duration_seconds': duration, 'status': 'uploading', 'started_utc': now()}
                registry['files'][filename] = record
                save(path, registry)
                try:
                    response = session.post(UPLOAD_URL, headers={'xi-api-key': key},
                        files={'file': (filename, raw, 'audio/mpeg')},
                        timeout=(30, 600), allow_redirects=False)
                except requests.RequestException as exc:
                    record.update(status='uncertain', error=str(exc).replace(key, '[REDACTED]'))
                    save(path, registry)
                    raise RuntimeError(f'{filename}: connection interrupted; outcome uncertain. Successful earlier uploads are saved.') from None
                if response.status_code != 200:
                    # Explicit client rejection can be retried after fixing its cause;
                    # server failures remain uncertain. No automatic repeat charges.
                    record.update(status='rejected' if 400 <= response.status_code < 500 else 'uncertain',
                        http_status=response.status_code, error=response.text[:2000].replace(key, '[REDACTED]'))
                    if 'copyrighted_material_detected' in record['error']:
                        record['status'] = 'copyright_rejected'
                    save(path, registry)
                    print(f'FAILED {filename}: HTTP {response.status_code}\n{record["error"]}', flush=True)
                    failed += 1
                    if response.status_code not in (400, 422):
                        break
                    continue
                try:
                    payload = response.json()
                    song_id = payload.get('song_id')
                    if not isinstance(song_id, str) or not song_id.strip():
                        raise ValueError('Response has no song_id')
                except (ValueError, AttributeError):
                    record.update(status='uncertain', error='Upload returned success but no valid song_id.')
                    save(path, registry)
                    raise RuntimeError(f'{filename}: success response had no valid song_id; inspect registry before retrying.') from None
                print(f'  song_id={song_id}', flush=True)
                record.update(status='uploaded', song_id=song_id, uploaded_utc=now())
                save(path, registry)
                uploaded += 1
        print(f'Finished. Uploaded: {uploaded}; reused: {skipped}; missing: {missing}; copyright skips: {blocked}; uncertain skips: {uncertain}; failed: {failed}\nRegistry: {path}')
        return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--retry-uncertain', action='append', default=[], choices=FILENAMES, metavar='FILENAME')
    args = parser.parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        print('\nStopped. Completed song IDs are saved; an in-flight upload may need review.', file=sys.stderr)
        return 130
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
