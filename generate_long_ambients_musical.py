#!/usr/bin/env python3
"""Generate 20 varied instrumental meditation pieces, each 2-5 minutes.

The filename is retained for convenience; this is a NEW musical batch, replacing
its former long-drone recipes. Run from your project root beside eleven-labs.txt.
Install: python -m pip install requests mutagen
Run all remaining: python generate_long_ambients.py
Try the first three contrasting pieces: python generate_long_ambients.py --limit 3
Offline preview: python generate_long_ambients.py --dry-run
Select one: python generate_long_ambients.py --only meditation-music-01-piano-cello

The goal is enjoyable standalone music: expressive melodies, musical phrasing,
harmonic movement and gentle rhythm, with a different musical identity per piece.
All 20 requests use force_instrumental: no sung lyrics or human voices. They do
not require reference uploads. Each piece is one complete Music API generation.
Musical quality still needs listening and curation; prompts cannot guarantee it.

Output: ceremonies/indian/ambients/activity/meditation-music-*.mp3
Separate musical-meditation-generation-manifest.json and lock, new batch ID,
and new filenames leave previous audio, manifests and curation ratings intact.
This script does NOT remove the earlier drone recordings from runtime selection;
stash those through the curator if you do not want them played.
New tracks enter the usual Unreviewed state. Refresh/restart apps to discover them.

Fixed batch and fixed durations: reruns skip finished or stashed pieces and
complete the remainder. --limit counts new requests; --only CLIP_ID is repeatable.
The key loads from eleven-labs.txt. Generation is sequential, with a one-hour
read timeout and no automatic paid retries. Song/request IDs and complete audio
transfers are checkpointed; completed partial downloads can be recovered.

Explicit HTTP rejections can be retried by rerunning after fixing the issue.
For ambiguous failures, inspect the manifest and ElevenLabs history first.
--retry-uncertain CLIP_ID authorizes a new paid attempt and preserves any partial
under a timestamped name. Nothing is deleted. OS locks release after a crash.
API reference: https://elevenlabs.io/docs/api-reference/music/compose
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / 'ceremonies' / 'indian' / 'ambients'
STASH_ROOT = ROOT / 'stash' / 'indian' / 'ambients'
API_KEY_PATH = ROOT / 'eleven-labs.txt'
MANIFEST_PATH = OUTPUT_ROOT / 'musical-meditation-generation-manifest.json'
API_URL = 'https://api.elevenlabs.io/v1/music'
MODEL_ID = 'music_v2_5'
OUTPUT_FORMAT = 'mp3_48000_192'
BATCH_ID = 'musical-meditation-20-v1'
READ_TIMEOUT_SECONDS = 3600

COMMON = (
    'Compose a beautiful, professionally arranged instrumental piece that is enjoyable '
    'to listen to on its own and suitable for relaxed meditation. '
    'Give it a memorable but unhurried melody, expressive human phrasing, satisfying '
    'harmonic movement and a coherent musical beginning, development and ending. '
    'Let the instruments converse and the arrangement develop naturally. A gentle '
    'pulse and flowing accompaniment are welcome. Keep the mood warm, reassuring '
    'and peaceful, with smooth dynamics, rounded tone and tasteful spacious reverb. '
    'Maintain musical substance throughout: avoid static drones, aimless sustained '
    'tones, long silent gaps and repetitive filler. Avoid aggressive percussion, '
    'piercing sounds and dramatic climaxes. Purely instrumental, with no voices '
    'or lyrics. Supply music only; brown noise and spatial movement are added later. '
)


@dataclass(frozen=True)
class AssetSpec:
    clip_id: str
    seconds: int
    direction: str
    category: str = 'activity'

    @property
    def relative_file(self):
        return Path(self.category) / (self.clip_id + '.mp3')


def build_asset_specs():
    # Different musical traditions, ensembles, meters and arrangements per piece.
    return [
        AssetSpec('meditation-music-01-piano-cello', 210,
            'Lyrical contemporary chamber music for acoustic piano and cello. A tender cello melody unfolds over flowing broken piano chords, then piano takes the melody while cello answers. Warm major and relative-minor colors, graceful rubato, a contrasting middle passage and a gentle return. Intimate, melodic and emotionally welcoming; no tragic film-score drama.'),
        AssetSpec('meditation-music-02-celtic-harp-flute', 180,
            'An original Celtic-inspired pastoral air for lever harp, wooden flute and light acoustic guitar. A lilting, singable flute tune, harp countermelodies and soft guitar support in relaxed compound meter. Alternate a welcoming main theme with a fresh answering theme. Green, open and hopeful in feeling; no jig-speed playing or marching drums.'),
        AssetSpec('meditation-music-03-brazilian-acoustic', 150,
            'A mellow Brazilian-inspired acoustic instrumental with nylon-string guitar, warm electric piano and soft upright bass. Relaxed bossa-nova phrasing, elegant gently extended chords and a clear melodic guitar line. A barely brushed rhythmic touch supports easy forward movement. Intimate and sunlit, with small tasteful melodic variations; no busy jazz solo or lounge spectacle.'),
        AssetSpec('meditation-music-04-indian-bansuri-santoor', 240,
            'A melodic Indian-inspired instrumental for low bansuri, softly played santoor and restrained tabla. A graceful flute theme answered by sparkling but rounded santoor phrases. Gentle coordinated rhythm, warm melodic ornament and an evolving acoustic arrangement. Let the melody remain central with soft accompaniment; avoid a long drone introduction, fast runs or percussion solos.'),
        AssetSpec('meditation-music-05-japanese-koto-shakuhachi', 180,
            'A lyrical Japanese-inspired duet for koto and softly voiced shakuhachi. Clear pentatonic melodic phrases over gently flowing plucked accompaniment, with thoughtful answering phrases and a contrasting middle section. Natural phrasing and warm resonance, sustained musical interest without extended empty pauses. Soft comfortable flute register, no breath blasts or sharp koto attacks.'),
        AssetSpec('meditation-music-06-andalusian-nylon-guitar', 150,
            'A peaceful Mediterranean guitar instrumental. Warm nylon-string guitar presents a lyrical melody with delicate fingerpicked accompaniment and occasional soft second-guitar harmony. Gentle Spanish-influenced chord colors, relaxed rubato and an intimate room sound. An expressive acoustic miniature with a rounded melodic ending; no rapid flamenco strumming, foot percussion or dramatic flourishes.'),
        AssetSpec('meditation-music-07-west-african-kora', 210,
            'An acoustic instrumental centered on kora, with a second quiet plucked-string part and very light hand percussion. Interlocking flowing figures support a distinct, warmly lyrical melody. Patiently vary the answering phrases and harmonic colors while maintaining a relaxed coordinated pulse. Bright-hearted and restful, with rounded string attacks; no static loop or energetic percussion break.'),
        AssetSpec('meditation-music-08-persian-santur-ney', 180,
            'A Persian-inspired melodic instrumental for softly struck santur, low ney and gentle plucked lute accompaniment. A graceful modal tune passes between flute and strings, with delicate ornament and quietly moving harmony. Smooth legato phrasing, warm consonant resting points and subtle rhythmic flow. Expressive and comforting rather than mournful or tense; no shrill flute or hammered-string barrage.'),
        AssetSpec('meditation-music-09-nordic-acoustic-lullaby', 240,
            'A Nordic-inspired original lullaby for warm viola, acoustic guitar and piano. A simple memorable folk-like melody develops through tender variations and a contrasting brighter phrase. Gently rocking accompaniment, open spacious harmonies and restrained bowing. Peaceful, intimate and reassuring throughout; no bleak sustained drone or swelling cinematic orchestra.'),
        AssetSpec('meditation-music-10-gentle-ambient-electronica', 240,
            'A melodic downtempo electronic meditation piece with warm electric piano, soft analog chord accompaniment, rounded plucked synth melody and a very light brushed electronic pulse. Develop a clear theme through changing chords and subtle answering motifs. Smooth, dreamlike and harmonically rich, with steady gentle movement. No beat drop, pumping bass, noisy sweep or extended pad-only section.'),
        AssetSpec('meditation-music-11-chinese-guzheng-xiao', 180,
            'A lyrical Chinese-inspired chamber instrumental with guzheng, low xiao and soft bowed-string accompaniment. An expressive pentatonic melody with gently rippling plucked responses. Gradually exchange lead and accompaniment roles, shape a contrasting middle phrase and return softly to the opening theme. Warm and flowing, no piercing register, sweeping virtuoso glissandi or theatrical climax.'),
        AssetSpec('meditation-music-12-hawaiian-slack-key', 210,
            'A relaxed Hawaiian slack-key-inspired acoustic guitar instrumental. Ringing open tunings, a melodic fingerpicked upper voice and a gently rocking bass pattern, played softly and naturally. Include a few tasteful second-guitar harmonies and graceful changes of voicing. An original welcoming melody with relaxed island phrasing; no surf effects, vocals, novelty sounds or showy soloing.'),
        AssetSpec('meditation-music-13-mellow-jazz-ballad', 180,
            'An intimate instrumental jazz ballad for warm piano, soft flugelhorn, upright bass and delicate brushed drums. A clear lyrical melody, rich reassuring chords and a short restrained melodic variation before the theme returns. Unhurried, tender and balanced, with mellow brass tone. Keep the tune accessible; no abstract improvisation, sharp accents or busy solos.'),
        AssetSpec('meditation-music-14-baroque-lute-recorder', 150,
            'A gentle pastoral chamber piece inspired by early European music, for lute, low wooden recorder and softly bowed viola da gamba. Tuneful recorder phrases, delicate plucked counterpoint and graceful flowing bass. A relaxed dance-like meter with two related melodic themes. Warm acoustic realism and tasteful room reverb, without a brisk court dance or shrill recorder.'),
        AssetSpec('meditation-music-15-andes-flute-guitar', 180,
            'An Andean-inspired pastoral instrumental with a softly played low wooden flute, delicate charango and nylon guitar. A warm, clearly shaped melody over a gentle swaying accompaniment. Let flute and plucked strings answer one another with modest variations, then settle into a satisfying ending. Relaxed and affectionate, without a driving festival rhythm, piercing panpipes or sound effects.'),
        AssetSpec('meditation-music-16-arabic-oud-chamber', 210,
            'An intimate Arabic-inspired instrumental for warm oud, soft qanun and low reed flute, with an understated frame-drum pulse. A lyrical modal theme, small graceful ornaments and conversational exchanges among the instruments. Natural phrase development and a gentle return to the main melody. Rounded attacks and calm dynamics; avoid dramatic lament, intense improvisation or rapid percussion.'),
        AssetSpec('meditation-music-17-indonesian-bamboo-ensemble', 180,
            'A soft Indonesian-inspired instrumental for bamboo flute, mellow wooden mallet instruments and gently plucked strings. Rounded interlocking accompaniment supports a clear flowing flute melody, with a relaxed coordinated pulse and changing answering phrases. Lush acoustic warmth and gentle pentatonic harmony. Keep metallic elements minimal, avoiding bright gongs, dense striking patterns or static repetition.'),
        AssetSpec('meditation-music-18-dreamy-new-age-piano', 300,
            'A beautifully melodic New Age instrumental for acoustic piano, gentle acoustic guitar and warm chamber strings. A memorable hopeful theme, flowing piano accompaniment and expressive instrumental responses. Develop the harmony through several related passages, returning to the theme with subtle variations. Dreamy and emotionally satisfying, with natural musical pacing; no choir pads, bombastic orchestration or endless drone.'),
        AssetSpec('meditation-music-19-handpan-guitar-duet', 120,
            'A melodic acoustic duet for mellow handpan and fingerpicked nylon-string guitar. Both share one gentle coordinated rhythm and a clear consonant tune, trading short musical phrases while the other accompanies. Develop the melody with graceful variations and warm harmonic changes. Rounded handpan strikes and relaxed timing, no competing rhythms, relentless ostinato or percussion solo.'),
        AssetSpec('meditation-music-20-acoustic-storybook-waltz', 180,
            'A tender instrumental waltz for warm clarinet, fingerpicked guitar and soft piano. A flowing, memorable clarinet melody over a light three-beat accompaniment, with graceful harmonic turns and a contrasting answering tune. Intimate chamber-folk character, gentle expressive phrasing and a satisfying return. Peaceful and quietly joyful; no theatrical accordion, oom-pah bass or sentimental crescendo.'),
    ]


def payload_for(spec):
    prompt = COMMON + f'Create one complete {spec.seconds}-second piece. ' + spec.direction
    if not 120 <= spec.seconds <= 300:
        raise ValueError(f'{spec.clip_id}: length must be 2-5 minutes')
    if len(prompt) > 4100:
        raise ValueError(f'{spec.clip_id}: prompt exceeds the API character limit')
    return {'model_id': MODEL_ID, 'store_for_inpainting': False,
            'prompt': prompt, 'music_length_ms': spec.seconds * 1000,
            'force_instrumental': True}


def now():
    return datetime.now(timezone.utc).isoformat()


def save_manifest(data):
    fd, name = tempfile.mkstemp(prefix='musical-meditation-', suffix='.tmp', dir=OUTPUT_ROOT)
    temporary = Path(name)
    with os.fdopen(fd, 'w', encoding='utf-8') as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    for attempt in range(21):
        try:
            temporary.replace(MANIFEST_PATH)
            return
        except PermissionError:
            if attempt == 20:
                raise RuntimeError(f'Manifest is locked. Latest checkpoint preserved at {temporary}')
            time.sleep(0.25)


@contextmanager
def batch_lock():
    # Keep this small lock file: deleting it can create two independently locked inodes.
    path = OUTPUT_ROOT / 'musical-meditation-generation.lock'
    with path.open('a+b') as handle:
        if path.stat().st_size == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError('Another copy of this generator is running.') from None
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def inspect_audio(path, spec):
    from mutagen.mp3 import MP3
    seconds = float(MP3(path).info.length)
    if not math.isfinite(seconds) or abs(seconds - spec.seconds) > 1.5:
        raise ValueError(f'{path.name}: requested {spec.seconds}s, received {seconds:.2f}s')
    if (spec.category == 'event' and not 0 < seconds < 10) or (
            spec.category == 'activity' and seconds <= 10):
        raise ValueError(f'{path.name}: duration {seconds:.2f}s is wrong for {spec.category}')
    return {'duration_seconds_actual': seconds, 'file_size_bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def publish_partial(partial, target):
    # Atomic no-clobber publication on NTFS/POSIX; the paid .part survives errors.
    os.link(partial, target)
    partial.unlink()


def reconcile(spec, record):
    target = OUTPUT_ROOT / spec.relative_file
    stashed = STASH_ROOT / spec.relative_file
    partial = target.with_suffix('.mp3.part')
    existing = [path for path in (target, stashed) if path.exists()]
    if len(existing) > 1:
        raise RuntimeError(f'Both active and stashed copies exist: {spec.clip_id}; review before continuing.')
    if existing:
        # Respect later curation/edits to a finished recording, including trimming.
        if record.get('status') == 'completed':
            return True
        details = inspect_audio(existing[0], spec)
        if record.get('status') != 'completed':
            record.update(status='completed', recovered_utc=now(), **details)
        return True
    if partial.exists():
        attempt = (record.get('attempts') or [{}])[-1]
        # MP3 headers can advertise the full duration of an incomplete download.
        # Never infer transfer completion from Mutagen's duration alone.
        transferred = (attempt.get('download_complete') is True
                       and attempt.get('bytes_received') == partial.stat().st_size)
        try:
            if not transferred:
                raise ValueError('Download completion was not confirmed; preserved for review')
            details = inspect_audio(partial, spec)
        except Exception as exc:
            print(f'  Partial audio needs review: {partial.name}: {exc}')
            record['status'] = 'uncertain'
        else:
            publish_partial(partial, target)
            record.update(status='completed', recovered_utc=now(), **details)
            print(f'  Recovered completed audio: {target.name}')
            return True
    if record.get('status') == 'completed':
        raise RuntimeError(f'Completed audio missing from active and stash folders: {target.name}. '
                           'Restore it; this script will not silently purchase it again.')
    return False


class RejectedRequest(RuntimeError):
    pass


def generate(session, key, payload, partial, record, attempt, manifest):
    with session.post(API_URL, headers={'xi-api-key': key, 'Accept': 'audio/mpeg'},
                      params={'output_format': OUTPUT_FORMAT}, json=payload,
                      timeout=(30, READ_TIMEOUT_SECONDS), stream=True,
                      allow_redirects=False) as response:
        if response.status_code != 200:
            message = f'HTTP {response.status_code}: {response.text[:3000].replace(key, "[REDACTED]")}'
            # No automatic retry in this run. A definite rejection is safe to retry next run.
            cls = RejectedRequest if response.status_code in (400, 401, 402, 403, 404, 413, 415, 422, 429) else RuntimeError
            raise cls(message)
        song_id = response.headers.get('song-id')
        attempt.update(song_id=song_id,
                       request_id=response.headers.get('request-id')
                                  or response.headers.get('x-request-id'))
        record['generated_song_id'] = song_id
        save_manifest(manifest)
        received = 0
        with partial.open('xb') as handle:
            for block in response.iter_content(chunk_size=256 * 1024):
                if block:
                    handle.write(block)
                    received += len(block)
            handle.flush()
            os.fsync(handle.fileno())
        if received == 0:
            raise RuntimeError('API returned empty audio.')
        # Requests checks broken chunked transfers; also check a declared body size.
        length = response.headers.get('Content-Length')
        if length and not response.headers.get('Content-Encoding') and received != int(length):
            raise RuntimeError(f'Incomplete download: expected {length} bytes, received {received}')
        attempt.update(download_complete=True, bytes_received=received)
        save_manifest(manifest)
        return song_id


def run(args):
    specs = build_asset_specs()
    chosen = [s for s in specs if not args.only or s.clip_id in args.only]
    # Validate every selected recipe before making any request.
    for spec in chosen:
        payload_for(spec)
    if args.dry_run:
        # A truly offline preview: no key, dependencies, lock, manifest, or paid requests.
        for spec in chosen:
            print(f'{spec.clip_id}: {spec.seconds / 60:g} minutes, instrumental')
            print(json.dumps(payload_for(spec), indent=2, ensure_ascii=False))
        print(f'{len(chosen)} recipes, {sum(s.seconds for s in chosen) / 60:.0f} minutes total. '
              'Existing completion state is not evaluated in preview.')
        return

    import requests
    from mutagen.mp3 import MP3  # Fail before any paid request if dependency is absent.
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with batch_lock():
        manifest = (json.loads(MANIFEST_PATH.read_text(encoding='utf-8-sig'))
                    if MANIFEST_PATH.exists() else
                    {'schema_version': 1, 'batch_id': BATCH_ID, 'clips': {}})
        if manifest.get('schema_version') != 1 or manifest.get('batch_id') != BATCH_ID or not isinstance(manifest.get('clips'), dict):
            raise ValueError(f'Unexpected manifest format: {MANIFEST_PATH}')
        records = manifest['clips']
        selected_ids = {s.clip_id for s in chosen}
        if any(clip_id not in selected_ids for clip_id in args.retry_uncertain):
            raise ValueError('--retry-uncertain must name one of the selected clips.')
        queue = []
        for spec in chosen:
            record = records.setdefault(spec.clip_id, {'status': 'planned', 'attempts': []})
            if reconcile(spec, record):
                print(f'SKIP completed (active or stashed): {spec.clip_id}')
                continue
            if record['status'] in ('pending', 'uncertain'):
                if spec.clip_id not in args.retry_uncertain:
                    save_manifest(manifest)
                    raise RuntimeError(f'Uncertain previous request: {spec.clip_id}. '
                        f'Review {MANIFEST_PATH} and any .mp3.part audio. '
                        f'After checking, explicitly retry with --retry-uncertain {spec.clip_id}')
                partial = (OUTPUT_ROOT / spec.relative_file).with_suffix('.mp3.part')
                if partial.exists():
                    archive = partial.with_name(partial.name + f'.saved-{time.time_ns()}')
                    partial.rename(archive)
                record['status'] = 'planned'
                record['retry_authorized_utc'] = now()
            queue.append(spec)
        save_manifest(manifest)
        if args.limit is not None:
            queue = queue[:args.limit]
        if not queue:
            print('Selected batch is complete. No generation needed.')
            return
        key = API_KEY_PATH.read_text(encoding='utf-8-sig').strip()
        if not key:
            raise ValueError(f'Empty API key file: {API_KEY_PATH}')
        print(f'{len(queue)} new music requests; {sum(s.seconds for s in queue) / 60:.0f} '
              f'minutes of requested audio; output: {OUTPUT_ROOT}', flush=True)
        with requests.Session() as session:
            for index, spec in enumerate(queue, 1):
                target = OUTPUT_ROOT / spec.relative_file
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_suffix('.mp3.part')
                if target.exists() or partial.exists() or (STASH_ROOT / spec.relative_file).exists():
                    raise RuntimeError(f'File appeared during this run; refusing to overwrite {target.name}')
                payload = payload_for(spec)
                record = records[spec.clip_id]
                attempt = {'started_utc': now(), 'status': 'pending', 'request': payload}
                record.update(status='pending', category=spec.category,
                              file=spec.relative_file.as_posix(), duration_seconds_requested=spec.seconds,
                              output_format=OUTPUT_FORMAT)
                record['attempts'].append(attempt)
                save_manifest(manifest)
                print(f'[{index}/{len(queue)}] {spec.clip_id} ({spec.seconds}s)', flush=True)
                try:
                    song_id = generate(session, key, payload, partial, record, attempt, manifest)
                    details = inspect_audio(partial, spec)
                    publish_partial(partial, target)
                    attempt.update(status='completed', finished_utc=now())
                    record.update(status='completed', generated_song_id=song_id, **details)
                    save_manifest(manifest)
                    print(f'  Saved {target.name} ({details["duration_seconds_actual"]:.2f}s)', flush=True)
                except Exception as exc:
                    error = str(exc).replace(key, '[REDACTED]')
                    status = 'rejected' if isinstance(exc, RejectedRequest) else 'uncertain'
                    attempt.update(status=status, error=error, finished_utc=now())
                    record.update(status=status, error=error)
                    save_manifest(manifest)
                    guidance = ('Fix the reported issue (or add credits), then rerun normally.'
                                if status == 'rejected' else
                                'Paid audio and partial files are preserved. Rerun to recover valid audio; '
                                'otherwise inspect the manifest before explicitly retrying.')
                    raise RuntimeError(f'{error}\n{guidance}') from None
                if index < len(queue):
                    time.sleep(args.pause)
        print('Done. Rerun normally to finish any remaining recipes.')
        print(f'Manifest: {MANIFEST_PATH}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--only', action='append', choices=[s.clip_id for s in build_asset_specs()],
                        metavar='CLIP_ID', help='Generate only this recipe; repeat to select several')
    parser.add_argument('--limit', type=int, help='Maximum new requests this run (default: all remaining)')
    parser.add_argument('--dry-run', action='store_true', help='Print recipes without writing files or calling the API')
    parser.add_argument('--pause', type=float, default=1.25, help='Seconds between successful requests')
    parser.add_argument('--retry-uncertain', action='append', default=[], metavar='CLIP_ID')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be at least 1')
    if not math.isfinite(args.pause) or args.pause < 0:
        parser.error('--pause must be a finite nonnegative number')
    run(args)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted. Rerun to recover completed audio; uncertain requests are not automatically retried.', file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
