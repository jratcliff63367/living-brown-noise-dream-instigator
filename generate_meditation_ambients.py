#!/usr/bin/env python3
"""Generate 20 Reiki/spa-style instrumental meditation pieces, 2-5 minutes each.

Place beside eleven-labs.txt in the project root. Same key and dependencies as
our previous generator: python -m pip install requests mutagen
Run all remaining: python generate_meditation_ambients.py
Audition first three: python generate_meditation_ambients.py --limit 3
Offline preview: python generate_meditation_ambients.py --dry-run
Select one: python generate_meditation_ambients.py --only reiki-meditation-01-piano-warm-strings

All recipes stay inside soothing Reiki/spa meditation music. Variety comes from
instrument combinations, melodic shapes, harmony, texture and arrangement.
They ask for flowing, pleasant music with gentle melodic and harmonic movement,
not static drones, concert solos, lounge music or a tour of musical genres.
All 20 use force_instrumental, with no human voices or lyrics. No references
are required. Each recording is one complete Music API generation.
Artistic quality still needs listening and curation; it is not guaranteed.

Output: ceremonies/indian/ambients/activity/reiki-meditation-*.mp3
A new batch ID, filenames, manifest and lock isolate this batch from both earlier
attempts. Earlier files and ratings remain untouched. Stash unwanted earlier
recordings in the curator to prevent playback. Refresh/restart apps to find new
recordings, which enter the usual Unreviewed state.

Reruns complete this fixed set of 20; they do not generate another set.
--limit counts NEW requests after skipping completed/stashed recordings.
--only CLIP_ID can be repeated. Durations and IDs are fixed for reliable resume.
Sequential generation, one-hour read timeout, no automatic paid retries.
Song/request IDs and successful downloads are checkpointed. Finished partial
files can be recovered; incomplete downloads remain preserved for review.

After an explicit HTTP rejection, fix the issue and rerun. For uncertain requests,
review the manifest and ElevenLabs history; --retry-uncertain CLIP_ID authorizes
another paid attempt and preserves partial audio under a timestamped filename.
No audio or ratings are deleted. The OS releases the lock after a crash.
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
MANIFEST_PATH = OUTPUT_ROOT / 'reiki-meditation-generation-manifest.json'
API_URL = 'https://api.elevenlabs.io/v1/music'
MODEL_ID = 'music_v2_5'
OUTPUT_FORMAT = 'mp3_48000_192'
BATCH_ID = 'reiki-meditation-20-v1'
READ_TIMEOUT_SECONDS = 3600

COMMON = (
    'Beautiful instrumental Reiki and spa meditation music, suitable for a peaceful '
    'massage room and immersive relaxation. This is the musical identity throughout. '
    'Create a soothing, flowing piece with gentle melodic phrases, welcoming consonant '
    'harmonies and delicate changes of arrangement. Let each phrase lead naturally into '
    'the next, with enough melodic and harmonic movement to be pleasant on its own. '
    'Keep all instruments softly blended, rounded and warm, with lush smooth reverb. '
    'The music should feel tender, spacious, restful and reassuring. Soft accompaniment '
    'can provide an easy breathing flow, without an insistent beat. '
    'Let the ending settle naturally and preserve continuity throughout the middle. '
    'Keep this firmly in meditation music: no jazz, lounge, bossa nova, pop, dance '
    'groove, showy soloing, cinematic drama or grand emotional climax. '
    'Avoid static drones, bare sustained tones, long empty pauses, piercing chimes '
    'and harsh attacks. Purely instrumental, no human voices or lyrics. '
    'Music only, without nature recordings or added noise; the playback system '
    'provides its own brown-noise bed and spatial movement. '
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
    # One shared meditation identity; variety through timbre and arrangement.
    return [
        AssetSpec('reiki-meditation-01-piano-warm-strings', 120,
            'Soft acoustic piano and a small cushion of warm strings. Simple descending piano phrases over gently changing consonant chords, with a tender answering string line. An intimate, flowing meditation arrangement; the piano remains blended rather than playing a featured concert solo.'),
        AssetSpec('reiki-meditation-02-bamboo-flute-harp', 150,
            'Low bamboo flute and softly plucked harp, supported by a faint warm synth harmony. A comforting pentatonic melody flows in relaxed breaths while the harp offers delicate answering notes. Gently change the chord voicings and maintain connected resonance through the flute rests.'),
        AssetSpec('reiki-meditation-03-dreamy-synth-melody', 180,
            'Warm rounded synthesizer chords and a delicate bell-soft keyboard melody in the middle register. A few graceful melodic notes develop through slowly changing harmonies, with subtle answering tones. Classic dreamy spa meditation music, with clear musical movement and no sequencer rhythm or pad-only drone.'),
        AssetSpec('reiki-meditation-04-nylon-guitar-piano', 210,
            'Soft nylon-string guitar and mellow piano. Gentle fingerpicked chord figures support a simple reassuring piano phrase, then the guitar quietly answers. Smooth consonant harmonic changes and spacious reverb create a continuous comforting meditation flow; no strumming groove or guitar virtuosity.'),
        AssetSpec('reiki-meditation-05-harp-cello', 240,
            'A softly plucked harp carries a delicate rising-and-falling melody over warm cello harmony. The cello occasionally answers with a short tender phrase while the harp maintains gentle flowing notes. Small changes in harmony and register sustain interest without orchestral swells or a dramatic story arc.'),
        AssetSpec('reiki-meditation-06-bansuri-santoor', 180,
            'Soft low bansuri and mellow santoor in a polished Reiki meditation arrangement. An easy pentatonic flute phrase is answered by rounded, widely spaced santoor notes over changing soft keyboard harmony. Emphasize reassuring melody and gentle continuity, with no tabla or drone introduction.'),
        AssetSpec('reiki-meditation-07-soft-handpan-strings', 150,
            'Mellow handpan, warm string accompaniment and a few quiet piano notes. Rounded handpan tones form a slow, clear melodic phrase, supported by gently changing chords. All parts share the same relaxed phrasing, creating a soft meditation flow without a percussion groove or repeating busy pattern.'),
        AssetSpec('reiki-meditation-08-koto-warm-keyboards', 180,
            'Soft koto and warm sustained keyboard chords with a gentle flute-like instrumental countermelody. Plucked pentatonic phrases develop patiently into related phrases as the harmony changes underneath. Keep the instruments softly blended into lush spa music, with no sharp string attacks or dramatic glissandi.'),
        AssetSpec('reiki-meditation-09-piano-harp-ripples', 240,
            'Mellow piano and harp with a quiet warm string layer. Gentle short harp figures ripple beneath a simple piano melody, then pause to let the piano harmony carry the flow. Introduce a related answering phrase halfway through and return naturally, with even soft dynamics throughout.'),
        AssetSpec('reiki-meditation-10-cello-piano-sanctuary', 210,
            'Warm cello sings a restrained, comforting instrumental melody over soft piano chords and delicate broken-chord accompaniment. Keep the register comfortable and the mood reassuring. Let the piano briefly take the phrase before the cello returns, always in a gentle spa meditation balance rather than concert chamber music.'),
        AssetSpec('reiki-meditation-11-wooden-flute-guitar', 180,
            'A low wooden flute above soft nylon-guitar arpeggios and a faint warm harmonic layer. Lyrical but understated phrases, gentle consonant chord changes and easy breathing spaces filled by guitar resonance. A peaceful meditation room atmosphere, without folk-dance rhythms or an assertive flute solo.'),
        AssetSpec('reiki-meditation-12-mellow-marimba-harp', 120,
            'Very softly played marimba and harp over warm keyboard harmony. Rounded wooden tones form small comforting melodic patterns that change as the chords progress. The harp adds delicate responses and resonance. Unhurried, softly blended spa music, with no mallet groove, rapid rolls or busy repetition.'),
        AssetSpec('reiki-meditation-13-soft-electric-piano', 180,
            'A mellow electric piano with warm acoustic strings and a delicate plucked synthesizer voice. Simple soothing melody, gently flowing straight phrasing and clear consonant harmonies. The plucked voice quietly answers between piano phrases. Maintain a dreamy meditation character with no jazz chords, swing, bass groove or drum kit.'),
        AssetSpec('reiki-meditation-14-harmonium-flute-harp', 210,
            'Mellow harmonium supplies softly changing chords beneath a gentle bamboo-flute melody and a few harp answers. The reeds support the musical phrases rather than holding an unchanging drone. Warm breathing continuity, modest melodic development and lush smooth reverb, with no reed buzz or chant.'),
        AssetSpec('reiki-meditation-15-singing-bowls-piano', 150,
            'Soft piano carries a reassuring melody over warm strings, with occasional low singing-bowl tones chosen to agree with the harmony. Bowls add rounded resonance at natural phrase boundaries; piano and strings provide the continuous musical substance. No repeated bell strikes, metallic whine or bowl-only drone.'),
        AssetSpec('reiki-meditation-16-gentle-plucked-strings', 180,
            'Soft harp-like plucked strings, nylon guitar and warm bowed-string accompaniment. A simple pentatonic melody passes between the plucked instruments while the chords change gently. Create fluid interconnected phrases and a subtle answering melody, all blended into an intimate meditation arrangement rather than rhythmic folk music.'),
        AssetSpec('reiki-meditation-17-floating-piano-flute', 240,
            'Soft piano, low wooden flute and delicate warm synthesizer chords. Begin with a clear soothing piano phrase; let the flute quietly answer and gradually share the melody. Slowly vary the harmony and accompaniment while preserving the same peaceful mood, with no silent breakdown or large build.'),
        AssetSpec('reiki-meditation-18-acoustic-guitar-cello', 210,
            'Warm nylon-string guitar and soft cello with a light harp accent. The guitar provides gently flowing broken chords; the cello offers a modest, hopeful melodic line and short answers. The arrangement breathes and evolves through related harmonies, always softly blended and restful with no tragic or theatrical expression.'),
        AssetSpec('reiki-meditation-19-luminous-harp-keyboards', 270,
            'A delicate harp melody above warm, smoothly voiced synthesizer harmonies and soft piano answers. Let a small melodic idea unfold in several related variations, gently changing the accompanying chord colors. Luminous, welcoming Reiki meditation music with rich but clear reverb; no high sparkling chime barrage or static pad passage.'),
        AssetSpec('reiki-meditation-20-flowing-meditation-ensemble', 300,
            'A cohesive spa meditation ensemble of soft piano, harp, low bamboo flute and warm strings. A simple comforting theme gradually passes between piano and flute while harp and strings provide flowing harmonic support. Over the piece, gently vary voicing and answering phrases; keep the ensemble intimate and even in energy, with no grand finale or new genre section.'),
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
    fd, name = tempfile.mkstemp(prefix='reiki-meditation-', suffix='.tmp', dir=OUTPUT_ROOT)
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
    path = OUTPUT_ROOT / 'reiki-meditation-generation.lock'
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
