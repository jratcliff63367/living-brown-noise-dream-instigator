#!/usr/bin/env python3
"""Generate 20 long instrumental ambients with ElevenLabs Music (music_v2_5).

Place this script beside eleven-labs.txt in the project root.
Install: python -m pip install requests mutagen
Run:     python generate_long_ambients.py
Audition: python generate_long_ambients.py --limit 1
Preview:  python generate_long_ambients.py --dry-run
Pick one: python generate_long_ambients.py --only long-ambient-04-quiet-pipe-organ

One original performance for each of the 20 ideas, with fixed durations of
5-10 minutes. Each is ONE music request: no loops, stitching, or references.
All requests use force_instrumental. Prompts favor continuous gentle evolution,
soft attacks, warm resonance, no vocals, no climaxes, and no embedded noise bed.
Actual musical quality and compliance with the artistic direction need curation.

Output: ceremonies/indian/ambients/activity/long-ambient-*.mp3
These join the ambient pool in the main app, not its ceremony recordings.
Restart/refresh the main app and curator to discover newly generated files.
New recordings have the curator's usual Unreviewed state until you rate them.

A fixed batch: reruns finish the remaining pieces and never start a second set.
Durations and IDs stay fixed so interrupted runs can resume reliably.
--limit counts NEW requests after skipping finished/stashed pieces.
--only CLIP_ID may be repeated to select particular pieces.
A separate long-ambient-generation-manifest.json and lock keep this batch
independent of the earlier short ambient generator and its outputs.

The API key is loaded from eleven-labs.txt exactly as in the supplied generator.
Only requests and mutagen are needed. The read timeout is one hour because
long music generations can take time; requests are sequential with no retries.
Song/request IDs are checkpointed as soon as response headers arrive, and audio
is written to a .mp3.part file as it downloads. Only validated complete audio is
published. Known-complete downloads can be recovered on the next run.

Explicit HTTP rejections, including insufficient credits, can be retried by
rerunning after fixing the issue. Ambiguous failures are not purchased again
automatically. Inspect the manifest and your ElevenLabs history, then use
--retry-uncertain CLIP_ID if another paid attempt is appropriate. Existing
partial audio is preserved with a timestamp. No audio or ratings are deleted.
The OS releases the process lock even if the generator crashes.

API docs (duration, prompt and instrumental options):
https://elevenlabs.io/docs/api-reference/music/compose
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
MANIFEST_PATH = OUTPUT_ROOT / 'long-ambient-generation-manifest.json'
API_URL = 'https://api.elevenlabs.io/v1/music'
MODEL_ID = 'music_v2_5'
OUTPUT_FORMAT = 'mp3_48000_192'
BATCH_ID = 'long-instrumental-ambients-20-v1'
READ_TIMEOUT_SECONDS = 3600

COMMON = (
    'An original, continuous long-form instrumental meditation. '
    'A beautiful inhabitable musical space, unhurried and quietly absorbing. '
    'Warm midrange detail, rounded attacks, stable consonant harmony, relaxed dynamics, '
    'and spacious warm reverberation with overlapping natural decays. '
    'Develop patiently through tiny changes in voicing, register and instrumental color; '
    'keep a coherent identity throughout. Sustain musical presence for the full duration. '
    'Quiet breaths between phrases are welcome, but keep resonance connecting them: '
    'no full-ensemble stop, false ending, silent breakdown, or restart in the middle. '
    'Allow only the final seconds to settle into a natural ending. '
    'No verse/chorus form, climaxes, dramatic crescendo, heavy beat, insistent ostinato, '
    'sudden entrances, piercing treble, harsh dissonance, or overpowering sub-bass. '
    'Strictly instrumental throughout: no female or male vocals, chanting, humming, '
    'choir, speech, or vocal samples. '
    'No added brown noise, rain, wind, surf, hiss, crowds, or other sound effects; '
    'the playback engine supplies its own living brown-noise bed. '
    'Render the instruments clearly with gentle detail and a stable spatial image. '
    'Spatial movement and gradual emergence are supplied during playback. '
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
    # Fixed, deliberately varied lengths; do not redraw durations on resume.
    return [
        AssetSpec('long-ambient-01-harmonium-wooden-room', 480,
            'Solo acoustic harmonium in a resonant wooden room. Mellow reeds and smooth '
            'bellows sustain open fifths and gentle added-note chords. Unforced breathing '
            'in the phrasing, without mechanical squeaks or rhythmic pumping. Over many '
            'minutes, exchange one inner note at a time and occasionally open the register, '
            'then return to intimate warmth. No nasal buzzing or lead melody.'),
        AssetSpec('long-ambient-02-bamboo-flute-drone', 360,
            'A low bamboo flute plays spacious, breath-soft pentatonic phrases above a '
            'quiet continuous shruti-like instrumental drone. Give the flute long rests '
            'while the drone and resonance remain. Vary phrase lengths and gently explore '
            'neighboring tones; never become virtuosic. Comfortable low-middle register, '
            'no piercing flute, excessive breath hiss, percussion, or escalating solo.'),
        AssetSpec('long-ambient-03-bowed-cello-meditation', 420,
            'A warm cello with a very soft second bowed-string layer. Long legato tones '
            'and occasional consonant double stops, rounded bow changes, human phrasing. '
            'A small melodic idea slowly changes shape while a held tone connects phrases. '
            'Restrained tenderness, no tragic film-score arc, abrasive bow scrape, '
            'dramatic vibrato, heavy bass rumble, or rhythmic accompaniment.'),
        AssetSpec('long-ambient-04-quiet-pipe-organ', 540,
            'A pipe organ using only soft flute-like stops in an immense reverberant hall. '
            'Quiet lower-middle chords and a few floating upper tones, sustained and '
            'seamlessly voiced. Slowly alter inner harmonies while preserving a peaceful '
            'tonal center. Convey architectural scale through resonance and spacing, '
            'never volume. No loud pedal bass, reeds, mixtures, hymn tune or grand finale.'),
        AssetSpec('long-ambient-05-underwater-harmonics', 600,
            'An underwater impression made entirely from rounded electronic tones, softly '
            'bowed metallic resonances, and slowly shifting consonant chords. Muted upper '
            'frequencies and liquid legato connections; tiny luminous details occasionally '
            'appear above the sustained harmony. Explore changes in depth and tonal color '
            'without increasing loudness. No water recordings, bubbles, sonar pings, '
            'horror atmosphere, sub-bass sweeps, or audible filter pumping.'),
        AssetSpec('long-ambient-06-gentle-string-harmonics', 420,
            'A small acoustic string ensemble weaving soft natural harmonics into warm '
            'middle-register sustained notes. Staggered bow changes keep the texture '
            'continuous. Let individual consonant partials become briefly distinct and '
            'blend back into the ensemble. Slow changes of voicing with no emotional '
            'build. Soft luminous warmth, never icy high strings or tense clusters.'),
        AssetSpec('long-ambient-07-felt-piano-space', 360,
            'An intimate felt piano improvisation of widely spaced soft chords and '
            'isolated middle-register notes. Rounded attacks, gently overlapping pedal '
            'resonance, warm room decay. Let simple two-note shapes return with modest '
            'harmonic variations and irregular pauses. Sustain the mood without a '
            'recognizable song, busy arpeggios, low pounding, mechanical key noise, '
            'or a sentimental crescendo.'),
        AssetSpec('long-ambient-08-bowed-vibraphone', 420,
            'Bowed vibraphone tones with nearly attackless onset and overlapping warm '
            'metallic sustain. Soft low-middle consonant intervals gradually change '
            'one pitch at a time. Occasionally expose a single rich overtone, then '
            'rejoin a gentle chord. An organic resonant performance with no motor '
            'tremolo, mallet beat, bell strikes, shrill ringing, or dissonant beating.'),
        AssetSpec('long-ambient-09-soft-horn-ensemble', 480,
            'A small ensemble of very softly played French horns and mellow muted '
            'flugelhorns. Warm blended lower-middle chords, staggered breaths, rounded '
            'entrances. Individual inner voices slowly shift through consonant harmony '
            'without becoming soloists. Maintain intimate pianissimo and warm room '
            'resonance. No fanfare, martial rhythm, cinematic swell, brassy edge, '
            'jazz lead, or human choir.'),
        AssetSpec('long-ambient-10-aeolian-harp', 540,
            'Aeolian-harp-inspired music: resonant open strings, softly excited '
            'harmonics, and occasional feather-light plucks with overlapping decays. '
            'Irregular clusters feel naturally stirred without a countable pulse. '
            'Keep one welcoming tonal family while string colors and spacing evolve. '
            'Only musical strings and their resonance, no actual wind recording, '
            'harp glissando flourishes, bright chime cascade, or rhythmic picking.'),
        AssetSpec('long-ambient-11-reverberant-guitar-swells', 480,
            'Clean electric guitar played with gentle volume swells that hide the pick '
            'attack. Warm consonant chord fragments overlap through long smooth '
            'reverberation. Occasional middle-register harmonics answer a sustained '
            'lower tone. Patiently alter chord inversions and sustain lengths. '
            'No drums, rock rhythm, distortion, soaring solo, delay ostinato, '
            'shrill feedback, or dramatic stereo movement.'),
        AssetSpec('long-ambient-12-tanpura-resonance', 600,
            'An intimate mellow tanpura study with richly ringing sympathetic strings. '
            'Gentle rounded plucks sustain a stable tonic and fifth with naturally '
            'unequal spacing. Over the full performance, subtly vary touch and overtone '
            'emphasis so the resonance stays alive. Spacious unforced continuity, '
            'no lead singer, sitar solo, tabla, bright buzzing treble, strict pulse '
            'or large harmonic departure.'),
        AssetSpec('long-ambient-13-bass-clarinet-strings', 360,
            'Soft bass clarinet in its comfortable lower-middle register above a '
            'quiet bed of bowed viola and cello. Rounded breath-rich phrases with '
            'generous rests, small intervals, and gentle consonant answers from '
            'the strings. A patient intimate conversation without increasing density. '
            'No growls, key clatter, jazzy runs, jaunty rhythm, orchestral climax '
            'or ominous low notes.'),
        AssetSpec('long-ambient-14-glass-harmonica', 300,
            'A glass harmonica with smooth singing instrumental tones in the warm '
            'middle register. Soft sustained consonant intervals overlap into slow '
            'chord changes, with occasional simple answering notes. Gentle human '
            'variation and luminous resonance. Keep the full performance comforting '
            'and rounded; no squeal, piercing upper register, horror dissonance, '
            'tinkling rhythm, or vocal imitation.'),
        AssetSpec('long-ambient-15-soft-marimba', 300,
            'A wooden marimba played with very soft mallets in the low-middle register. '
            'Rounded resonant notes and tiny answering phrases with relaxed unequal '
            'spacing; notes occasionally overlap in gentle consonant intervals. '
            'Evolve a small pentatonic idea by changing one note or pause at a time. '
            'No rigid ostinato, rapid rolls, dance groove, sharp attacks, bass thumps '
            'or increasing rhythmic complexity.'),
        AssetSpec('long-ambient-16-warm-analog-landscape', 600,
            'Warm analog synthesizer pads with rounded oscillator tones, slow '
            'consonant voice-leading, and subtle timbral drift. A few delicate melodic '
            'fragments emerge within the harmony and dissolve back into it. '
            'Change color and register patiently while retaining a calm tonal center '
            'and steady energy. No sequencer, arpeggiator, drums, white-noise layer, '
            'sci-fi alarms, resonant sweeps, sub drops or cinematic build.'),
        AssetSpec('long-ambient-17-open-tuning-acoustic-guitar', 360,
            'A warm acoustic guitar in a consonant open tuning. Sparse gentle '
            'fingerpicked chord fragments, ringing open strings and occasional '
            'natural harmonics. Loosely timed phrases with resonance bridging '
            'the spaces; small changes of voicing over several minutes. '
            'Intimate and unhurried, without repetitive strumming, flashy runs, '
            'string squeaks, tapping, percussion or an insistent folk-song rhythm.'),
        AssetSpec('long-ambient-18-orchestra-through-mist', 540,
            'A restrained chamber orchestra of soft strings, low flutes and mellow '
            'clarinets. Blend them into slow consonant harmonies with rounded '
            'entrances. Individual instruments briefly become distinguishable, '
            'offer a tiny phrase, and blend back into sustained harmony. Spacious '
            'warm reverberation, clear musical texture with softened edges. '
            'No actual fog or wind sound, percussion, brass climax, tension cue '
            'or dramatic orchestral development.'),
        AssetSpec('long-ambient-19-gong-resonance', 480,
            'A restrained resonance meditation using gently excited low gongs '
            'and occasional mellow singing bowls chosen for compatible pitches. '
            'Soft mallet friction and feather-light contacts produce overlapping '
            'rounded blooms, not crashing strikes. Let warm overtones slowly '
            'change prominence while keeping energy even. No thunderous bass, '
            'metallic screech, ominous roar, sharp chimes, strong beating '
            'dissonance or escalating gong wash.'),
        AssetSpec('long-ambient-20-harmonium-bamboo-duet', 540,
            'An acoustic duet of mellow harmonium and low bamboo flute. The '
            'harmonium establishes continuous warm harmony; the flute offers '
            'widely spaced gentle phrases and sometimes rests for a long stretch '
            'while the reeds quietly change their voicing. The two instruments '
            'trade musical attention naturally across the full performance. '
            'Small consonant ideas, smooth continuity, soft breath and room '
            'resonance. No busy accompaniment, high flute runs or climactic duet.'),
    ]


def payload_for(spec):
    prompt = (
        COMMON + f'Create one uninterrupted {spec.seconds // 60}-minute performance. '
        + spec.direction
    )
    if not 300 <= spec.seconds <= 600:
        raise ValueError(f'{spec.clip_id}: length must be 5-10 minutes')
    if len(prompt) > 4100:
        raise ValueError(f'{spec.clip_id}: prompt exceeds the API character limit')
    return {'model_id': MODEL_ID, 'store_for_inpainting': False,
            'prompt': prompt, 'music_length_ms': spec.seconds * 1000,
            'force_instrumental': True}


def now():
    return datetime.now(timezone.utc).isoformat()


def save_manifest(data):
    fd, name = tempfile.mkstemp(prefix='long-ambient-', suffix='.tmp', dir=OUTPUT_ROOT)
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
    path = OUTPUT_ROOT / 'long-ambient-generation.lock'
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
            print(f'{spec.clip_id}: {spec.seconds // 60} minutes, instrumental')
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
