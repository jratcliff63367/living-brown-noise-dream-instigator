#!/usr/bin/env python3
"""Generate 47 ambient clips using ElevenLabs MUSIC (music_v2_5).
Includes the original 17 plus 10 new events and 20 new activities.

Place this script beside eleven-labs.txt in the project root.
Install: python -m pip install requests mutagen
Run:     python generate_new_ambients.py
Audition: python generate_new_ambients.py --only event --limit 2
Preview:  python generate_new_ambients.py --dry-run

Events: 6-9 seconds; activities: 24-38 seconds. No space clips are generated.
New activities include 14 vocal textures and 6 instrumentals. Vocal textures
use humming, wordless chanting, or simple mantras, never lyrical verses.
Uses text prompts with force_instrumental for instruments, and single-chunk
composition plans for mantra and wordless vocal clips.
No reference uploads or additional API calls are required.

This is a fixed 47-clip batch. If the original 17 are complete, the next run
generates only the 30 additions. Reruns finish the batch, not start another.
--limit counts NEW requests this run, after skipping completed/stashed files.
New music-* filenames and new-music-generation-manifest.json preserve all old
samples, the original manifest, and curator ratings. Completed files moved to
stash are recognized and will not be regenerated. Nothing is deleted.

Audio is validated before publication. Interrupted .mp3.part files can be
recovered automatically when valid. Explicit HTTP rejections (including credit
rejections) can be retried on the next run. Ambiguous network/server failures
are NOT retried automatically: inspect the manifest and your ElevenLabs history.
To explicitly retry one uncertain request: --retry-uncertain CLIP_ID
This may charge again. Existing partial audio is preserved under a timestamped
name before retrying. The lock is released by the OS even after a crash.

API docs: https://elevenlabs.io/docs/api-reference/music/compose
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
MANIFEST_PATH = OUTPUT_ROOT / 'new-music-generation-manifest.json'
API_URL = 'https://api.elevenlabs.io/v1/music'
MODEL_ID = 'music_v2_5'
OUTPUT_FORMAT = 'mp3_48000_192'
# Preserve this identity so completed clips from the first batch remain completed.
BATCH_ID = 'new-musical-ambients-17-v1'

COMMON = (
    'Intimate meditative acoustic music miniature, restful and gently expressive. '
    'Warm rounded timbres, restrained dynamics, natural human timing. '
    'A small coherent musical gesture, not a dramatic song or cinematic cue. '
    'Soft onset, warm spacious reverberation with a clear natural decay. '
    'No build-up, climax, abrupt change, harsh attack, piercing treble, distortion, '
    'strong bass thump, busy rhythm, applause, crowd chatter, narration, or nature effects. '
    'Do not add brown noise, rain, hiss, or wind; this will be mixed into a separate noise bed. '
    'Keep the performance present and clean; spatial distance is applied during playback. '
)
NEGATIVE = [
    'melodic lead singing', 'song verses', 'pop chorus', 'English speech',
    'invented words', 'gibberish', 'spoken instructions', 'shouting', 'whisper ASMR',
    'sharp sibilance', 'mouth clicks', 'dramatic crescendo', 'driving beat',
    'metallic screech', 'piercing high notes', 'distortion', 'crowd noise',
    'brown noise', 'rain', 'wind', 'abrupt restart',
]


@dataclass(frozen=True)
class AssetSpec:
    clip_id: str
    category: str
    seconds: int
    direction: str
    mantra: str = ''
    wordless: bool = False

    @property
    def relative_file(self):
        return Path(self.category) / (self.clip_id + '.mp3')


def build_asset_specs():
    return [
        AssetSpec('music-event-01-warm-temple-bell', 'event', 8,
                  'One low warm bronze temple bell softly struck near the beginning. '
                  'Let its rich rounded resonance decay through the remaining time. '
                  'Solo bell, no accompaniment, no repeated strikes, no bright clang.'),
        AssetSpec('music-event-02-soft-bell-answer', 'event', 9,
                  'Two very gentle mid-low temple bell tones separated by a generous breath. '
                  'Consonant pitches, the second softer than the first, lingering warm decay. '
                  'Solo bells, no rhythmic pattern, no sharp metallic transients.'),
        AssetSpec('music-event-03-bronze-bell-bloom', 'event', 9,
                  'A single softly felted bronze bell stroke with a mellow singing-bowl-like bloom. '
                  'A stable low-mid tone and gentle overtones, long warm room resonance. '
                  'No rubbing squeal, repeated pulse, sharp shimmer, or accompaniment.'),
        AssetSpec('music-event-04-tabla-fingertip-gesture', 'event', 6,
                  'One understated solo tabla fingertip phrase, three or four feather-light touches. '
                  'Rounded intimate hand percussion, leisurely spacing, a soft final decay. '
                  'No drum roll, fast flourish, heavy bass stroke, backing track, or steady beat.'),
        AssetSpec('music-event-05-tabla-soft-response', 'event', 8,
                  'Two tiny answering gestures on solo tabla, a soft rounded treble touch answered '
                  'by a restrained mellow lower tone. Unhurried and sparse, quiet resonance between '
                  'touches. No energetic solo, insistent rhythm, slap, or bass boom.'),
        AssetSpec('music-event-06-om-shanti-murmur', 'event', 8,
                  'One calm human voice quietly intones Om shanti on a narrow near-monotone pitch. '
                  'Natural gentle devotional chant, soft voiced tone rather than whispering. '
                  'No melodic lead, ornament, emotional swell, or instruments.', 'Om shanti'),
        AssetSpec('music-event-07-shanti-small-group', 'event', 9,
                  'Two or three softly blended voices gently intone shanti together twice. '
                  'Near-monotone mantra murmuring, calm natural breath, subtle warm room resonance. '
                  'No crowd bustle, theatrical choir, whispering, lead singer, or accompaniment.',
                  'Shanti\nShanti'),
        AssetSpec('music-activity-01-harmonium-stillness', 'activity', 32,
                  'Solo warm harmonium, a gently breathing tonic and fifth with tiny organic changes '
                  'in reed color. Stable consonance, no insistent pumping, buzzy nasal tone, or tune. '
                  'Begin softly and leave a gentle resonant ending.'),
        AssetSpec('music-activity-02-handpan-open-space', 'activity', 30,
                  'Solo mellow handpan in a consonant pentatonic mode. A handful of widely spaced '
                  'rounded notes and simple two-note responses, with long natural resonance. '
                  'Free timing, no ostinato, fast pattern, sharp slap, or rhythm section.'),
        AssetSpec('music-activity-03-bansuri-over-drone', 'activity', 34,
                  'Low-register soft bansuri over a barely present warm harmonium drone. '
                  'Two or three unhurried flute phrases, limited pitch range, relaxed breath. '
                  'No piercing high register, virtuosic runs, excessive breath hiss, or percussion.'),
        AssetSpec('music-activity-04-tanpura-resonance', 'activity', 28,
                  'A warm restrained tanpura tonic-and-fifth texture. Sparse gentle plucks with '
                  'overlapping resonances, stable tuning and a soft rounded attack. '
                  'No sharp buzzing, dissonance, new melody, rhythm section, or large changes.'),
        AssetSpec('music-activity-05-low-bowl-stillness', 'activity', 26,
                  'Two or three widely spaced low singing-bowl strokes, mellow stable consonant tones. '
                  'Let each resonance breathe before the next arrives, soft warm room tail. '
                  'No rim scraping, squealing, high metallic whine, beating dissonance, or busy chimes.'),
        AssetSpec('music-activity-06-tabla-and-harmonium', 'activity', 30,
                  'A warm sustained harmonium under very sparse gentle tabla fingertip touches. '
                  'The drone dominates; percussion is a quiet occasional detail with relaxed timing. '
                  'No drum solo, claps, insistent groove, pounding bass, or energetic development.'),
        AssetSpec('music-activity-07-handpan-warm-drone', 'activity', 36,
                  'Sparse low-mid handpan notes over a very quiet harmonium tonic drone in the same key. '
                  'Simple consonant intervals, relaxed uneven spacing and long ringing decays. '
                  'The handpan remains soft; no repeated ostinato or dramatic melodic arc.'),
        AssetSpec('music-activity-08-prayer-room-mantra', 'activity', 38,
                  'Warm harmonium drone with two softly blended human voices repeating Om shanti. '
                  'Voices are a subordinate near-monotone devotional texture, unhurried with natural '
                  'breathing spaces. No sung melody, soloist, vocal acrobatics, or percussion.',
                  'Om shanti\nOm shanti\nOm shanti\nOm shanti'),
        AssetSpec('music-activity-09-evening-mantra-and-bell', 'activity', 34,
                  'Quiet harmonium supports a small softly blended group intoning shanti, with only '
                  'one or two very mellow low bell accents. Near-monotone prayer texture and steady '
                  'restrained energy. No celebratory aarti rhythm, clapping, lead singing, or chatter.',
                  'Shanti\nShanti\nShanti\nShanti'),
        AssetSpec('music-activity-10-harmonium-flute-answer', 'activity', 24,
                  'A soft low harmonium chord gently opens into one brief low bansuri response, then '
                  'settles into lingering warm resonance. Minimal consonant chamber music, intimate '
                  'and restrained throughout. No percussion, bright flute register, or finale.'),
        AssetSpec('music-event-08-low-bell-soft-mallet', 'event', 8, 'One soft padded-mallet strike on a low bronze temple bell. Rounded onset, velvety resonant body, long natural decay. No bright edge or accompaniment.', '', False),
        AssetSpec('music-event-09-bell-fifth-response', 'event', 9, 'Two quiet bronze bell notes a consonant fifth apart, softly answering each other. Generous space and a lingering warm tail, no chiming pattern.', '', False),
        AssetSpec('music-event-10-tabla-three-soft-touches', 'event', 6, 'Three feather-light solo tabla finger touches, loose unhurried spacing, rounded resonant tone. No slap, roll, driving pulse, or heavy bass.', '', False),
        AssetSpec('music-event-11-tabla-brushed-resonance', 'event', 7, 'A tiny solo tabla gesture with soft finger pads and one restrained mellow bass response. Quiet hand contact, gentle decay; no sharp transient or virtuosic flourish.', '', False),
        AssetSpec('music-event-12-handpan-two-note-sigh', 'event', 9, 'Two softly played low-mid handpan notes, consonant and widely spaced, with overlapping warm resonance. A complete miniature gesture without a repeating rhythm.', '', False),
        AssetSpec('music-event-13-bowl-single-warm-bloom', 'event', 9, 'One soft low singing-bowl stroke, stable consonant overtones, a rounded warm bloom fading naturally. No scraping, rim rubbing, bright chimes, or beating dissonance.', '', False),
        AssetSpec('music-event-14-female-hum-breath', 'event', 8, 'One soft female closed-mouth hum on a comfortable low-mid pitch, a natural breath then gentle release. Entirely wordless, no open-vowel solo melody or whispering.', '', True),
        AssetSpec('music-event-15-two-voice-hummed-fifth', 'event', 9, 'Two quiet voices hum a stable consonant fifth with closed mouths. One sustained shared breath and a gentle release, no words, melody, or theatrical choir.', '', True),
        AssetSpec('music-event-16-om-warm-resonance', 'event', 8, 'One quietly voiced Om in a comfortable middle register, steady near-monotone intonation with warm room resonance. No booming bass, overtone growl, or dramatic swell.', 'Om', False),
        AssetSpec('music-event-17-shanti-soft-answer', 'event', 9, 'Two softly blended voices intone shanti as a tiny calm response, narrow near-monotone pitch and natural breathing. No lead singer, whisper, or percussion.', 'Shanti\nShanti', False),
        AssetSpec('music-activity-11-female-hum-harmonium', 'activity', 34, 'A warm harmonium drone under soft female closed-mouth humming. Short sustained tones separated by breaths, stable tonic and fifth, voice blended into the reeds. No words or melodic lead.', '', True),
        AssetSpec('music-activity-12-low-hum-tanpura', 'activity', 32, 'A gentle low-mid male closed-mouth hum with mellow tanpura resonance. Relaxed comfortable register, no throat growl or exaggerated bass. Stable tonal center, no words or solo melody.', '', True),
        AssetSpec('music-activity-13-hummed-duet-open-space', 'activity', 36, 'Two softly blended female voices hum a consonant unison opening briefly to a fifth. Natural staggered breathing and spacious warm reverberation, entirely wordless and unaccompanied. No swelling choir.', '', True),
        AssetSpec('music-activity-14-handpan-and-hummed-tonic', 'activity', 30, 'Widely spaced mellow handpan notes with a barely present closed-mouth hum holding the tonic. The voice is a soft sustained texture, with natural breathing, no lyrics or melodic phrasing.', '', True),
        AssetSpec('music-activity-15-bansuri-and-hum', 'activity', 34, 'A low bansuri plays two unhurried phrases over quiet closed-mouth humming on a stable pitch. Gentle consonance and warm air, no high flute register, word formation, or vocal lead.', '', True),
        AssetSpec('music-activity-16-shruti-wordless-chant', 'activity', 38, 'Warm shruti-box tonic and fifth with a restrained near-monotone open-vowel chant. Only sustained ah and oo vowel sounds, natural soft human tone, no syllabic language, invented words, or lyrical melody.', '', True),
        AssetSpec('music-activity-17-om-harmonium-breaths', 'activity', 36, 'Warm harmonium beneath repeated quiet Om intonations by one soft female voice. Each near-monotone chant has a natural breathing space. No lyrical verse, vocal ornament, or build-up.', 'Om\nOm\nOm\nOm', False),
        AssetSpec('music-activity-18-shanti-small-room', 'activity', 32, 'Three quietly blended voices intone shanti over a faint harmonium. Intimate group prayer texture in a small resonant room, narrow pitch range, no lead singing or crowd chatter.', 'Shanti\nShanti\nShanti\nShanti', False),
        AssetSpec('music-activity-19-hummed-voices-bowl', 'activity', 30, 'A soft closed-mouth female hum and only two low mellow singing-bowl strokes in the same tonal center. Long breath and resonance, no words, metallic whine, or dramatic entrances.', '', True),
        AssetSpec('music-activity-20-tabla-and-hummed-drone', 'activity', 34, 'A quiet sustained hummed tonic beneath sparse rounded tabla fingertip touches. Warm human voice with natural breathing; percussion stays delicate, no lyrics, groove, or heavy bass strokes.', '', True),
        AssetSpec('music-activity-21-wordless-chapel-breath', 'activity', 38, 'Two soft female voices sustain near-monotone oo vowel tones with staggered breaths in a warm reverberant stone room. Consonant intervals, no words, invented syllables, melody, or cinematic crescendo.', '', True),
        AssetSpec('music-activity-22-om-shanti-tanpura', 'activity', 36, 'Mellow tanpura and two softly blended voices intoning Om shanti. Relaxed near-monotone prayer, gentle natural pauses, no sung verses, dramatic expression, or percussion.', 'Om shanti\nOm shanti\nOm shanti\nOm shanti', False),
        AssetSpec('music-activity-23-mixed-hummed-unison', 'activity', 32, 'One soft male and one soft female voice hum together in comfortable registers, gentle octave blending. A faint harmonium supports them. Closed-mouth wordless warmth, no booming bass, lead tune, or swelling choir.', '', True),
        AssetSpec('music-activity-24-hummed-response-harmonium', 'activity', 34, 'A warm harmonium drone with two understated closed-mouth hums softly answering in the same register. Spacious breaths, nearly static pitch, natural human tone. No words, scat, catchy motif, or lead singing.', '', True),
        AssetSpec('music-activity-25-harmonium-slow-fifths', 'activity', 30, 'Solo warm harmonium, tonic and fifth gently changing reed color across quiet sustained breaths. One subtle consonant voicing change; no insistent pumping, nasal buzz, rhythmic pattern, or melody.', '', False),
        AssetSpec('music-activity-26-handpan-unhurried-drops', 'activity', 32, 'Solo low-mid handpan, a few rounded pentatonic notes with generous irregular spacing. Natural decay and quiet touch, no ostinato, busy pattern, slap, or dramatic arc.', '', False),
        AssetSpec('music-activity-27-bansuri-held-tone', 'activity', 28, 'Low bansuri over a barely audible shruti drone, one long gentle tone and a small consonant response. Relaxed breath, limited range, no piercing register, runs, or excessive hiss.', '', False),
        AssetSpec('music-activity-28-bowl-harmonium-rest', 'activity', 36, 'Two low softly struck singing-bowl tones over a warm quiet harmonium tonic. Stable consonance with gentle lingering resonance; no rim scrape, repeated chimes, dissonant beating, or crescendo.', '', False),
        AssetSpec('music-activity-29-tanpura-reed-warmth', 'activity', 34, 'Mellow tanpura resonance woven into a soft harmonium tonic and fifth. Slow organic changes in timbre, tiny rounded plucks, no new melody, buzzy treble, or strong pulse.', '', False),
        AssetSpec('music-activity-30-handpan-tabla-whisperlight', 'activity', 30, 'Sparse mellow handpan notes with just a few feather-light tabla fingertip responses. Both share relaxed free timing and a warm acoustic space. No busy percussion, sharp accents, heavy bass, or groove.', '', False),
    ]


def payload_for(spec):
    base = {'model_id': MODEL_ID, 'store_for_inpainting': False}
    if not spec.mantra and not spec.wordless:
        return {**base, 'prompt': COMMON + spec.direction +
                ' Strictly instrumental: no voice, humming, singing, or chanting.',
                'music_length_ms': spec.seconds * 1000, 'force_instrumental': True}
    return {**base, 'composition_plan': {'chunks': [{
        'text': '[Wordless vocal texture]' if spec.wordless else spec.mantra,
        'duration_ms': spec.seconds * 1000,
        'positive_styles': [COMMON, spec.direction,
                            'quiet wordless vocal texture' if spec.wordless else 'quiet devotional mantra',
                            'natural human voices', 'narrow near-monotone pitch range',
                            ('No lyrics or language; only the humming or sustained vowel tones described'
                             if spec.wordless else
                             'Only intone the supplied Sanskrit mantra text; no invented syllables'),
                            'restrained acoustic performance', 'warm natural reverberation'],
        'negative_styles': NEGATIVE,
        'context_adherence': 'high',
    }]}}


def now():
    return datetime.now(timezone.utc).isoformat()


def save_manifest(data):
    fd, name = tempfile.mkstemp(prefix='new-music-', suffix='.tmp', dir=OUTPUT_ROOT)
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
    path = OUTPUT_ROOT / 'new-music-generation.lock'
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
        details = inspect_audio(existing[0], spec)
        if record.get('status') != 'completed':
            record.update(status='completed', recovered_utc=now(), **details)
        return True
    if partial.exists():
        try:
            details = inspect_audio(partial, spec)
        except Exception as exc:
            print(f'  Partial audio needs review: {partial.name}: {exc}')
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


def generate(session, key, payload):
    with session.post(API_URL, headers={'xi-api-key': key, 'Accept': 'audio/mpeg'},
                      params={'output_format': OUTPUT_FORMAT}, json=payload,
                      timeout=(30, 1200), allow_redirects=False) as response:
        if response.status_code != 200:
            message = f'HTTP {response.status_code}: {response.text[:3000].replace(key, "[REDACTED]")}'
            # No automatic retry in this run. A definite rejection is safe to retry next run.
            cls = RejectedRequest if response.status_code in (400, 401, 402, 403, 404, 413, 415, 422, 429) else RuntimeError
            raise cls(message)
        if not response.content:
            raise RuntimeError('API returned empty audio.')
        return response.content, response.headers.get('song-id')


def run(args):
    specs = build_asset_specs()
    chosen = [s for s in specs if args.only is None or s.category == args.only]
    if args.dry_run:
        # A truly offline preview: no key, dependencies, lock, manifest, or paid requests.
        for spec in chosen:
            print(f'{spec.clip_id}: {spec.seconds}s, {"wordless vocal" if spec.wordless else "mantra" if spec.mantra else "instrumental"}')
            print(json.dumps(payload_for(spec), indent=2, ensure_ascii=False))
        print(f'{len(chosen)} recipes shown; existing completion state is not evaluated in preview.')
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
            raise ValueError('--retry-uncertain must name a clip in the selected category.')
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
        print(f'{len(queue)} new music requests; output: {OUTPUT_ROOT}', flush=True)
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
                    audio, song_id = generate(session, key, payload)
                    attempt['song_id'] = song_id
                    with partial.open('xb') as handle:
                        handle.write(audio)
                        handle.flush()
                        os.fsync(handle.fileno())
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
    parser.add_argument('--only', choices=['event', 'activity'])
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
