#!/usr/bin/env python3
"""Generate 50 Latin female vocal tracks: the original 40 plus ten female-reverb tracks.
The female-reverb additions use Byzantine-inspired singing and warm spacious reverberation.
Requested duration: 120-300 seconds. Existing vocal files and manifests are preserved.
Place beside eleven-labs.txt in the project root.
Install: python -m pip install requests mutagen
Run: python generate_indian_vocals.py
Optional: --count 1 for audition, --dry-run to inspect plans without API calls.
COUNT is the target total, capped at 50. Reruns finish the batch rather than adding 50 more.
Reuses the original Latin manifest: 40 completed tracks leave only ten additions.
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

NUM_CLIPS_TO_GENERATE = 50
MIN_DURATION_SECONDS = 120
MAX_DURATION_SECONDS = 300
MODEL_ID = 'music_v2_5'
OUTPUT_FORMAT = 'mp3_48000_192'
REFERENCE_SLICE_SECONDS = 18
ROOT = Path(__file__).resolve().parent
REFERENCE_DIR = ROOT / 'ceremonies/indian/vocal-reference'
OUTPUT_DIR = ROOT / 'ceremonies/indian/vocals'
API_URL = 'https://api.elevenlabs.io/v1/music'

# Musical style names describe arrangements, not an imitation of a particular voice.
# The existing female references supply vocal character. All lyrics are Latin.
STYLE_PROFILES = {
    'indian': [
        'one natural solo female voice, Indian devotional and classical inspired phrasing',
        'gentle raga-like melodic exploration, meend-like glides and graceful restrained melisma',
        'free unhurried rhythm, warm intimate delivery, no supporting singers',
    ],
    'celtic-new-age': [
        'ethereal Celtic and New Age female vocal arrangement',
        'floating modal melody, soft layered female harmonies and long overlapping legato phrases',
        'spacious reverberation with clear human vocal detail, luminous restrained atmosphere',
    ],
    'norwegian': [
        'Norwegian folk-inspired female singing, intimate reflective Nordic lullaby character',
        'clear natural solo tone, modal melodic contours, subtle folk ornaments and flexible unmetered phrasing',
        'restrained vibrato, open acoustic space, quiet warmth rather than theatrical Nordic epic music',
        'sing in Latin, not Norwegian; the Norwegian influence is musical only; no loud herding calls',
    ],
    'medieval-quartet': [
        'four natural female voices in an intimate unaccompanied medieval early-music ensemble',
        'Latin plainchant alternating with delicate two-to-four-part medieval-inspired polyphony',
        'pure blended tone, restrained vibrato, open fifths and octaves, independent gently moving vocal lines',
        'small reverberant stone chapel, clear diction, contemplative chamber scale rather than a large choir',
    ],
}
VARIATIONS = {
    'long-arcs': 'Long flowing melodic arcs, gradual development and gently sustained phrase endings.',
    'low-warmth': 'Favor comfortable lower female registers and rounded warmth, with only occasional higher answers.',
    'luminous': 'Favor a light clear middle-to-upper register, delicate entrances and soft unforced sustained notes.',
    'returning-melody': 'Revisit a simple lyrical melodic idea with small graceful changes; never a rigid loop.',
    'quiet-lament': 'Tender descending phrases and restrained wistful expression, peaceful rather than distressed.',
    'spacious': 'Sparse unhurried phrases with short natural breathing spaces and lingering connected resonance.',
    'gentle-ornaments': 'Small expressive turns and subtle ornaments around sustained notes, never rapid or showy.',
    'prayerful': 'Simple sincere prayer-like delivery with clear sung words and gentle melodic motion.',
    'rising-falling': 'Slow small rising and falling melodic contours with even dynamics and no dramatic peak.',
    'intimate': 'Private tender delivery, delicate breath-supported tone and understated emotional expression.',
}
# Ten original short Latin verses, shared across the styles for comparison.
# English meanings are metadata only and are NEVER submitted as lyrics.
LATIN_VERSES = [
    ('Nox tranquilla nos circumdat.\nLuna clara super nos lucet.\nCor in pace requiescit.\nSomnus lenis ad nos venit.', 'Quiet night surrounds us; the bright moon shines above us; the heart rests in peace; gentle sleep comes to us.'),
    ('Stellae lucent in caelo.\nVentus lenis inter arbores spirat.\nTerra tacet sub luna.\nAnima mea requiescit.', 'Stars shine in the sky; a gentle wind breathes among the trees; the earth is silent beneath the moon; my soul rests.'),
    ('Mare placidum lente movetur.\nUnda mollis litus tangit.\nLuna super aquas lucet.\nPax profunda in corde manet.', 'The calm sea moves slowly; a gentle wave touches the shore; the moon shines over the waters; deep peace remains in the heart.'),
    ('Lux mitis per noctem fulget.\nSpes quieta in nobis manet.\nOmnis cura paulatim abit.\nCor apertum pacem invenit.', 'A gentle light shines through the night; quiet hope remains in us; every care gradually departs; an open heart finds peace.'),
    ('Sub arboribus umbra iacet.\nFolia leniter moventur.\nFons inter lapides murmurat.\nHic in pace requiescimus.', 'Shade lies beneath the trees; leaves move gently; a spring murmurs among the stones; here we rest in peace.'),
    ('Aurora longe adhuc latet.\nNox amica nobiscum manet.\nOculi fessi iam clauduntur.\nDulcis somnus nos amplectitur.', 'Dawn is still hidden far away; friendly night stays with us; tired eyes now close; sweet sleep embraces us.'),
    ('Pax in terra, pax in corde.\nLux in nocte, spes in vita.\nAmor mitis nos custodit.\nAnima quieta requiescit.', 'Peace on earth, peace in the heart; light in the night, hope in life; gentle love watches over us; a quiet soul rests.'),
    ('Flumen lente ad mare fluit.\nTempus sine voce transit.\nSidera vias nostras servant.\nNos sub caelo requiescimus.', 'The river flows slowly to the sea; time passes without a voice; the stars watch over our paths; we rest beneath the sky.'),
    ('Vox quieta per noctem sonat.\nCantus lenis corda mulcet.\nSpes et amor nobiscum manent.\nPax nos omnes circumdat.', 'A quiet voice sounds through the night; a gentle song soothes hearts; hope and love remain with us; peace surrounds us all.'),
    ('Luna candida, stella clara.\nAura mitis, terra cara.\nCurae longe iam recedunt.\nCorda nostra requiescunt.', 'Bright moon, clear star; gentle breeze, dear earth; cares now retreat far away; our hearts rest.'),
]
# Added to the existing batch without renaming any original recipes or state files.
STYLE_PROFILES['female-reverb'] = [
    'Byzantine-inspired female modal chant, a creative setting of the supplied Latin text rather than a historical reconstruction',
    'Slow unfolding modal phrases, sustained low female vocal drones and gentle ornamental turns, free unhurried rhythm',
    'Lean into long warm cathedral reverberation as part of the musical texture, softened diffuse reflections and rich lingering tails',
    'Let each sung phrase leave a luminous reverberant trail that overlaps the next phrase while the foreground voice remains recognizably human',
    'Spacious enveloping stone acoustics, smooth dark reverberant decay without brittle brightness, distinct rhythmic echoes or booming bass',
    'Keep the same acoustic space and ensemble throughout the recording; no sudden changes in reverb or density',
]
REVERB_ENSEMBLES = [
    'One intimate solo female voice in a vast softly reverberant stone chamber; only her reverberation answers her',
    'A low female soloist with a very soft sustained female vocal drone beneath her, warm spacious resonance',
    'A clear female soloist over two restrained sustained female harmony layers, spacious but transparent',
    'A small three-voice female ensemble with overlapping modal phrases and a shared lingering acoustic tail',
    'One tender solo female voice with distant quiet female responses, their phrase tails gently interweaving',
    'One sparse solo female voice in a large warm chapel; preserve the long natural decay between phrases',
    'Two gently layered female voices with restrained ornaments and slowly shifting consonant intervals',
    'A small four-voice female ensemble, prayerful and blended, with a stable low vocal drone and clear upper melody',
    'Three soft female voices entering gradually around a sustained tonal center, without building intensity',
    'A close tender female soloist surrounded by faint sustained female harmonies and warm distant reflections',
]
RECIPES = {
    f'{style}-{variation}': profile + [direction]
    for style, profile in STYLE_PROFILES.items()
    for variation, direction in VARIATIONS.items()
}
RECIPE_STYLE = {f'{style}-{v}': style for style in STYLE_PROFILES for v in VARIATIONS}
RECIPE_VERSE = {f'{style}-{v}': i for style in STYLE_PROFILES for i, v in enumerate(VARIATIONS)}
for _index, _variation in enumerate(VARIATIONS):
    RECIPES[f'female-reverb-{_variation}'] += [REVERB_ENSEMBLES[_index]]

ANCHOR = [
    'Natural acoustic female singing, human breath support, expressive formants and subtle natural timing variation',
    'Sing only the supplied Latin lyrics, with consistent ecclesiastical Latin pronunciation and clearly formed words',
    'Repeat the written Latin verses as needed; sustain their vowels melodically without inventing words or replacing lyrics with gibberish',
    'The reference provides female vocal character; follow the requested musical style and Latin lyrics rather than copying the reference language or melody',
    'Unaccompanied voices, no instruments or percussion, quiet restrained dynamics suitable for restful listening',
    'One continuous performance across sections; preserve tonal center, ensemble and recording perspective without an internal ending or restart',
]
NEGATIVE = [
    'English lyrics', 'Norwegian lyrics', 'Sanskrit mantras', 'invented words', 'gibberish',
    'nonlexical scat singing', 'spoken narration', 'spoken instructions',
    'male vocals', 'instruments', 'percussion', 'drums', 'synthesizers',
    'robotic voice', 'vocoder', 'autotune', 'synthetic vocal pads',
    'shouting', 'belting', 'piercing high notes', 'dramatic crescendos',
    'large cinematic choir', 'driving beat', 'pop chorus',
    'long internal silence', 'section restart', 'fade out and restart',
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
        verse = LATIN_VERSES[RECIPE_VERSE[recipe]][0]
        # Enough real text for each section; repeat complete verses without stage labels.
        lyrics = '\n\n'.join([verse] * max(2, (seconds + 29) // 30))
        chunks.append({'text': lyrics, 'duration_ms': seconds * 1000,
                       'positive_styles': ANCHOR + RECIPES[recipe],
                       'negative_styles': NEGATIVE, 'context_adherence': 'high'})
    chunks[0]['conditioning_ref'] = {'song_id': ref['song_id'], 'range': {'start_ms': 0, 'end_ms': ref['end_ms']}}
    chunks[0]['condition_strength'] = ('high' if RECIPE_STYLE[recipe] == 'indian' else 'medium')
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


def is_credit_rejection(error):
    """Recognize explicit payment/quota rejections, never ambiguous timeouts."""
    message = str(error).strip()
    match = re.match(r'^HTTP (4[0-9]{2}):\s*(.*)', message, re.DOTALL)
    if not match:
        return False
    if int(match.group(1)) == 402:
        return True
    try:
        body = json.loads(match.group(2))
    except (ValueError, TypeError):
        return False
    detail = body.get('detail', body) if isinstance(body, dict) else None
    if not isinstance(detail, dict):
        return False
    return any(detail.get(field) in {
        'quota_exceeded', 'insufficient_credits', 'insufficient_credit',
        'insufficient_quota', 'payment_required',
    } for field in ('status', 'code', 'type'))


def recover_credit_rejections(records):
    """Migrate old failed entries only when rejection is explicit and no audio exists."""
    changed = False
    for record in records:
        if record.get('status') != 'failed_or_uncertain':
            continue
        target = OUTPUT_DIR / record['filename']
        if (is_credit_rejection(record.get('error', ''))
                and not target.exists()
                and not target.with_suffix('.mp3.part').exists()):
            record.update(status='rejected_credits', resolved_utc=now())
            print(f'Recognized credit rejection: {target.name}; eligible for a new attempt.')
            changed = True
    return changed


def run(args):
    import requests
    from mutagen.mp3 import MP3
    refs = references()  # Validate every reference before any paid calls.
    state_path = OUTPUT_DIR / 'latin-generation-pool-state.json'
    manifest_path = OUTPUT_DIR / 'latin-generation-manifest.json'
    state = load(state_path, {})
    manifest = load(manifest_path, {'schema_version': 1, 'style': 'indian', 'batch': 'latin-four-styles-40', 'generations': []})
    records = manifest['generations']
    if recover_credit_rejections(records) and not args.dry_run:
        save(manifest_path, manifest)
    unresolved = [r for r in records if r.get('status') in ('pending', 'failed_or_uncertain')]
    if unresolved:
        raise RuntimeError(
            'A previous request has an uncertain outcome. Review latin-generation-manifest.json '
            'and any .mp3.part files before retrying. Recover the audio and mark completed, '
            'or mark abandoned only after confirming it cannot be recovered. '
            'Unresolved details: ' + '; '.join(
                f"{r.get('filename')}: {r.get('error', 'no saved error; request was interrupted')}"
                for r in unresolved
            )
        )
    completed = [r for r in records if r.get('status') == 'completed']
    missing = [r['filename'] for r in completed if not (OUTPUT_DIR / r['filename']).is_file()]
    if missing:
        raise RuntimeError('Completed files are missing; restore them before continuing: ' + ', '.join(missing))
    completed_recipes = {r['archetype'] for r in completed}
    available_recipes = [r for r in RECIPES if r not in completed_recipes]
    if len(completed_recipes) != len(completed) or not completed_recipes <= set(RECIPES):
        raise RuntimeError('Latin manifest contains duplicate or unknown recipes; review it before continuing.')
    remaining = max(0, args.count - len(completed))
    if not remaining:
        print(f'Target already met: {len(completed)} completed Latin vocal samples. No generation requested.')
        return
    key = '' if args.dry_run else (ROOT / 'eleven-labs.txt').read_text(encoding='utf-8-sig').strip()
    if not args.dry_run and not key:
        raise ValueError('eleven-labs.txt is empty.')
    indices = [int(m.group(1)) for p in OUTPUT_DIR.iterdir() if (m := re.match(r'^latin-vocal-(\d+)-', p.name))]
    index = max(indices, default=0) + 1
    print(f'{remaining} new Latin vocal samples to reach {args.count} total; {len(refs)} references; output: {OUTPUT_DIR}')
    with requests.Session() as session:
        for i in range(remaining):
            ref = refs[draw(state, 'reference_pool', list(refs))]
            recipe = draw(state, 'prompt_pool', available_recipes)
            available_recipes.remove(recipe)
            duration = random.randint(MIN_DURATION_SECONDS, MAX_DURATION_SECONDS)
            plan = plan_for(ref, recipe, duration)
            fingerprint = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:12]
            filename = f'latin-vocal-{index:03d}-{recipe}-{fingerprint}.mp3'
            print(f'[{i+1}/{remaining}] {filename}\n  Reference: {ref["filename"]}; {duration}s\n  ' + ' '.join(RECIPES[recipe]), flush=True)
            if args.dry_run:
                index += 1
                continue
            record = {'created_utc': now(), 'filename': filename, 'status': 'pending',
                      'reference_filename': ref['filename'], 'reference_song_id': ref['song_id'],
                      'reference_sha256': ref['sha256'], 'archetype': recipe,
                      'duration_seconds_requested': duration,
                      'musical_style': RECIPE_STYLE[recipe], 'language': 'Latin',
                      'lyrics': LATIN_VERSES[RECIPE_VERSE[recipe]][0],
                      'lyrics_english_meaning': LATIN_VERSES[RECIPE_VERSE[recipe]][1],
                      'condition_strength': plan['chunks'][0]['condition_strength'],
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
                error = str(exc).replace(key, '[REDACTED]')
                target = OUTPUT_DIR / filename
                rejected = (is_credit_rejection(error) and not target.exists()
                            and not target.with_suffix('.mp3.part').exists())
                record.update(status='rejected_credits' if rejected else 'failed_or_uncertain',
                              error=error)
                save(manifest_path, manifest)
                guidance = ('Add credits, then rerun normally to finish the batch.' if rejected
                            else 'Check the manifest before rerunning; the request outcome is uncertain.')
                raise RuntimeError(f'Generation stopped: {error}. Completed samples are preserved. {guidance}') from None
            index += 1
            if i + 1 < remaining:
                time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=NUM_CLIPS_TO_GENERATE, help='Target total completed samples, 1-50 (default: 50)')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.count <= 50:
        parser.error('--count must be between 1 and 50')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lock = OUTPUT_DIR / 'latin-generation.lock'
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
