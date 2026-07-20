# Living Brown Noise / Dream Instigator

A Python audio experiment for creating long-duration brown-noise soundscapes that feel **alive rather than static**.

Most brown-noise recordings optimize the first few minutes: one pleasant spectrum, one stereo image, and one fixed texture repeated for hours. This project takes a different approach. It treats brown noise as a continuously evolving acoustic environment that moves through a broad range of pleasant brown-noise styles without calling attention to the transitions.

The longer-term project also includes a separate **Dream Instigator** system: optional ambient loops and discrete sound cues that can range from nearly subliminal to deliberately foregrounded.

## Current status

The current prototype is a single Python application:

```text
living-brown-noise-dream-instigator.py
```

It includes:

- Real-time brown-noise playback
- A GUI for isolating and tuning individual systems
- Persistent settings
- Offline WAV export from five minutes to six hours
- A fixed-anchor spectral engine designed to evolve without clicks or zipper noise
- Brown-noise spectral evolution
- Slowly drifting stereo correlation
- Breath modulation based on a multi-stage physiological timing model
- Long-term breath-prominence evolution
- Rare body-movement events that perturb the living system into a new equilibrium

The next major Living Brown Noise feature is a separate low-frequency heartbeat or pulse layer.

The Dream Instigator package/editor system is planned but not yet implemented.

## Design goals

### Living Brown Noise

The goal is not to generate one ideal brown noise. The goal is to generate an **endless distribution of excellent brown noises**.

A conventional track may sound good initially but become fatiguing through sameness. Living Brown Noise is intended to remain unobtrusive while continually changing just enough that the listener's auditory system never completely finishes adapting to it.

The core design principles are:

- **Variation without distraction**  
  Changes should usually be perceived only in retrospect. The listener may notice that the sound feels different from several minutes ago without hearing an identifiable transition.

- **A broad but safe spectral landscape**  
  The engine moves through a tested range of brown-noise styles, from deep and heavy to lighter and more textured, while avoiding settings that become harsh, weak, excessively subsonic, or rhythmically uneven.

- **Multiple timescales of life**  
  The sound changes at several nested rates:
  - Brown-noise randomness at the sample level
  - Breathing over seconds
  - Stereo and spectral evolution over minutes
  - Breath prominence over longer periods
  - Rare body-movement events over tens of minutes or hours

- **A resting-organism metaphor**  
  The sensory reference is lying against a large sleeping dog. The brown noise is the body; breathing, pulse, stereo depth, spectral motion, and occasional repositioning make it feel like a coherent living presence rather than a collection of unrelated audio effects.

- **No obvious looping or repeating modulation**  
  Organic stochastic motion is preferred over simple periodic LFO behavior.

- **Click-free evolution**  
  The current engine uses permanently configured spectral anchors and sample-ramped equal-power mixing. Runtime evolution changes mixer gains rather than repeatedly rebuilding or swapping active filters.

- **Real-time and offline parity**  
  The same engine is used for interactive playback and accelerated WAV export.

### Body movement

Normal evolution is intentionally slow and continuous. Body-movement events are different: they are rare, discrete state changes.

A movement does not play a literal rustling sound. Instead, it perturbs the organism:

- Spectral position changes
- Weight and texture shift
- Stereo equilibrium changes
- The system settles into a different evolving state

This creates an occasional perceptible reconfiguration without turning the track into a sequence of sound effects.

### Dream Instigator

Dream Instigator is a separate experience layered over Living Brown Noise.

It is expected to support reusable soundscape packages containing:

- One or more looping ambient tracks
- Groups of discrete one-shot effects
- Metadata controlling gain, timing, probability, spacing, fades, stereo placement, distance, overlap, and attention level
- Very slow crossfades between environmental identities rather than abrupt theme changes

An eventual **Attention** control should span two equally valid listening modes:

- Barely perceptible: “Wait, did I hear something?”
- Foregrounded and obvious: active listening or meditation

The first planned package is likely an ocean environment, using surf and water-lapping loops with one-shot sounds such as gulls, foghorns, buoy bells, distant boats, and shoreline events.

## Requirements

- Python 3.10 or later
- Windows, macOS, or Linux with an available audio output device
- The following Python packages:
  - NumPy
  - SciPy
  - sounddevice
  - PySide6

## Installation

Standard installation:

```bash
python -m pip install numpy scipy sounddevice PySide6
```

On the Windows development machine used for this project, explicitly invoking the intended Python 3.10 interpreter avoids Visual Studio Code selecting the wrong environment:

```powershell
& "C:\Users\jratc\AppData\Local\Programs\Python\Python310\python.exe" -m pip install numpy scipy sounddevice PySide6
```

If Python is installed elsewhere, replace that path with the path to the desired `python.exe`.

## Running

```bash
python living-brown-noise-dream-instigator.py
```

Using the explicit Windows interpreter:

```powershell
& "C:\Users\jratc\AppData\Local\Programs\Python\Python310\python.exe" .\living-brown-noise-dream-instigator.py
```

## Recommended first use

1. Start playback with the default settings.
2. Use the checkboxes to isolate individual systems.
3. Disable evolution before manually tuning the fixed brown-noise style controls.
4. Use high evolution rates only for accelerated testing.
5. Return evolution and body-movement frequency to slow values before exporting an overnight track.
6. Export a long WAV and evaluate it during a real listening or sleep session rather than relying only on short daytime tests.

## Brown-noise style controls

The current tuning surface focuses on four perceptual dimensions:

- **Body / spectral position**  
  Moves through the accepted range from deep/heavy to lighter/more present, with automatic gain compensation.

- **Slope strength**  
  Changes how soft versus detailed the brown-noise spectrum feels.

- **Low-end emphasis**  
  Adds broad weight without simply shifting the entire spectrum downward.

- **Upper texture**  
  Adds character and presence from a brighter parallel spectral branch.

These controls define the safe parameter landscape used by the evolution engine.

## Project philosophy

This is an experimental sound-design tool, not a medical device and not a claim about sleep science.

Its purpose is artistic and practical:

> Create a long-duration brown-noise environment that remains comforting because it changes, not despite changing.

The Dream Instigator portion extends that idea by placing optional, tunable environmental and narrative cues inside the living foundation.
