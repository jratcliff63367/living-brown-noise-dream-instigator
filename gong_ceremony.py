
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, replace

import numpy as np

from nonlinear_resonant_body import ResonantBodySpec, StatefulModalNetwork
from synthesized_sound_source import OrganicWanderer1D, SmoothedValue, db_to_linear


@dataclass(frozen=True, slots=True)
class GongSpec:
    base_hz: float = 42.0
    size: float = 1.0
    decay_seconds: float = 46.0
    strike_strength: float = 0.52
    darkness: float = 0.56
    cascade: float = 0.90
    chaos: float = 0.72
    bloom: float = 0.86

    friction_level: float = 0.0
    friction_pressure: float = 0.52
    friction_speed: float = 0.42
    friction_brightness: float = 0.64
    friction_instability: float = 0.72

    hand_level: float = 0.0
    hand_pressure: float = 0.62
    hand_position: float = 0.70

    output_gain_db: float = -8.5

    def validated(self):
        return self


class GongState:
    def __init__(self, spec: GongSpec):
        self._lock = threading.Lock()
        self._spec = spec.validated()

    def get(self):
        with self._lock:
            return self._spec

    def update(self, **changes):
        with self._lock:
            self._spec = replace(self._spec, **changes).validated()


class ProceduralGongGenerator:
    """
    Third-generation gong model.

    Key change from v2:
    a strike creates a short dispersive metallic plate disturbance before the
    sound resolves into the modal field. The modal network itself is also more
    turbulent and less "sine-bank" clean.
    """

    def __init__(self, sample_rate, state, *, seed=904101):
        self.sample_rate = float(sample_rate)
        self.state = state
        self.rng = np.random.default_rng(seed)

        self.gain = SmoothedValue(
            self.sample_rate,
            db_to_linear(state.get().output_gain_db),
            0.14,
        )
        self.friction_smoother = SmoothedValue(
            self.sample_rate,
            state.get().friction_level,
            0.45,
        )
        self.hand_smoother = SmoothedValue(
            self.sample_rate,
            state.get().hand_level,
            0.35,
        )

        self.friction_wander = OrganicWanderer1D(
            seed=seed + 20,
            natural_period_seconds=6.0,
            damping_ratio=0.56,
            drive_strength=1.0,
            drive_smoothing_seconds=1.8,
        )
        self.hand_wander = OrganicWanderer1D(
            seed=seed + 21,
            natural_period_seconds=4.8,
            damping_ratio=0.50,
            drive_strength=1.10,
            drive_smoothing_seconds=1.3,
        )

        self.network = self._build_network(state.get(), seed)
        self.impact_events = []
        self.friction_phase = 0.0
        self.hand_phase = 0.0

    def _build_network(self, spec, seed):
        rng = np.random.default_rng(seed + 1)

        # Significantly denser field than v2.
        low = np.geomspace(spec.base_hz, 340.0, 26)
        mid = np.geomspace(120.0, 2200.0, 46)
        high = np.geomspace(650.0, 6200.0, 34)

        f = np.concatenate([low, mid, high])
        fam = np.concatenate([
            np.zeros(len(low), dtype=np.int32),
            np.ones(len(mid), dtype=np.int32),
            np.full(len(high), 2, dtype=np.int32),
        ])

        # Irregular plate geometry and intentional local clustering.
        f *= np.exp(
            rng.normal(
                0.0,
                0.013 + 0.024 * spec.chaos,
                len(f),
            )
        )

        # Add close splittings to many modes.
        cluster = rng.random(len(f)) < 0.42
        f[cluster] *= 1.0 + rng.normal(
            0.0,
            0.0022 + 0.0055 * spec.chaos,
            int(cluster.sum()),
        )

        order = np.argsort(f)
        f = f[order]
        fam = fam[order]

        x = np.log(f / f.min()) / np.log(f.max() / f.min())
        decay = (
            spec.decay_seconds
            * (1.16 - 0.72 * x)
            * np.exp(rng.normal(0.0, 0.33, len(f)))
        )
        decay = np.clip(decay, 2.5, 100.0)

        radiation = (
            0.55
            + 0.95
            * np.exp(
                -0.5 * ((np.log(f) - np.log(720.0)) / 1.20) ** 2
            )
        )
        radiation *= np.power(
            f / max(35.0, spec.base_hz),
            -0.24 * spec.darkness,
        )
        radiation *= rng.uniform(0.66, 1.34, len(f))

        body_spec = ResonantBodySpec(
            cascade_threshold=0.022 + 0.040 * (1.0 - spec.cascade),
            cascade_rate=0.34 + 1.10 * spec.cascade,
            diffusion_rate=0.05 + 0.13 * spec.chaos,
            nonlinear_enter=0.20,
            nonlinear_leave=0.095,
            frequency_pull=0.0024 + 0.0058 * spec.chaos,
            sideband_amount=0.14 + 0.34 * spec.chaos,
            roughness_amount=0.10 + 0.18 * spec.chaos,
            coherence_loss=0.10 + 0.22 * spec.chaos,
            saturation=0.98,
        )

        return StatefulModalNetwork(
            self.sample_rate,
            f,
            decay,
            radiation,
            fam,
            spec=body_spec,
            seed=seed + 2,
        )

    def _strike_weights(self, location, hardness):
        f = self.network.frequencies

        cutoff = 260.0 + 3400.0 * hardness
        spectral = 1.0 / (
            1.0 + (f / cutoff) ** (2.2 + 1.6 * (1.0 - hardness))
        )

        idx = np.arange(len(f), dtype=np.float64)
        spatial = (
            0.12
            + np.sin(
                math.pi * (
                    0.42
                    + 5.4 * location
                    + 0.117 * idx
                )
            ) ** 2
        )

        family_gate = np.where(
            self.network.family == 0,
            1.0,
            np.where(
                self.network.family == 1,
                0.34 + 0.52 * hardness,
                0.05 + 0.25 * hardness,
            ),
        )

        return spectral * spatial * family_gate + 1e-8

    def strike(self, strength=None, location=None, *, hardness=None):
        spec = self.state.get()

        if strength is None:
            strength = spec.strike_strength
        if location is None:
            location = float(self.rng.uniform(0.22, 0.86))
        if hardness is None:
            hardness = float(
                np.clip(
                    self.rng.normal(
                        0.24 + 0.18 * strength,
                        0.07,
                    ),
                    0.10,
                    0.56,
                )
            )

        strength = float(np.clip(strength, 0.0, 1.2))
        location = float(np.clip(location, 0.0, 1.0))
        hardness = float(np.clip(hardness, 0.0, 1.0))

        self.network.inject(
            self._strike_weights(location, hardness),
            0.10 + 1.05 * strength * strength,
            phase_randomization=0.30 + 0.20 * hardness,
        )

        # Dispersive plate disturbance: very short broadband metallic event with
        # several chirping/decaying bands. This is the "sheet of metal was
        # actually struck" cue missing from v2.
        self.impact_events.append({
            "age": 0.0,
            "strength": strength,
            "hardness": hardness,
            "seed_phase": self.rng.uniform(0, 2 * math.pi, 6),
        })

    def clear(self):
        self.network.clear()
        self.impact_events.clear()

    def _render_plate_disturbance(self, frame_count):
        if not self.impact_events:
            return np.zeros(frame_count, dtype=np.float64)

        out = np.zeros(frame_count, dtype=np.float64)
        retained = []
        n = np.arange(frame_count, dtype=np.float64)
        dt_sample = 1.0 / self.sample_rate

        for event in self.impact_events:
            t = event["age"] + n * dt_sample
            strength = event["strength"]
            hardness = event["hardness"]
            phases = event["seed_phase"]

            env_fast = np.exp(-t / (0.030 + 0.030 * (1.0 - hardness)))
            env_slow = np.exp(-t / (0.16 + 0.20 * (1.0 - hardness)))

            # Dispersive chirps: high bands fall rapidly toward modal regions.
            bands = (
                (4200.0, 980.0),
                (3000.0, 760.0),
                (2200.0, 580.0),
                (1500.0, 440.0),
                (950.0, 320.0),
                (620.0, 220.0),
            )

            metallic = np.zeros(frame_count, dtype=np.float64)
            for i, (f0, f1) in enumerate(bands):
                tau = 0.045 + 0.020 * i
                f_inst = f1 + (f0 - f1) * np.exp(-t / tau)
                # Approximate phase integral using local frequency.
                phase = (
                    phases[i]
                    + 2 * math.pi * f_inst * t
                )
                metallic += (
                    (0.42 / (1.0 + 0.22 * i))
                    * np.sin(phase)
                )

            noise = self.rng.standard_normal(frame_count)
            out += strength * (
                0.070 * env_fast * noise
                + 0.034 * env_slow * metallic
            )

            event["age"] += frame_count / self.sample_rate
            if event["age"] < 0.8:
                retained.append(event)

        self.impact_events = retained
        return out

    def _friction_drive(self, spec, frame_count):
        level = self.friction_smoother.ramp(
            spec.friction_level, frame_count
        )
        mean = float(np.mean(level))
        if mean < 1e-6:
            return np.zeros(frame_count, dtype=np.float64)

        dt = frame_count / self.sample_rate
        wander = self.friction_wander.advance(dt)
        pressure = np.clip(
            spec.friction_pressure + 0.12 * wander, 0, 1
        )
        speed = np.clip(
            spec.friction_speed + 0.10 * wander, 0, 1
        )

        f = self.network.frequencies
        center = (
            220.0
            + 2500.0 * spec.friction_brightness
            + 420.0 * wander
        )
        width = 0.70 + 0.90 * spec.friction_instability
        weights = np.exp(
            -0.5
            * (
                np.log2(np.maximum(f, 1.0) / max(35.0, center))
                / width
            ) ** 2
        )
        weights *= np.where(
            self.network.family == 0, 0.30, 1.0
        )

        self.network.inject(
            weights,
            mean
            * (0.018 + 0.13 * pressure)
            * dt
            * 60.0,
            phase_randomization=0.03 + 0.06 * spec.friction_instability,
        )

        n = np.arange(frame_count, dtype=np.float64)
        slip_hz = 8.0 + 38.0 * speed
        phase = (
            self.friction_phase
            + 2 * math.pi * slip_hz * n / self.sample_rate
        )
        slip = np.tanh(
            (1.4 + 6.0 * pressure) * np.sin(phase)
        )
        self.friction_phase = float(
            (phase[-1] + 2 * math.pi * slip_hz / self.sample_rate)
            % (2 * math.pi)
        )
        noise = self.rng.standard_normal(frame_count)
        return mean * (
            0.0025 * slip
            + 0.0030 * spec.friction_instability * noise
        )

    def _hand_drive(self, spec, frame_count):
        level = self.hand_smoother.ramp(
            spec.hand_level, frame_count
        )
        mean = float(np.mean(level))
        if mean < 1e-6:
            return np.zeros(frame_count, dtype=np.float64)

        dt = frame_count / self.sample_rate
        wander = self.hand_wander.advance(dt)
        pressure = np.clip(
            spec.hand_pressure + 0.16 * wander, 0, 1
        )
        position = np.clip(
            spec.hand_position + 0.12 * wander, 0.05, 0.98
        )

        center = (
            110.0
            + 1900.0 * position ** 1.55
            + 380.0 * wander * pressure
        )
        f = self.network.frequencies
        width = 0.48 + 0.70 * (1.0 - pressure)
        weights = np.exp(
            -0.5
            * (
                np.log2(np.maximum(f, 1.0) / max(35.0, center))
                / width
            ) ** 2
        )
        second = center * (1.68 + 0.30 * wander)
        weights += 0.40 * np.exp(
            -0.5
            * (
                np.log2(np.maximum(f, 1.0) / max(35.0, second))
                / 0.64
            ) ** 2
        )

        self.network.inject(
            weights,
            mean
            * (0.020 + 0.15 * pressure)
            * dt
            * 60.0,
            phase_randomization=0.02 + 0.04 * pressure,
        )

        n = np.arange(frame_count, dtype=np.float64)
        carrier_hz = (
            24.0
            + 200.0 * position
            + 105.0 * pressure
            + 36.0 * wander
        )
        phase = (
            self.hand_phase
            + 2 * math.pi * carrier_hz * n / self.sample_rate
        )
        carrier = np.tanh(
            (1.8 + 7.5 * pressure) * np.sin(phase)
        )
        self.hand_phase = float(
            (phase[-1] + 2 * math.pi * carrier_hz / self.sample_rate)
            % (2 * math.pi)
        )
        noise = self.rng.standard_normal(frame_count)
        return mean * (0.0032 * carrier + 0.0022 * noise)

    def generate(self, frame_count):
        spec = self.state.get()
        plate = self._render_plate_disturbance(frame_count)
        friction = self._friction_drive(spec, frame_count)
        hand = self._hand_drive(spec, frame_count)

        body = self.network.render(
            frame_count,
            external_drive=(friction + hand),
            external_drive_gain=1.0,
        ).astype(np.float64)

        # The impact disturbance leads, then folds into the persistent body.
        out = body + plate

        gain = self.gain.ramp(
            db_to_linear(spec.output_gain_db),
            frame_count,
        )
        out *= gain
        out = 0.97 * np.tanh(out * 1.02)
        return out.astype(np.float32, copy=False)


@dataclass(frozen=True, slots=True)
class GongCeremonySpec:
    enabled: bool = False
    duration_minutes: float = 30.0
    intensity: float = 0.64
    friction_presence: float = 0.82
    hand_magic: float = 0.88
    spatiality: float = 0.62

    def validated(self):
        return self


class GongCeremonyState:
    def __init__(self, spec):
        self._lock = threading.Lock()
        self._spec = spec.validated()

    def get(self):
        with self._lock:
            return self._spec

    def update(self, **changes):
        with self._lock:
            self._spec = replace(self._spec, **changes).validated()


@dataclass(frozen=True, slots=True)
class GongProfile:
    name: str
    base_hz: float
    size: float
    decay_seconds: float
    darkness: float
    cascade: float
    chaos: float
    output_gain_db: float
    x: float
    y: float
    z: float


@dataclass(slots=True)
class GongVoice:
    profile: GongProfile
    state: GongState
    generator: ProceduralGongGenerator
    position: np.ndarray
    move_start: np.ndarray
    move_target: np.ndarray
    move_elapsed: float
    move_duration: float
    next_strike: float
    strikes: int = 0


class GongCeremonyController:
    PHASES = (
        ("arrival", 0.00, 0.08),
        ("grounding", 0.08, 0.20),
        ("expansion", 0.20, 0.40),
        ("immersion", 0.40, 0.65),
        ("alchemy", 0.65, 0.79),
        ("integration", 0.79, 0.90),
        ("closing", 0.90, 0.97),
        ("final silence", 0.97, 1.00),
    )

    PROFILES = (
        GongProfile(
            "Large tam-tam",
            42.0, 1.00, 48.0, 0.56, 0.92, 0.74, -8.5,
            0.0, -0.30, -2.70,
        ),
        GongProfile(
            "Medium symphonic gong",
            61.0, 0.76, 38.0, 0.48, 0.86, 0.68, -10.5,
            -1.00, 0.18, -2.25,
        ),
        GongProfile(
            "Bright friction gong",
            82.0, 0.52, 30.0, 0.34, 0.78, 0.76, -12.0,
            1.00, 0.48, -1.95,
        ),
    )

    def __init__(self, sample_rate, state, *, seed=904500):
        self.sample_rate = float(sample_rate)
        self.state = state
        self.rng = np.random.default_rng(seed)

        self.voices = []
        for i, p in enumerate(self.PROFILES):
            gs = GongState(
                GongSpec(
                    base_hz=p.base_hz,
                    size=p.size,
                    decay_seconds=p.decay_seconds,
                    darkness=p.darkness,
                    cascade=p.cascade,
                    chaos=p.chaos,
                    output_gain_db=p.output_gain_db,
                )
            )
            g = ProceduralGongGenerator(
                self.sample_rate, gs, seed=seed + 100 + i * 37
            )
            home = np.array([p.x, p.y, p.z], dtype=np.float64)
            self.voices.append(
                GongVoice(
                    p, gs, g,
                    home.copy(), home.copy(), home.copy(),
                    0.0, 28.0,
                    2.0 + i * 2.0,
                )
            )

        self.elapsed_seconds = 0.0
        self.phase = "arrival"
        self.phase_progress = 0.0
        self.performance_progress = 0.0
        self.running = False
        self.complete = False
        self._opening = False
        self._closing_cues = set()

        self.energy_wander = OrganicWanderer1D(
            seed=seed + 50,
            natural_period_seconds=40.0,
            damping_ratio=0.64,
            drive_strength=0.92,
            drive_smoothing_seconds=12.0,
        )
        self.magic_wander = OrganicWanderer1D(
            seed=seed + 51,
            natural_period_seconds=16.0,
            damping_ratio=0.56,
            drive_strength=1.04,
            drive_smoothing_seconds=4.8,
        )

    def restart(self):
        self.elapsed_seconds = 0.0
        self.phase = "arrival"
        self.phase_progress = 0.0
        self.performance_progress = 0.0
        self.running = True
        self.complete = False
        self._opening = False
        self._closing_cues.clear()
        for i, v in enumerate(self.voices):
            v.generator.clear()
            v.state.update(friction_level=0.0, hand_level=0.0)
            v.next_strike = 7.0 + i * 3.0
            v.strikes = 0

    def stop(self):
        self.running = False
        for v in self.voices:
            v.state.update(friction_level=0.0, hand_level=0.0)

    def _phase_for_progress(self, progress):
        for name, start, end in self.PHASES:
            if progress < end:
                return name, float(
                    np.clip((progress - start) / (end - start), 0, 1)
                )
        return "complete", 1.0

    def _energy(self):
        p = self.phase_progress
        if self.phase == "arrival":
            return 0.18 + 0.18 * p
        if self.phase == "grounding":
            return 0.36 + 0.20 * p
        if self.phase == "expansion":
            return 0.54 + 0.24 * p
        if self.phase == "immersion":
            return 0.82 + 0.14 * math.sin(math.pi * p)
        if self.phase == "alchemy":
            return 0.90 + 0.08 * math.sin(math.pi * p)
        if self.phase == "integration":
            return 0.72 - 0.28 * p
        if self.phase == "closing":
            return 0.38 - 0.26 * p
        return 0.0

    def _weight(self, i):
        p = self.phase_progress
        if self.phase == "arrival":
            return (1.0, 0.0, 0.0)[i]
        if self.phase == "grounding":
            return (1.0, 0.58, 0.0)[i]
        if self.phase == "expansion":
            return (0.98, 0.86, 0.38 + 0.46 * p)[i]
        if self.phase == "immersion":
            return (0.96, 1.0, 0.82)[i]
        if self.phase == "alchemy":
            return (0.82, 0.88, 1.0)[i]
        if self.phase == "integration":
            return (0.90, 0.68, 0.44)[i]
        if self.phase == "closing":
            return (1.0, 0.16 * (1.0 - p), 0.0)[i]
        return 0.0

    def _next_strike(self, voice, energy, weight, intensity):
        effective = max(
            0.08,
            energy * weight * (0.65 + 0.65 * intensity),
        )
        size = voice.profile.size
        low = (2.6 + 5.2 * size) / (0.58 + effective)
        high = (6.0 + 12.0 * size) / (0.55 + effective)
        low = float(np.clip(low, 2.0, 14.0))
        high = float(np.clip(high, low + 1.0, 28.0))
        return float(
            math.exp(
                self.rng.uniform(math.log(low), math.log(high))
            )
        )

    def advance(self, dt):
        spec = self.state.get()
        if not spec.enabled or not self.running or self.complete:
            return

        dt = max(0.0, float(dt))
        total = spec.duration_minutes * 60.0
        self.elapsed_seconds += dt
        self.performance_progress = float(
            np.clip(self.elapsed_seconds / max(1.0, total), 0, 1)
        )
        self.phase, self.phase_progress = self._phase_for_progress(
            self.performance_progress
        )

        if self.phase == "complete":
            self.complete = True
            self.running = False
            self.stop()
            return

        energy = self._energy()
        energy *= (
            0.90
            + 0.16 * (0.5 + 0.5 * self.energy_wander.advance(dt))
        )
        energy = float(np.clip(energy, 0, 1))
        magic = 0.5 + 0.5 * self.magic_wander.advance(dt)

        if (
            self.phase == "arrival"
            and not self._opening
            and self.elapsed_seconds >= 1.5
        ):
            self.voices[0].generator.strike(
                0.54, 0.48, hardness=0.18
            )
            self.voices[0].strikes += 1
            self._opening = True

        for i, v in enumerate(self.voices):
            weight = self._weight(i)

            if self.phase in {"arrival", "final silence"}:
                friction_phase = hand_phase = 0.0
            elif self.phase == "grounding":
                friction_phase, hand_phase = 0.10, 0.0
            elif self.phase == "expansion":
                friction_phase, hand_phase = 0.26, 0.10
            elif self.phase == "immersion":
                friction_phase, hand_phase = 0.52, 0.30
            elif self.phase == "alchemy":
                friction_phase, hand_phase = 0.70, 0.74
            elif self.phase == "integration":
                friction_phase, hand_phase = 0.30, 0.16
            else:
                friction_phase, hand_phase = (
                    0.08 * (1.0 - self.phase_progress),
                    0.0,
                )

            friction_pref = (0.38, 0.80, 1.0)[i]
            hand_pref = (0.15, 0.62, 1.0)[i]

            friction = float(np.clip(
                spec.friction_presence
                * friction_phase
                * friction_pref
                * weight
                * energy
                * (0.30 + 0.70 * magic),
                0, 0.86,
            ))
            hand = float(np.clip(
                spec.hand_magic
                * hand_phase
                * hand_pref
                * weight
                * energy
                * (0.22 + 0.78 * magic ** 1.8),
                0, 0.90,
            ))

            v.state.update(
                friction_level=friction,
                friction_pressure=float(np.clip(
                    0.44 + 0.34 * energy + 0.12 * magic, 0, 1
                )),
                friction_speed=float(np.clip(
                    0.30 + 0.34 * magic, 0, 1
                )),
                friction_brightness=float(np.clip(
                    0.48
                    + 0.34 * (1.0 - v.profile.size)
                    + 0.16 * magic,
                    0, 1
                )),
                friction_instability=float(np.clip(
                    0.48 + 0.38 * magic, 0, 1
                )),
                hand_level=hand,
                hand_pressure=float(np.clip(
                    0.50 + 0.38 * magic, 0, 1
                )),
                hand_position=float(np.clip(
                    0.46
                    + 0.28 * math.sin(self.elapsed_seconds * 0.029 + i)
                    + 0.10 * magic,
                    0.10, 0.94
                )),
            )

            if self.phase in {"closing", "final silence"}:
                continue

            v.next_strike -= dt
            if v.next_strike <= 0.0 and weight > 0.05:
                if self.rng.random() < (
                    0.05 + 0.14 * energy * spec.intensity
                ):
                    strength = float(np.clip(
                        self.rng.normal(0.62 + 0.18 * energy, 0.08),
                        0.42, 0.94
                    ))
                else:
                    strength = float(np.clip(
                        self.rng.normal(0.28 + 0.22 * energy, 0.08),
                        0.14, 0.62
                    ))

                v.generator.strike(
                    strength,
                    float(self.rng.uniform(0.20, 0.88)),
                )
                v.strikes += 1
                v.next_strike = self._next_strike(
                    v, energy, weight, spec.intensity
                )

        if self.phase == "closing":
            cues = (
                (0.12, 0.44),
                (0.48, 0.31),
                (0.78, 0.20),
            )
            for cue_index, (cue, strength) in enumerate(cues):
                if (
                    self.phase_progress >= cue
                    and cue_index not in self._closing_cues
                ):
                    self.voices[0].generator.strike(
                        strength, 0.50, hardness=0.16
                    )
                    self.voices[0].strikes += 1
                    self._closing_cues.add(cue_index)

    def render_mono(self, frame_count):
        return [v.generator.generate(frame_count) for v in self.voices]

    @property
    def remaining_seconds(self):
        return max(
            0.0,
            self.state.get().duration_minutes * 60.0 - self.elapsed_seconds,
        )
