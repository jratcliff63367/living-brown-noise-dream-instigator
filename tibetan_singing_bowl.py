
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, replace
import numpy as np

from nonlinear_resonant_body import ResonantBodySpec, StatefulModalNetwork
from synthesized_sound_source import OrganicWanderer1D, SmoothedValue, db_to_linear


@dataclass(frozen=True, slots=True)
class SingingBowlSpec:
    fundamental_hz: float = 185.0
    decay_seconds: float = 20.0
    strike_strength: float = 0.62
    brightness: float = 0.50
    asymmetry: float = 0.44
    coupling: float = 0.34
    body: float = 0.70
    rub_level: float = 0.0
    rub_motion: float = 0.40
    rub_pressure: float = 0.56
    rub_speed: float = 0.46
    output_gain_db: float = -11.0

    def validated(self):
        return self


class SingingBowlState:
    def __init__(self, spec):
        self._lock = threading.Lock()
        self._spec = spec.validated()

    def get(self):
        with self._lock:
            return self._spec

    def update(self, **changes):
        with self._lock:
            self._spec = replace(self._spec, **changes).validated()


class TibetanSingingBowlGenerator:
    """
    Third-generation bowl model.

    The dominant measured mode families remain, but each pair is now surrounded
    by weak satellite modes so the bowl keeps its recognizable pitch identity
    while acquiring much greater depth and metallic complexity.
    """

    MEASURED = np.array(
        [1.0, 2.77828, 5.18099, 8.16289, 11.66063, 15.63801, 19.99],
        dtype=np.float64,
    )

    def __init__(self, sample_rate, state, *, seed=602701):
        self.sample_rate = float(sample_rate)
        self.state = state
        self.rng = np.random.default_rng(seed)

        self.gain = SmoothedValue(
            self.sample_rate,
            db_to_linear(state.get().output_gain_db),
            0.14,
        )
        self.rub = SmoothedValue(
            self.sample_rate,
            state.get().rub_level,
            0.42,
        )

        self.rub_wander = OrganicWanderer1D(
            seed=seed + 100,
            natural_period_seconds=7.8,
            damping_ratio=0.60,
            drive_strength=0.88,
            drive_smoothing_seconds=2.6,
        )
        self.contact_wander = OrganicWanderer1D(
            seed=seed + 101,
            natural_period_seconds=12.0,
            damping_ratio=0.70,
            drive_strength=0.72,
            drive_smoothing_seconds=4.0,
        )

        self.contact_angle = float(
            self.rng.uniform(0, 2 * math.pi)
        )
        self.impact_events = []

        self.network = self._build_network(state.get(), seed)

    def _build_network(self, spec, seed):
        rng = np.random.default_rng(seed + 1)

        freqs = []
        families = []
        radiation = []
        decays = []

        for k, ratio in enumerate(self.MEASURED):
            base = spec.fundamental_hz * ratio

            cents = 1.0 + spec.asymmetry * (3.0 + 2.0 * k)
            split = 2.0 ** (cents / 1200.0)

            dominant_a = base / math.sqrt(split)
            dominant_b = base * math.sqrt(split)

            # Dominant pair.
            local = [dominant_a, dominant_b]

            # Satellites near each dominant mode. They remain much weaker than
            # the main pair, preserving bowl identity while adding depth.
            satellite_cents = (
                -18.0 - 5.0 * k,
                -7.0 - 2.5 * k,
                8.5 + 2.0 * k,
                20.0 + 4.0 * k,
            )
            for c in satellite_cents:
                local.append(
                    base * 2.0 ** (c / 1200.0)
                )

            for j, f in enumerate(local):
                freqs.append(f)
                families.append(k)

                high = k / max(1, len(self.MEASURED) - 1)
                if j < 2:
                    r = (
                        1.00
                        * (1.00 - 0.22 * high)
                        * (0.78 + 0.55 * spec.brightness * high)
                    )
                    d = (
                        spec.decay_seconds
                        * (1.00 - 0.42 * high)
                        * rng.uniform(0.92, 1.12)
                    )
                else:
                    r = (
                        0.10
                        + 0.14 * spec.brightness
                        + 0.08 * (1.0 - high)
                    )
                    d = (
                        spec.decay_seconds
                        * (0.38 + 0.24 * (1.0 - high))
                        * rng.uniform(0.70, 1.20)
                    )

                radiation.append(r)
                decays.append(max(1.6, d))

        f = np.array(freqs, dtype=np.float64)
        fam = np.array(families, dtype=np.int32)
        radiation = np.array(radiation, dtype=np.float64)
        decays = np.array(decays, dtype=np.float64)

        # Tiny manufacturing irregularities.
        f *= np.exp(
            rng.normal(
                0.0,
                0.0015 + 0.0030 * spec.asymmetry,
                len(f),
            )
        )

        order = np.argsort(f)
        f = f[order]
        fam = fam[order]
        radiation = radiation[order]
        decays = decays[order]

        body_spec = ResonantBodySpec(
            cascade_threshold=0.050 + 0.045 * (1.0 - spec.coupling),
            cascade_rate=0.05 + 0.22 * spec.coupling,
            diffusion_rate=0.012 + 0.035 * spec.coupling,
            nonlinear_enter=0.20,
            nonlinear_leave=0.10,
            frequency_pull=0.0006 + 0.0010 * spec.coupling,
            sideband_amount=0.018 + 0.065 * spec.coupling,
            roughness_amount=0.018 + 0.040 * spec.brightness,
            coherence_loss=0.012 + 0.050 * spec.asymmetry,
            saturation=0.94,
        )

        return StatefulModalNetwork(
            self.sample_rate,
            f,
            decays,
            radiation,
            fam,
            spec=body_spec,
            seed=seed + 2,
        )

    def _strike_weights(self, angle):
        f = self.network.frequencies
        fam = self.network.family

        weights = np.zeros(len(f), dtype=np.float64)
        for i, family in enumerate(fam):
            n = int(family) + 2
            weights[i] = (
                0.16
                + 0.84 * abs(math.cos(n * angle + 0.31 * i))
            )

        # Suppress satellites slightly by using distance to nearest measured
        # family center.
        centers = self.state.get().fundamental_hz * self.MEASURED
        nearest = np.min(
            np.abs(np.log(f[:, None] / centers[None, :])),
            axis=1,
        )
        dominant_bias = np.exp(-nearest / 0.012)
        weights *= 0.34 + 0.66 * dominant_bias
        return weights + 1e-8

    def strike(self, strength=None):
        spec = self.state.get()
        if strength is None:
            strength = spec.strike_strength
        strength = float(np.clip(strength, 0.0, 1.2))

        angle = float(self.rng.uniform(0, 2 * math.pi))
        self.network.inject(
            self._strike_weights(angle),
            0.10 + 0.78 * strength * strength,
            phase_randomization=0.14,
        )

        # Very subtle bronze contact smear: enough to prevent "pure oscillator"
        # onset without making the bowl cymbal-like.
        self.impact_events.append({
            "age": 0.0,
            "strength": strength,
            "phase": self.rng.uniform(0, 2 * math.pi, 4),
        })

    def clear(self):
        self.network.clear()
        self.impact_events.clear()

    def _render_impact(self, frame_count):
        if not self.impact_events:
            return np.zeros(frame_count, dtype=np.float64)

        n = np.arange(frame_count, dtype=np.float64)
        t_step = 1.0 / self.sample_rate
        out = np.zeros(frame_count, dtype=np.float64)
        retained = []

        for e in self.impact_events:
            t = e["age"] + n * t_step
            env = np.exp(-t / 0.045)
            env2 = np.exp(-t / 0.13)

            f0 = self.state.get().fundamental_hz
            bands = (
                2.4 * f0,
                4.6 * f0,
                7.8 * f0,
                12.5 * f0,
            )

            metal = np.zeros(frame_count, dtype=np.float64)
            for i, freq in enumerate(bands):
                metal += (
                    0.24 / (1 + 0.35 * i)
                ) * np.sin(
                    e["phase"][i]
                    + 2 * math.pi * freq * t
                )

            noise = self.rng.standard_normal(frame_count)
            out += e["strength"] * (
                0.012 * env * noise
                + 0.010 * env2 * metal
            )

            e["age"] += frame_count / self.sample_rate
            if e["age"] < 0.45:
                retained.append(e)

        self.impact_events = retained
        return out

    def _rub_drive(self, spec, frame_count):
        level = self.rub.ramp(spec.rub_level, frame_count)
        mean = float(np.mean(level))
        if mean < 1e-6:
            return np.zeros(frame_count, dtype=np.float64)

        dt = frame_count / self.sample_rate
        wander = self.rub_wander.advance(
            dt * (0.4 + 2.0 * spec.rub_motion)
        )
        contact = self.contact_wander.advance(dt)

        turns = (
            0.10
            + 0.60 * spec.rub_speed
            + 0.08 * wander
        )
        self.contact_angle = (
            self.contact_angle
            + 2 * math.pi * turns * dt
        ) % (2 * math.pi)

        f = self.network.frequencies
        fam = self.network.family
        weights = np.zeros(len(f), dtype=np.float64)
        phase_target = np.zeros(len(f), dtype=np.float64)

        for i, family in enumerate(fam):
            n_order = int(family) + 2
            tangent = 1.0 / n_order
            order_tilt = math.exp(-0.36 * int(family))

            weights[i] = (
                tangent
                * order_tilt
                * (
                    0.16
                    + abs(
                        math.sin(
                            n_order * self.contact_angle + 0.29 * i
                        )
                    )
                )
            )
            phase_target[i] = (
                n_order * self.contact_angle + 0.31 * i
            ) % (2 * math.pi)

        pressure = float(np.clip(
            spec.rub_pressure * (0.90 + 0.14 * contact),
            0, 1
        ))

        self.network.inject(
            weights,
            mean
            * (0.010 + 0.070 * pressure)
            * dt
            * 60.0,
            phase_randomization=0.010,
            phase_target=phase_target,
            phase_lock=0.04 + 0.16 * pressure,
        )

        # Quiet physical contact texture only.
        n = np.arange(frame_count, dtype=np.float64)
        slip_hz = 16.0 + 48.0 * spec.rub_speed + 5.0 * wander
        slip = np.tanh(
            (1.7 + 4.5 * pressure)
            * np.sin(2 * math.pi * slip_hz * n / self.sample_rate)
        )
        noise = self.rng.standard_normal(frame_count)

        return mean * (
            0.0013 * slip + 0.0014 * noise
        )

    def generate(self, frame_count):
        spec = self.state.get()
        rub = self._rub_drive(spec, frame_count)
        body = self.network.render(
            frame_count,
            external_drive=rub,
            external_drive_gain=1.0,
        ).astype(np.float64)
        impact = self._render_impact(frame_count)

        out = body + impact
        out *= self.gain.ramp(
            db_to_linear(spec.output_gain_db),
            frame_count,
        )
        out = 0.95 * np.tanh(out * 0.96)
        return out.astype(np.float32, copy=False)


@dataclass(frozen=True, slots=True)
class BowlCeremonySpec:
    enabled: bool = False
    duration_minutes: float = 30.0
    intensity: float = 0.62
    spatiality: float = 0.88
    rubbing: float = 0.78

    def validated(self):
        return self


class BowlCeremonyState:
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
class BowlProfile:
    name: str
    fundamental_hz: float
    decay_seconds: float
    brightness: float
    asymmetry: float
    coupling: float
    body: float
    output_gain_db: float
    size_class: float


@dataclass(slots=True)
class BowlVoice:
    profile: BowlProfile
    state: SingingBowlState
    generator: TibetanSingingBowlGenerator
    position: np.ndarray
    next_strike_seconds: float
    current_rub: float = 0.0
    strikes: int = 0


class BowlCeremonyController:
    PROFILES = (
        BowlProfile(
            "Large grounding bowl",
            107.3, 31.0, 0.30, 0.42, 0.30, 0.92, -11.8, 1.00
        ),
        BowlProfile(
            "Low-mid bowl",
            143.8, 26.0, 0.42, 0.44, 0.34, 0.80, -12.8, 0.78
        ),
        BowlProfile(
            "Middle singing bowl",
            191.6, 22.0, 0.56, 0.48, 0.38, 0.66, -13.4, 0.55
        ),
        BowlProfile(
            "Small clear bowl",
            286.7, 17.0, 0.72, 0.52, 0.40, 0.48, -14.4, 0.32
        ),
    )

    def __init__(self, sample_rate, ceremony_state, *, seed=602900):
        self.sample_rate = float(sample_rate)
        self.ceremony_state = ceremony_state
        self.rng = np.random.default_rng(seed)
        homes = (
            np.array([0.0, -0.70, -2.45]),
            np.array([-1.05, -0.22, -1.95]),
            np.array([0.95, 0.20, -1.62]),
            np.array([0.42, 0.78, -1.35]),
        )

        self.voices = []
        for i, p in enumerate(self.PROFILES):
            s = SingingBowlState(
                SingingBowlSpec(
                    fundamental_hz=p.fundamental_hz,
                    decay_seconds=p.decay_seconds,
                    brightness=p.brightness,
                    asymmetry=p.asymmetry,
                    coupling=p.coupling,
                    body=p.body,
                    output_gain_db=p.output_gain_db,
                )
            )
            g = TibetanSingingBowlGenerator(
                self.sample_rate, s, seed=seed + 100 + i * 31
            )
            self.voices.append(
                BowlVoice(
                    p, s, g, homes[i].astype(np.float64),
                    1.8 + i * 1.5
                )
            )

        self.elapsed_seconds = 0.0
        self.phase = "arrival"
        self.phase_progress = 0.0
        self.performance_progress = 0.0
        self.running = False
        self.complete = False
        self._opening_large = False
        self._opening_mid = False
        self._closing_cues = set()

        self.dynamic_wander = OrganicWanderer1D(
            seed=seed + 40,
            natural_period_seconds=48.0,
            damping_ratio=0.68,
            drive_strength=0.82,
            drive_smoothing_seconds=14.0,
        )
        self.rub_wander = OrganicWanderer1D(
            seed=seed + 41,
            natural_period_seconds=61.0,
            damping_ratio=0.76,
            drive_strength=0.72,
            drive_smoothing_seconds=17.0,
        )

    def restart(self):
        self.elapsed_seconds = 0.0
        self.phase = "arrival"
        self.phase_progress = 0.0
        self.performance_progress = 0.0
        self.running = True
        self.complete = False
        self._opening_large = False
        self._opening_mid = False
        self._closing_cues.clear()

        for i, v in enumerate(self.voices):
            v.generator.clear()
            v.state.update(rub_level=0.0)
            v.next_strike_seconds = 8.0 + i * 3.0
            v.current_rub = 0.0
            v.strikes = 0

    def stop(self):
        self.running = False
        for v in self.voices:
            v.state.update(rub_level=0.0)

    def _phase_for_progress(self, p):
        phases = (
            ("arrival", 0.00, 0.08),
            ("grounding", 0.08, 0.22),
            ("opening", 0.22, 0.42),
            ("immersion", 0.42, 0.72),
            ("integration", 0.72, 0.88),
            ("closing", 0.88, 0.96),
            ("final silence", 0.96, 1.00),
        )
        for name, start, end in phases:
            if p < end:
                return name, float(
                    np.clip((p - start) / (end - start), 0, 1)
                )
        return "complete", 1.0

    def _energy(self):
        p = self.phase_progress
        if self.phase == "arrival":
            return 0.18 + 0.15 * p
        if self.phase == "grounding":
            return 0.34 + 0.18 * p
        if self.phase == "opening":
            return 0.50 + 0.24 * p
        if self.phase == "immersion":
            return 0.78 + 0.16 * math.sin(math.pi * p)
        if self.phase == "integration":
            return 0.68 - 0.28 * p
        if self.phase == "closing":
            return 0.36 - 0.24 * p
        return 0.0

    def _weight(self, i):
        p = self.phase_progress
        if self.phase == "arrival":
            return (1.0, 0.0, 0.0, 0.0)[i]
        if self.phase == "grounding":
            return (1.0, 0.64, 0.08, 0.0)[i]
        if self.phase == "opening":
            return (0.96, 0.88, 0.62 + 0.24 * p, 0.22 + 0.50 * p)[i]
        if self.phase == "immersion":
            return (0.90, 1.0, 0.96, 0.82)[i]
        if self.phase == "integration":
            return (0.92, 0.74, 0.58, 0.38)[i]
        if self.phase == "closing":
            return (1.0, 0.18 * (1.0 - p), 0.08 * (1.0 - p), 0.0)[i]
        return 0.0

    def _next_strike(self, voice, energy, weight, intensity):
        size = voice.profile.size_class
        effective = max(
            0.05,
            energy * weight * (0.60 + 0.65 * intensity),
        )
        low = (4.0 + 10.0 * size) / (0.42 + effective)
        high = (10.0 + 26.0 * size) / (0.40 + effective)
        low = float(np.clip(low, 3.5, 30.0))
        high = float(np.clip(high, low + 2.0, 65.0))
        return float(
            math.exp(
                self.rng.uniform(math.log(low), math.log(high))
            )
        )

    def advance(self, dt):
        spec = self.ceremony_state.get()
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
            + 0.16 * (0.5 + 0.5 * self.dynamic_wander.advance(dt))
        )
        energy = float(np.clip(energy, 0, 1))
        rub_shape = 0.5 + 0.5 * self.rub_wander.advance(dt)

        if (
            self.phase == "arrival"
            and not self._opening_large
            and self.elapsed_seconds >= 1.5
        ):
            self.voices[0].generator.strike(0.68)
            self.voices[0].strikes += 1
            self._opening_large = True

        if (
            self.phase == "arrival"
            and not self._opening_mid
            and self.elapsed_seconds >= 5.0
        ):
            self.voices[2].generator.strike(0.36)
            self.voices[2].strikes += 1
            self._opening_mid = True

        for i, v in enumerate(self.voices):
            weight = self._weight(i)

            if self.phase in {"arrival", "final silence"}:
                rub_phase = 0.0
            elif self.phase == "grounding":
                rub_phase = 0.18
            elif self.phase == "opening":
                rub_phase = 0.42
            elif self.phase == "immersion":
                rub_phase = 0.70
            elif self.phase == "integration":
                rub_phase = 0.40
            else:
                rub_phase = 0.10 * (1.0 - self.phase_progress)

            rub_pref = (0.34, 0.92, 1.0, 0.66)[i]
            target = float(np.clip(
                spec.rubbing
                * rub_phase
                * rub_pref
                * weight
                * energy
                * (0.34 + 0.66 * rub_shape),
                0, 0.82
            ))

            v.current_rub += (
                target - v.current_rub
            ) * (
                1.0
                - math.exp(
                    -dt / (1.0 if target > v.current_rub else 2.8)
                )
            )

            v.state.update(
                rub_level=v.current_rub,
                rub_pressure=float(np.clip(
                    0.44 + 0.30 * energy + 0.12 * rub_shape,
                    0, 1
                )),
                rub_speed=float(np.clip(
                    0.34
                    + 0.30 * rub_shape
                    + 0.08 * (1.0 - v.profile.size_class),
                    0, 1
                )),
            )

            if self.phase in {"closing", "final silence"}:
                continue

            v.next_strike_seconds -= dt
            if (
                v.next_strike_seconds <= 0.0
                and weight > 0.05
            ):
                strength = float(np.clip(
                    self.rng.normal(
                        0.22 + 0.52 * energy, 0.09
                    ),
                    0.14, 0.90
                ))
                v.generator.strike(strength)
                v.strikes += 1
                v.next_strike_seconds = self._next_strike(
                    v, energy, weight, spec.intensity
                )

        if self.phase == "closing":
            cues = (
                (0.10, 0.46),
                (0.43, 0.34),
                (0.73, 0.24),
            )
            for cue_index, (cue, strength) in enumerate(cues):
                if (
                    self.phase_progress >= cue
                    and cue_index not in self._closing_cues
                ):
                    self.voices[0].generator.strike(strength)
                    self.voices[0].strikes += 1
                    self._closing_cues.add(cue_index)

    def render_mono(self, frame_count):
        return [v.generator.generate(frame_count) for v in self.voices]

    @property
    def remaining_seconds(self):
        return max(
            0.0,
            self.ceremony_state.get().duration_minutes * 60.0
            - self.elapsed_seconds,
        )
