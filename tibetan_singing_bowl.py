from __future__ import annotations

import math
import threading
from dataclasses import dataclass, replace

import numpy as np

from synthesized_sound_source import OrganicWanderer1D, SmoothedValue, db_to_linear


@dataclass(frozen=True, slots=True)
class SingingBowlSpec:
    """Artistically controllable modal singing-bowl model."""

    fundamental_hz: float = 185.0
    decay_seconds: float = 14.0
    strike_strength: float = 0.62
    brightness: float = 0.48
    inharmonicity: float = 0.42
    beating: float = 0.34
    body: float = 0.68
    rub_level: float = 0.0
    rub_motion: float = 0.38
    output_gain_db: float = -12.0

    def validated(self) -> "SingingBowlSpec":
        if not 70.0 <= self.fundamental_hz <= 700.0:
            raise ValueError("fundamental_hz must be between 70 and 700")
        if not 1.0 <= self.decay_seconds <= 60.0:
            raise ValueError("decay_seconds must be between 1 and 60")
        for name in (
            "strike_strength", "brightness", "inharmonicity", "beating",
            "body", "rub_level", "rub_motion",
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if not -48.0 <= self.output_gain_db <= 6.0:
            raise ValueError("output_gain_db must be between -48 and +6")
        return self


class SingingBowlState:
    def __init__(self, spec: SingingBowlSpec) -> None:
        self._lock = threading.Lock()
        self._spec = spec.validated()

    def get(self) -> SingingBowlSpec:
        with self._lock:
            return self._spec

    def set(self, spec: SingingBowlSpec) -> None:
        with self._lock:
            self._spec = spec.validated()

    def update(self, **changes) -> None:
        with self._lock:
            self._spec = replace(self._spec, **changes).validated()


@dataclass(slots=True)
class _StrikeEvent:
    age_seconds: float
    strength: float
    mode_phase: np.ndarray
    pitch_scale: float


class TibetanSingingBowlGenerator:
    """
    Mono procedural singing bowl.

    Strikes excite a bank of slowly decaying inharmonic modes. Rubbing injects
    a continuous low-level excitation into the same modal family. Closely split
    modes create controllable acoustic beating without looping a sample.
    """

    BASE_MODE_RATIOS = np.array(
        [1.00, 2.01, 2.98, 4.12, 5.37, 6.84, 8.36, 9.93],
        dtype=np.float64,
    )
    MODE_DETUNE_SHAPE = np.array(
        [0.00, 0.02, -0.03, 0.055, -0.045, 0.072, -0.062, 0.085],
        dtype=np.float64,
    )
    MODE_AMPLITUDES = np.array(
        [1.00, 0.72, 0.54, 0.41, 0.30, 0.22, 0.16, 0.11],
        dtype=np.float64,
    )
    MODE_DECAY_MULTIPLIERS = np.array(
        [1.00, 0.93, 0.82, 0.74, 0.66, 0.58, 0.50, 0.44],
        dtype=np.float64,
    )

    def __init__(
        self,
        sample_rate: float,
        state: SingingBowlState,
        *,
        seed: int = 602_701,
    ) -> None:
        self.sample_rate = float(sample_rate)
        self.state = state
        self.rng = np.random.default_rng(seed)
        self.events: list[_StrikeEvent] = []

        self.rub_phases = self.rng.uniform(
            0.0, 2.0 * math.pi, len(self.BASE_MODE_RATIOS)
        )
        self.rub_split_phases = self.rng.uniform(
            0.0, 2.0 * math.pi, len(self.BASE_MODE_RATIOS)
        )
        self.rub_energy = 0.0

        spec = state.get()
        self.gain_smoother = SmoothedValue(
            self.sample_rate,
            db_to_linear(spec.output_gain_db),
            0.12,
        )
        self.rub_smoother = SmoothedValue(
            self.sample_rate,
            spec.rub_level,
            0.35,
        )

        self.rub_wander = OrganicWanderer1D(
            seed=seed + 100,
            natural_period_seconds=8.0,
            damping_ratio=0.62,
            drive_strength=0.85,
            drive_smoothing_seconds=2.8,
        )
        self.pitch_wander = OrganicWanderer1D(
            seed=seed + 101,
            natural_period_seconds=17.0,
            damping_ratio=0.78,
            drive_strength=0.35,
            drive_smoothing_seconds=5.5,
        )

        self._transient_state = 0.0

    @staticmethod
    def _mode_frequencies(
        spec: SingingBowlSpec,
        pitch_scale: float = 1.0,
    ) -> np.ndarray:
        ratios = (
            TibetanSingingBowlGenerator.BASE_MODE_RATIOS
            * (
                1.0
                + spec.inharmonicity
                * TibetanSingingBowlGenerator.MODE_DETUNE_SHAPE
            )
        )
        return spec.fundamental_hz * pitch_scale * ratios

    def strike(self, strength: float | None = None) -> None:
        spec = self.state.get()
        if strength is None:
            strength = spec.strike_strength
        strength = float(np.clip(strength, 0.0, 1.5))

        pitch_scale = float(
            np.clip(
                self.rng.normal(1.0, 0.004 + 0.010 * spec.beating),
                0.965,
                1.035,
            )
        )
        phases = self.rng.uniform(
            -0.18, 0.18, len(self.BASE_MODE_RATIOS)
        )
        self.events.append(
            _StrikeEvent(
                age_seconds=0.0,
                strength=strength,
                mode_phase=phases,
                pitch_scale=pitch_scale,
            )
        )
        self._transient_state += 0.35 * strength

    def clear(self) -> None:
        self.events.clear()
        self.rub_energy = 0.0
        self._transient_state = 0.0

    def _render_strike(
        self,
        event: _StrikeEvent,
        spec: SingingBowlSpec,
        frame_count: int,
    ) -> tuple[np.ndarray, float]:
        times = (
            np.arange(frame_count, dtype=np.float64) / self.sample_rate
            + event.age_seconds
        )
        frequencies = self._mode_frequencies(spec, event.pitch_scale)

        brightness_tilt = np.linspace(
            1.0 - 0.18 * spec.brightness,
            0.22 + 1.15 * spec.brightness,
            len(frequencies),
        )
        body_tilt = np.linspace(
            1.15 + 0.55 * spec.body,
            0.78 - 0.28 * spec.body,
            len(frequencies),
        )

        output = np.zeros(frame_count, dtype=np.float64)
        for index, frequency in enumerate(frequencies):
            decay = max(
                0.25,
                spec.decay_seconds * self.MODE_DECAY_MULTIPLIERS[index],
            )
            envelope = np.exp(-times / decay)
            amplitude = (
                event.strength
                * self.MODE_AMPLITUDES[index]
                * brightness_tilt[index]
                * body_tilt[index]
            )

            split_cents = (
                1.2
                + 8.0 * spec.beating
                * (0.35 + index / max(1, len(frequencies) - 1))
            )
            split_ratio = 2.0 ** (split_cents / 1200.0)
            main = np.sin(
                2.0 * math.pi * frequency * times
                + event.mode_phase[index]
            )
            split = np.sin(
                2.0 * math.pi * frequency * split_ratio * times
                - 0.73 * event.mode_phase[index]
            )

            split_mix = 0.10 + 0.26 * spec.beating
            output += (
                amplitude
                * envelope
                * ((1.0 - split_mix) * main + split_mix * split)
            )

        return output, event.age_seconds + frame_count / self.sample_rate

    def _render_rub(
        self,
        spec: SingingBowlSpec,
        frame_count: int,
    ) -> np.ndarray:
        rub_level = self.rub_smoother.ramp(spec.rub_level, frame_count)
        if float(np.max(rub_level)) <= 1.0e-6 and self.rub_energy < 1.0e-5:
            return np.zeros(frame_count, dtype=np.float64)

        elapsed = frame_count / self.sample_rate
        wander = self.rub_wander.advance(
            elapsed * (0.35 + 2.2 * spec.rub_motion)
        )
        pitch_wander = self.pitch_wander.advance(elapsed)

        target_energy = float(
            np.mean(rub_level)
            * (0.70 + 0.30 * (0.5 + 0.5 * wander))
        )
        catch_time = 1.4 - 1.0 * spec.rub_motion
        amount = 1.0 - math.exp(-elapsed / max(0.12, catch_time))
        self.rub_energy += (target_energy - self.rub_energy) * amount

        frequencies = self._mode_frequencies(
            spec,
            pitch_scale=1.0 + 0.0015 * pitch_wander,
        )
        sample_indices = np.arange(frame_count, dtype=np.float64)
        output = np.zeros(frame_count, dtype=np.float64)
        rub_mode_weights = np.array(
            [1.0, 0.82, 0.62, 0.43, 0.30, 0.20, 0.13, 0.08],
            dtype=np.float64,
        )
        high_tilt = np.linspace(
            0.75,
            0.25 + 1.25 * spec.brightness,
            len(frequencies),
        )

        for index, frequency in enumerate(frequencies):
            phase_step = 2.0 * math.pi * frequency / self.sample_rate
            phases = self.rub_phases[index] + phase_step * sample_indices

            split_cents = 0.8 + 5.5 * spec.beating
            split_frequency = frequency * (2.0 ** (split_cents / 1200.0))
            split_step = 2.0 * math.pi * split_frequency / self.sample_rate
            split_phases = (
                self.rub_split_phases[index] + split_step * sample_indices
            )

            split_mix = 0.08 + 0.18 * spec.beating
            component = (
                (1.0 - split_mix) * np.sin(phases)
                + split_mix * np.sin(split_phases)
            )
            output += (
                self.rub_energy
                * rub_mode_weights[index]
                * high_tilt[index]
                * component
            )

            self.rub_phases[index] = float(
                (phases[-1] + phase_step) % (2.0 * math.pi)
            )
            self.rub_split_phases[index] = float(
                (split_phases[-1] + split_step) % (2.0 * math.pi)
            )

        noise = self.rng.standard_normal(frame_count)
        kernel = np.ones(8, dtype=np.float64) / 8.0
        roughness = noise - np.convolve(noise, kernel, mode="same")
        output += (
            roughness
            * self.rub_energy
            * (0.002 + 0.012 * spec.brightness)
        )
        return output

    def generate(self, frame_count: int) -> np.ndarray:
        spec = self.state.get()
        frame_count = int(frame_count)
        if frame_count <= 0:
            return np.zeros(0, dtype=np.float32)

        output = np.zeros(frame_count, dtype=np.float64)
        retained: list[_StrikeEvent] = []

        for event in self.events:
            rendered, new_age = self._render_strike(
                event, spec, frame_count
            )
            output += rendered
            event.age_seconds = new_age
            if event.age_seconds < spec.decay_seconds * 6.0:
                retained.append(event)
        self.events = retained

        output += self._render_rub(spec, frame_count)

        transient = np.empty(frame_count, dtype=np.float64)
        value = self._transient_state
        decay = math.exp(-1.0 / (0.010 * self.sample_rate))
        for i in range(frame_count):
            transient[i] = value
            value *= decay
        self._transient_state = value

        if float(np.max(np.abs(transient))) > 1.0e-8:
            contact = self.rng.standard_normal(frame_count)
            output += 0.055 * transient * contact

        gain = self.gain_smoother.ramp(
            db_to_linear(spec.output_gain_db),
            frame_count,
        )
        output *= gain
        output = 0.92 * np.tanh(output * 0.72)
        return output.astype(np.float32, copy=False)


@dataclass(frozen=True, slots=True)
class BowlCeremonySpec:
    enabled: bool = False
    activity: float = 0.35
    evolution: float = 0.45
    strike_probability: float = 0.65
    rub_probability: float = 0.35

    def validated(self) -> "BowlCeremonySpec":
        for name in (
            "activity", "evolution", "strike_probability", "rub_probability"
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        return self


class BowlCeremonyState:
    def __init__(self, spec: BowlCeremonySpec) -> None:
        self._lock = threading.Lock()
        self._spec = spec.validated()

    def get(self) -> BowlCeremonySpec:
        with self._lock:
            return self._spec

    def update(self, **changes) -> None:
        with self._lock:
            self._spec = replace(self._spec, **changes).validated()


class BowlCeremonyController:
    """Slow, non-grid generative behavior for the bowl instrument."""

    def __init__(
        self,
        bowl_state: SingingBowlState,
        ceremony_state: BowlCeremonyState,
        *,
        seed: int = 602_900,
    ) -> None:
        self.bowl_state = bowl_state
        self.ceremony_state = ceremony_state
        self.rng = np.random.default_rng(seed)

        self.activity_wander = OrganicWanderer1D(
            seed=seed,
            natural_period_seconds=38.0,
            damping_ratio=0.66,
            drive_strength=0.95,
            drive_smoothing_seconds=11.0,
        )
        self.rub_wander = OrganicWanderer1D(
            seed=seed + 1,
            natural_period_seconds=52.0,
            damping_ratio=0.72,
            drive_strength=0.85,
            drive_smoothing_seconds=15.0,
        )

        self.elapsed_to_next_strike = 1.5
        self.current_activity = 0.0
        self.current_rub = 0.0

    def _schedule_next_strike(self, activity: float) -> None:
        minimum = 2.8 + (1.0 - activity) * 8.0
        maximum = 10.0 + (1.0 - activity) * 45.0
        self.elapsed_to_next_strike = float(
            math.exp(
                self.rng.uniform(math.log(minimum), math.log(maximum))
            )
        )

    def advance(
        self,
        elapsed_seconds: float,
        generator: TibetanSingingBowlGenerator,
    ) -> None:
        spec = self.ceremony_state.get()
        if not spec.enabled:
            return

        speed = 0.25 + 2.5 * spec.evolution
        activity_shape = 0.5 + 0.5 * self.activity_wander.advance(
            elapsed_seconds * speed
        )
        rub_shape = 0.5 + 0.5 * self.rub_wander.advance(
            elapsed_seconds * speed * 0.72
        )

        activity = float(
            np.clip(
                0.15 * spec.activity
                + 0.85 * spec.activity * activity_shape,
                0.0,
                1.0,
            )
        )
        self.current_activity = activity

        self.elapsed_to_next_strike -= elapsed_seconds
        if self.elapsed_to_next_strike <= 0.0:
            if self.rng.random() < spec.strike_probability:
                generator.strike(
                    float(
                        np.clip(
                            self.rng.normal(
                                0.38 + 0.52 * activity,
                                0.12,
                            ),
                            0.16,
                            1.0,
                        )
                    )
                )
            self._schedule_next_strike(activity)

        target_rub = (
            spec.rub_probability
            * activity
            * (rub_shape ** 1.6)
        )
        self.current_rub = float(np.clip(target_rub, 0.0, 1.0))

        bowl_spec = self.bowl_state.get()
        self.bowl_state.set(
            replace(
                bowl_spec,
                rub_level=self.current_rub,
            ).validated()
        )
