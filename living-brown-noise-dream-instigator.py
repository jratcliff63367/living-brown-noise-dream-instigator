from __future__ import annotations

import json
import math
import sys
import threading
import time
import wave
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import ClassVar

import numpy as np
import sounddevice as sd
from scipy import signal
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


# =============================================================================
# Organic motion generator
# =============================================================================

@dataclass(frozen=True, slots=True)
class OrganicMotionSpec:
    """
    Settings for a bounded stochastic spring.

    natural_period_seconds:
        The approximate period the system would have if disturbed and then
        left alone. Lower values react quickly; higher values feel heavier.

    damping_ratio:
        1.0 is critically damped. Lower values permit more overshoot and sway.
        Higher values settle more directly.

    drive_strength:
        Strength of the slowly changing random force.

    drive_smoothing_seconds:
        How rapidly the random force itself changes.

    soft_limit:
        Controls the gentle compression of raw spring position into [-1, +1].
    """

    natural_period_seconds: float = 2.5
    damping_ratio: float = 0.72
    drive_strength: float = 1.10
    drive_smoothing_seconds: float = 0.90
    soft_limit: float = 1.35

    def validated(self) -> OrganicMotionSpec:
        if not 0.05 <= self.natural_period_seconds <= 120.0:
            raise ValueError(
                "natural_period_seconds must be between 0.05 and 120"
            )
        if not 0.05 <= self.damping_ratio <= 4.0:
            raise ValueError("damping_ratio must be between 0.05 and 4")
        if not 0.0 <= self.drive_strength <= 10.0:
            raise ValueError("drive_strength must be between 0 and 10")
        if not 0.01 <= self.drive_smoothing_seconds <= 120.0:
            raise ValueError(
                "drive_smoothing_seconds must be between 0.01 and 120"
            )
        if not 0.1 <= self.soft_limit <= 10.0:
            raise ValueError("soft_limit must be between 0.1 and 10")
        return self


class OrganicMotionState:
    """Thread-safe live organic-motion settings."""

    def __init__(self, spec: OrganicMotionSpec) -> None:
        self._lock = threading.Lock()
        self._spec = spec.validated()

    def get(self) -> OrganicMotionSpec:
        with self._lock:
            return self._spec

    def set(self, spec: OrganicMotionSpec) -> None:
        with self._lock:
            self._spec = spec.validated()

    def update(self, **changes: float) -> None:
        with self._lock:
            self._spec = replace(
                self._spec,
                **changes,
            ).validated()


class OrganicMotion1D:
    """
    A second-order stochastic system with inertia, damping and a smoothly
    wandering random force.

    Unlike Perlin noise, it owns velocity and momentum. It can overshoot,
    hesitate, settle and reverse direction in a physically continuous way.
    """

    def __init__(
        self,
        motion_state: OrganicMotionState,
        seed: int = 12345,
    ) -> None:
        self.motion_state = motion_state
        self.rng = np.random.default_rng(seed)

        self.position = 0.0
        self.velocity = 0.0
        self.drive = 0.0

    def advance(self, elapsed_seconds: float) -> tuple[float, float]:
        """
        Advance the simulation and return start/end values in [-1, +1].
        """
        spec = self.motion_state.get()
        start_value = math.tanh(
            self.position / spec.soft_limit
        )

        remaining = max(0.0, float(elapsed_seconds))
        maximum_step = 1.0 / 120.0

        omega = 2.0 * math.pi / spec.natural_period_seconds
        spring_k = omega * omega
        damping_c = 2.0 * spec.damping_ratio * omega

        while remaining > 0.0:
            dt = min(maximum_step, remaining)
            remaining -= dt

            # Smooth random force rather than independent impulses.
            drive_decay = math.exp(
                -dt / spec.drive_smoothing_seconds
            )
            drive_variance = math.sqrt(
                max(0.0, 1.0 - drive_decay * drive_decay)
            )
            self.drive = (
                self.drive * drive_decay
                + float(self.rng.standard_normal()) * drive_variance
            )

            acceleration = (
                -spring_k * self.position
                - damping_c * self.velocity
                + spec.drive_strength * spring_k * self.drive
            )

            # Semi-implicit Euler integration.
            self.velocity += acceleration * dt
            self.position += self.velocity * dt

        end_value = math.tanh(
            self.position / spec.soft_limit
        )

        return start_value, end_value


# =============================================================================
# Breath
# =============================================================================

@dataclass(frozen=True, slots=True)
class BreathSpec:
    """
    Biological breath-state parameters.

    The four stage means describe a slow, relaxed resting cycle. Timing varies
    modestly from breath to breath, but long-term Breath Evolution does not
    drive respiratory rate; it primarily changes how prominent the complete
    breath effect is in the mix.

    Rare event probabilities are evaluated once per complete breath cycle.
    """

    inhale_mean_seconds: float = 1.30
    hold_mean_seconds: float = 0.10
    exhale_mean_seconds: float = 2.30
    rest_mean_seconds: float = 0.80

    timing_variation: float = 0.08
    timing_memory: float = 0.82

    depth_variation: float = 0.12
    depth_memory: float = 0.75

    deep_breath_probability: float = 0.012
    deep_breath_scale: float = 1.45

    long_rest_probability: float = 0.008
    long_rest_scale: float = 2.20

    shallow_breath_probability: float = 0.020
    shallow_breath_scale: float = 0.72

    gain_range_db: float = 4.5
    spectral_depth: float = 0.35
    width_depth: float = 0.18

    def validated(self) -> BreathSpec:
        for name, value in (
            ("inhale_mean_seconds", self.inhale_mean_seconds),
            ("hold_mean_seconds", self.hold_mean_seconds),
            ("exhale_mean_seconds", self.exhale_mean_seconds),
            ("rest_mean_seconds", self.rest_mean_seconds),
        ):
            if not 0.01 <= value <= 60.0:
                raise ValueError(
                    f"{name} must be between 0.01 and 60 seconds"
                )

        for name, value in (
            ("timing_variation", self.timing_variation),
            ("depth_variation", self.depth_variation),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

        for name, value in (
            ("timing_memory", self.timing_memory),
            ("depth_memory", self.depth_memory),
        ):
            if not 0.0 <= value <= 0.99:
                raise ValueError(f"{name} must be between 0 and 0.99")

        for name, value in (
            ("deep_breath_probability", self.deep_breath_probability),
            ("long_rest_probability", self.long_rest_probability),
            ("shallow_breath_probability", self.shallow_breath_probability),
        ):
            if not 0.0 <= value <= 0.25:
                raise ValueError(f"{name} must be between 0 and 0.25")

        for name, value in (
            ("deep_breath_scale", self.deep_breath_scale),
            ("long_rest_scale", self.long_rest_scale),
            ("shallow_breath_scale", self.shallow_breath_scale),
        ):
            if not 0.1 <= value <= 5.0:
                raise ValueError(f"{name} must be between 0.1 and 5")

        if not 0.0 <= self.gain_range_db <= 12.0:
            raise ValueError("gain_range_db must be between 0 and 12")
        if not 0.0 <= self.spectral_depth <= 1.0:
            raise ValueError("spectral_depth must be between 0 and 1")
        if not 0.0 <= self.width_depth <= 1.0:
            raise ValueError("width_depth must be between 0 and 1")

        return self


class BreathState:
    """Thread-safe live breath specification shared by GUI and audio engine."""

    def __init__(self, spec: BreathSpec) -> None:
        self._lock = threading.Lock()
        self._spec = spec.validated()
        self._version = 0

    def get(self) -> tuple[BreathSpec, int]:
        with self._lock:
            return self._spec, self._version

    def set(self, spec: BreathSpec) -> None:
        spec = spec.validated()
        with self._lock:
            self._spec = spec
            self._version += 1

    def update(self, **changes: float) -> None:
        with self._lock:
            updated = replace(self._spec, **changes).validated()
            self._spec = updated
            self._version += 1


class BreathEnvelope:
    """
    Explicit inhale/hold/exhale/rest state machine.

    Cycle timing and depth use correlated random walks, so adjacent breaths
    resemble one another. Rare events add deep breaths, shallow breaths and
    longer rests without imposing a repeating pattern.
    """

    STAGE_INHALE = "inhale"
    STAGE_HOLD = "hold"
    STAGE_EXHALE = "exhale"
    STAGE_REST = "rest"

    def __init__(
        self,
        sample_rate: float,
        breath_state: BreathState,
        seed: int = 112233,
    ) -> None:
        self.sample_rate = float(sample_rate)
        self.breath_state = breath_state
        self.rng = np.random.default_rng(seed)

        self.stage = self.STAGE_INHALE
        self.stage_position_samples = 0
        self._seen_spec_version = -1

        self._timing_latent = 0.0
        self._depth_latent = 0.0
        self._cycle_timing_scale = 1.0
        self._cycle_depth_scale = 1.0
        self._cycle_rest_scale = 1.0
        self.current_event = "normal"

        spec, version = self.breath_state.get()
        self._seen_spec_version = version
        self._choose_new_cycle(spec)
        self.stage_duration_samples = self._duration_for_stage(
            self.stage,
            spec,
        )

        self.current_value = 0.0

    @staticmethod
    def _correlated_step(
        previous: float,
        memory: float,
        rng: np.random.Generator,
    ) -> float:
        innovation_scale = math.sqrt(
            max(0.0, 1.0 - memory * memory)
        )
        return (
            memory * previous
            + innovation_scale * float(rng.standard_normal())
        )

    def _choose_new_cycle(self, spec: BreathSpec) -> None:
        self._timing_latent = self._correlated_step(
            self._timing_latent,
            spec.timing_memory,
            self.rng,
        )
        self._depth_latent = self._correlated_step(
            self._depth_latent,
            spec.depth_memory,
            self.rng,
        )

        self._cycle_timing_scale = float(
            np.clip(
                math.exp(
                    self._timing_latent * spec.timing_variation
                ),
                0.45,
                2.2,
            )
        )
        self._cycle_depth_scale = float(
            np.clip(
                math.exp(
                    self._depth_latent * spec.depth_variation
                ),
                0.55,
                1.65,
            )
        )
        self._cycle_rest_scale = 1.0
        self.current_event = "normal"

        event_roll = float(self.rng.random())

        if event_roll < spec.deep_breath_probability:
            self._cycle_timing_scale *= spec.deep_breath_scale
            self._cycle_depth_scale *= spec.deep_breath_scale
            self.current_event = "deep"

        elif event_roll < (
            spec.deep_breath_probability
            + spec.long_rest_probability
        ):
            self._cycle_rest_scale *= spec.long_rest_scale
            self.current_event = "long rest"

        elif event_roll < (
            spec.deep_breath_probability
            + spec.long_rest_probability
            + spec.shallow_breath_probability
        ):
            self._cycle_timing_scale *= spec.shallow_breath_scale
            self._cycle_depth_scale *= spec.shallow_breath_scale
            self.current_event = "shallow"

        self._cycle_timing_scale = float(
            np.clip(self._cycle_timing_scale, 0.35, 3.0)
        )
        self._cycle_depth_scale = float(
            np.clip(self._cycle_depth_scale, 0.35, 1.8)
        )

    def _duration_for_stage(
        self,
        stage: str,
        spec: BreathSpec,
    ) -> int:
        if stage == self.STAGE_INHALE:
            seconds = (
                spec.inhale_mean_seconds
                * self._cycle_timing_scale
            )
        elif stage == self.STAGE_HOLD:
            seconds = (
                spec.hold_mean_seconds
                * self._cycle_timing_scale
            )
        elif stage == self.STAGE_EXHALE:
            seconds = (
                spec.exhale_mean_seconds
                * self._cycle_timing_scale
            )
        elif stage == self.STAGE_REST:
            seconds = (
                spec.rest_mean_seconds
                * self._cycle_timing_scale
                * self._cycle_rest_scale
            )
        else:
            raise ValueError(f"Unknown breath stage: {stage}")

        # Tiny per-stage variation prevents every phase from scaling in
        # perfect lockstep while retaining the same overall cycle identity.
        seconds *= float(
            np.clip(
                math.exp(float(self.rng.normal(0.0, 0.035))),
                0.88,
                1.14,
            )
        )

        return max(1, int(max(0.01, seconds) * self.sample_rate))

    def _apply_live_timing_change(
        self,
        spec: BreathSpec,
        version: int,
    ) -> None:
        if version == self._seen_spec_version:
            return

        old_duration = max(1, self.stage_duration_samples)
        progress = min(
            1.0,
            self.stage_position_samples / old_duration,
        )

        self.stage_duration_samples = self._duration_for_stage(
            self.stage,
            spec,
        )
        self.stage_position_samples = int(
            progress * self.stage_duration_samples
        )
        self._seen_spec_version = version

    def _advance_stage(self, spec: BreathSpec) -> None:
        if self.stage == self.STAGE_INHALE:
            self.stage = self.STAGE_HOLD
        elif self.stage == self.STAGE_HOLD:
            self.stage = self.STAGE_EXHALE
        elif self.stage == self.STAGE_EXHALE:
            self.stage = self.STAGE_REST
        else:
            self.stage = self.STAGE_INHALE
            self._choose_new_cycle(spec)

        self.stage_position_samples = 0
        self.stage_duration_samples = self._duration_for_stage(
            self.stage,
            spec,
        )

    @staticmethod
    def _inhale_curve(x: np.ndarray) -> np.ndarray:
        # Gentle start, fuller finish.
        return np.power(
            0.5 - 0.5 * np.cos(np.pi * x),
            1.10,
        )

    @staticmethod
    def _exhale_curve(x: np.ndarray) -> np.ndarray:
        # Relaxed release: initially faster, with a long soft tail.
        smooth = 0.5 - 0.5 * np.cos(np.pi * x)
        return 1.0 - np.power(smooth, 0.78)

    def generate(self, frame_count: int) -> np.ndarray:
        spec, version = self.breath_state.get()
        self._apply_live_timing_change(spec, version)

        output = np.empty(frame_count, dtype=np.float32)
        write_position = 0

        while write_position < frame_count:
            remaining = (
                self.stage_duration_samples
                - self.stage_position_samples
            )
            chunk_size = min(frame_count - write_position, remaining)

            start_fraction = (
                self.stage_position_samples
                / self.stage_duration_samples
            )
            end_fraction = (
                self.stage_position_samples + chunk_size
            ) / self.stage_duration_samples

            fractions = np.linspace(
                start_fraction,
                end_fraction,
                chunk_size,
                endpoint=False,
                dtype=np.float64,
            )

            if self.stage == self.STAGE_INHALE:
                values = self._inhale_curve(fractions)
            elif self.stage == self.STAGE_HOLD:
                values = np.ones(chunk_size, dtype=np.float64)
            elif self.stage == self.STAGE_EXHALE:
                values = self._exhale_curve(fractions)
            else:
                values = np.zeros(chunk_size, dtype=np.float64)

            values *= self._cycle_depth_scale
            np.clip(values, 0.0, 1.8, out=values)

            output[
                write_position:write_position + chunk_size
            ] = values

            write_position += chunk_size
            self.stage_position_samples += chunk_size

            if self.stage_position_samples >= self.stage_duration_samples:
                spec, version = self.breath_state.get()
                self._seen_spec_version = version
                self._advance_stage(spec)

        self.current_value = float(output[-1])
        return output



# =============================================================================
# Breath prominence evolution
# =============================================================================

@dataclass(frozen=True, slots=True)
class BreathEvolutionSpec:
    """
    Slowly evolves only the prominence of the complete breath effect.

    It does not alter inhale, hold, exhale, or rest timing. Those remain under
    the biological state machine, with only modest breath-to-breath variation.

    multiplier_min / multiplier_max:
        Scale applied to gain, spectral and width breath depths.

        0.0 means the breath disappears into the background.
        1.0 means the breath uses the values shown in Breath parameters.
        1.0 means the full configured breath values.

    period_min_seconds / period_max_seconds:
        Duration of one complete low -> high -> low prominence cycle.
        A new duration is chosen for every cycle.

    curve_power:
        Controls how long the breath spends near the quiet end.
        1.0 is a raised cosine. Higher values keep it subdued longer and
        create shorter periods of strong prominence.
    """

    enabled: bool = True
    multiplier_min: float = 0.0
    multiplier_max: float = 1.0
    period_min_seconds: float = 180.0
    period_max_seconds: float = 480.0
    curve_power: float = 1.35

    def validated(self) -> BreathEvolutionSpec:
        if not 0.0 <= self.multiplier_min <= 1.0:
            raise ValueError("multiplier_min must be between 0 and 1")
        if not 0.0 <= self.multiplier_max <= 1.0:
            raise ValueError("multiplier_max must be between 0 and 1")
        if self.multiplier_min > self.multiplier_max:
            raise ValueError(
                "multiplier_min cannot exceed multiplier_max"
            )

        if not 1.0 <= self.period_min_seconds <= 86400.0:
            raise ValueError(
                "period_min_seconds must be between 1 and 86400"
            )
        if not 1.0 <= self.period_max_seconds <= 86400.0:
            raise ValueError(
                "period_max_seconds must be between 1 and 86400"
            )
        if self.period_min_seconds > self.period_max_seconds:
            raise ValueError(
                "period_min_seconds cannot exceed period_max_seconds"
            )

        if not 0.1 <= self.curve_power <= 8.0:
            raise ValueError("curve_power must be between 0.1 and 8")

        return self


class BreathEvolutionState:
    """Thread-safe live breath-evolution settings."""

    def __init__(self, spec: BreathEvolutionSpec) -> None:
        self._lock = threading.Lock()
        self._spec = spec.validated()

    def get(self) -> BreathEvolutionSpec:
        with self._lock:
            return self._spec

    def set(self, spec: BreathEvolutionSpec) -> None:
        with self._lock:
            self._spec = spec.validated()

    def update(self, **changes) -> None:
        with self._lock:
            self._spec = replace(
                self._spec,
                **changes,
            ).validated()


class BreathProminenceOscillator:
    """
    Smooth low -> high -> low oscillator whose period changes each cycle.

    The oscillator is intentionally separate from the biological breath state
    machine. It controls how visible that breath is over much longer spans.
    """

    def __init__(
        self,
        sample_rate: float,
        evolution_state: BreathEvolutionState,
        seed: int = 556677,
    ) -> None:
        self.sample_rate = float(sample_rate)
        self.evolution_state = evolution_state
        self.rng = np.random.default_rng(seed)

        self.phase = 0.0
        self.current_period_seconds = 1.0
        self.current_multiplier = 1.0

        spec = self.evolution_state.get()
        self.current_period_seconds = self._choose_period(spec)

    def _choose_period(self, spec: BreathEvolutionSpec) -> float:
        if spec.period_min_seconds == spec.period_max_seconds:
            return spec.period_min_seconds

        # Log-uniform selection prevents the upper end from dominating when
        # the allowed period range is wide.
        low = math.log(spec.period_min_seconds)
        high = math.log(spec.period_max_seconds)
        return float(math.exp(self.rng.uniform(low, high)))

    @staticmethod
    def _shape(phase: np.ndarray, power: float) -> np.ndarray:
        # phase 0..1 maps to quiet -> strong -> quiet.
        raised_cosine = 0.5 - 0.5 * np.cos(2.0 * np.pi * phase)
        return np.power(raised_cosine, power)

    def generate(self, frame_count: int) -> np.ndarray:
        spec = self.evolution_state.get()

        if not spec.enabled:
            output = np.ones(frame_count, dtype=np.float32)
            self.current_multiplier = 1.0
            return output

        output = np.empty(frame_count, dtype=np.float32)
        write_position = 0

        while write_position < frame_count:
            samples_per_cycle = max(
                1,
                int(self.current_period_seconds * self.sample_rate),
            )

            samples_remaining = max(
                1,
                int((1.0 - self.phase) * samples_per_cycle),
            )

            chunk_size = min(
                frame_count - write_position,
                samples_remaining,
            )

            phase_increment = chunk_size / samples_per_cycle
            phases = np.linspace(
                self.phase,
                self.phase + phase_increment,
                chunk_size,
                endpoint=False,
                dtype=np.float64,
            )

            shaped = self._shape(
                np.mod(phases, 1.0),
                spec.curve_power,
            )

            multipliers = (
                spec.multiplier_min
                + shaped
                * (spec.multiplier_max - spec.multiplier_min)
            )

            output[
                write_position:write_position + chunk_size
            ] = multipliers.astype(np.float32)

            write_position += chunk_size
            self.phase += phase_increment

            if self.phase >= 1.0 - 1e-12:
                self.phase = 0.0
                self.current_period_seconds = self._choose_period(spec)

        self.current_multiplier = float(output[-1])
        return output


# =============================================================================
# Noise source
# =============================================================================

@dataclass(frozen=True, slots=True)
class BrownNoiseSpec:
    """
    Final perceptual brown-noise controls.

    body:
        0.0 maps to the lowest accepted spectral shift (0.50x).
        1.0 maps to the highest accepted spectral shift (2.20x).

    slope_strength:
        Accepted range is intentionally narrow: 0.75 through 1.00.

    low_end_emphasis_db:
        Broad fixed-frequency low shelf from 0 through +8 dB.

    upper_texture:
        Blend of the brighter filtered branch from 0 through 1.
    """

    body: float = 0.50
    slope_strength: float = 1.00
    low_end_emphasis_db: float = 0.0
    upper_texture: float = 0.0

    filter_transition_seconds: float = 0.35

    # Hidden implementation constants defining the established baseline.
    base_highpass_hz: float = 11.0
    base_lowpass_1_hz: float = 32.3
    base_lowpass_2_hz: float = 270.3
    base_lowpass_3_hz: float = 338.1
    base_gain_db: float = 13.25
    bright_filter_scale: float = 1.35
    low_shelf_hz: float = 90.0

    BODY_MIN_SHIFT: ClassVar[float] = 0.50
    BODY_MAX_SHIFT: ClassVar[float] = 2.20
    BODY_CURVE_POWER: ClassVar[float] = 0.72

    def validated(self, sample_rate: float) -> BrownNoiseSpec:
        if not 0.0 <= self.body <= 1.0:
            raise ValueError("body must be between 0 and 1")
        if not 0.75 <= self.slope_strength <= 1.0:
            raise ValueError(
                "slope_strength must be between 0.75 and 1.0"
            )
        if not 0.0 <= self.low_end_emphasis_db <= 8.0:
            raise ValueError(
                "low_end_emphasis_db must be between 0 and 8"
            )
        if not 0.0 <= self.upper_texture <= 1.0:
            raise ValueError("upper_texture must be between 0 and 1")
        if not 0.01 <= self.filter_transition_seconds <= 5.0:
            raise ValueError(
                "filter_transition_seconds must be between 0.01 and 5"
            )
        if sample_rate <= 1000:
            raise ValueError("sample_rate is invalid")
        return self

    @staticmethod
    def _shape_body(body: float) -> float:
        """
        Compress the pathological bottom end while retaining the full control
        range. A power below 1.0 moves low slider values upward, so evolution
        spends less time close to the minimum spectral shift.
        """
        body = float(np.clip(body, 0.0, 1.0))
        return body ** BrownNoiseSpec.BODY_CURVE_POWER

    @property
    def spectral_shift(self) -> float:
        """
        Logarithmic spectral mapping after a gentle low-end compression curve.
        """
        shaped_body = self._shape_body(self.body)
        low = math.log(self.BODY_MIN_SHIFT)
        high = math.log(self.BODY_MAX_SHIFT)
        return math.exp(low + shaped_body * (high - low))

    @property
    def body_compensation_db(self) -> float:
        """
        Psychoacoustic compensation for spectral shift.

        The first low-tail curve was too aggressive: at 0.20x it pushed the
        generator near 38 dB total gain and caused severe hard clipping.

        The usable range now bottoms out at 0.50x and the slider mapping is
        compressed near that boundary. Only a small additional low-tail boost
        remains necessary.
        """
        shift = self.spectral_shift
        octave_term = -5.0 * math.log2(shift)

        if shift < 0.70:
            low_tail = 1.5 * ((0.70 / shift) - 1.0)
            low_tail = min(low_tail, 1.5)
        else:
            low_tail = 0.0

        requested_total_gain = (
            self.base_gain_db
            + octave_term
            + low_tail
            + self.weight_compensation_db
            + self.texture_compensation_db
        )

        max_generator_gain_db = 24.0

        allowed_body_compensation = (
            max_generator_gain_db
            - self.base_gain_db
            - self.weight_compensation_db
            - self.texture_compensation_db
        )

        return min(
            octave_term + low_tail,
            allowed_body_compensation,
        )

    @property
    def weight_compensation_db(self) -> float:
        """
        Increasing the low shelf raises measured level. Subtract enough to
        keep the change primarily about weight rather than loudness.
        """
        return -0.42 * self.low_end_emphasis_db

    @property
    def texture_compensation_db(self) -> float:
        """
        Full upper-texture blend was roughly a couple dB louder in testing.
        """
        return -2.0 * self.upper_texture

    @property
    def compensated_gain_db(self) -> float:
        return (
            self.base_gain_db
            + self.body_compensation_db
            + self.weight_compensation_db
            + self.texture_compensation_db
        )


class BrownNoiseState:
    """Thread-safe live spectral settings shared by every noise source."""

    def __init__(
        self,
        sample_rate: float,
        spec: BrownNoiseSpec,
    ) -> None:
        self.sample_rate = float(sample_rate)
        self._lock = threading.Lock()
        self._spec = spec.validated(self.sample_rate)
        self._version = 0

    def get(self) -> tuple[BrownNoiseSpec, int]:
        with self._lock:
            return self._spec, self._version

    def set(self, spec: BrownNoiseSpec) -> None:
        spec = spec.validated(self.sample_rate)
        with self._lock:
            self._spec = spec
            self._version += 1

    def update(self, **changes: float) -> None:
        with self._lock:
            updated = replace(
                self._spec,
                **changes,
            ).validated(self.sample_rate)
            self._spec = updated
            self._version += 1


@dataclass(slots=True)
class FixedAnchorVoice:
    """
    One permanently configured spectral voice.

    Every voice runs continuously, even when its current mixer gain is zero,
    so bringing it into the mix can never expose an uninitialized filter state.
    """

    rng: np.random.Generator

    dark_sos: np.ndarray
    dark_state: np.ndarray

    bright_sos: np.ndarray
    bright_state: np.ndarray

    dark_weight_alpha: float
    dark_weight_state: float = 0.0

    bright_weight_alpha: float = 0.0
    bright_weight_state: float = 0.0


class BrownNoiseInstance:
    """
    Fixed spectral-anchor implementation.

    Body and Slope are represented by a permanent two-dimensional grid of
    already-valid brown-noise voices. The filters never change after startup.

    Evolution performs only continuous equal-power gain changes between
    neighboring anchors. Weight, Texture and compensated gain are also ramped
    sample-by-sample. Therefore there are:

      * no runtime coefficient changes;
      * no filter-state reinterpretation;
      * no filter-bank replacement;
      * no zipper ticks from buffer-boundary updates.

    Correlation remains downstream in LivingBrownNoiseMixer and has no effect
    on the anchor filters.
    """

    BODY_ANCHOR_COUNT = 7
    SLOPE_ANCHORS = (0.75, 1.00)

    def __init__(
        self,
        sample_rate: float,
        noise_state: BrownNoiseState,
        seed: int | None = None,
    ) -> None:
        self.sample_rate = float(sample_rate)
        self.noise_state = noise_state
        self.seed = int(seed or 0)

        spec, _ = self.noise_state.get()

        self.body_anchor_values = np.linspace(
            0.0,
            1.0,
            self.BODY_ANCHOR_COUNT,
            dtype=np.float64,
        )

        self.voices: list[list[FixedAnchorVoice]] = []

        voice_number = 0
        for body_value in self.body_anchor_values:
            row: list[FixedAnchorVoice] = []

            for slope_value in self.SLOPE_ANCHORS:
                voice_seed = self.seed + 1009 * voice_number + 17
                row.append(
                    self._build_voice(
                        spec=spec,
                        body=float(body_value),
                        slope=float(slope_value),
                        seed=voice_seed,
                    )
                )
                voice_number += 1

            self.voices.append(row)

        self.current_body = spec.body
        self.current_slope = spec.slope_strength
        self.current_weight = spec.low_end_emphasis_db
        self.current_texture = spec.upper_texture

        # This smooths targets received at buffer boundaries, while the actual
        # gains are linearly ramped for every sample in the requested buffer.
        self.parameter_smoothing_seconds = 0.12

    @staticmethod
    def _one_pole_alpha(
        cutoff_hz: float,
        sample_rate: float,
    ) -> float:
        cutoff_hz = float(
            np.clip(cutoff_hz, 0.1, sample_rate * 0.45)
        )
        return math.exp(
            -2.0 * math.pi * cutoff_hz / sample_rate
        )

    @staticmethod
    def _body_to_shift(
        body: float,
        spec: BrownNoiseSpec,
    ) -> float:
        shaped_body = spec._shape_body(body)
        low = math.log(spec.BODY_MIN_SHIFT)
        high = math.log(spec.BODY_MAX_SHIFT)
        return math.exp(low + shaped_body * (high - low))

    def _low_shelf_sos(
        self,
        frequency_hz: float,
        gain_db: float,
    ) -> np.ndarray:
        if abs(gain_db) < 1e-12:
            return np.array(
                [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            )

        frequency_hz = float(
            np.clip(
                frequency_hz,
                1.0,
                self.sample_rate * 0.45,
            )
        )

        amplitude = 10.0 ** (gain_db / 40.0)
        omega = 2.0 * math.pi * frequency_hz / self.sample_rate
        cosine = math.cos(omega)
        sine = math.sin(omega)

        alpha = (
            sine
            / 2.0
            * math.sqrt(
                (amplitude + 1.0 / amplitude) * 2.0
            )
        )
        two_sqrt_a_alpha = (
            2.0 * math.sqrt(amplitude) * alpha
        )

        b0 = amplitude * (
            (amplitude + 1.0)
            - (amplitude - 1.0) * cosine
            + two_sqrt_a_alpha
        )
        b1 = 2.0 * amplitude * (
            (amplitude - 1.0)
            - (amplitude + 1.0) * cosine
        )
        b2 = amplitude * (
            (amplitude + 1.0)
            - (amplitude - 1.0) * cosine
            - two_sqrt_a_alpha
        )
        a0 = (
            (amplitude + 1.0)
            + (amplitude - 1.0) * cosine
            + two_sqrt_a_alpha
        )
        a1 = -2.0 * (
            (amplitude - 1.0)
            + (amplitude + 1.0) * cosine
        )
        a2 = (
            (amplitude + 1.0)
            + (amplitude - 1.0) * cosine
            - two_sqrt_a_alpha
        )

        return np.array(
            [[
                b0 / a0,
                b1 / a0,
                b2 / a0,
                1.0,
                a1 / a0,
                a2 / a0,
            ]],
            dtype=np.float64,
        )

    def _build_fixed_filter(
        self,
        spec: BrownNoiseSpec,
        body: float,
        slope: float,
        bright_scale: float,
    ) -> np.ndarray:
        sections: list[np.ndarray] = []

        shift = self._body_to_shift(body, spec)
        nyquist_safe = self.sample_rate * 0.45

        highpass_hz = min(
            max(0.5, spec.base_highpass_hz * shift),
            nyquist_safe,
        )

        sections.append(
            signal.butter(
                1,
                highpass_hz,
                btype="highpass",
                fs=self.sample_rate,
                output="sos",
            )
        )

        base_cutoffs = (
            spec.base_lowpass_1_hz,
            spec.base_lowpass_2_hz,
            spec.base_lowpass_3_hz,
        )

        for index, base_cutoff in enumerate(base_cutoffs):
            shifted = min(
                max(
                    2.0,
                    base_cutoff * shift * bright_scale,
                ),
                nyquist_safe,
            )

            if index > 0:
                log_a = math.log(shifted)
                log_b = math.log(nyquist_safe)
                shifted = math.exp(
                    slope * log_a
                    + (1.0 - slope) * log_b
                )

            sections.append(
                signal.butter(
                    1,
                    shifted,
                    btype="lowpass",
                    fs=self.sample_rate,
                    output="sos",
                )
            )

        return np.vstack(sections)

    def _build_voice(
        self,
        spec: BrownNoiseSpec,
        body: float,
        slope: float,
        seed: int,
    ) -> FixedAnchorVoice:
        dark_sos = self._build_fixed_filter(
            spec,
            body,
            slope,
            bright_scale=1.0,
        )
        bright_sos = self._build_fixed_filter(
            spec,
            body,
            slope,
            bright_scale=spec.bright_filter_scale,
        )

        weight_alpha = self._one_pole_alpha(
            spec.low_shelf_hz,
            self.sample_rate,
        )

        return FixedAnchorVoice(
            rng=np.random.default_rng(seed),
            dark_sos=dark_sos,
            dark_state=signal.sosfilt_zi(dark_sos) * 0.0,
            bright_sos=bright_sos,
            bright_state=signal.sosfilt_zi(bright_sos) * 0.0,
            dark_weight_alpha=weight_alpha,
            bright_weight_alpha=weight_alpha,
        )

    @staticmethod
    def _fixed_lowpass(
        samples: np.ndarray,
        alpha: float,
        previous_output: float,
    ) -> tuple[np.ndarray, float]:
        b = np.array([1.0 - alpha], dtype=np.float64)
        a = np.array([1.0, -alpha], dtype=np.float64)

        output, final_state = signal.lfilter(
            b,
            a,
            samples,
            zi=np.array([alpha * previous_output]),
        )

        return output, float(output[-1])

    def _process_voice(
        self,
        voice: FixedAnchorVoice,
        frame_count: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        white = voice.rng.standard_normal(frame_count)

        dark, voice.dark_state = signal.sosfilt(
            voice.dark_sos,
            white,
            zi=voice.dark_state,
        )
        bright, voice.bright_state = signal.sosfilt(
            voice.bright_sos,
            white,
            zi=voice.bright_state,
        )

        dark_low, voice.dark_weight_state = self._fixed_lowpass(
            dark,
            voice.dark_weight_alpha,
            voice.dark_weight_state,
        )
        bright_low, voice.bright_weight_state = self._fixed_lowpass(
            bright,
            voice.bright_weight_alpha,
            voice.bright_weight_state,
        )

        return dark, bright, dark_low, bright_low

    def _ramped_parameter(
        self,
        current: float,
        target: float,
        frame_count: int,
    ) -> tuple[np.ndarray, float]:
        elapsed = frame_count / self.sample_rate
        smoothing = 1.0 - math.exp(
            -elapsed / self.parameter_smoothing_seconds
        )
        end = current + (target - current) * smoothing

        ramp = np.linspace(
            current,
            end,
            frame_count,
            endpoint=False,
            dtype=np.float64,
        )

        return ramp, float(end)

    @staticmethod
    def _equal_power_pair(
        normalized_position: np.ndarray,
        count: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scaled = np.clip(
            normalized_position,
            0.0,
            1.0,
        ) * (count - 1)

        lower = np.floor(scaled).astype(np.int32)
        upper = np.minimum(lower + 1, count - 1)
        fraction = scaled - lower

        lower_gain = np.cos(0.5 * np.pi * fraction)
        upper_gain = np.sin(0.5 * np.pi * fraction)

        same = lower == upper
        lower_gain[same] = 1.0
        upper_gain[same] = 0.0

        return lower, lower_gain, upper_gain

    @staticmethod
    def _compensated_gain_array(
        spec: BrownNoiseSpec,
        body: np.ndarray,
        weight_db: np.ndarray,
        texture: np.ndarray,
    ) -> np.ndarray:
        low_log = math.log(spec.BODY_MIN_SHIFT)
        high_log = math.log(spec.BODY_MAX_SHIFT)

        shaped_body = np.power(
            np.clip(body, 0.0, 1.0),
            spec.BODY_CURVE_POWER,
        )

        shift = np.exp(
            low_log + shaped_body * (high_log - low_log)
        )

        octave_term = -5.0 * np.log2(shift)

        low_tail = np.where(
            shift < 0.70,
            np.minimum(
                1.5 * ((0.70 / shift) - 1.0),
                1.5,
            ),
            0.0,
        )

        weight_comp = -0.42 * weight_db
        texture_comp = -2.0 * texture

        allowed_body = (
            24.0
            - spec.base_gain_db
            - weight_comp
            - texture_comp
        )

        body_comp = np.minimum(
            octave_term + low_tail,
            allowed_body,
        )

        gain_db = (
            spec.base_gain_db
            + body_comp
            + weight_comp
            + texture_comp
        )

        return np.power(10.0, gain_db / 20.0)

    def generate(
        self,
        frame_count: int,
        spec_snapshot: BrownNoiseSpec | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if spec_snapshot is None:
            spec_snapshot, _ = self.noise_state.get()

        body, self.current_body = self._ramped_parameter(
            self.current_body,
            spec_snapshot.body,
            frame_count,
        )
        slope, self.current_slope = self._ramped_parameter(
            self.current_slope,
            spec_snapshot.slope_strength,
            frame_count,
        )
        weight_db, self.current_weight = self._ramped_parameter(
            self.current_weight,
            spec_snapshot.low_end_emphasis_db,
            frame_count,
        )
        texture, self.current_texture = self._ramped_parameter(
            self.current_texture,
            spec_snapshot.upper_texture,
            frame_count,
        )

        body_lower, body_lower_gain, body_upper_gain = (
            self._equal_power_pair(
                body,
                self.BODY_ANCHOR_COUNT,
            )
        )
        body_upper = np.minimum(
            body_lower + 1,
            self.BODY_ANCHOR_COUNT - 1,
        )

        slope_position = np.clip(
            (slope - self.SLOPE_ANCHORS[0])
            / (
                self.SLOPE_ANCHORS[1]
                - self.SLOPE_ANCHORS[0]
            ),
            0.0,
            1.0,
        )
        slope_low_gain = np.cos(
            0.5 * np.pi * slope_position
        )
        slope_high_gain = np.sin(
            0.5 * np.pi * slope_position
        )

        weight_linear = np.power(
            10.0,
            weight_db / 20.0,
        )
        weight_amount = weight_linear - 1.0

        mixed_dark = np.zeros(frame_count, dtype=np.float64)
        mixed_bright = np.zeros(frame_count, dtype=np.float64)

        # Every voice runs on every buffer so all filter states remain warm.
        for body_index, row in enumerate(self.voices):
            active_body_gain = np.where(
                body_lower == body_index,
                body_lower_gain,
                0.0,
            )
            active_body_gain += np.where(
                body_upper == body_index,
                body_upper_gain,
                0.0,
            )

            for slope_index, voice in enumerate(row):
                dark, bright, dark_low, bright_low = (
                    self._process_voice(
                        voice,
                        frame_count,
                    )
                )

                if slope_index == 0:
                    slope_gain = slope_low_gain
                else:
                    slope_gain = slope_high_gain

                voice_gain = active_body_gain * slope_gain

                weighted_dark = (
                    dark + weight_amount * dark_low
                )
                weighted_bright = (
                    bright + weight_amount * bright_low
                )

                mixed_dark += voice_gain * weighted_dark
                mixed_bright += voice_gain * weighted_bright

        baseline = (
            mixed_dark
            + (mixed_bright - mixed_dark) * texture
        )

        gain = self._compensated_gain_array(
            spec_snapshot,
            body,
            weight_db,
            texture,
        )

        baseline *= gain
        mixed_bright *= gain

        return (
            baseline.astype(np.float32, copy=False),
            mixed_bright.astype(np.float32, copy=False),
        )


# =============================================================================
# Brown-noise spectral evolution
# =============================================================================

@dataclass(frozen=True, slots=True)
class BrownNoiseEvolutionSpec:
    """
    One global speed control drives four independent organic wanderers.

    rate:
        0.0 = very slow overnight evolution
        1.0 = intentionally rapid testing mode

    Each parameter remains inside the already-approved perceptual range.
    """

    enabled: bool = True
    rate: float = 0.16

    def validated(self) -> BrownNoiseEvolutionSpec:
        if not 0.0 <= self.rate <= 1.0:
            raise ValueError("rate must be between 0 and 1")
        return self

    @property
    def time_scale(self) -> float:
        """
        Logarithmic mapping.

        rate 0.0 -> about 45 minutes per broad excursion
        rate 0.5 -> about 70 seconds
        rate 1.0 -> about 3 seconds
        """
        slow = 2700.0
        fast = 3.0
        return math.exp(
            math.log(slow)
            + self.rate * (math.log(fast) - math.log(slow))
        )


class BrownNoiseEvolutionState:
    def __init__(self, spec: BrownNoiseEvolutionSpec) -> None:
        self._lock = threading.Lock()
        self._spec = spec.validated()

    def get(self) -> BrownNoiseEvolutionSpec:
        with self._lock:
            return self._spec

    def set(self, spec: BrownNoiseEvolutionSpec) -> None:
        with self._lock:
            self._spec = spec.validated()

    def update(self, **changes) -> None:
        with self._lock:
            self._spec = replace(
                self._spec,
                **changes,
            ).validated()


class BoundedOrganicWanderer:
    """
    Low-dimensional stochastic spring mapped into 0..1.

    Each instance has different inertia and random forcing, so the four
    perceptual parameters do not rise and fall together.
    """

    def __init__(
        self,
        seed: int,
        period_multiplier: float,
        damping_ratio: float,
        drive_smoothing_multiplier: float,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.period_multiplier = period_multiplier
        self.damping_ratio = damping_ratio
        self.drive_smoothing_multiplier = drive_smoothing_multiplier

        self.position = 0.0
        self.velocity = 0.0
        self.drive = 0.0

    def set_normalized_position(self, value: float) -> None:
        value = float(np.clip(value, 0.001, 0.999))
        bipolar = 2.0 * value - 1.0
        self.position = 1.15 * math.atanh(bipolar)
        self.velocity = 0.0
        self.drive = 0.0

    def advance(
        self,
        elapsed_seconds: float,
        time_scale: float,
    ) -> float:
        period = max(
            0.20,
            time_scale * self.period_multiplier,
        )
        smoothing = max(
            0.10,
            time_scale * self.drive_smoothing_multiplier,
        )

        omega = 2.0 * math.pi / period
        spring_k = omega * omega
        damping_c = 2.0 * self.damping_ratio * omega

        remaining = max(0.0, elapsed_seconds)

        # At extreme test speeds, integrate finely enough to avoid instability.
        maximum_step = min(
            1.0 / 60.0,
            period / 80.0,
        )

        while remaining > 0.0:
            dt = min(maximum_step, remaining)
            remaining -= dt

            drive_decay = math.exp(-dt / smoothing)
            drive_variance = math.sqrt(
                max(0.0, 1.0 - drive_decay * drive_decay)
            )
            self.drive = (
                self.drive * drive_decay
                + float(self.rng.standard_normal()) * drive_variance
            )

            acceleration = (
                -spring_k * self.position
                - damping_c * self.velocity
                + spring_k * 1.25 * self.drive
            )

            self.velocity += acceleration * dt
            self.position += self.velocity * dt

        # Soft bounded mapping. Most time is spent away from exact limits,
        # while the full approved range remains reachable.
        return 0.5 + 0.5 * math.tanh(self.position / 1.15)


class BrownNoiseEvolution:
    def __init__(
        self,
        evolution_state: BrownNoiseEvolutionState,
    ) -> None:
        self.evolution_state = evolution_state

        self.body = BoundedOrganicWanderer(
            seed=41001,
            period_multiplier=1.00,
            damping_ratio=0.72,
            drive_smoothing_multiplier=0.34,
        )
        self.slope = BoundedOrganicWanderer(
            seed=41002,
            period_multiplier=1.37,
            damping_ratio=0.88,
            drive_smoothing_multiplier=0.48,
        )
        self.weight = BoundedOrganicWanderer(
            seed=41003,
            period_multiplier=0.83,
            damping_ratio=0.78,
            drive_smoothing_multiplier=0.31,
        )
        self.texture = BoundedOrganicWanderer(
            seed=41004,
            period_multiplier=1.61,
            damping_ratio=0.68,
            drive_smoothing_multiplier=0.42,
        )

        self.current_body = 0.50
        self.current_slope = 1.00
        self.current_weight = 0.0
        self.current_texture = 0.0
        self._initialized = False

    def _initialize_from_static(
        self,
        static_spec: BrownNoiseSpec,
    ) -> None:
        self.body.set_normalized_position(static_spec.body)
        self.slope.set_normalized_position(
            (static_spec.slope_strength - 0.75) / 0.25
        )
        self.weight.set_normalized_position(
            static_spec.low_end_emphasis_db / 8.0
        )
        self.texture.set_normalized_position(
            static_spec.upper_texture
        )
        self._initialized = True

    def advance(
        self,
        elapsed_seconds: float,
        static_spec: BrownNoiseSpec,
    ) -> BrownNoiseSpec:
        spec = self.evolution_state.get()

        if not self._initialized:
            self._initialize_from_static(static_spec)

        if not spec.enabled:
            self.current_body = static_spec.body
            self.current_slope = static_spec.slope_strength
            self.current_weight = static_spec.low_end_emphasis_db
            self.current_texture = static_spec.upper_texture
            return static_spec

        time_scale = spec.time_scale

        body_n = self.body.advance(elapsed_seconds, time_scale)
        slope_n = self.slope.advance(elapsed_seconds, time_scale)
        weight_n = self.weight.advance(elapsed_seconds, time_scale)
        texture_n = self.texture.advance(elapsed_seconds, time_scale)

        self.current_body = body_n
        self.current_slope = 0.75 + 0.25 * slope_n
        self.current_weight = 8.0 * weight_n
        self.current_texture = texture_n

        return replace(
            static_spec,
            body=self.current_body,
            slope_strength=self.current_slope,
            low_end_emphasis_db=self.current_weight,
            upper_texture=self.current_texture,
        )



# =============================================================================
# Body movement events
# =============================================================================

@dataclass(frozen=True, slots=True)
class BodyMovementSpec:
    """Rare discrete perturbations of the spectral-evolution system."""

    enabled: bool = True
    frequency: float = 0.08

    def validated(self) -> BodyMovementSpec:
        if not 0.0 <= self.frequency <= 1.0:
            raise ValueError("frequency must be between 0 and 1")
        return self

    @property
    def interval_range_seconds(self) -> tuple[float, float]:
        # 0.0: roughly 30–90 minutes. 1.0: roughly 2–6 seconds.
        slow_min, slow_max = 1800.0, 5400.0
        fast_min, fast_max = 2.0, 6.0
        f = self.frequency
        minimum = math.exp(math.log(slow_min) + f * (math.log(fast_min) - math.log(slow_min)))
        maximum = math.exp(math.log(slow_max) + f * (math.log(fast_max) - math.log(slow_max)))
        return minimum, maximum


class BodyMovementState:
    def __init__(self, spec: BodyMovementSpec) -> None:
        self._lock = threading.Lock()
        self._spec = spec.validated()

    def get(self) -> BodyMovementSpec:
        with self._lock:
            return self._spec

    def set(self, spec: BodyMovementSpec) -> None:
        with self._lock:
            self._spec = spec.validated()

    def update(self, **changes) -> None:
        with self._lock:
            self._spec = replace(self._spec, **changes).validated()


class BodyMovementScheduler:
    """Applies occasional bounded impulses to existing organic wanderers."""

    def __init__(self, state: BodyMovementState, seed: int = 70001) -> None:
        self.state = state
        self.rng = np.random.default_rng(seed)
        self.elapsed = 0.0
        self.next_event = 1.0
        self.event_count = 0
        self.last_strength = 0.0
        self.age = 0.0
        self.reschedule()

    def reschedule(self) -> None:
        minimum, maximum = self.state.get().interval_range_seconds
        self.next_event = float(math.exp(self.rng.uniform(math.log(minimum), math.log(maximum))))
        self.elapsed = 0.0

    def advance(self, elapsed_seconds: float, evolution: BrownNoiseEvolution) -> bool:
        self.age += elapsed_seconds
        if not self.state.get().enabled:
            return False
        self.elapsed += elapsed_seconds
        if self.elapsed < self.next_event:
            return False
        self._trigger(evolution)
        self.reschedule()
        return True

    def _trigger(self, evolution: BrownNoiseEvolution) -> None:
        strength = float(np.clip(self.rng.lognormal(-0.35, 0.38), 0.30, 1.35))
        self.event_count += 1
        self.last_strength = strength
        self.age = 0.0
        scales = (
            (evolution.body, 0.95, 0.60),
            (evolution.slope, 0.28, 0.18),
            (evolution.weight, 0.75, 0.45),
            (evolution.texture, 0.55, 0.35),
        )
        for wanderer, position_scale, velocity_scale in scales:
            direction = float(self.rng.choice((-1.0, 1.0)))
            wanderer.position += direction * strength * position_scale * float(self.rng.uniform(0.55, 1.0))
            wanderer.velocity += direction * strength * velocity_scale * float(self.rng.uniform(0.40, 1.0))
            wanderer.position = float(np.clip(wanderer.position, -2.8, 2.8))
            wanderer.velocity = float(np.clip(wanderer.velocity, -3.5, 3.5))


# =============================================================================
# Mixer controls
# =============================================================================

@dataclass(frozen=True, slots=True)
class EngineModes:
    stereo_enabled: bool = True
    correlation_enabled: bool = True
    breath_enabled: bool = True


class ModeState:
    def __init__(self, modes: EngineModes | None = None) -> None:
        self._lock = threading.Lock()
        self._modes = modes or EngineModes()

    def get(self) -> EngineModes:
        with self._lock:
            return self._modes

    def set(
        self,
        *,
        stereo_enabled: bool | None = None,
        correlation_enabled: bool | None = None,
        breath_enabled: bool | None = None,
    ) -> None:
        with self._lock:
            current = self._modes
            self._modes = EngineModes(
                stereo_enabled=(
                    current.stereo_enabled
                    if stereo_enabled is None
                    else bool(stereo_enabled)
                ),
                correlation_enabled=(
                    current.correlation_enabled
                    if correlation_enabled is None
                    else bool(correlation_enabled)
                ),
                breath_enabled=(
                    current.breath_enabled
                    if breath_enabled is None
                    else bool(breath_enabled)
                ),
            )


@dataclass(frozen=True, slots=True)
class MixerSpec:
    correlation_min: float = 0.0
    correlation_max: float = 1.0
    master_gain_db: float = -3.0
    toggle_smoothing_seconds: float = 0.25


class LivingBrownNoiseMixer:
    def __init__(
        self,
        sample_rate: float,
        common: BrownNoiseInstance,
        independent_left: BrownNoiseInstance,
        independent_right: BrownNoiseInstance,
        mode_state: ModeState,
        noise_state: BrownNoiseState,
        noise_evolution_state: BrownNoiseEvolutionState,
        body_movement_state: BodyMovementState,
        breath_state: BreathState,
        breath_evolution_state: BreathEvolutionState,
        motion_state: OrganicMotionState,
        mixer_spec: MixerSpec,
    ) -> None:
        self.sample_rate = float(sample_rate)
        self.common = common
        self.independent_left = independent_left
        self.independent_right = independent_right
        self.mode_state = mode_state
        self.noise_state = noise_state
        self.noise_evolution_state = noise_evolution_state
        self.noise_evolution = BrownNoiseEvolution(
            noise_evolution_state
        )
        self.body_movement_state = body_movement_state
        self.body_movement = BodyMovementScheduler(body_movement_state)
        self.breath_state = breath_state
        self.breath_evolution_state = breath_evolution_state
        self.motion_state = motion_state
        self.mixer_spec = mixer_spec

        self.correlation_motion = OrganicMotion1D(
            motion_state=motion_state,
            seed=987654,
        )
        self.breath = BreathEnvelope(
            sample_rate=self.sample_rate,
            breath_state=breath_state,
        )
        self.breath_prominence = BreathProminenceOscillator(
            sample_rate=self.sample_rate,
            evolution_state=breath_evolution_state,
        )

        initial_modes = self.mode_state.get()
        self.stereo_mix = 1.0 if initial_modes.stereo_enabled else 0.0
        self.correlation_mix = (
            1.0 if initial_modes.correlation_enabled else 0.0
        )
        self.breath_mix = 1.0 if initial_modes.breath_enabled else 0.0

        self.current_correlation = 0.536
        self.current_breath = 0.0
        self.current_breath_stage = BreathEnvelope.STAGE_INHALE
        self.current_breath_prominence = 1.0
        self.current_breath_evolution_period = 0.0
        self.current_noise_body = 0.50
        self.current_noise_slope = 1.00
        self.current_noise_weight = 0.0
        self.current_noise_texture = 0.0
        self.current_body_movement_count = 0
        self.current_body_movement_strength = 0.0
        self.current_body_movement_age = 0.0

    def _approach_target(
        self,
        current: float,
        target: float,
        frame_count: int,
    ) -> np.ndarray:
        smoothing_samples = max(
            1,
            int(
                self.mixer_spec.toggle_smoothing_seconds
                * self.sample_rate
            ),
        )
        maximum_change = frame_count / smoothing_samples
        end = current + np.clip(
            target - current,
            -maximum_change,
            maximum_change,
        )
        return np.linspace(
            current,
            end,
            frame_count,
            endpoint=False,
            dtype=np.float32,
        )

    def _noise_to_correlation(self, noise_value: float) -> float:
        normalized = float(
            np.clip(noise_value * 0.5 + 0.5, 0.0, 1.0)
        )
        return (
            self.mixer_spec.correlation_min
            + normalized
            * (
                self.mixer_spec.correlation_max
                - self.mixer_spec.correlation_min
            )
        )

    @staticmethod
    def _blend(
        dark: np.ndarray,
        bright: np.ndarray,
        amount: np.ndarray,
    ) -> np.ndarray:
        return dark + (bright - dark) * amount

    def generate(self, frame_count: int) -> np.ndarray:
        modes = self.mode_state.get()
        elapsed_seconds = frame_count / self.sample_rate
        static_noise_spec, _ = self.noise_state.get()
        evolved_noise_spec = self.noise_evolution.advance(
            elapsed_seconds,
            static_noise_spec,
        )
        self.body_movement.advance(
            elapsed_seconds,
            self.noise_evolution,
        )
        self.current_body_movement_count = self.body_movement.event_count
        self.current_body_movement_strength = self.body_movement.last_strength
        self.current_body_movement_age = self.body_movement.age
        self.current_noise_body = evolved_noise_spec.body
        self.current_noise_slope = evolved_noise_spec.slope_strength
        self.current_noise_weight = (
            evolved_noise_spec.low_end_emphasis_db
        )
        self.current_noise_texture = evolved_noise_spec.upper_texture
        breath_spec, _ = self.breath_state.get()

        stereo_curve = self._approach_target(
            self.stereo_mix,
            1.0 if modes.stereo_enabled else 0.0,
            frame_count,
        )
        correlation_curve = self._approach_target(
            self.correlation_mix,
            1.0 if modes.correlation_enabled else 0.0,
            frame_count,
        )
        breath_curve = self._approach_target(
            self.breath_mix,
            1.0 if modes.breath_enabled else 0.0,
            frame_count,
        )

        self.stereo_mix = float(stereo_curve[-1])
        self.correlation_mix = float(correlation_curve[-1])
        self.breath_mix = float(breath_curve[-1])

        raw_breath = self.breath.generate(frame_count)
        prominence = self.breath_prominence.generate(frame_count)
        active_breath = raw_breath * breath_curve

        self.current_breath = float(active_breath[-1])
        self.current_breath_stage = self.breath.stage
        self.current_breath_prominence = float(prominence[-1])
        self.current_breath_evolution_period = (
            self.breath_prominence.current_period_seconds
        )

        evolved_breath = active_breath * prominence

        # Deep-breath cycles can exceed 1.0 by design. Bound the final
        # modulation signal so gain, spectrum and width cannot combine into
        # a severe overload when Body is also at a high-compensation setting.
        bounded_breath = np.clip(
            evolved_breath,
            0.0,
            1.25,
        )

        spectral_amount = (
            bounded_breath * breath_spec.spectral_depth
        )

        common_dark, common_bright = self.common.generate(
            frame_count,
            spec_snapshot=evolved_noise_spec,
        )
        left_dark, left_bright = self.independent_left.generate(
            frame_count,
            spec_snapshot=evolved_noise_spec,
        )
        right_dark, right_bright = self.independent_right.generate(
            frame_count,
            spec_snapshot=evolved_noise_spec,
        )

        common = self._blend(
            common_dark,
            common_bright,
            spectral_amount,
        )
        independent_left = self._blend(
            left_dark,
            left_bright,
            spectral_amount,
        )
        independent_right = self._blend(
            right_dark,
            right_bright,
            spectral_amount,
        )

        motion_start, motion_end = self.correlation_motion.advance(
            frame_count / self.sample_rate
        )

        correlation_start = self._noise_to_correlation(
            motion_start
        )
        correlation_end = self._noise_to_correlation(
            motion_end
        )

        evolving_correlation = np.linspace(
            correlation_start,
            correlation_end,
            frame_count,
            endpoint=False,
            dtype=np.float32,
        )

        correlation = evolving_correlation * correlation_curve
        correlation -= bounded_breath * breath_spec.width_depth
        np.clip(correlation, 0.0, 1.0, out=correlation)

        self.current_correlation = float(correlation[-1])

        common_gain = np.sqrt(correlation)
        independent_gain = np.sqrt(1.0 - correlation)

        stereo_left = (
            common_gain * common
            + independent_gain * independent_left
        )
        stereo_right = (
            common_gain * common
            + independent_gain * independent_right
        )

        mono = common
        left = mono + (stereo_left - mono) * stereo_curve
        right = mono + (stereo_right - mono) * stereo_curve

        stereo = np.column_stack((left, right))

        breath_gain_db = (
            bounded_breath - 0.5 * breath_curve * prominence
        ) * breath_spec.gain_range_db

        stereo *= np.power(
            10.0,
            breath_gain_db / 20.0,
        )[:, np.newaxis]

        stereo *= 10.0 ** (
            self.mixer_spec.master_gain_db / 20.0
        )

        # Smooth safety limiter. For ordinary levels tanh is effectively
        # linear; during an overload it rounds peaks rather than chopping them
        # into the crackling flat tops produced by a hard clipper.
        stereo = 0.98 * np.tanh(stereo / 0.98)

        return stereo.astype(np.float32, copy=False)


# =============================================================================
# Audio output
# =============================================================================

class AudioEngine:
    def __init__(
        self,
        mixer: LivingBrownNoiseMixer,
        sample_rate: int = 44_100,
        block_size: int = 2_048,
        device: int | str | None = None,
    ) -> None:
        self.mixer = mixer
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.device = device

        self.stream: sd.OutputStream | None = None
        self.callback_error: Exception | None = None

    @property
    def is_running(self) -> bool:
        return self.stream is not None

    def _callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info,
        status: sd.CallbackFlags,
    ) -> None:
        del time_info, status

        try:
            outdata[:] = self.mixer.generate(frames)
        except Exception as exc:
            self.callback_error = exc
            outdata.fill(0.0)

    def start(self) -> None:
        if self.stream is not None:
            return

        self.callback_error = None
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            device=self.device,
            channels=2,
            dtype="float32",
            latency="high",
            callback=self._callback,
        )
        self.stream.start()

    def stop(self) -> None:
        if self.stream is None:
            return

        self.stream.stop()
        self.stream.close()
        self.stream = None



# =============================================================================
# Settings persistence
# =============================================================================

class SettingsStore:
    """JSON settings stored in the user's home directory."""

    def __init__(self) -> None:
        self.path = (
            Path.home()
            / ".living_brown_noise"
            / "settings.json"
        )

    def load(self) -> dict:
        try:
            if not self.path.exists():
                return {}
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            # A malformed settings file should never prevent startup.
            return {}

    def save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(data, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(self.path)


def build_mixer(
    *,
    sample_rate: int,
    modes: EngineModes,
    noise_spec: BrownNoiseSpec,
    noise_evolution_spec: BrownNoiseEvolutionSpec,
    body_movement_spec: BodyMovementSpec,
    breath_spec: BreathSpec,
    breath_evolution_spec: BreathEvolutionSpec,
    motion_spec: OrganicMotionSpec,
    seed_base: int,
) -> tuple[
    LivingBrownNoiseMixer,
    ModeState,
    BrownNoiseState,
    BrownNoiseEvolutionState,
    BodyMovementState,
    BreathState,
    BreathEvolutionState,
    OrganicMotionState,
]:
    noise_state = BrownNoiseState(sample_rate, noise_spec)
    noise_evolution_state = BrownNoiseEvolutionState(
        noise_evolution_spec
    )
    body_movement_state = BodyMovementState(body_movement_spec)

    common = BrownNoiseInstance(
        sample_rate,
        noise_state,
        seed=seed_base + 1,
    )
    independent_left = BrownNoiseInstance(
        sample_rate,
        noise_state,
        seed=seed_base + 2,
    )
    independent_right = BrownNoiseInstance(
        sample_rate,
        noise_state,
        seed=seed_base + 3,
    )

    mode_state = ModeState(modes)
    breath_state = BreathState(breath_spec)
    breath_evolution_state = BreathEvolutionState(
        breath_evolution_spec
    )
    motion_state = OrganicMotionState(motion_spec)

    mixer = LivingBrownNoiseMixer(
        sample_rate=sample_rate,
        common=common,
        independent_left=independent_left,
        independent_right=independent_right,
        mode_state=mode_state,
        noise_state=noise_state,
        noise_evolution_state=noise_evolution_state,
        body_movement_state=body_movement_state,
        breath_state=breath_state,
        breath_evolution_state=breath_evolution_state,
        motion_state=motion_state,
        mixer_spec=MixerSpec(),
    )

    return (
        mixer,
        mode_state,
        noise_state,
        noise_evolution_state,
        body_movement_state,
        breath_state,
        breath_evolution_state,
        motion_state,
    )


# =============================================================================
# Offline export
# =============================================================================

class ExportWorker(QThread):
    progress_changed = Signal(int)
    export_finished = Signal(str)
    export_failed = Signal(str)
    export_cancelled = Signal()

    def __init__(
        self,
        *,
        output_path: str,
        duration_minutes: int,
        sample_rate: int,
        modes: EngineModes,
        noise_spec: BrownNoiseSpec,
        noise_evolution_spec: BrownNoiseEvolutionSpec,
        body_movement_spec: BodyMovementSpec,
        breath_spec: BreathSpec,
        breath_evolution_spec: BreathEvolutionSpec,
        motion_spec: OrganicMotionSpec,
    ) -> None:
        super().__init__()
        self.output_path = output_path
        self.duration_minutes = duration_minutes
        self.sample_rate = sample_rate
        self.modes = modes
        self.noise_spec = noise_spec
        self.noise_evolution_spec = noise_evolution_spec
        self.body_movement_spec = body_movement_spec
        self.breath_spec = breath_spec
        self.breath_evolution_spec = breath_evolution_spec
        self.motion_spec = motion_spec
        self._cancel_requested = threading.Event()

    def request_cancel(self) -> None:
        self._cancel_requested.set()

    def run(self) -> None:
        try:
            total_frames = int(
                self.duration_minutes * 60 * self.sample_rate
            )
            chunk_frames = max(2048, self.sample_rate // 2)

            seed_base = int(time.time_ns() & 0x7FFFFFFF)
            mixer, _, _, _, _, _, _, _ = build_mixer(
                sample_rate=self.sample_rate,
                modes=self.modes,
                noise_spec=self.noise_spec,
                noise_evolution_spec=self.noise_evolution_spec,
                body_movement_spec=self.body_movement_spec,
                breath_spec=self.breath_spec,
                breath_evolution_spec=self.breath_evolution_spec,
                motion_spec=self.motion_spec,
                seed_base=seed_base,
            )

            output = Path(self.output_path)
            output.parent.mkdir(parents=True, exist_ok=True)

            frames_written = 0

            with wave.open(str(output), "wb") as wav:
                wav.setnchannels(2)
                wav.setsampwidth(2)  # 16-bit PCM
                wav.setframerate(self.sample_rate)

                while frames_written < total_frames:
                    if self._cancel_requested.is_set():
                        raise InterruptedError

                    frame_count = min(
                        chunk_frames,
                        total_frames - frames_written,
                    )

                    audio = mixer.generate(frame_count)

                    # Convert float [-1, 1] to little-endian signed PCM16.
                    pcm = np.clip(audio, -1.0, 1.0)
                    pcm = np.round(pcm * 32767.0).astype("<i2")
                    wav.writeframesraw(pcm.tobytes())

                    frames_written += frame_count
                    percent = int(
                        frames_written * 100 / total_frames
                    )
                    self.progress_changed.emit(percent)

                wav.writeframes(b"")

            self.progress_changed.emit(100)
            self.export_finished.emit(str(output))

        except InterruptedError:
            try:
                Path(self.output_path).unlink(missing_ok=True)
            except Exception:
                pass
            self.export_cancelled.emit()

        except Exception as exc:
            try:
                Path(self.output_path).unlink(missing_ok=True)
            except Exception:
                pass
            self.export_failed.emit(str(exc))


# =============================================================================
# GUI helper: linked slider and spin box
# =============================================================================

class FloatControl(QWidget):
    """
    Horizontal slider with a precise numeric spin box.

    The slider and spin box stay synchronized and emit values through the
    supplied callback.
    """

    def __init__(
        self,
        *,
        minimum: float,
        maximum: float,
        value: float,
        step: float,
        decimals: int,
        suffix: str,
        on_change,
    ) -> None:
        super().__init__()

        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.step = float(step)
        self.on_change = on_change
        self._updating = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(
            0,
            round((self.maximum - self.minimum) / self.step),
        )

        self.spin = QDoubleSpinBox()
        self.spin.setRange(self.minimum, self.maximum)
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(self.step)
        self.spin.setSuffix(suffix)
        self.spin.setMinimumWidth(105)

        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin)

        self.slider.valueChanged.connect(self._slider_changed)
        self.spin.valueChanged.connect(self._spin_changed)

        self.set_value(value, notify=False)

    def _slider_to_float(self, slider_value: int) -> float:
        return self.minimum + slider_value * self.step

    def _float_to_slider(self, value: float) -> int:
        return round((value - self.minimum) / self.step)

    def _slider_changed(self, slider_value: int) -> None:
        if self._updating:
            return

        value = self._slider_to_float(slider_value)

        self._updating = True
        self.spin.setValue(value)
        self._updating = False

        self.on_change(value)

    def _spin_changed(self, value: float) -> None:
        if self._updating:
            return

        self._updating = True
        self.slider.setValue(self._float_to_slider(value))
        self._updating = False

        self.on_change(float(value))

    def set_value(self, value: float, notify: bool = True) -> None:
        value = float(np.clip(value, self.minimum, self.maximum))

        self._updating = True
        self.slider.setValue(self._float_to_slider(value))
        self.spin.setValue(value)
        self._updating = False

        if notify:
            self.on_change(value)


# =============================================================================
# GUI
# =============================================================================

class MainWindow(QMainWindow):
    def __init__(
        self,
        engine: AudioEngine,
        mode_state: ModeState,
        noise_state: BrownNoiseState,
        noise_evolution_state: BrownNoiseEvolutionState,
        body_movement_state: BodyMovementState,
        breath_state: BreathState,
        breath_evolution_state: BreathEvolutionState,
        motion_state: OrganicMotionState,
        mixer: LivingBrownNoiseMixer,
        settings_store: SettingsStore,
        loaded_settings: dict,
    ) -> None:
        super().__init__()

        self.engine = engine
        self.mode_state = mode_state
        self.noise_state = noise_state
        self.noise_evolution_state = noise_evolution_state
        self.body_movement_state = body_movement_state
        self.breath_state = breath_state
        self.breath_evolution_state = breath_evolution_state
        self.motion_state = motion_state
        self.mixer = mixer
        self.settings_store = settings_store
        self.loaded_settings = loaded_settings
        self.export_worker: ExportWorker | None = None

        self.settings_save_timer = QTimer(self)
        self.settings_save_timer.setSingleShot(True)
        self.settings_save_timer.timeout.connect(self._save_settings)

        self.default_breath_spec = BreathSpec()

        self.setWindowTitle(
            "Living Brown Noise — Deconstruction Lab v1.6"
        )
        self.resize(840, 1040)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        explanation = QLabel(
            "Toggle layers for A/B testing, then open Breath parameters "
            "to tune the breath live. Timing changes preserve the current "
            "stage's approximate progress."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        # ------------------------------------------------------------------
        # Engine layer switches
        # ------------------------------------------------------------------

        controls_box = QGroupBox("Engine layers")
        controls_layout = QVBoxLayout(controls_box)

        self.stereo_checkbox = QCheckBox(
            "Stereo — off duplicates one mono source to both ears"
        )
        self.stereo_checkbox.setChecked(
            self.mode_state.get().stereo_enabled
        )

        self.correlation_checkbox = QCheckBox(
            "Correlation mixing — off uses fully independent L/R noise"
        )
        self.correlation_checkbox.setChecked(
            self.mode_state.get().correlation_enabled
        )

        self.breath_checkbox = QCheckBox(
            "Breath algorithm — gain + spectral + width modulation"
        )
        self.breath_checkbox.setChecked(
            self.mode_state.get().breath_enabled
        )

        self.noise_expand_button = QToolButton()
        self.noise_expand_button.setText(
            "Brown-noise style parameters"
        )
        self.noise_expand_button.setCheckable(True)
        self.noise_expand_button.setChecked(
            bool(
                self.loaded_settings.get(
                    "noise_panel_expanded",
                    True,
                )
            )
        )
        self.noise_expand_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.noise_expand_button.setArrowType(
            Qt.ArrowType.RightArrow
        )
        controls_layout.addWidget(self.noise_expand_button)

        self.noise_panel = QWidget()
        noise_form = QFormLayout(self.noise_panel)
        noise_form.setContentsMargins(24, 4, 0, 8)
        self.noise_panel.setVisible(
            self.noise_expand_button.isChecked()
        )

        noise_spec, _ = self.noise_state.get()

        self.noise_body_control = FloatControl(
            minimum=0.0,
            maximum=1.0,
            value=noise_spec.body,
            step=0.01,
            decimals=2,
            suffix="",
            on_change=lambda value: self._update_noise_parameter(
                body=value
            ),
        )
        noise_form.addRow(
            "Body / spectral position:",
            self.noise_body_control,
        )

        self.noise_slope_control = FloatControl(
            minimum=0.75,
            maximum=1.0,
            value=noise_spec.slope_strength,
            step=0.01,
            decimals=2,
            suffix="",
            on_change=lambda value: self._update_noise_parameter(
                slope_strength=value
            ),
        )
        noise_form.addRow(
            "Slope strength:",
            self.noise_slope_control,
        )

        self.noise_low_end_control = FloatControl(
            minimum=0.0,
            maximum=8.0,
            value=noise_spec.low_end_emphasis_db,
            step=0.1,
            decimals=1,
            suffix=" dB",
            on_change=lambda value: self._update_noise_parameter(
                low_end_emphasis_db=value
            ),
        )
        noise_form.addRow(
            "Low-end emphasis:",
            self.noise_low_end_control,
        )

        self.noise_texture_control = FloatControl(
            minimum=0.0,
            maximum=1.0,
            value=noise_spec.upper_texture,
            step=0.01,
            decimals=2,
            suffix="",
            on_change=lambda value: self._update_noise_parameter(
                upper_texture=value
            ),
        )
        noise_form.addRow(
            "Upper texture:",
            self.noise_texture_control,
        )

        self.noise_body_status = QLabel("")
        noise_form.addRow(
            "Resolved body:",
            self.noise_body_status,
        )

        reset_noise_button = QPushButton(
            "Reset brown-noise defaults"
        )
        reset_noise_button.clicked.connect(
            self._reset_noise_defaults
        )
        noise_form.addRow("", reset_noise_button)

        evolution_spec = self.noise_evolution_state.get()

        self.noise_evolution_checkbox = QCheckBox(
            "Brown-noise evolution — wander through accepted styles"
        )
        self.noise_evolution_checkbox.setChecked(
            evolution_spec.enabled
        )
        noise_form.addRow(
            "",
            self.noise_evolution_checkbox,
        )

        self.noise_evolution_rate_control = FloatControl(
            minimum=0.0,
            maximum=1.0,
            value=evolution_spec.rate,
            step=0.01,
            decimals=2,
            suffix="",
            on_change=lambda value: self._update_noise_evolution(
                rate=value
            ),
        )
        noise_form.addRow(
            "Evolution rate:",
            self.noise_evolution_rate_control,
        )

        self.noise_evolution_status = QLabel("")
        noise_form.addRow(
            "Current evolved style:",
            self.noise_evolution_status,
        )

        movement_spec = self.body_movement_state.get()
        self.body_movement_checkbox = QCheckBox(
            "Body movement — occasional discrete repositioning"
        )
        self.body_movement_checkbox.setChecked(movement_spec.enabled)
        noise_form.addRow("", self.body_movement_checkbox)

        self.body_movement_frequency_control = FloatControl(
            minimum=0.0,
            maximum=1.0,
            value=movement_spec.frequency,
            step=0.01,
            decimals=2,
            suffix="",
            on_change=lambda value: self._update_body_movement(
                frequency=value
            ),
        )
        noise_form.addRow(
            "Body movement frequency:",
            self.body_movement_frequency_control,
        )
        self.body_movement_status = QLabel("")
        noise_form.addRow("Body movement status:", self.body_movement_status)

        controls_layout.addWidget(self.noise_panel)
        controls_layout.addWidget(self.stereo_checkbox)
        controls_layout.addWidget(self.correlation_checkbox)

        self.motion_expand_button = QToolButton()
        self.motion_expand_button.setText(
            "Organic motion parameters"
        )
        self.motion_expand_button.setCheckable(True)
        self.motion_expand_button.setChecked(
            bool(
                self.loaded_settings.get(
                    "motion_panel_expanded",
                    False,
                )
            )
        )
        self.motion_expand_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.motion_expand_button.setArrowType(
            Qt.ArrowType.RightArrow
        )
        controls_layout.addWidget(self.motion_expand_button)

        self.motion_panel = QWidget()
        motion_form = QFormLayout(self.motion_panel)
        motion_form.setContentsMargins(24, 4, 0, 8)
        self.motion_panel.setVisible(
            self.motion_expand_button.isChecked()
        )

        motion_spec = self.motion_state.get()

        self.motion_period_control = FloatControl(
            minimum=0.05,
            maximum=30.0,
            value=motion_spec.natural_period_seconds,
            step=0.05,
            decimals=2,
            suffix=" s",
            on_change=lambda value: self._update_motion_parameter(
                natural_period_seconds=value
            ),
        )
        motion_form.addRow(
            "Natural period:",
            self.motion_period_control,
        )

        self.motion_damping_control = FloatControl(
            minimum=0.05,
            maximum=3.0,
            value=motion_spec.damping_ratio,
            step=0.01,
            decimals=2,
            suffix="",
            on_change=lambda value: self._update_motion_parameter(
                damping_ratio=value
            ),
        )
        motion_form.addRow(
            "Damping ratio:",
            self.motion_damping_control,
        )

        self.motion_drive_control = FloatControl(
            minimum=0.0,
            maximum=5.0,
            value=motion_spec.drive_strength,
            step=0.01,
            decimals=2,
            suffix="",
            on_change=lambda value: self._update_motion_parameter(
                drive_strength=value
            ),
        )
        motion_form.addRow(
            "Drive strength:",
            self.motion_drive_control,
        )

        self.motion_smoothing_control = FloatControl(
            minimum=0.01,
            maximum=20.0,
            value=motion_spec.drive_smoothing_seconds,
            step=0.01,
            decimals=2,
            suffix=" s",
            on_change=lambda value: self._update_motion_parameter(
                drive_smoothing_seconds=value
            ),
        )
        motion_form.addRow(
            "Drive smoothing:",
            self.motion_smoothing_control,
        )

        self.motion_limit_control = FloatControl(
            minimum=0.1,
            maximum=5.0,
            value=motion_spec.soft_limit,
            step=0.01,
            decimals=2,
            suffix="",
            on_change=lambda value: self._update_motion_parameter(
                soft_limit=value
            ),
        )
        motion_form.addRow(
            "Soft limit:",
            self.motion_limit_control,
        )

        reset_motion_button = QPushButton(
            "Reset organic motion defaults"
        )
        reset_motion_button.clicked.connect(
            self._reset_motion_defaults
        )
        motion_form.addRow("", reset_motion_button)

        controls_layout.addWidget(self.motion_panel)
        controls_layout.addWidget(self.breath_checkbox)

        self.breath_evolution_checkbox = QCheckBox(
            "Breath evolution — slowly fades breath prominence in and out"
        )
        self.breath_evolution_checkbox.setChecked(
            self.breath_evolution_state.get().enabled
        )
        controls_layout.addWidget(self.breath_evolution_checkbox)

        self.breath_evolution_expand_button = QToolButton()
        self.breath_evolution_expand_button.setText(
            "Breath evolution parameters"
        )
        self.breath_evolution_expand_button.setCheckable(True)
        self.breath_evolution_expand_button.setChecked(
            bool(
                self.loaded_settings.get(
                    "breath_evolution_panel_expanded",
                    False,
                )
            )
        )
        self.breath_evolution_expand_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.breath_evolution_expand_button.setArrowType(
            Qt.ArrowType.RightArrow
        )
        controls_layout.addWidget(
            self.breath_evolution_expand_button
        )

        self.breath_evolution_panel = QWidget()
        evolution_form = QFormLayout(
            self.breath_evolution_panel
        )
        evolution_form.setContentsMargins(24, 4, 0, 8)
        self.breath_evolution_panel.setVisible(
            self.breath_evolution_expand_button.isChecked()
        )

        evolution_spec = self.breath_evolution_state.get()

        self.evolution_min_control = FloatControl(
            minimum=0.0,
            maximum=1.0,
            value=evolution_spec.multiplier_min,
            step=0.05,
            decimals=2,
            suffix="×",
            on_change=lambda value: self._update_breath_evolution(
                multiplier_min=value
            ),
        )
        evolution_form.addRow(
            "Minimum prominence:",
            self.evolution_min_control,
        )

        self.evolution_max_control = FloatControl(
            minimum=0.0,
            maximum=1.0,
            value=evolution_spec.multiplier_max,
            step=0.05,
            decimals=2,
            suffix="×",
            on_change=lambda value: self._update_breath_evolution(
                multiplier_max=value
            ),
        )
        evolution_form.addRow(
            "Maximum prominence:",
            self.evolution_max_control,
        )

        self.evolution_period_min_control = FloatControl(
            minimum=5.0,
            maximum=21600.0,
            value=evolution_spec.period_min_seconds,
            step=5.0,
            decimals=0,
            suffix=" s",
            on_change=lambda value: self._update_breath_evolution(
                period_min_seconds=value
            ),
        )
        evolution_form.addRow(
            "Minimum cycle:",
            self.evolution_period_min_control,
        )

        self.evolution_period_max_control = FloatControl(
            minimum=5.0,
            maximum=21600.0,
            value=evolution_spec.period_max_seconds,
            step=5.0,
            decimals=0,
            suffix=" s",
            on_change=lambda value: self._update_breath_evolution(
                period_max_seconds=value
            ),
        )
        evolution_form.addRow(
            "Maximum cycle:",
            self.evolution_period_max_control,
        )

        self.evolution_curve_control = FloatControl(
            minimum=0.1,
            maximum=8.0,
            value=evolution_spec.curve_power,
            step=0.05,
            decimals=2,
            suffix="",
            on_change=lambda value: self._update_breath_evolution(
                curve_power=value
            ),
        )
        evolution_form.addRow(
            "Quiet-time bias:",
            self.evolution_curve_control,
        )

        reset_evolution_button = QPushButton(
            "Reset breath evolution defaults"
        )
        reset_evolution_button.clicked.connect(
            self._reset_breath_evolution_defaults
        )
        evolution_form.addRow("", reset_evolution_button)

        controls_layout.addWidget(self.breath_evolution_panel)

        # Collapsible breath controls, directly beneath the breath checkbox.
        self.breath_expand_button = QToolButton()
        self.breath_expand_button.setText("Breath parameters")
        self.breath_expand_button.setCheckable(True)
        self.breath_expand_button.setChecked(
            bool(self.loaded_settings.get("breath_panel_expanded", False))
        )
        self.breath_expand_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.breath_expand_button.setArrowType(
            Qt.ArrowType.RightArrow
        )

        controls_layout.addWidget(self.breath_expand_button)

        self.breath_panel = QWidget()
        breath_form = QFormLayout(self.breath_panel)
        breath_form.setContentsMargins(24, 4, 0, 8)
        self.breath_panel.setVisible(
            self.breath_expand_button.isChecked()
        )

        spec, _ = self.breath_state.get()

        # Depth controls
        self.gain_depth_control = FloatControl(
            minimum=0.0,
            maximum=12.0,
            value=spec.gain_range_db,
            step=0.1,
            decimals=1,
            suffix=" dB",
            on_change=lambda value: self._update_breath_parameter(
                gain_range_db=value
            ),
        )
        breath_form.addRow(
            "Gain range:",
            self.gain_depth_control,
        )

        self.spectral_depth_control = FloatControl(
            minimum=0.0,
            maximum=1.0,
            value=spec.spectral_depth,
            step=0.01,
            decimals=2,
            suffix="",
            on_change=lambda value: self._update_breath_parameter(
                spectral_depth=value
            ),
        )
        breath_form.addRow(
            "Spectral depth:",
            self.spectral_depth_control,
        )

        self.width_depth_control = FloatControl(
            minimum=0.0,
            maximum=1.0,
            value=spec.width_depth,
            step=0.01,
            decimals=2,
            suffix="",
            on_change=lambda value: self._update_breath_parameter(
                width_depth=value
            ),
        )
        breath_form.addRow(
            "Width depth:",
            self.width_depth_control,
        )

        # Biological cycle timing
        self.inhale_mean_control = self._make_breath_control(
            minimum=0.05,
            maximum=10.0,
            value=spec.inhale_mean_seconds,
            step=0.05,
            decimals=2,
            suffix=" s",
            field_name="inhale_mean_seconds",
        )
        breath_form.addRow(
            "Mean inhale:",
            self.inhale_mean_control,
        )

        self.hold_mean_control = self._make_breath_control(
            minimum=0.01,
            maximum=5.0,
            value=spec.hold_mean_seconds,
            step=0.01,
            decimals=2,
            suffix=" s",
            field_name="hold_mean_seconds",
        )
        breath_form.addRow(
            "Mean hold:",
            self.hold_mean_control,
        )

        self.exhale_mean_control = self._make_breath_control(
            minimum=0.05,
            maximum=15.0,
            value=spec.exhale_mean_seconds,
            step=0.05,
            decimals=2,
            suffix=" s",
            field_name="exhale_mean_seconds",
        )
        breath_form.addRow(
            "Mean exhale:",
            self.exhale_mean_control,
        )

        self.rest_mean_control = self._make_breath_control(
            minimum=0.01,
            maximum=10.0,
            value=spec.rest_mean_seconds,
            step=0.05,
            decimals=2,
            suffix=" s",
            field_name="rest_mean_seconds",
        )
        breath_form.addRow(
            "Mean rest:",
            self.rest_mean_control,
        )

        self.timing_variation_control = self._make_breath_control(
            minimum=0.0,
            maximum=1.0,
            value=spec.timing_variation,
            step=0.01,
            decimals=2,
            suffix="",
            field_name="timing_variation",
        )
        breath_form.addRow(
            "Timing variation:",
            self.timing_variation_control,
        )

        self.timing_memory_control = self._make_breath_control(
            minimum=0.0,
            maximum=0.99,
            value=spec.timing_memory,
            step=0.01,
            decimals=2,
            suffix="",
            field_name="timing_memory",
        )
        breath_form.addRow(
            "Timing memory:",
            self.timing_memory_control,
        )

        self.depth_variation_control = self._make_breath_control(
            minimum=0.0,
            maximum=1.0,
            value=spec.depth_variation,
            step=0.01,
            decimals=2,
            suffix="",
            field_name="depth_variation",
        )
        breath_form.addRow(
            "Depth variation:",
            self.depth_variation_control,
        )

        self.depth_memory_control = self._make_breath_control(
            minimum=0.0,
            maximum=0.99,
            value=spec.depth_memory,
            step=0.01,
            decimals=2,
            suffix="",
            field_name="depth_memory",
        )
        breath_form.addRow(
            "Depth memory:",
            self.depth_memory_control,
        )

        self.deep_probability_control = self._make_breath_control(
            minimum=0.0,
            maximum=0.10,
            value=spec.deep_breath_probability,
            step=0.001,
            decimals=3,
            suffix="",
            field_name="deep_breath_probability",
        )
        breath_form.addRow(
            "Deep-breath chance:",
            self.deep_probability_control,
        )

        self.deep_scale_control = self._make_breath_control(
            minimum=1.0,
            maximum=3.0,
            value=spec.deep_breath_scale,
            step=0.01,
            decimals=2,
            suffix="×",
            field_name="deep_breath_scale",
        )
        breath_form.addRow(
            "Deep-breath scale:",
            self.deep_scale_control,
        )

        self.long_rest_probability_control = self._make_breath_control(
            minimum=0.0,
            maximum=0.10,
            value=spec.long_rest_probability,
            step=0.001,
            decimals=3,
            suffix="",
            field_name="long_rest_probability",
        )
        breath_form.addRow(
            "Long-rest chance:",
            self.long_rest_probability_control,
        )

        self.long_rest_scale_control = self._make_breath_control(
            minimum=1.0,
            maximum=5.0,
            value=spec.long_rest_scale,
            step=0.05,
            decimals=2,
            suffix="×",
            field_name="long_rest_scale",
        )
        breath_form.addRow(
            "Long-rest scale:",
            self.long_rest_scale_control,
        )

        self.shallow_probability_control = self._make_breath_control(
            minimum=0.0,
            maximum=0.10,
            value=spec.shallow_breath_probability,
            step=0.001,
            decimals=3,
            suffix="",
            field_name="shallow_breath_probability",
        )
        breath_form.addRow(
            "Shallow-breath chance:",
            self.shallow_probability_control,
        )

        self.shallow_scale_control = self._make_breath_control(
            minimum=0.2,
            maximum=1.0,
            value=spec.shallow_breath_scale,
            step=0.01,
            decimals=2,
            suffix="×",
            field_name="shallow_breath_scale",
        )
        breath_form.addRow(
            "Shallow-breath scale:",
            self.shallow_scale_control,
        )

        reset_button = QPushButton("Reset breath defaults")
        reset_button.clicked.connect(self._reset_breath_defaults)
        breath_form.addRow("", reset_button)

        controls_layout.addWidget(self.breath_panel)
        layout.addWidget(controls_box)

        # ------------------------------------------------------------------
        # Transport
        # ------------------------------------------------------------------

        transport = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)

        transport.addWidget(self.start_button)
        transport.addWidget(self.stop_button)
        layout.addLayout(transport)

        # ------------------------------------------------------------------
        # Offline export
        # ------------------------------------------------------------------

        export_box = QGroupBox("Offline export")
        export_layout = QVBoxLayout(export_box)

        duration_row = QHBoxLayout()
        duration_row.addWidget(QLabel("Duration:"))

        self.export_duration_slider = QSlider(
            Qt.Orientation.Horizontal
        )
        self.export_duration_slider.setRange(5, 360)
        self.export_duration_slider.setSingleStep(5)
        self.export_duration_slider.setPageStep(15)
        self.export_duration_slider.setValue(
            int(self.loaded_settings.get("export_duration_minutes", 60))
        )

        self.export_duration_label = QLabel("")
        self.export_duration_label.setMinimumWidth(80)

        duration_row.addWidget(self.export_duration_slider, 1)
        duration_row.addWidget(self.export_duration_label)
        export_layout.addLayout(duration_row)

        export_buttons = QHBoxLayout()
        self.export_button = QPushButton("Export audio…")
        self.cancel_export_button = QPushButton("Cancel export")
        self.cancel_export_button.setEnabled(False)

        export_buttons.addWidget(self.export_button)
        export_buttons.addWidget(self.cancel_export_button)
        export_layout.addLayout(export_buttons)

        self.export_progress = QProgressBar()
        self.export_progress.setRange(0, 100)
        self.export_progress.setValue(0)
        self.export_progress.setTextVisible(True)
        export_layout.addWidget(self.export_progress)

        self.export_status_label = QLabel(
            "Exports the current settings as stereo 16-bit WAV. "
            "Rendering runs faster than real time."
        )
        self.export_status_label.setWordWrap(True)
        export_layout.addWidget(self.export_status_label)

        layout.addWidget(export_box)

        # ------------------------------------------------------------------
        # Status
        # ------------------------------------------------------------------

        status_box = QGroupBox("Live status")
        status_form = QFormLayout(status_box)

        self.playback_label = QLabel("Stopped")
        self.mode_label = QLabel("")
        self.correlation_label = QLabel("—")
        self.breath_label = QLabel("—")
        self.breath_evolution_label = QLabel("—")

        status_form.addRow("Playback:", self.playback_label)
        status_form.addRow("Active path:", self.mode_label)
        status_form.addRow("Correlation:", self.correlation_label)
        self.pipeline_label = QLabel(
            "Fixed spectral anchors; sample-ramped mixing; correlation afterward"
        )
        status_form.addRow("DSP pipeline:", self.pipeline_label)
        status_form.addRow("Breath:", self.breath_label)
        status_form.addRow(
            "Breath prominence:",
            self.breath_evolution_label,
        )

        layout.addWidget(status_box)
        layout.addStretch()

        # ------------------------------------------------------------------
        # Signals
        # ------------------------------------------------------------------

        self.noise_expand_button.toggled.connect(
            self._toggle_noise_panel
        )
        self.noise_evolution_checkbox.toggled.connect(
            self._on_noise_evolution_toggled
        )
        self.body_movement_checkbox.toggled.connect(
            self._on_body_movement_toggled
        )
        self.stereo_checkbox.toggled.connect(self._on_modes_changed)
        self.correlation_checkbox.toggled.connect(
            self._on_modes_changed
        )
        self.breath_checkbox.toggled.connect(self._on_modes_changed)
        self.breath_evolution_checkbox.toggled.connect(
            self._on_breath_evolution_toggled
        )

        self.motion_expand_button.toggled.connect(
            self._toggle_motion_panel
        )
        self.breath_evolution_expand_button.toggled.connect(
            self._toggle_breath_evolution_panel
        )
        self.breath_expand_button.toggled.connect(
            self._toggle_breath_panel
        )

        self.start_button.clicked.connect(self._start)
        self.stop_button.clicked.connect(self._stop)
        self.export_button.clicked.connect(self._start_export)
        self.cancel_export_button.clicked.connect(self._cancel_export)
        self.export_duration_slider.valueChanged.connect(
            self._on_export_duration_changed
        )

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh_status)
        self.timer.start(100)

        self._toggle_noise_panel(
            self.noise_expand_button.isChecked()
        )
        self._toggle_motion_panel(
            self.motion_expand_button.isChecked()
        )
        self._toggle_breath_evolution_panel(
            self.breath_evolution_expand_button.isChecked()
        )
        self._toggle_breath_panel(
            self.breath_expand_button.isChecked()
        )
        self._on_export_duration_changed(
            self.export_duration_slider.value()
        )
        self._set_manual_noise_controls_enabled(
            not self.noise_evolution_checkbox.isChecked()
        )
        self._update_noise_status()
        self._on_modes_changed()

    def _update_body_movement(self, **changes) -> None:
        self.body_movement_state.update(**changes)
        self.mixer.body_movement.reschedule()
        self._update_noise_status()
        self._schedule_settings_save()

    def _on_body_movement_toggled(self, checked: bool) -> None:
        self.body_movement_state.update(enabled=bool(checked))
        if checked:
            self.mixer.body_movement.reschedule()
        self._update_noise_status()
        self._schedule_settings_save()

    def _update_noise_evolution(self, **changes) -> None:
        self.noise_evolution_state.update(**changes)
        self._update_noise_status()
        self._schedule_settings_save()

    def _set_manual_noise_controls_enabled(
        self,
        enabled: bool,
    ) -> None:
        for control in (
            self.noise_body_control,
            self.noise_slope_control,
            self.noise_low_end_control,
            self.noise_texture_control,
        ):
            control.setEnabled(enabled)

    def _on_noise_evolution_toggled(self, checked: bool) -> None:
        self.noise_evolution_state.update(enabled=bool(checked))
        self._set_manual_noise_controls_enabled(not checked)
        self._update_noise_status()
        self._schedule_settings_save()

    def _update_noise_parameter(self, **changes: float) -> None:
        self.noise_state.update(**changes)
        self._update_noise_status()
        self._schedule_settings_save()

    def _update_noise_status(self) -> None:
        spec, _ = self.noise_state.get()
        evolution = self.noise_evolution_state.get()

        gain_ceiling_note = (
            " (ceiling)"
            if spec.compensated_gain_db >= 23.999
            else ""
        )
        self.noise_body_status.setText(
            f"{spec.spectral_shift:.3f}× shift, "
            f"{spec.compensated_gain_db:.1f} dB generator gain"
            f"{gain_ceiling_note}"
        )

        if evolution.enabled:
            self.noise_evolution_status.setText(
                f"enabled; broad timescale ≈ "
                f"{evolution.time_scale:.1f} s"
            )
        else:
            self.noise_evolution_status.setText("disabled")

        movement = self.body_movement_state.get()
        minimum, maximum = movement.interval_range_seconds
        if movement.enabled:
            self.body_movement_status.setText(
                f"enabled; interval ≈ {minimum:.1f}–{maximum:.1f} s"
            )
        else:
            self.body_movement_status.setText("disabled")

    def _toggle_noise_panel(self, expanded: bool) -> None:
        self.noise_panel.setVisible(expanded)
        self.noise_expand_button.setArrowType(
            Qt.ArrowType.DownArrow
            if expanded
            else Qt.ArrowType.RightArrow
        )
        self._schedule_settings_save()

    def _reset_noise_defaults(self) -> None:
        spec = BrownNoiseSpec()
        evolution = BrownNoiseEvolutionSpec()
        movement = BodyMovementSpec()

        self.noise_state.set(spec)
        self.noise_evolution_state.set(evolution)
        self.body_movement_state.set(movement)

        controls = (
            (self.noise_body_control, spec.body),
            (self.noise_slope_control, spec.slope_strength),
            (
                self.noise_low_end_control,
                spec.low_end_emphasis_db,
            ),
            (
                self.noise_texture_control,
                spec.upper_texture,
            ),
            (
                self.noise_evolution_rate_control,
                evolution.rate,
            ),
            (
                self.body_movement_frequency_control,
                movement.frequency,
            ),
        )

        for control, value in controls:
            control.set_value(value, notify=False)

        self.noise_evolution_checkbox.setChecked(
            evolution.enabled
        )
        self.body_movement_checkbox.setChecked(movement.enabled)
        self.mixer.body_movement.reschedule()

        self._update_noise_status()
        self._schedule_settings_save()

    def _update_motion_parameter(self, **changes: float) -> None:
        self.motion_state.update(**changes)
        self._schedule_settings_save()

    def _toggle_motion_panel(self, expanded: bool) -> None:
        self.motion_panel.setVisible(expanded)
        self.motion_expand_button.setArrowType(
            Qt.ArrowType.DownArrow
            if expanded
            else Qt.ArrowType.RightArrow
        )
        self._schedule_settings_save()

    def _reset_motion_defaults(self) -> None:
        spec = OrganicMotionSpec()
        self.motion_state.set(spec)

        self.motion_period_control.set_value(
            spec.natural_period_seconds,
            notify=False,
        )
        self.motion_damping_control.set_value(
            spec.damping_ratio,
            notify=False,
        )
        self.motion_drive_control.set_value(
            spec.drive_strength,
            notify=False,
        )
        self.motion_smoothing_control.set_value(
            spec.drive_smoothing_seconds,
            notify=False,
        )
        self.motion_limit_control.set_value(
            spec.soft_limit,
            notify=False,
        )

        self._schedule_settings_save()

    def _update_breath_evolution(self, **changes) -> None:
        spec = self.breath_evolution_state.get()

        # Keep min/max pairs valid during live slider movement.
        if "multiplier_min" in changes:
            value = float(changes["multiplier_min"])
            if value > spec.multiplier_max:
                self.evolution_max_control.set_value(
                    value,
                    notify=False,
                )
                changes["multiplier_max"] = value

        if "multiplier_max" in changes:
            value = float(changes["multiplier_max"])
            if value < spec.multiplier_min:
                self.evolution_min_control.set_value(
                    value,
                    notify=False,
                )
                changes["multiplier_min"] = value

        if "period_min_seconds" in changes:
            value = float(changes["period_min_seconds"])
            if value > spec.period_max_seconds:
                self.evolution_period_max_control.set_value(
                    value,
                    notify=False,
                )
                changes["period_max_seconds"] = value

        if "period_max_seconds" in changes:
            value = float(changes["period_max_seconds"])
            if value < spec.period_min_seconds:
                self.evolution_period_min_control.set_value(
                    value,
                    notify=False,
                )
                changes["period_min_seconds"] = value

        self.breath_evolution_state.update(**changes)
        self._schedule_settings_save()

    def _on_breath_evolution_toggled(self, checked: bool) -> None:
        self.breath_evolution_state.update(
            enabled=bool(checked)
        )
        self._schedule_settings_save()

    def _toggle_breath_evolution_panel(
        self,
        expanded: bool,
    ) -> None:
        self.breath_evolution_panel.setVisible(expanded)
        self.breath_evolution_expand_button.setArrowType(
            Qt.ArrowType.DownArrow
            if expanded
            else Qt.ArrowType.RightArrow
        )
        self._schedule_settings_save()

    def _reset_breath_evolution_defaults(self) -> None:
        spec = BreathEvolutionSpec()
        self.breath_evolution_state.set(spec)

        self.breath_evolution_checkbox.setChecked(spec.enabled)
        self.evolution_min_control.set_value(
            spec.multiplier_min,
            notify=False,
        )
        self.evolution_max_control.set_value(
            spec.multiplier_max,
            notify=False,
        )
        self.evolution_period_min_control.set_value(
            spec.period_min_seconds,
            notify=False,
        )
        self.evolution_period_max_control.set_value(
            spec.period_max_seconds,
            notify=False,
        )
        self.evolution_curve_control.set_value(
            spec.curve_power,
            notify=False,
        )

        self._schedule_settings_save()

    def _update_breath_parameter(self, **changes: float) -> None:
        self.breath_state.update(**changes)
        self._schedule_settings_save()

    def _make_breath_control(
        self,
        *,
        minimum: float,
        maximum: float,
        value: float,
        step: float,
        decimals: int,
        suffix: str,
        field_name: str,
    ) -> FloatControl:
        return FloatControl(
            minimum=minimum,
            maximum=maximum,
            value=value,
            step=step,
            decimals=decimals,
            suffix=suffix,
            on_change=lambda new_value, name=field_name: (
                self._update_breath_parameter(
                    **{name: new_value}
                )
            ),
        )

    def _toggle_breath_panel(self, expanded: bool) -> None:
        self.breath_panel.setVisible(expanded)
        self.breath_expand_button.setArrowType(
            Qt.ArrowType.DownArrow
            if expanded
            else Qt.ArrowType.RightArrow
        )
        self._schedule_settings_save()

    def _reset_breath_defaults(self) -> None:
        spec = self.default_breath_spec
        self.breath_state.set(spec)

        self.gain_depth_control.set_value(
            spec.gain_range_db,
            notify=False,
        )
        self.spectral_depth_control.set_value(
            spec.spectral_depth,
            notify=False,
        )
        self.width_depth_control.set_value(
            spec.width_depth,
            notify=False,
        )

        biological_controls = (
            (
                self.inhale_mean_control,
                spec.inhale_mean_seconds,
            ),
            (
                self.hold_mean_control,
                spec.hold_mean_seconds,
            ),
            (
                self.exhale_mean_control,
                spec.exhale_mean_seconds,
            ),
            (
                self.rest_mean_control,
                spec.rest_mean_seconds,
            ),
            (
                self.timing_variation_control,
                spec.timing_variation,
            ),
            (
                self.timing_memory_control,
                spec.timing_memory,
            ),
            (
                self.depth_variation_control,
                spec.depth_variation,
            ),
            (
                self.depth_memory_control,
                spec.depth_memory,
            ),
            (
                self.deep_probability_control,
                spec.deep_breath_probability,
            ),
            (
                self.deep_scale_control,
                spec.deep_breath_scale,
            ),
            (
                self.long_rest_probability_control,
                spec.long_rest_probability,
            ),
            (
                self.long_rest_scale_control,
                spec.long_rest_scale,
            ),
            (
                self.shallow_probability_control,
                spec.shallow_breath_probability,
            ),
            (
                self.shallow_scale_control,
                spec.shallow_breath_scale,
            ),
        )

        for control, value in biological_controls:
            control.set_value(value, notify=False)

        self._schedule_settings_save()

    def _on_modes_changed(self) -> None:
        stereo = self.stereo_checkbox.isChecked()

        self.mode_state.set(
            stereo_enabled=stereo,
            correlation_enabled=self.correlation_checkbox.isChecked(),
            breath_enabled=self.breath_checkbox.isChecked(),
        )

        if not stereo:
            path = "Mono duplicated L/R"
        elif self.correlation_checkbox.isChecked():
            path = (
                "Stereo: shared + independent, evolving correlation"
            )
        else:
            path = "Stereo: fully independent left/right"

        if self.breath_checkbox.isChecked():
            path += " + breath"

        self.mode_label.setText(path)
        self._schedule_settings_save()

    def _schedule_settings_save(self) -> None:
        self.settings_save_timer.start(250)

    def _save_settings(self) -> None:
        noise_spec, _ = self.noise_state.get()
        noise_evolution_spec = self.noise_evolution_state.get()
        body_movement_spec = self.body_movement_state.get()
        breath_spec, _ = self.breath_state.get()
        breath_evolution_spec = self.breath_evolution_state.get()
        motion_spec = self.motion_state.get()
        modes = self.mode_state.get()

        data = {
            "version": 2,
            "modes": asdict(modes),
            "brown_noise": asdict(noise_spec),
            "brown_noise_evolution": asdict(noise_evolution_spec),
            "body_movement": asdict(body_movement_spec),
            "noise_panel_expanded": (
                self.noise_expand_button.isChecked()
            ),
            "breath": asdict(breath_spec),
            "breath_evolution": asdict(breath_evolution_spec),
            "breath_evolution_panel_expanded": (
                self.breath_evolution_expand_button.isChecked()
            ),
            "organic_motion": asdict(motion_spec),
            "motion_panel_expanded": (
                self.motion_expand_button.isChecked()
            ),
            "breath_panel_expanded": (
                self.breath_expand_button.isChecked()
            ),
            "export_duration_minutes": (
                self.export_duration_slider.value()
            ),
        }

        try:
            self.settings_store.save(data)
        except Exception as exc:
            self.export_status_label.setText(
                f"Could not save settings: {exc}"
            )

    @staticmethod
    def _format_duration(minutes: int) -> str:
        if minutes < 60:
            return f"{minutes} min"

        hours, remainder = divmod(minutes, 60)
        if remainder == 0:
            return f"{hours} h"

        return f"{hours} h {remainder} min"

    def _on_export_duration_changed(self, minutes: int) -> None:
        self.export_duration_label.setText(
            self._format_duration(minutes)
        )
        self._schedule_settings_save()

    def _start_export(self) -> None:
        if self.export_worker is not None:
            return

        suggested_name = (
            f"living-brown-noise-"
            f"{self.export_duration_slider.value()}min.wav"
        )

        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export living brown noise",
            str(Path.home() / suggested_name),
            "Wave audio (*.wav)",
        )

        if not output_path:
            return

        if not output_path.lower().endswith(".wav"):
            output_path += ".wav"

        # Free the audio device and CPU for the renderer.
        self._stop()

        modes = self.mode_state.get()
        noise_spec, _ = self.noise_state.get()
        noise_evolution_spec = self.noise_evolution_state.get()
        body_movement_spec = self.body_movement_state.get()
        breath_spec, _ = self.breath_state.get()
        breath_evolution_spec = self.breath_evolution_state.get()
        motion_spec = self.motion_state.get()

        self.export_worker = ExportWorker(
            output_path=output_path,
            duration_minutes=self.export_duration_slider.value(),
            sample_rate=44_100,
            modes=modes,
            noise_spec=noise_spec,
            noise_evolution_spec=noise_evolution_spec,
            body_movement_spec=body_movement_spec,
            breath_spec=breath_spec,
            breath_evolution_spec=breath_evolution_spec,
            motion_spec=motion_spec,
        )

        self.export_worker.progress_changed.connect(
            self.export_progress.setValue
        )
        self.export_worker.export_finished.connect(
            self._export_finished
        )
        self.export_worker.export_failed.connect(
            self._export_failed
        )
        self.export_worker.export_cancelled.connect(
            self._export_cancelled
        )

        self.export_button.setEnabled(False)
        self.cancel_export_button.setEnabled(True)
        self.start_button.setEnabled(False)
        self.export_progress.setValue(0)
        self.export_status_label.setText(
            "Rendering current settings…"
        )

        self.export_worker.start()

    def _cancel_export(self) -> None:
        if self.export_worker is not None:
            self.export_worker.request_cancel()
            self.cancel_export_button.setEnabled(False)
            self.export_status_label.setText(
                "Cancelling after the current render chunk…"
            )

    def _finish_export_ui(self) -> None:
        worker = self.export_worker
        self.export_worker = None

        self.export_button.setEnabled(True)
        self.cancel_export_button.setEnabled(False)
        self.start_button.setEnabled(True)

        if worker is not None:
            worker.deleteLater()

    def _export_finished(self, output_path: str) -> None:
        self.export_progress.setValue(100)
        self.export_status_label.setText(
            f"Export complete: {output_path}"
        )
        self._finish_export_ui()

        QMessageBox.information(
            self,
            "Export complete",
            f"Audio written to:\n{output_path}",
        )

    def _export_failed(self, message: str) -> None:
        self.export_status_label.setText(
            f"Export failed: {message}"
        )
        self._finish_export_ui()

        QMessageBox.critical(
            self,
            "Export failed",
            message,
        )

    def _export_cancelled(self) -> None:
        self.export_progress.setValue(0)
        self.export_status_label.setText("Export cancelled.")
        self._finish_export_ui()

    def _start(self) -> None:
        try:
            self.engine.start()
        except Exception as exc:
            self.playback_label.setText(f"Start failed: {exc}")
            return

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.playback_label.setText("Running")

    def _stop(self) -> None:
        self.engine.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.playback_label.setText("Stopped")

    def _refresh_status(self) -> None:
        if self.engine.callback_error is not None:
            self.playback_label.setText(
                f"Audio error: {self.engine.callback_error}"
            )

        self.correlation_label.setText(
            f"{self.mixer.current_correlation:.3f}"
        )
        self.breath_label.setText(
            f"{self.mixer.current_breath:.3f} "
            f"({self.mixer.current_breath_stage}; "
            f"{self.mixer.breath.current_event})"
        )
        self.breath_evolution_label.setText(
            f"{self.mixer.current_breath_prominence:.2f}× "
            f"({self.mixer.current_breath_evolution_period:.0f} s cycle)"
        )
        evolution = self.noise_evolution_state.get()
        if evolution.enabled:
            self.noise_evolution_status.setText(
                f"Body {self.mixer.current_noise_body:.2f}, "
                f"Slope {self.mixer.current_noise_slope:.2f}, "
                f"Weight {self.mixer.current_noise_weight:.1f} dB, "
                f"Texture {self.mixer.current_noise_texture:.2f}"
            )

        if self.body_movement_state.get().enabled:
            if self.mixer.current_body_movement_count == 0:
                self.body_movement_status.setText("waiting for first movement")
            else:
                self.body_movement_status.setText(
                    f"events {self.mixer.current_body_movement_count}; "
                    f"last strength {self.mixer.current_body_movement_strength:.2f}; "
                    f"{self.mixer.current_body_movement_age:.1f} s ago"
                )

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._save_settings()
        self.engine.stop()
        if self.export_worker is not None:
            self.export_worker.request_cancel()
            self.export_worker.wait(5000)
        event.accept()


# =============================================================================
# Main
# =============================================================================

def build_application() -> tuple[QApplication, MainWindow]:
    sample_rate = 44_100

    app = QApplication(sys.argv)

    settings_store = SettingsStore()
    loaded = settings_store.load()

    default_modes = EngineModes()
    mode_data = loaded.get("modes", {})
    try:
        modes = EngineModes(
            stereo_enabled=bool(
                mode_data.get(
                    "stereo_enabled",
                    default_modes.stereo_enabled,
                )
            ),
            correlation_enabled=bool(
                mode_data.get(
                    "correlation_enabled",
                    default_modes.correlation_enabled,
                )
            ),
            breath_enabled=bool(
                mode_data.get(
                    "breath_enabled",
                    default_modes.breath_enabled,
                )
            ),
        )
    except Exception:
        modes = default_modes

    default_noise = BrownNoiseSpec()
    noise_data = loaded.get("brown_noise", {})
    try:
        noise_spec = BrownNoiseSpec(
            **{
                field_name: noise_data.get(
                    field_name,
                    getattr(default_noise, field_name),
                )
                for field_name in asdict(default_noise)
            }
        ).validated(sample_rate)
    except Exception:
        noise_spec = default_noise

    default_noise_evolution = BrownNoiseEvolutionSpec()
    noise_evolution_data = loaded.get(
        "brown_noise_evolution",
        {},
    )
    try:
        noise_evolution_spec = BrownNoiseEvolutionSpec(
            **{
                field_name: noise_evolution_data.get(
                    field_name,
                    getattr(default_noise_evolution, field_name),
                )
                for field_name in asdict(default_noise_evolution)
            }
        ).validated()
    except Exception:
        noise_evolution_spec = default_noise_evolution

    default_body_movement = BodyMovementSpec()
    body_movement_data = loaded.get("body_movement", {})
    try:
        body_movement_spec = BodyMovementSpec(
            **{
                field_name: body_movement_data.get(
                    field_name,
                    getattr(default_body_movement, field_name),
                )
                for field_name in asdict(default_body_movement)
            }
        ).validated()
    except Exception:
        body_movement_spec = default_body_movement

    default_breath_evolution = BreathEvolutionSpec()
    breath_evolution_data = loaded.get(
        "breath_evolution",
        {},
    )
    try:
        breath_evolution_spec = BreathEvolutionSpec(
            **{
                field_name: breath_evolution_data.get(
                    field_name,
                    getattr(default_breath_evolution, field_name),
                )
                for field_name in asdict(default_breath_evolution)
            }
        ).validated()
    except Exception:
        breath_evolution_spec = default_breath_evolution

    default_motion = OrganicMotionSpec()
    motion_data = loaded.get("organic_motion", {})
    try:
        motion_spec = OrganicMotionSpec(
            **{
                field_name: motion_data.get(
                    field_name,
                    getattr(default_motion, field_name),
                )
                for field_name in asdict(default_motion)
            }
        ).validated()
    except Exception:
        motion_spec = default_motion

    default_breath = BreathSpec()
    breath_data = dict(loaded.get("breath", {}))

    old_breath_defaults = {
        "inhale_mean_seconds": 1.05,
        "hold_mean_seconds": 0.08,
        "exhale_mean_seconds": 1.65,
        "rest_mean_seconds": 0.50,
        "timing_variation": 0.18,
        "timing_memory": 0.72,
    }

    new_breath_defaults = {
        "inhale_mean_seconds": default_breath.inhale_mean_seconds,
        "hold_mean_seconds": default_breath.hold_mean_seconds,
        "exhale_mean_seconds": default_breath.exhale_mean_seconds,
        "rest_mean_seconds": default_breath.rest_mean_seconds,
        "timing_variation": default_breath.timing_variation,
        "timing_memory": default_breath.timing_memory,
    }

    for field_name, old_value in old_breath_defaults.items():
        saved_value = breath_data.get(field_name)
        if (
            isinstance(saved_value, (int, float))
            and abs(float(saved_value) - old_value) < 1e-9
        ):
            breath_data[field_name] = new_breath_defaults[field_name]

    try:
        breath_spec = BreathSpec(
            **{
                field_name: breath_data.get(
                    field_name,
                    getattr(default_breath, field_name),
                )
                for field_name in asdict(default_breath)
            }
        ).validated()
    except Exception:
        breath_spec = default_breath

    (
        mixer,
        mode_state,
        noise_state,
        noise_evolution_state,
        body_movement_state,
        breath_state,
        breath_evolution_state,
        motion_state,
    ) = build_mixer(
        sample_rate=sample_rate,
        modes=modes,
        noise_spec=noise_spec,
        noise_evolution_spec=noise_evolution_spec,
        body_movement_spec=body_movement_spec,
        breath_spec=breath_spec,
        breath_evolution_spec=breath_evolution_spec,
        motion_spec=motion_spec,
        seed_base=1000,
    )

    engine = AudioEngine(
        mixer=mixer,
        sample_rate=sample_rate,
        block_size=2_048,
    )

    window = MainWindow(
        engine=engine,
        mode_state=mode_state,
        noise_state=noise_state,
        noise_evolution_state=noise_evolution_state,
        body_movement_state=body_movement_state,
        breath_state=breath_state,
        breath_evolution_state=breath_evolution_state,
        motion_state=motion_state,
        mixer=mixer,
        settings_store=settings_store,
        loaded_settings=loaded,
    )

    return app, window


def main() -> int:
    app, window = build_application()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
