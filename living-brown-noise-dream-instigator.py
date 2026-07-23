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
import av
from steam_audio_renderer import SteamAudioRenderer, Vector3
from scipy import signal
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
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


# Requires phonon.dll and steam_audio_renderer.py beside this script.
#
# Edit this path to point at the folder containing the WAV files you want
# available in the Soundscape Sample Test dropdown.
SOUND_EFFECTS_DIRECTORY = Path(
    r"D:\github\living-brown-noise-dream-instigator\sounds"
)

SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".oga",
    ".m4a",
    ".aac",
    ".aiff",
    ".aif",
}

# Files at or below this duration are catalogued as layered motif events.
DREAM_MOTIF_LAYER_THRESHOLD_SECONDS = 10.0

# =============================================================================
# Steam Audio dual brown-source baseline
# =============================================================================

STEAM_SPATIAL_FRAME_SIZE = 2_048
STEAM_DEFAULT_SOURCE_POSITION = Vector3(0.0, 0.0, -2.0)

# Fixed, deliberately wide positions for proving the dual-body architecture.
# The viscous-fluid motion system will replace these constants later.
STEAM_BROWN_LEFT_POSITION = Vector3(-2.75, 0.0, -2.0)
STEAM_BROWN_RIGHT_POSITION = Vector3(2.75, 0.0, -2.0)

STEAM_HEARTBEAT_SPATIAL_BLEND = 1.0
STEAM_SOUNDSCAPE_SPATIAL_AMOUNT = 0.06

HEARTBEAT_DISTANCE_MIN_METERS = 0.15
HEARTBEAT_DISTANCE_MAX_METERS = 4.0
HEARTBEAT_DISTANCE_DEFAULT_METERS = 0.75
HEARTBEAT_HORIZONTAL_MIN_METERS = -2.5
HEARTBEAT_HORIZONTAL_MAX_METERS = 2.5
HEARTBEAT_HORIZONTAL_DEFAULT_METERS = 0.0
HEARTBEAT_VERTICAL_MIN_METERS = -2.0
HEARTBEAT_VERTICAL_MAX_METERS = 2.0
HEARTBEAT_VERTICAL_DEFAULT_METERS = -0.25

# The moving 3D bodies are an additive texture over the complete correlated
# stereo foundation, not a replacement for it. Their amount is controlled live.


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

        # Metabolism may change this later, but it must exist before the first
        # call to _duration_for_stage during construction.
        self.external_tempo_multiplier = 1.0

        spec, version = self.breath_state.get()
        self._seen_spec_version = version
        self._choose_new_cycle(spec)
        self.stage_duration_samples = self._duration_for_stage(
            self.stage,
            spec,
        )

        self.current_value = 0.0

    def set_external_tempo_multiplier(
        self,
        multiplier: float,
    ) -> None:
        multiplier = float(
            np.clip(multiplier, 0.25, 5.0)
        )

        if abs(
            multiplier - self.external_tempo_multiplier
        ) < 1e-6:
            return

        old_duration = max(1, self.stage_duration_samples)
        progress = min(
            1.0,
            self.stage_position_samples / old_duration,
        )

        self.external_tempo_multiplier = multiplier

        spec, _ = self.breath_state.get()
        self.stage_duration_samples = self._duration_for_stage(
            self.stage,
            spec,
        )
        self.stage_position_samples = int(
            progress * self.stage_duration_samples
        )

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

        seconds *= self.external_tempo_multiplier

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
# Heartbeat / pulse layer
# =============================================================================

@dataclass(frozen=True, slots=True)
class HeartbeatSpec:
    """
    Production heartbeat configuration.

    The user-facing controls are intentionally limited to an enable checkbox.
    Heart rate and prominence evolve independently inside these fixed,
    artistically approved ranges.
    """

    rate_min_bpm: float = 30.0
    rate_max_bpm: float = 60.0

    prominence_min: float = 0.0
    prominence_max: float = 0.68

    rate_evolution_min_seconds: float = 120.0
    rate_evolution_max_seconds: float = 420.0

    prominence_evolution_min_seconds: float = 90.0
    prominence_evolution_max_seconds: float = 360.0

    def validated(self) -> HeartbeatSpec:
        if not 20.0 <= self.rate_min_bpm < self.rate_max_bpm <= 100.0:
            raise ValueError("invalid heartbeat rate range")
        if not 0.0 <= self.prominence_min < self.prominence_max <= 1.0:
            raise ValueError("invalid heartbeat prominence range")
        if not (
            1.0
            <= self.rate_evolution_min_seconds
            <= self.rate_evolution_max_seconds
        ):
            raise ValueError("invalid heartbeat rate evolution range")
        if not (
            1.0
            <= self.prominence_evolution_min_seconds
            <= self.prominence_evolution_max_seconds
        ):
            raise ValueError("invalid heartbeat prominence evolution range")
        return self


class HeartbeatState:
    def __init__(self, spec: HeartbeatSpec) -> None:
        self._lock = threading.Lock()
        self._spec = spec.validated()

    def get(self) -> HeartbeatSpec:
        with self._lock:
            return self._spec

    def set(self, spec: HeartbeatSpec) -> None:
        with self._lock:
            self._spec = spec.validated()


class SmoothRandomJourney:
    """
    Continuous random travel between targets.

    Each segment uses a cosine ease, so value and slope are both continuous at
    the ends. A beta distribution can bias the journey toward a preferred part
    of its range without creating a fixed center or periodic oscillator.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        initial_value: float,
        minimum: float,
        maximum: float,
        duration_min_seconds: float,
        duration_max_seconds: float,
        beta_a: float,
        beta_b: float,
    ) -> None:
        self.rng = rng
        self.minimum = minimum
        self.maximum = maximum
        self.duration_min_seconds = duration_min_seconds
        self.duration_max_seconds = duration_max_seconds
        self.beta_a = beta_a
        self.beta_b = beta_b

        self.start_value = initial_value
        self.current_value = initial_value
        self.target_value = initial_value
        self.elapsed = 0.0
        self.duration = 1.0
        self._choose_next_target(initial=True)

    def _choose_next_target(self, initial: bool = False) -> None:
        if not initial:
            self.start_value = self.current_value

        normalized = float(
            self.rng.beta(self.beta_a, self.beta_b)
        )
        self.target_value = (
            self.minimum
            + normalized * (self.maximum - self.minimum)
        )

        self.duration = float(
            self.rng.uniform(
                self.duration_min_seconds,
                self.duration_max_seconds,
            )
        )
        self.elapsed = 0.0

    def advance(self, elapsed_seconds: float) -> float:
        remaining = max(0.0, elapsed_seconds)

        while remaining > 0.0:
            available = self.duration - self.elapsed
            step = min(remaining, available)
            self.elapsed += step
            remaining -= step

            position = min(1.0, self.elapsed / self.duration)
            blend = 0.5 - 0.5 * math.cos(math.pi * position)

            self.current_value = (
                self.start_value
                + (self.target_value - self.start_value) * blend
            )

            if self.elapsed >= self.duration - 1e-9:
                self.current_value = self.target_value
                self._choose_next_target()

        return self.current_value


class HeartbeatGenerator:
    """
    Procedural resonant heartbeat instrument.

    Each cardiac cycle creates a low, physical "lub" followed by a smaller,
    slightly brighter "dub". The sound is synthesized from decaying resonant
    modes plus extremely faint valve and turbulence detail. Beat strength,
    timing, pitch, decay, and lub/dub spacing vary subtly so long playback does
    not expose a repeated sample.

    The existing slow rate and prominence journeys remain intact.
    """

    def __init__(
        self,
        sample_rate: float,
        heartbeat_state: HeartbeatState,
        seed: int = 88001,
    ) -> None:
        self.sample_rate = float(sample_rate)
        self.heartbeat_state = heartbeat_state
        self.rng = np.random.default_rng(seed)

        spec = self.heartbeat_state.get()

        self.absolute_sample = 0
        self.next_beat_sample = 0
        self.active_beats: list[dict[str, float]] = []

        self.current_envelope = 0.0
        self.current_rate_bpm = 50.0
        self.current_prominence = 0.24
        self.current_interval_seconds = 60.0 / self.current_rate_bpm

        self.rate_journey = SmoothRandomJourney(
            rng=self.rng,
            initial_value=50.0,
            minimum=spec.rate_min_bpm,
            maximum=spec.rate_max_bpm,
            duration_min_seconds=spec.rate_evolution_min_seconds,
            duration_max_seconds=spec.rate_evolution_max_seconds,
            beta_a=4.0,
            beta_b=3.2,
        )
        self.prominence_journey = SmoothRandomJourney(
            rng=self.rng,
            initial_value=0.24,
            minimum=spec.prominence_min,
            maximum=spec.prominence_max,
            duration_min_seconds=spec.prominence_evolution_min_seconds,
            duration_max_seconds=spec.prominence_evolution_max_seconds,
            beta_a=1.25,
            beta_b=2.15,
        )

    def _schedule_next_beat(self) -> None:
        mean_interval = 60.0 / max(1.0, self.current_rate_bpm)
        jitter = float(
            np.clip(
                self.rng.normal(0.0, 0.010),
                -0.022,
                0.022,
            )
        )
        interval = mean_interval * (1.0 + jitter)
        self.current_interval_seconds = interval
        self.next_beat_sample += max(
            1,
            int(interval * self.sample_rate),
        )

    def _new_beat(self, sample_index: int) -> dict[str, float]:
        strength = float(
            np.clip(self.rng.normal(1.0, 0.055), 0.84, 1.17)
        )
        pitch_scale = float(
            np.clip(self.rng.normal(1.0, 0.025), 0.94, 1.07)
        )
        decay_scale = float(
            np.clip(self.rng.normal(1.0, 0.075), 0.82, 1.20)
        )
        dub_delay = float(
            np.clip(self.rng.normal(0.185, 0.012), 0.155, 0.220)
        )
        dub_strength = float(
            np.clip(self.rng.normal(0.60, 0.055), 0.46, 0.73)
        )
        phase = float(self.rng.uniform(-0.10, 0.10))

        return {
            "sample": float(sample_index),
            "strength": strength,
            "pitch_scale": pitch_scale,
            "decay_scale": decay_scale,
            "dub_delay": dub_delay,
            "dub_strength": dub_strength,
            "phase": phase,
            "detail_seed": float(self.rng.uniform(0.0, 2.0 * math.pi)),
        }

    @staticmethod
    def _attack_decay(
        age: np.ndarray,
        attack_seconds: float,
        decay_seconds: float,
    ) -> np.ndarray:
        active = age >= 0.0
        envelope = np.zeros_like(age, dtype=np.float64)
        if not np.any(active):
            return envelope

        active_age = age[active]
        attack = 1.0 - np.exp(
            -active_age / max(1e-5, attack_seconds)
        )
        decay = np.exp(
            -active_age / max(1e-5, decay_seconds)
        )
        envelope[active] = attack * decay
        return envelope

    @staticmethod
    def _resonance(
        age: np.ndarray,
        frequency_hz: float,
        attack_seconds: float,
        decay_seconds: float,
        phase: float = 0.0,
    ) -> np.ndarray:
        envelope = HeartbeatGenerator._attack_decay(
            age,
            attack_seconds,
            decay_seconds,
        )
        return envelope * np.sin(
            2.0 * np.pi * frequency_hz * np.maximum(age, 0.0)
            + phase
        )

    def _render_beat(
        self,
        beat: dict[str, float],
        absolute_samples: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        beat_sample = int(beat["sample"])
        age = (
            absolute_samples - beat_sample
        ) / self.sample_rate

        strength = beat["strength"]
        pitch = beat["pitch_scale"]
        decay = beat["decay_scale"]
        phase = beat["phase"]

        # LUB: broad chest/body impact with several modes that settle at
        # different rates. The low modes provide physical weight; the upper
        # mode gives enough identity to remain perceptible in dense noise.
        lub = (
            1.00
            * self._resonance(
                age,
                38.0 * pitch,
                0.006,
                0.205 * decay,
                phase,
            )
            + 0.64
            * self._resonance(
                age,
                58.0 * pitch,
                0.004,
                0.145 * decay,
                phase * 0.7,
            )
            + 0.25
            * self._resonance(
                age,
                91.0 * pitch,
                0.003,
                0.088 * decay,
                phase * 0.35,
            )
        )

        # A non-oscillatory pressure component makes the first sound read as
        # a physical contraction rather than merely a low musical tone.
        pressure = self._attack_decay(
            age,
            0.0035,
            0.090 * decay,
        )
        pressure *= np.exp(
            -np.maximum(age, 0.0) / (0.18 * decay)
        )

        # DUB: delayed, lighter, and slightly brighter.
        dub_age = age - beat["dub_delay"]
        dub = beat["dub_strength"] * (
            0.82
            * self._resonance(
                dub_age,
                48.0 * pitch,
                0.004,
                0.125 * decay,
                -phase,
            )
            + 0.48
            * self._resonance(
                dub_age,
                76.0 * pitch,
                0.003,
                0.090 * decay,
                phase * 0.4,
            )
            + 0.18
            * self._resonance(
                dub_age,
                118.0 * pitch,
                0.002,
                0.052 * decay,
                -phase * 0.3,
            )
        )

        # Very faint valve detail. It is deliberately tonal/noisy enough to
        # identify the events, but far below the low resonant body.
        # Soft valve detail. Both components begin at a zero crossing and use
        # a several-millisecond attack so they add definition without producing
        # a phase-dependent click or pop at onset.
        click_lub = 0.022 * self._resonance(
            age,
            310.0 * pitch,
            0.0040,
            0.018,
            0.0,
        )
        click_dub = 0.014 * self._resonance(
            dub_age,
            390.0 * pitch,
            0.0045,
            0.015,
            0.0,
        )

        waveform = strength * (
            0.76 * lub
            + 0.28 * pressure
            + 0.72 * dub
            + click_lub
            + click_dub
        )

        event_envelope = np.maximum(
            self._attack_decay(age, 0.003, 0.24 * decay),
            self._attack_decay(dub_age, 0.002, 0.16 * decay),
        )

        # The resonances still contain a small amount of energy after their
        # audible body has ended. Never discard that tail abruptly. Fade every
        # beat smoothly to an exact zero between 0.58 and 0.92 seconds.
        release_start = 0.58
        release_end = 0.92
        release_position = np.clip(
            (age - release_start)
            / (release_end - release_start),
            0.0,
            1.0,
        )
        terminal_gain = 0.5 + 0.5 * np.cos(
            np.pi * release_position
        )
        terminal_gain[age < 0.0] = 0.0
        terminal_gain[age >= release_end] = 0.0

        waveform *= terminal_gain
        event_envelope *= terminal_gain

        return waveform, event_envelope

    def generate(self, frame_count: int) -> np.ndarray:
        elapsed_seconds = frame_count / self.sample_rate

        self.current_rate_bpm = self.rate_journey.advance(
            elapsed_seconds
        )
        self.current_prominence = self.prominence_journey.advance(
            elapsed_seconds
        )

        buffer_start = self.absolute_sample
        buffer_end = buffer_start + frame_count

        while self.next_beat_sample < buffer_end:
            self.active_beats.append(
                self._new_beat(self.next_beat_sample)
            )
            self._schedule_next_beat()

        absolute = np.arange(
            buffer_start,
            buffer_end,
            dtype=np.int64,
        )

        output = np.zeros(frame_count, dtype=np.float64)
        envelope = np.zeros(frame_count, dtype=np.float64)
        retained: list[dict[str, float]] = []

        # Must extend beyond the terminal fade's exact-zero endpoint.
        tail_seconds = 0.96

        for beat in self.active_beats:
            rendered, beat_envelope = self._render_beat(
                beat,
                absolute,
            )
            output += rendered
            envelope = np.maximum(envelope, beat_envelope)

            if (
                int(beat["sample"])
                + int(tail_seconds * self.sample_rate)
                >= buffer_end
            ):
                retained.append(beat)

        self.active_beats = retained
        self.absolute_sample = buffer_end
        self.current_envelope = float(envelope[-1])

        # Keep the existing long-form prominence evolution, but make the new
        # instrument substantially more audible than the former noise pulse.
        prominence = self.current_prominence
        gain = (
            0.18 * prominence
            + 2.85 * prominence * prominence
        )

        output *= gain

        # Gentle saturation supplies chest-like density and prevents rare
        # overlapping events from producing hard digital peaks.
        output = 0.88 * np.tanh(output * 1.35)

        return output.astype(np.float32, copy=False)


# =============================================================================
# Dream motif catalogue
# =============================================================================

@dataclass(frozen=True, slots=True)
class DreamMotifAsset:
    path: Path
    duration_seconds: float
    is_layered_event: bool


@dataclass(frozen=True, slots=True)
class DreamMotif:
    name: str
    directory: Path
    ambient_assets: tuple[DreamMotifAsset, ...]
    layered_assets: tuple[DreamMotifAsset, ...]

    @property
    def total_assets(self) -> int:
        return len(self.ambient_assets) + len(self.layered_assets)


class DreamMotifCatalog:
    """Scans immediate subfolders beneath the configured sounds directory."""

    def __init__(
        self,
        root_directory: Path,
        layer_threshold_seconds: float,
    ) -> None:
        self.root_directory = root_directory
        self.layer_threshold_seconds = float(layer_threshold_seconds)
        self.motifs: tuple[DreamMotif, ...] = ()
        self.errors: tuple[str, ...] = ()

    @staticmethod
    def _probe_duration(path: Path) -> float:
        container = av.open(str(path))
        try:
            streams = [s for s in container.streams if s.type == "audio"]
            if not streams:
                raise ValueError("no audio stream")

            stream = streams[0]
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
                if duration > 0.0:
                    return duration

            if container.duration is not None:
                duration = float(container.duration) / float(av.time_base)
                if duration > 0.0:
                    return duration

            # Metadata can be absent in some MP3/VBR files. Decode only enough
            # to determine the complete timestamp range as a reliable fallback.
            end_seconds = 0.0
            sample_rate = int(
                stream.codec_context.sample_rate
                or stream.rate
                or 44_100
            )
            decoded_samples = 0
            for frame in container.decode(stream):
                decoded_samples += int(frame.samples)
                if frame.pts is not None and frame.time_base is not None:
                    frame_end = float(frame.pts * frame.time_base)
                    frame_end += frame.samples / sample_rate
                    end_seconds = max(end_seconds, frame_end)

            if end_seconds > 0.0:
                return end_seconds
            if decoded_samples > 0:
                return decoded_samples / sample_rate
            raise ValueError("decoded no audio frames")
        finally:
            container.close()

    def scan(self) -> tuple[DreamMotif, ...]:
        self.root_directory.mkdir(parents=True, exist_ok=True)
        motifs: list[DreamMotif] = []
        errors: list[str] = []

        directories = sorted(
            (p for p in self.root_directory.iterdir() if p.is_dir()),
            key=lambda p: p.name.lower(),
        )

        for directory in directories:
            ambient: list[DreamMotifAsset] = []
            layered: list[DreamMotifAsset] = []

            files = sorted(
                (
                    p
                    for p in directory.iterdir()
                    if p.is_file()
                    and p.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
                ),
                key=lambda p: p.name.lower(),
            )

            for path in files:
                try:
                    duration = self._probe_duration(path)
                    asset = DreamMotifAsset(
                        path=path,
                        duration_seconds=duration,
                        is_layered_event=(
                            duration
                            <= self.layer_threshold_seconds
                        ),
                    )
                    if asset.is_layered_event:
                        layered.append(asset)
                    else:
                        ambient.append(asset)
                except Exception as exc:
                    errors.append(
                        f"{directory.name}/{path.name}: {exc}"
                    )

            motifs.append(
                DreamMotif(
                    name=directory.name,
                    directory=directory,
                    ambient_assets=tuple(ambient),
                    layered_assets=tuple(layered),
                )
            )

        self.motifs = tuple(motifs)
        self.errors = tuple(errors)
        return self.motifs

    def find(self, name: str) -> DreamMotif | None:
        for motif in self.motifs:
            if motif.name == name:
                return motif
        return None


# =============================================================================
# Soundscape sample test layer
# =============================================================================

@dataclass(frozen=True, slots=True)
class AmbientSampleSpec:
    selected_filename: str = ""

    fade_in_seconds: float = 8.0
    fade_out_seconds: float = 12.0

    duration_min_seconds: float = 45.0
    duration_max_seconds: float = 150.0

    silence_min_seconds: float = 20.0
    silence_max_seconds: float = 75.0

    volume_min_db: float = -34.0
    volume_max_db: float = -18.0

    # Time taken to drift between independently chosen volume targets.
    volume_walk_min_seconds: float = 20.0
    volume_walk_max_seconds: float = 75.0

    def validated(self) -> AmbientSampleSpec:
        if not 0.05 <= self.fade_in_seconds <= 600.0:
            raise ValueError("invalid fade-in time")
        if not 0.05 <= self.fade_out_seconds <= 600.0:
            raise ValueError("invalid fade-out time")
        if not (
            0.1
            <= self.duration_min_seconds
            <= self.duration_max_seconds
            <= 86_400.0
        ):
            raise ValueError("invalid active duration range")
        if not (
            0.0
            <= self.silence_min_seconds
            <= self.silence_max_seconds
            <= 86_400.0
        ):
            raise ValueError("invalid silence duration range")
        if not (
            -80.0
            <= self.volume_min_db
            <= self.volume_max_db
            <= 12.0
        ):
            raise ValueError("invalid volume range")
        if not (
            1.0
            <= self.volume_walk_min_seconds
            <= self.volume_walk_max_seconds
            <= 86_400.0
        ):
            raise ValueError("invalid volume walk range")
        return self


class AmbientSampleState:
    """
    Thread-safe settings and decoded WAV data.

    WAV decoding and resampling occur on the GUI or export setup thread, never
    inside the real-time callback.
    """

    def __init__(
        self,
        sample_rate: int,
        spec: AmbientSampleSpec,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self._lock = threading.Lock()
        self._spec = spec.validated()
        self._audio: np.ndarray | None = None
        self._loaded_filename = ""
        self._load_error = ""

        self._source_typical_dbfs = -80.0
        self._normalization_gain_db = 0.0
        self._normalized_typical_dbfs = -80.0
        self._normalized_peak_dbfs = -80.0
        self._audio_generation = 0

    def get(
        self,
    ) -> tuple[
        AmbientSampleSpec,
        np.ndarray | None,
        str,
        str,
        int,
    ]:
        with self._lock:
            return (
                self._spec,
                self._audio,
                self._loaded_filename,
                self._load_error,
                self._audio_generation,
            )

    def set_spec(self, spec: AmbientSampleSpec) -> None:
        with self._lock:
            self._spec = spec.validated()

    def update(self, **changes) -> None:
        with self._lock:
            self._spec = replace(
                self._spec,
                **changes,
            ).validated()

    @staticmethod
    def _convert_pcm_to_float(data: np.ndarray) -> np.ndarray:
        if np.issubdtype(data.dtype, np.floating):
            return data.astype(np.float32, copy=False)

        if data.dtype == np.uint8:
            return (
                data.astype(np.float32) - 128.0
            ) / 128.0

        info = np.iinfo(data.dtype)
        scale = float(max(abs(info.min), abs(info.max)))
        return data.astype(np.float32) / scale

    @staticmethod
    def _dbfs(value: float) -> float:
        return 20.0 * math.log10(max(1e-12, value))

    @staticmethod
    def _measure_typical_active_rms(
        data: np.ndarray,
        sample_rate: int,
    ) -> float:
        """
        Estimate a recording's typical active loudness.

        Peak normalization is unsuitable for field recordings because one
        chair scrape, shout, or dropped object can determine the level of an
        otherwise quiet ambience. Instead:

          * collapse stereo to energy-preserving mono;
          * measure non-overlapping 400 ms RMS windows;
          * ignore effectively silent windows below -55 dBFS;
          * use the 70th percentile of the remaining windows.

        The percentile favors the recording's normal active texture without
        allowing its single loudest event to dominate.
        """
        energy_mono = np.sqrt(
            np.mean(
                np.square(data.astype(np.float64)),
                axis=1,
            )
        )

        window_frames = max(1, int(0.400 * sample_rate))
        complete_windows = len(energy_mono) // window_frames

        if complete_windows == 0:
            return float(
                np.sqrt(np.mean(np.square(energy_mono)))
            )

        trimmed = energy_mono[
            : complete_windows * window_frames
        ]
        windows = trimmed.reshape(
            complete_windows,
            window_frames,
        )
        rms_values = np.sqrt(
            np.mean(np.square(windows), axis=1)
        )

        active_threshold = 10.0 ** (-55.0 / 20.0)
        active = rms_values[rms_values >= active_threshold]

        if len(active) == 0:
            active = rms_values

        return float(np.percentile(active, 70.0))

    @classmethod
    def _normalize_field_recording(
        cls,
        data: np.ndarray,
        sample_rate: int,
    ) -> tuple[np.ndarray, float, float, float, float]:
        """
        Normalize typical active loudness while preserving dynamics.

        The target is deliberately moderate because the UI applies an
        additional negative mix range afterward. Gain is bounded, and a final
        peak ceiling prevents unexpectedly loud transients from clipping.
        """
        target_typical_dbfs = -20.0
        maximum_boost_db = 18.0
        maximum_cut_db = -30.0
        peak_ceiling_dbfs = -1.0

        source_typical_rms = cls._measure_typical_active_rms(
            data,
            sample_rate,
        )
        source_typical_dbfs = cls._dbfs(source_typical_rms)

        requested_gain_db = (
            target_typical_dbfs - source_typical_dbfs
        )
        gain_db = float(
            np.clip(
                requested_gain_db,
                maximum_cut_db,
                maximum_boost_db,
            )
        )

        normalized = (
            data.astype(np.float64)
            * (10.0 ** (gain_db / 20.0))
        )

        peak = float(np.max(np.abs(normalized)))
        peak_ceiling = 10.0 ** (peak_ceiling_dbfs / 20.0)

        if peak > peak_ceiling:
            ceiling_adjustment_db = cls._dbfs(
                peak_ceiling / peak
            )
            normalized *= 10.0 ** (
                ceiling_adjustment_db / 20.0
            )
            gain_db += ceiling_adjustment_db

        normalized_typical_rms = (
            cls._measure_typical_active_rms(
                normalized,
                sample_rate,
            )
        )
        normalized_peak = float(
            np.max(np.abs(normalized))
        )

        return (
            np.ascontiguousarray(
                normalized,
                dtype=np.float32,
            ),
            source_typical_dbfs,
            gain_db,
            cls._dbfs(normalized_typical_rms),
            cls._dbfs(normalized_peak),
        )

    def normalization_info(
        self,
    ) -> tuple[float, float, float, float]:
        with self._lock:
            return (
                self._source_typical_dbfs,
                self._normalization_gain_db,
                self._normalized_typical_dbfs,
                self._normalized_peak_dbfs,
            )

    @staticmethod
    def _decode_audio_file(
        path: Path,
    ) -> tuple[np.ndarray, int]:
        """
        Decode supported audio formats through PyAV/FFmpeg.

        The decoder always returns float32 stereo in frames-by-channels layout.
        Source sample rate is preserved here; the existing resampler converts
        it to the engine rate afterward.
        """
        container = av.open(str(path))

        try:
            audio_streams = [
                stream
                for stream in container.streams
                if stream.type == "audio"
            ]
            if not audio_streams:
                raise ValueError(
                    "File contains no audio stream"
                )

            stream = audio_streams[0]
            input_rate = int(
                stream.codec_context.sample_rate
                or stream.rate
                or 44_100
            )

            chunks: list[np.ndarray] = []

            # Convert every decoded frame to planar float stereo through
            # PyAV's resampler. Some PyAV releases do not accept a `format`
            # keyword on AudioFrame.to_ndarray(), so conversion belongs here.
            resampler = av.audio.resampler.AudioResampler(
                format="fltp",
                layout="stereo",
                rate=input_rate,
            )

            def append_frame(converted_frame) -> None:
                array = converted_frame.to_ndarray()

                # Planar float output is channels x frames.
                if array.ndim == 1:
                    array = array[np.newaxis, :]

                if array.shape[0] == 1:
                    array = np.repeat(array, 2, axis=0)
                elif array.shape[0] > 2:
                    array = array[:2]

                chunks.append(
                    np.asarray(
                        array.T,
                        dtype=np.float32,
                    )
                )

            for frame in container.decode(stream):
                converted_frames = resampler.resample(frame)

                if converted_frames is None:
                    continue

                if not isinstance(converted_frames, list):
                    converted_frames = [converted_frames]

                for converted_frame in converted_frames:
                    append_frame(converted_frame)

            # Flush any delayed samples retained by the resampler.
            flushed_frames = resampler.resample(None)
            if flushed_frames is not None:
                if not isinstance(flushed_frames, list):
                    flushed_frames = [flushed_frames]

                for converted_frame in flushed_frames:
                    append_frame(converted_frame)

            if not chunks:
                raise ValueError(
                    "Audio stream decoded no frames"
                )

            data = np.concatenate(chunks, axis=0)

            # Protect against malformed files returning NaN/Inf.
            data = np.nan_to_num(
                data,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32, copy=False)

            return data, input_rate

        finally:
            container.close()

    def load_file(self, path: Path | None) -> None:
        if path is None:
            with self._lock:
                self._audio = None
                self._loaded_filename = ""
                self._load_error = ""
                self._source_typical_dbfs = -80.0
                self._normalization_gain_db = 0.0
                self._normalized_typical_dbfs = -80.0
                self._normalized_peak_dbfs = -80.0
                self._audio_generation += 1
            return

        try:
            data, input_rate = self._decode_audio_file(path)

            if int(input_rate) != self.sample_rate:
                divisor = math.gcd(int(input_rate), self.sample_rate)
                up = self.sample_rate // divisor
                down = int(input_rate) // divisor
                data = signal.resample_poly(
                    data,
                    up,
                    down,
                    axis=0,
                ).astype(np.float32)

            if len(data) < 2:
                raise ValueError(
                    "Audio file contains no usable audio"
                )

            (
                data,
                source_typical_dbfs,
                normalization_gain_db,
                normalized_typical_dbfs,
                normalized_peak_dbfs,
            ) = self._normalize_field_recording(
                data,
                self.sample_rate,
            )

            with self._lock:
                self._audio = data
                self._loaded_filename = path.name
                self._load_error = ""
                self._source_typical_dbfs = (
                    source_typical_dbfs
                )
                self._normalization_gain_db = (
                    normalization_gain_db
                )
                self._normalized_typical_dbfs = (
                    normalized_typical_dbfs
                )
                self._normalized_peak_dbfs = (
                    normalized_peak_dbfs
                )
                self._audio_generation += 1

        except Exception as exc:
            with self._lock:
                self._audio = None
                self._loaded_filename = ""
                self._load_error = str(exc)
                self._source_typical_dbfs = -80.0
                self._normalization_gain_db = 0.0
                self._normalized_typical_dbfs = -80.0
                self._normalized_peak_dbfs = -80.0
                self._audio_generation += 1


class AmbientSamplePlayer:
    STAGE_SILENCE = 0
    STAGE_FADE_IN = 1
    STAGE_HOLD = 2
    STAGE_FADE_OUT = 3

    def __init__(
        self,
        sample_rate: int,
        state: AmbientSampleState,
        seed: int = 99001,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.state = state
        self.rng = np.random.default_rng(seed)

        self.stage = self.STAGE_SILENCE
        self.stage_elapsed = 0
        self.stage_total = 1

        self.read_position = 0
        self.current_gain_linear = 0.0
        self.target_gain_linear = 0.0
        self.loaded_identity = -1

        self.current_stage_name = "silent"
        self.current_gain_db = -80.0

        spec, _, _, _, _ = self.state.get()
        initial_volume_db = 0.5 * (
            spec.volume_min_db + spec.volume_max_db
        )
        self.volume_journey = SmoothRandomJourney(
            rng=self.rng,
            initial_value=initial_volume_db,
            minimum=spec.volume_min_db,
            maximum=spec.volume_max_db,
            duration_min_seconds=spec.volume_walk_min_seconds,
            duration_max_seconds=spec.volume_walk_max_seconds,
            beta_a=1.0,
            beta_b=1.0,
        )
        self._volume_signature = (
            spec.volume_min_db,
            spec.volume_max_db,
            spec.volume_walk_min_seconds,
            spec.volume_walk_max_seconds,
        )

        self._begin_silence(initial=True)

    def _reset_for_new_audio(
        self,
        audio_length: int,
        generation: int,
        spec: AmbientSampleSpec,
    ) -> None:
        """
        A file selection is a user audition action, so the new sample should
        become audible immediately rather than inherit the previous file's
        silent interval or partially completed fade.
        """
        self.loaded_identity = generation

        self.stage = self.STAGE_FADE_IN
        self.stage_elapsed = 0
        self.stage_total = max(
            1,
            int(spec.fade_in_seconds * self.sample_rate),
        )

        self.read_position = int(
            self.rng.integers(0, max(1, audio_length))
        )
        self.current_stage_name = "fading in"

        initial_volume_db = float(
            np.clip(
                0.5 * (
                    spec.volume_min_db + spec.volume_max_db
                ),
                spec.volume_min_db,
                spec.volume_max_db,
            )
        )
        self.current_gain_db = initial_volume_db
        self.volume_journey = SmoothRandomJourney(
            rng=self.rng,
            initial_value=initial_volume_db,
            minimum=spec.volume_min_db,
            maximum=spec.volume_max_db,
            duration_min_seconds=spec.volume_walk_min_seconds,
            duration_max_seconds=spec.volume_walk_max_seconds,
            beta_a=1.0,
            beta_b=1.0,
        )
        self._volume_signature = (
            spec.volume_min_db,
            spec.volume_max_db,
            spec.volume_walk_min_seconds,
            spec.volume_walk_max_seconds,
        )

    def _random_seconds(
        self,
        minimum: float,
        maximum: float,
    ) -> float:
        if maximum <= minimum:
            return minimum
        return float(self.rng.uniform(minimum, maximum))

    def _begin_silence(self, initial: bool = False) -> None:
        spec, _, _, _, _ = self.state.get()
        self.stage = self.STAGE_SILENCE
        self.stage_elapsed = 0

        seconds = 0.0 if initial else self._random_seconds(
            spec.silence_min_seconds,
            spec.silence_max_seconds,
        )
        self.stage_total = max(
            1,
            int(seconds * self.sample_rate),
        )
        self.current_stage_name = "silent"

    def _begin_fade_in(self, audio_length: int) -> None:
        spec, _, _, _, _ = self.state.get()

        self.stage = self.STAGE_FADE_IN
        self.stage_elapsed = 0
        self.stage_total = max(
            1,
            int(spec.fade_in_seconds * self.sample_rate),
        )

        # Volume itself is continuously evolved. Fade-in only controls the
        # layer's awareness envelope and does not choose a fixed level.
        self.read_position = int(
            self.rng.integers(0, max(1, audio_length))
        )
        self.current_stage_name = "fading in"

    def _begin_hold(self) -> None:
        spec, _, _, _, _ = self.state.get()
        self.stage = self.STAGE_HOLD
        self.stage_elapsed = 0
        self.stage_total = max(
            1,
            int(
                self._random_seconds(
                    spec.duration_min_seconds,
                    spec.duration_max_seconds,
                )
                * self.sample_rate
            ),
        )
        self.current_stage_name = "present"

    def _begin_fade_out(self) -> None:
        spec, _, _, _, _ = self.state.get()
        self.stage = self.STAGE_FADE_OUT
        self.stage_elapsed = 0
        self.stage_total = max(
            1,
            int(spec.fade_out_seconds * self.sample_rate),
        )
        self.current_stage_name = "fading out"

    def _next_audio(
        self,
        audio: np.ndarray,
        count: int,
    ) -> np.ndarray:
        length = len(audio)
        indices = (
            np.arange(count, dtype=np.int64)
            + self.read_position
        ) % length
        self.read_position = int(
            (self.read_position + count) % length
        )
        return audio[indices]

    def _render_stage_gain(self, count: int) -> np.ndarray:
        start = self.stage_elapsed
        end = start + count

        if self.stage == self.STAGE_FADE_IN:
            positions = np.arange(start, end, dtype=np.float64)
            normalized = np.clip(
                positions / max(1, self.stage_total),
                0.0,
                1.0,
            )
            shape = 0.5 - 0.5 * np.cos(np.pi * normalized)
            return shape.astype(np.float32)

        if self.stage == self.STAGE_HOLD:
            return np.ones(
                count,
                dtype=np.float32,
            )

        if self.stage == self.STAGE_FADE_OUT:
            positions = np.arange(start, end, dtype=np.float64)
            normalized = np.clip(
                positions / max(1, self.stage_total),
                0.0,
                1.0,
            )
            shape = 0.5 + 0.5 * np.cos(np.pi * normalized)
            return shape.astype(np.float32)

        return np.zeros(count, dtype=np.float32)

    def _refresh_volume_journey(
        self,
        spec: AmbientSampleSpec,
    ) -> None:
        signature = (
            spec.volume_min_db,
            spec.volume_max_db,
            spec.volume_walk_min_seconds,
            spec.volume_walk_max_seconds,
        )
        if signature == self._volume_signature:
            return

        current = float(
            np.clip(
                self.current_gain_db,
                spec.volume_min_db,
                spec.volume_max_db,
            )
        )
        self.volume_journey = SmoothRandomJourney(
            rng=self.rng,
            initial_value=current,
            minimum=spec.volume_min_db,
            maximum=spec.volume_max_db,
            duration_min_seconds=spec.volume_walk_min_seconds,
            duration_max_seconds=spec.volume_walk_max_seconds,
            beta_a=1.0,
            beta_b=1.0,
        )
        self._volume_signature = signature

    def _volume_gain_curve(
        self,
        frame_count: int,
        spec: AmbientSampleSpec,
    ) -> np.ndarray:
        self._refresh_volume_journey(spec)

        start_db = self.volume_journey.current_value
        end_db = self.volume_journey.advance(
            frame_count / self.sample_rate
        )

        db_curve = np.linspace(
            start_db,
            end_db,
            frame_count,
            endpoint=False,
            dtype=np.float64,
        )
        self.current_gain_db = float(end_db)

        return np.power(
            10.0,
            db_curve / 20.0,
        ).astype(np.float32)

    def generate(self, frame_count: int) -> np.ndarray:
        spec, audio, _, _, generation = self.state.get()

        if audio is None:
            self.loaded_identity = generation
            self.current_stage_name = "no audio loaded"
            return np.zeros((frame_count, 2), dtype=np.float32)

        if generation != self.loaded_identity:
            self._reset_for_new_audio(
                audio_length=len(audio),
                generation=generation,
                spec=spec,
            )

        result = np.zeros((frame_count, 2), dtype=np.float32)
        written = 0

        while written < frame_count:
            if self.stage_elapsed >= self.stage_total:
                if self.stage == self.STAGE_SILENCE:
                    self._begin_fade_in(len(audio))
                elif self.stage == self.STAGE_FADE_IN:
                    self._begin_hold()
                elif self.stage == self.STAGE_HOLD:
                    self._begin_fade_out()
                else:
                    self._begin_silence()

            remaining_stage = self.stage_total - self.stage_elapsed
            count = min(
                frame_count - written,
                max(1, remaining_stage),
            )

            source = self._next_audio(audio, count)
            gain = self._render_stage_gain(count)
            result[written : written + count] = (
                source * gain[:, np.newaxis]
            )

            self.stage_elapsed += count
            written += count

        volume_gain = self._volume_gain_curve(
            frame_count,
            spec,
        )
        result *= volume_gain[:, np.newaxis]

        return result


# =============================================================================
# Mixer controls
# =============================================================================

@dataclass(frozen=True, slots=True)
class EngineModes:
    base_enabled: bool = True
    stereo_enabled: bool = True
    correlation_enabled: bool = True
    breath_enabled: bool = True
    heartbeat_enabled: bool = True
    soundscape_enabled: bool = False


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
        base_enabled: bool | None = None,
        stereo_enabled: bool | None = None,
        correlation_enabled: bool | None = None,
        breath_enabled: bool | None = None,
        heartbeat_enabled: bool | None = None,
        soundscape_enabled: bool | None = None,
    ) -> None:
        with self._lock:
            current = self._modes
            self._modes = EngineModes(
                base_enabled=(
                    current.base_enabled
                    if base_enabled is None
                    else bool(base_enabled)
                ),
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
                heartbeat_enabled=(
                    current.heartbeat_enabled
                    if heartbeat_enabled is None
                    else bool(heartbeat_enabled)
                ),
                soundscape_enabled=(
                    current.soundscape_enabled
                    if soundscape_enabled is None
                    else bool(soundscape_enabled)
                ),
            )



@dataclass(frozen=True, slots=True)
class DualBrownMotionSpec:
    """Live controls for the two brown-noise bodies moving on a sphere."""

    layer_enabled: bool = True
    layer_amount: float = 0.35
    enabled: bool = True
    sphere_radius: float = 2.75
    center_distance: float = 2.0
    evolution_rate: float = 0.32

    def validated(self) -> "DualBrownMotionSpec":
        if not 0.0 <= self.layer_amount <= 1.5:
            raise ValueError(
                "layer_amount must be between 0 and 1.5"
            )
        if not 0.0 <= self.sphere_radius <= 10.0:
            raise ValueError(
                "sphere_radius must be between 0 and 10 meters"
            )
        if not 0.05 <= self.center_distance <= 12.0:
            raise ValueError(
                "center_distance must be between 0.05 and 12 meters"
            )
        if not 0.0 <= self.evolution_rate <= 1.0:
            raise ValueError(
                "evolution_rate must be between 0 and 1"
            )
        return self

    @property
    def simulation_speed(self) -> float:
        if self.evolution_rate <= 0.0:
            return 0.0

        slow = 0.055
        fast = 5.0
        return math.exp(
            math.log(slow)
            + self.evolution_rate
            * (math.log(fast) - math.log(slow))
        )


class DualBrownMotionState:
    """Thread-safe motion settings shared by GUI and audio engine."""

    def __init__(self, spec: DualBrownMotionSpec) -> None:
        self._lock = threading.Lock()
        self._spec = spec.validated()

    def get(self) -> DualBrownMotionSpec:
        with self._lock:
            return self._spec

    def set(self, spec: DualBrownMotionSpec) -> None:
        with self._lock:
            self._spec = spec.validated()

    def update(self, **changes) -> None:
        with self._lock:
            self._spec = replace(
                self._spec,
                **changes,
            ).validated()


class DualBrownFluidMotion:
    """
    Soft-coupled lava-lamp motion over a sphere.

    There is no orbit or destination. Both bodies have tangential velocity and
    inertia. A shared slowly wandering angular current carries them, independent
    local eddies introduce lag and flex, viscous drag dissipates momentum, and
    a soft opposition spring prevents the stereo field from collapsing onto one
    side without forcing an exact rigid diameter.
    """

    def __init__(
        self,
        state: DualBrownMotionState,
        seed: int = 920_117,
    ) -> None:
        self.state = state
        self.rng = np.random.default_rng(seed)

        self.left_direction = np.array(
            [-1.0, 0.0, 0.0],
            dtype=np.float64,
        )
        self.right_direction = np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float64,
        )
        self.left_velocity = np.zeros(3, dtype=np.float64)
        self.right_velocity = np.zeros(3, dtype=np.float64)

        self.shared_omega = np.array(
            [0.08, 0.22, -0.05],
            dtype=np.float64,
        )
        self.left_local_omega = np.zeros(3, dtype=np.float64)
        self.right_local_omega = np.zeros(3, dtype=np.float64)

        self.current_separation_degrees = 180.0
        self.current_left_position = Vector3(-2.75, 0.0, -2.0)
        self.current_right_position = Vector3(2.75, 0.0, -2.0)

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        length = float(np.linalg.norm(vector))
        if length <= 1e-12:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return vector / length

    @staticmethod
    def _project_tangent(
        vector: np.ndarray,
        direction: np.ndarray,
    ) -> np.ndarray:
        return vector - direction * float(
            np.dot(vector, direction)
        )

    def _advance_flow_field(
        self,
        current: np.ndarray,
        dt: float,
        smoothing_seconds: float,
        scale: float,
    ) -> np.ndarray:
        decay = math.exp(-dt / smoothing_seconds)
        innovation = math.sqrt(
            max(0.0, 1.0 - decay * decay)
        )
        return (
            current * decay
            + self.rng.standard_normal(3)
            * innovation
            * scale
        )

    def _integrate_body(
        self,
        direction: np.ndarray,
        velocity: np.ndarray,
        other_direction: np.ndarray,
        flow_omega: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        desired_flow = np.cross(flow_omega, direction)

        desired_opposite = -other_direction
        opposition_error = self._project_tangent(
            desired_opposite,
            direction,
        )

        dot_value = float(
            np.clip(
                np.dot(direction, other_direction),
                -1.0,
                1.0,
            )
        )

        # Permissive while broadly opposite; much firmer if both bodies begin
        # collapsing into the same hemisphere.
        collapse_amount = float(
            np.clip((dot_value + 0.78) / 1.78, 0.0, 1.0)
        )
        spring_strength = 0.75 + 5.0 * collapse_amount

        acceleration = (
            desired_flow * 2.0
            + opposition_error * spring_strength
            - velocity * 1.35
        )

        velocity = velocity + acceleration * dt
        velocity = self._project_tangent(velocity, direction)

        direction = self._normalize(
            direction + velocity * dt
        )
        velocity = self._project_tangent(velocity, direction)

        return direction, velocity

    def advance(
        self,
        elapsed_seconds: float,
        override_spec: DualBrownMotionSpec | None = None,
    ) -> tuple[Vector3, Vector3]:
        spec = (
            override_spec
            if override_spec is not None
            else self.state.get()
        )
        simulated_seconds = (
            max(0.0, float(elapsed_seconds))
            * spec.simulation_speed
        )

        if spec.enabled and simulated_seconds > 0.0:
            remaining = simulated_seconds

            while remaining > 0.0:
                dt = min(1.0 / 120.0, remaining)
                remaining -= dt

                self.shared_omega = self._advance_flow_field(
                    self.shared_omega,
                    dt,
                    smoothing_seconds=5.5,
                    scale=0.48,
                )
                self.left_local_omega = self._advance_flow_field(
                    self.left_local_omega,
                    dt,
                    smoothing_seconds=2.4,
                    scale=0.17,
                )
                self.right_local_omega = self._advance_flow_field(
                    self.right_local_omega,
                    dt,
                    smoothing_seconds=2.9,
                    scale=0.17,
                )

                self.left_direction, self.left_velocity = (
                    self._integrate_body(
                        self.left_direction,
                        self.left_velocity,
                        self.right_direction,
                        self.shared_omega
                        + self.left_local_omega,
                        dt,
                    )
                )
                self.right_direction, self.right_velocity = (
                    self._integrate_body(
                        self.right_direction,
                        self.right_velocity,
                        self.left_direction,
                        self.shared_omega
                        + self.right_local_omega,
                        dt,
                    )
                )

        center = np.array(
            [0.0, 0.0, -spec.center_distance],
            dtype=np.float64,
        )
        left = center + self.left_direction * spec.sphere_radius
        right = center + self.right_direction * spec.sphere_radius

        self.current_left_position = Vector3(
            float(left[0]),
            float(left[1]),
            float(left[2]),
        )
        self.current_right_position = Vector3(
            float(right[0]),
            float(right[1]),
            float(right[2]),
        )

        separation_dot = float(
            np.clip(
                np.dot(
                    self.left_direction,
                    self.right_direction,
                ),
                -1.0,
                1.0,
            )
        )
        self.current_separation_degrees = math.degrees(
            math.acos(separation_dot)
        )

        return (
            self.current_left_position,
            self.current_right_position,
        )


@dataclass(frozen=True, slots=True)
class HeartbeatSpatialSpec:
    distance: float = HEARTBEAT_DISTANCE_DEFAULT_METERS
    horizontal: float = HEARTBEAT_HORIZONTAL_DEFAULT_METERS
    vertical: float = HEARTBEAT_VERTICAL_DEFAULT_METERS
    level_db: float = 12.0

    def validated(self) -> "HeartbeatSpatialSpec":
        if not HEARTBEAT_DISTANCE_MIN_METERS <= self.distance <= HEARTBEAT_DISTANCE_MAX_METERS:
            raise ValueError("heartbeat distance outside range")
        if not HEARTBEAT_HORIZONTAL_MIN_METERS <= self.horizontal <= HEARTBEAT_HORIZONTAL_MAX_METERS:
            raise ValueError("heartbeat horizontal outside range")
        if not HEARTBEAT_VERTICAL_MIN_METERS <= self.vertical <= HEARTBEAT_VERTICAL_MAX_METERS:
            raise ValueError("heartbeat vertical outside range")
        if not -24.0 <= self.level_db <= 24.0:
            raise ValueError("heartbeat level must be between -24 and +24 dB")
        return self

    @property
    def position(self) -> Vector3:
        return Vector3(self.horizontal, self.vertical, -self.distance)


class HeartbeatSpatialState:
    def __init__(self, spec: HeartbeatSpatialSpec) -> None:
        self._lock = threading.Lock()
        self._spec = spec.validated()

    def get(self) -> HeartbeatSpatialSpec:
        with self._lock:
            return self._spec

    def set(self, spec: HeartbeatSpatialSpec) -> None:
        with self._lock:
            self._spec = spec.validated()

    def update(self, **changes) -> None:
        with self._lock:
            self._spec = replace(self._spec, **changes).validated()



@dataclass(frozen=True, slots=True)
class MetabolismSpec:
    """Independent ranges for the central living-system controller."""

    enabled: bool = False
    phase_min_minutes: float = 8.0
    phase_max_minutes: float = 35.0

    # Percentage preference for resting states. Zero leaves the activity
    # drive linear; 100 strongly favors rest while preserving rare excursions.
    resting_tendency_percent: float = 75.0

    brown_body_min: float = 0.0
    brown_body_max: float = 1.0
    brown_slope_min: float = 0.75
    brown_slope_max: float = 1.0
    brown_low_end_min_db: float = 0.0
    brown_low_end_max_db: float = 8.0
    brown_texture_min: float = 0.0
    brown_texture_max: float = 1.0

    breath_prominence_min: float = 0.02
    breath_prominence_max: float = 0.85
    breath_tempo_min: float = 1.0
    breath_tempo_max: float = 2.6
    breath_gain_min_db: float = 0.25
    breath_gain_max_db: float = 4.5
    breath_spectral_min: float = 0.05
    breath_spectral_max: float = 0.35
    breath_width_min: float = 0.03
    breath_width_max: float = 0.18

    heartbeat_distance_min: float = 0.75
    heartbeat_distance_max: float = 4.0
    heartbeat_level_min_db: float = 0.0
    heartbeat_level_max_db: float = 18.0

    brown_3d_amount_min: float = 0.02
    brown_3d_amount_max: float = 0.55
    brown_radius_min: float = 0.25
    brown_radius_max: float = 5.0
    brown_center_distance_min: float = 0.5
    brown_center_distance_max: float = 5.0
    brown_evolution_min: float = 0.02
    brown_evolution_max: float = 0.65

    def validated(self) -> "MetabolismSpec":
        if not 0.0 <= self.resting_tendency_percent <= 100.0:
            raise ValueError(
                "resting_tendency_percent must be between 0 and 100"
            )

        pairs = (
            ("phase_min_minutes", "phase_max_minutes", 0.25, 240.0),
            ("brown_body_min", "brown_body_max", 0.0, 1.0),
            ("brown_slope_min", "brown_slope_max", 0.75, 1.0),
            ("brown_low_end_min_db", "brown_low_end_max_db", 0.0, 8.0),
            ("brown_texture_min", "brown_texture_max", 0.0, 1.0),
            ("breath_prominence_min", "breath_prominence_max", 0.0, 1.5),
            ("breath_tempo_min", "breath_tempo_max", 0.25, 5.0),
            ("breath_gain_min_db", "breath_gain_max_db", 0.0, 12.0),
            ("breath_spectral_min", "breath_spectral_max", 0.0, 1.0),
            ("breath_width_min", "breath_width_max", 0.0, 1.0),
            ("heartbeat_distance_min", "heartbeat_distance_max", 0.15, 4.0),
            ("heartbeat_level_min_db", "heartbeat_level_max_db", -24.0, 24.0),
            ("brown_3d_amount_min", "brown_3d_amount_max", 0.0, 1.5),
            ("brown_radius_min", "brown_radius_max", 0.0, 10.0),
            ("brown_center_distance_min", "brown_center_distance_max", 0.05, 12.0),
            ("brown_evolution_min", "brown_evolution_max", 0.0, 1.0),
        )
        for lo_name, hi_name, lo_bound, hi_bound in pairs:
            lo = float(getattr(self, lo_name))
            hi = float(getattr(self, hi_name))
            if not lo_bound <= lo <= hi_bound:
                raise ValueError(f"{lo_name} outside range")
            if not lo_bound <= hi <= hi_bound:
                raise ValueError(f"{hi_name} outside range")
            if lo > hi:
                raise ValueError(f"{lo_name} cannot exceed {hi_name}")
        return self


class MetabolismState:
    """Thread-safe metabolism settings with interactive min/max normalization."""

    _PAIRS = (
        ("phase_min_minutes", "phase_max_minutes"),
        ("brown_body_min", "brown_body_max"),
        ("brown_slope_min", "brown_slope_max"),
        ("brown_low_end_min_db", "brown_low_end_max_db"),
        ("brown_texture_min", "brown_texture_max"),
        ("breath_prominence_min", "breath_prominence_max"),
        ("breath_tempo_min", "breath_tempo_max"),
        ("breath_gain_min_db", "breath_gain_max_db"),
        ("breath_spectral_min", "breath_spectral_max"),
        ("breath_width_min", "breath_width_max"),
        ("heartbeat_distance_min", "heartbeat_distance_max"),
        ("heartbeat_level_min_db", "heartbeat_level_max_db"),
        ("brown_3d_amount_min", "brown_3d_amount_max"),
        ("brown_radius_min", "brown_radius_max"),
        ("brown_center_distance_min", "brown_center_distance_max"),
        ("brown_evolution_min", "brown_evolution_max"),
    )

    def __init__(self, spec: MetabolismSpec) -> None:
        self._lock = threading.Lock()
        self._spec = spec.validated()

    def get(self) -> MetabolismSpec:
        with self._lock:
            return self._spec

    def set(self, spec: MetabolismSpec) -> None:
        with self._lock:
            self._spec = spec.validated()

    def update(self, **changes) -> None:
        with self._lock:
            values = asdict(self._spec)
            values.update(changes)

            for minimum_name, maximum_name in self._PAIRS:
                minimum = float(values[minimum_name])
                maximum = float(values[maximum_name])

                if minimum > maximum:
                    if minimum_name in changes:
                        values[maximum_name] = minimum
                    else:
                        values[minimum_name] = maximum

            self._spec = MetabolismSpec(**values).validated()


@dataclass(frozen=True, slots=True)
class MetabolismValues:
    # Raw metabolic position drives texture and spatial shape.
    activity: float

    # Quiet-weighted activity drives prominence, urgency and audible density.
    activity_drive: float

    brown_body: float
    brown_slope: float
    brown_low_end_db: float
    brown_texture: float
    breath_prominence: float
    breath_tempo: float
    breath_gain_db: float
    breath_spectral_depth: float
    breath_width_depth: float
    heartbeat_distance: float
    heartbeat_level_db: float
    brown_3d_amount: float
    brown_radius: float
    brown_center_distance: float
    brown_evolution: float


class MetabolismEngine:
    """Smooth, nonperiodic travel through the independent metabolism envelope."""

    def __init__(
        self,
        state: MetabolismState,
        seed: int = 730221,
    ) -> None:
        self.state = state
        self.rng = np.random.default_rng(seed)

        self.start_activity = 0.30
        self.current_activity = 0.30
        self.target_activity = 0.30
        self.elapsed = 0.0
        self.duration = 1.0
        self._was_enabled = False

        self._choose_target(initial=True)

    def _choose_target(self, initial: bool = False) -> None:
        spec = self.state.get()

        if not initial:
            self.start_activity = self.current_activity

        # The raw metabolic state explores the complete range without a quiet
        # bias. Texture and spatial shape therefore remain fully dynamic even
        # while the audible activity drive is predominantly subdued.
        self.target_activity = float(self.rng.random())
        self.duration = float(
            self.rng.uniform(
                spec.phase_min_minutes * 60.0,
                spec.phase_max_minutes * 60.0,
            )
        )
        self.elapsed = 0.0

    @staticmethod
    def _smoothstep5(value: float) -> float:
        value = float(np.clip(value, 0.0, 1.0))
        return value ** 3 * (
            value * (value * 6.0 - 15.0) + 10.0
        )

    @staticmethod
    def _map(
        activity: float,
        minimum: float,
        maximum: float,
    ) -> float:
        return minimum + activity * (maximum - minimum)

    def advance(
        self,
        elapsed_seconds: float,
    ) -> MetabolismValues | None:
        spec = self.state.get()

        if not spec.enabled:
            self._was_enabled = False
            return None

        if not self._was_enabled:
            self._was_enabled = True
            self.start_activity = self.current_activity
            self._choose_target()

        remaining = max(0.0, float(elapsed_seconds))

        while remaining > 0.0:
            available = max(0.0, self.duration - self.elapsed)
            step = min(remaining, available)
            self.elapsed += step
            remaining -= step

            progress = self.elapsed / max(1e-9, self.duration)
            blend = self._smoothstep5(progress)
            self.current_activity = (
                self.start_activity
                + (self.target_activity - self.start_activity) * blend
            )

            if self.elapsed >= self.duration - 1e-9:
                self.current_activity = self.target_activity
                self._choose_target()

        activity = float(
            np.clip(self.current_activity, 0.0, 1.0)
        )

        # Shape only the dimensions associated with loudness, prominence,
        # urgency, or moving-information density. At zero bias this is linear.
        # At higher values, most of the journey remains near rest, while an
        # exact high state can still reach the full configured maxima.
        resting_tendency = (
            spec.resting_tendency_percent / 100.0
        )
        quiet_exponent = 1.0 + 5.0 * resting_tendency
        activity_drive = activity ** quiet_exponent

        texture_wave = 0.5 + 0.5 * math.sin(
            2.0 * math.pi * (activity + 0.17)
        )
        body_wave = 0.5 + 0.5 * math.sin(
            2.0 * math.pi * (activity * 0.73 + 0.41)
        )
        slope_wave = 0.5 + 0.5 * math.sin(
            2.0 * math.pi * (activity * 0.61 + 0.08)
        )
        spatial_wave = 0.5 + 0.5 * math.sin(
            2.0 * math.pi * (activity * 0.83 + 0.29)
        )

        return MetabolismValues(
            activity=activity,
            activity_drive=activity_drive,
            brown_body=self._map(
                body_wave, spec.brown_body_min, spec.brown_body_max
            ),
            brown_slope=self._map(
                slope_wave, spec.brown_slope_min, spec.brown_slope_max
            ),
            brown_low_end_db=self._map(
                texture_wave,
                spec.brown_low_end_min_db,
                spec.brown_low_end_max_db,
            ),
            brown_texture=self._map(
                1.0 - texture_wave,
                spec.brown_texture_min,
                spec.brown_texture_max,
            ),
            breath_prominence=self._map(
                activity_drive,
                spec.breath_prominence_min,
                spec.breath_prominence_max,
            ),
            breath_tempo=self._map(
                1.0 - activity_drive,
                spec.breath_tempo_min,
                spec.breath_tempo_max,
            ),
            breath_gain_db=self._map(
                activity_drive,
                spec.breath_gain_min_db,
                spec.breath_gain_max_db,
            ),
            breath_spectral_depth=self._map(
                activity_drive,
                spec.breath_spectral_min,
                spec.breath_spectral_max,
            ),
            breath_width_depth=self._map(
                activity_drive,
                spec.breath_width_min,
                spec.breath_width_max,
            ),
            heartbeat_distance=self._map(
                1.0 - activity_drive,
                spec.heartbeat_distance_min,
                spec.heartbeat_distance_max,
            ),
            heartbeat_level_db=self._map(
                activity_drive,
                spec.heartbeat_level_min_db,
                spec.heartbeat_level_max_db,
            ),
            brown_3d_amount=self._map(
                activity_drive,
                spec.brown_3d_amount_min,
                spec.brown_3d_amount_max,
            ),
            brown_radius=self._map(
                spatial_wave,
                spec.brown_radius_min,
                spec.brown_radius_max,
            ),
            brown_center_distance=self._map(
                1.0 - spatial_wave,
                spec.brown_center_distance_min,
                spec.brown_center_distance_max,
            ),
            brown_evolution=self._map(
                activity_drive,
                spec.brown_evolution_min,
                spec.brown_evolution_max,
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
        heartbeat_state: HeartbeatState,
        ambient_sample_state: AmbientSampleState,
        breath_state: BreathState,
        breath_evolution_state: BreathEvolutionState,
        motion_state: OrganicMotionState,
        brown_motion_spec: DualBrownMotionSpec,
        heartbeat_spatial_spec: HeartbeatSpatialSpec,
        metabolism_spec: MetabolismSpec,
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
        self.heartbeat_state = heartbeat_state
        self.heartbeat = HeartbeatGenerator(
            sample_rate=self.sample_rate,
            heartbeat_state=heartbeat_state,
        )
        self.ambient_sample_state = ambient_sample_state
        self.ambient_sample = AmbientSamplePlayer(
            sample_rate=int(self.sample_rate),
            state=ambient_sample_state,
        )

        self.spatial_renderer = SteamAudioRenderer(
            sample_rate=int(self.sample_rate),
            frame_size=STEAM_SPATIAL_FRAME_SIZE,
            validation_enabled=False,
            log_messages=False,
        )
        self.brown_noise_left_spatial = (
            self.spatial_renderer.create_source(
                position=STEAM_BROWN_LEFT_POSITION,
                spatial_blend=1.0,
                distance_attenuation_enabled=True,
            )
        )
        self.brown_noise_right_spatial = (
            self.spatial_renderer.create_source(
                position=STEAM_BROWN_RIGHT_POSITION,
                spatial_blend=1.0,
                distance_attenuation_enabled=True,
            )
        )
        self.heartbeat_spatial_state = HeartbeatSpatialState(
            heartbeat_spatial_spec
        )
        self.heartbeat_spatial_rng = np.random.default_rng(881731)
        self.heartbeat_spatial = self.spatial_renderer.create_source(
            position=heartbeat_spatial_spec.position,
            spatial_blend=STEAM_HEARTBEAT_SPATIAL_BLEND,
            distance_attenuation_enabled=True,
        )
        self.soundscape_spatial = self.spatial_renderer.create_source(
            position=STEAM_DEFAULT_SOURCE_POSITION,
            spatial_blend=1.0,
        )

        self.breath_state = breath_state
        self.breath_evolution_state = breath_evolution_state
        self.motion_state = motion_state
        self.metabolism_state = MetabolismState(metabolism_spec)
        self.metabolism = MetabolismEngine(self.metabolism_state)
        self.current_metabolism_activity = 0.0
        self.current_metabolism_values: MetabolismValues | None = None

        self.brown_motion_state = DualBrownMotionState(
            brown_motion_spec
        )
        self.brown_motion = DualBrownFluidMotion(
            self.brown_motion_state
        )
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
        self.base_mix = 1.0 if initial_modes.base_enabled else 0.0
        self.stereo_mix = 1.0 if initial_modes.stereo_enabled else 0.0
        self.correlation_mix = (
            1.0 if initial_modes.correlation_enabled else 0.0
        )
        self.breath_mix = 1.0 if initial_modes.breath_enabled else 0.0
        self.heartbeat_mix = (
            1.0 if initial_modes.heartbeat_enabled else 0.0
        )
        self.soundscape_mix = (
            1.0 if initial_modes.soundscape_enabled else 0.0
        )

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
        self.current_heartbeat = 0.0
        self.current_heart_interval = 60.0 / 50.0
        self.current_heartbeat_position = heartbeat_spatial_spec.position
        self.current_soundscape_stage = "disabled"
        self.current_soundscape_gain_db = -80.0
        self.current_brown_3d_mix = 1.0
        self.current_brown_motion_separation = 180.0
        self.current_brown_left_position = STEAM_BROWN_LEFT_POSITION
        self.current_brown_right_position = STEAM_BROWN_RIGHT_POSITION

    def set_heartbeat_position(self, **changes) -> None:
        if changes:
            self.heartbeat_spatial_state.update(**changes)
        spec = self.heartbeat_spatial_state.get()
        self.current_heartbeat_position = spec.position
        self.heartbeat_spatial.set_position_vector(spec.position)

    def _randomize_heartbeat_position(self) -> None:
        current = self.heartbeat_spatial_state.get()
        spec = HeartbeatSpatialSpec(
            distance=float(self.heartbeat_spatial_rng.uniform(
                HEARTBEAT_DISTANCE_MIN_METERS,
                HEARTBEAT_DISTANCE_MAX_METERS,
            )),
            horizontal=float(self.heartbeat_spatial_rng.uniform(
                HEARTBEAT_HORIZONTAL_MIN_METERS,
                HEARTBEAT_HORIZONTAL_MAX_METERS,
            )),
            vertical=float(self.heartbeat_spatial_rng.uniform(
                HEARTBEAT_VERTICAL_MIN_METERS,
                HEARTBEAT_VERTICAL_MAX_METERS,
            )),
            level_db=current.level_db,
        )
        self.heartbeat_spatial_state.set(spec)
        self.current_heartbeat_position = spec.position
        self.heartbeat_spatial.set_position_vector(spec.position)

    def close(self) -> None:
        self.spatial_renderer.close()

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

        metabolism_values = self.metabolism.advance(elapsed_seconds)
        self.current_metabolism_values = metabolism_values

        if metabolism_values is None:
            self.current_metabolism_activity = 0.0
            self.breath.set_external_tempo_multiplier(1.0)
        else:
            self.current_metabolism_activity = metabolism_values.activity
            self.breath.set_external_tempo_multiplier(
                metabolism_values.breath_tempo
            )
        static_noise_spec, _ = self.noise_state.get()
        if metabolism_values is None:
            evolved_noise_spec = self.noise_evolution.advance(
                elapsed_seconds,
                static_noise_spec,
            )
        else:
            evolved_noise_spec = replace(
                static_noise_spec,
                body=metabolism_values.brown_body,
                slope_strength=metabolism_values.brown_slope,
                low_end_emphasis_db=metabolism_values.brown_low_end_db,
                upper_texture=metabolism_values.brown_texture,
            ).validated(self.sample_rate)
        body_movement_triggered = self.body_movement.advance(
            elapsed_seconds,
            self.noise_evolution,
        )
        if body_movement_triggered:
            self._randomize_heartbeat_position()
        self.current_body_movement_count = self.body_movement.event_count
        self.current_body_movement_strength = self.body_movement.last_strength
        self.current_body_movement_age = self.body_movement.age
        self.current_noise_body = evolved_noise_spec.body
        self.current_noise_slope = evolved_noise_spec.slope_strength
        self.current_noise_weight = (
            evolved_noise_spec.low_end_emphasis_db
        )
        self.current_noise_texture = evolved_noise_spec.upper_texture
        manual_breath_spec, _ = self.breath_state.get()
        if metabolism_values is None:
            breath_spec = manual_breath_spec
        else:
            breath_spec = replace(
                manual_breath_spec,
                gain_range_db=metabolism_values.breath_gain_db,
                spectral_depth=metabolism_values.breath_spectral_depth,
                width_depth=metabolism_values.breath_width_depth,
            ).validated()

        base_curve = self._approach_target(
            self.base_mix,
            1.0 if modes.base_enabled else 0.0,
            frame_count,
        )
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
        heartbeat_curve = self._approach_target(
            self.heartbeat_mix,
            1.0 if modes.heartbeat_enabled else 0.0,
            frame_count,
        )
        soundscape_curve = self._approach_target(
            self.soundscape_mix,
            1.0 if modes.soundscape_enabled else 0.0,
            frame_count,
        )

        self.base_mix = float(base_curve[-1])
        self.stereo_mix = float(stereo_curve[-1])
        self.correlation_mix = float(correlation_curve[-1])
        self.breath_mix = float(breath_curve[-1])
        self.heartbeat_mix = float(heartbeat_curve[-1])
        self.soundscape_mix = float(soundscape_curve[-1])

        raw_breath = self.breath.generate(frame_count)
        prominence = self.breath_prominence.generate(frame_count)
        if metabolism_values is not None:
            prominence *= metabolism_values.breath_prominence

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

        # The complete pre-existing Living Brown Noise bus can be suppressed
        # for heartbeat debugging without stopping its internal state.
        stereo *= base_curve[:, np.newaxis]

        manual_brown_motion_spec = self.brown_motion_state.get()

        if metabolism_values is None:
            effective_brown_motion_spec = manual_brown_motion_spec
        else:
            effective_brown_motion_spec = DualBrownMotionSpec(
                layer_enabled=True,
                layer_amount=metabolism_values.brown_3d_amount,
                enabled=True,
                sphere_radius=metabolism_values.brown_radius,
                center_distance=(
                    metabolism_values.brown_center_distance
                ),
                evolution_rate=metabolism_values.brown_evolution,
            ).validated()

        left_position, right_position = self.brown_motion.advance(
            elapsed_seconds,
            override_spec=effective_brown_motion_spec,
        )
        self.brown_noise_left_spatial.set_position_vector(
            left_position
        )
        self.brown_noise_right_spatial.set_position_vector(
            right_position
        )
        self.current_brown_left_position = left_position
        self.current_brown_right_position = right_position
        self.current_brown_motion_separation = (
            self.brown_motion.current_separation_degrees
        )

        # The spatial bodies use the independent generators directly. The
        # correlated stereo bed above remains completely intact.
        brown_left_3d = self.brown_noise_left_spatial.process_mono(
            independent_left
        )
        brown_right_3d = self.brown_noise_right_spatial.process_mono(
            independent_right
        )
        brown_3d = brown_left_3d + brown_right_3d

        brown_motion_spec = effective_brown_motion_spec
        brown_3d_curve = self._approach_target(
            self.current_brown_3d_mix,
            1.0 if brown_motion_spec.layer_enabled else 0.0,
            frame_count,
        )
        self.current_brown_3d_mix = float(brown_3d_curve[-1])

        # The 3D layer shares the organic breath modulation, but it is a
        # separate bus and must not inherit the 2D foundation's enable curve.
        brown_3d *= np.power(
            10.0,
            breath_gain_db / 20.0,
        )[:, np.newaxis]

        stereo += (
            brown_3d
            * brown_3d_curve[:, np.newaxis]
            * brown_motion_spec.layer_amount
        )

        heartbeat = self.heartbeat.generate(frame_count)
        manual_heartbeat_position_spec = (
            self.heartbeat_spatial_state.get()
        )
        if metabolism_values is None:
            heartbeat_position_spec = manual_heartbeat_position_spec
        else:
            heartbeat_position_spec = replace(
                manual_heartbeat_position_spec,
                distance=metabolism_values.heartbeat_distance,
                level_db=metabolism_values.heartbeat_level_db,
            ).validated()

        heartbeat_level = 10.0 ** (
            heartbeat_position_spec.level_db / 20.0
        )
        active_heartbeat = (
            heartbeat
            * heartbeat_curve
            * heartbeat_level
        )

        self.current_heartbeat = float(
            self.heartbeat.current_envelope
            * heartbeat_curve[-1]
        )
        self.current_heart_interval = (
            self.heartbeat.current_interval_seconds
        )

        self.current_heartbeat_position = heartbeat_position_spec.position
        self.heartbeat_spatial.set_position_vector(
            heartbeat_position_spec.position
        )
        spatial_heartbeat = self.heartbeat_spatial.process_mono(
            active_heartbeat
        )
        stereo += spatial_heartbeat

        soundscape = self.ambient_sample.generate(frame_count)
        soundscape *= soundscape_curve[:, np.newaxis]
        spatial_soundscape = self.soundscape_spatial.process_stereo_bed(
            soundscape,
            spatial_amount=STEAM_SOUNDSCAPE_SPATIAL_AMOUNT,
        )
        stereo += spatial_soundscape

        self.current_soundscape_stage = (
            self.ambient_sample.current_stage_name
        )
        self.current_soundscape_gain_db = (
            self.ambient_sample.current_gain_db
        )

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
    heartbeat_spec: HeartbeatSpec,
    ambient_sample_spec: AmbientSampleSpec,
    sound_effects_directory: Path,
    breath_spec: BreathSpec,
    breath_evolution_spec: BreathEvolutionSpec,
    motion_spec: OrganicMotionSpec,
    brown_motion_spec: DualBrownMotionSpec,
    heartbeat_spatial_spec: HeartbeatSpatialSpec,
    metabolism_spec: MetabolismSpec,
    seed_base: int,
) -> tuple[
    LivingBrownNoiseMixer,
    ModeState,
    BrownNoiseState,
    BrownNoiseEvolutionState,
    BodyMovementState,
    HeartbeatState,
    AmbientSampleState,
    BreathState,
    BreathEvolutionState,
    OrganicMotionState,
]:
    noise_state = BrownNoiseState(sample_rate, noise_spec)
    noise_evolution_state = BrownNoiseEvolutionState(
        noise_evolution_spec
    )
    body_movement_state = BodyMovementState(body_movement_spec)
    heartbeat_state = HeartbeatState(heartbeat_spec)
    ambient_sample_state = AmbientSampleState(
        sample_rate,
        ambient_sample_spec,
    )
    if ambient_sample_spec.selected_filename:
        ambient_sample_state.load_file(
            sound_effects_directory
            / ambient_sample_spec.selected_filename
        )

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
        heartbeat_state=heartbeat_state,
        ambient_sample_state=ambient_sample_state,
        breath_state=breath_state,
        breath_evolution_state=breath_evolution_state,
        motion_state=motion_state,
        brown_motion_spec=brown_motion_spec,
        heartbeat_spatial_spec=heartbeat_spatial_spec,
        metabolism_spec=metabolism_spec,
        mixer_spec=MixerSpec(),
    )

    return (
        mixer,
        mode_state,
        noise_state,
        noise_evolution_state,
        body_movement_state,
        heartbeat_state,
        ambient_sample_state,
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
        heartbeat_spec: HeartbeatSpec,
        ambient_sample_spec: AmbientSampleSpec,
        sound_effects_directory: Path,
        breath_spec: BreathSpec,
        breath_evolution_spec: BreathEvolutionSpec,
        motion_spec: OrganicMotionSpec,
        brown_motion_spec: DualBrownMotionSpec,
        heartbeat_spatial_spec: HeartbeatSpatialSpec,
        metabolism_spec: MetabolismSpec,
    ) -> None:
        super().__init__()
        self.output_path = output_path
        self.duration_minutes = duration_minutes
        self.sample_rate = sample_rate
        self.modes = modes
        self.noise_spec = noise_spec
        self.noise_evolution_spec = noise_evolution_spec
        self.body_movement_spec = body_movement_spec
        self.heartbeat_spec = heartbeat_spec
        self.ambient_sample_spec = ambient_sample_spec
        self.sound_effects_directory = sound_effects_directory
        self.breath_spec = breath_spec
        self.breath_evolution_spec = breath_evolution_spec
        self.motion_spec = motion_spec
        self.brown_motion_spec = brown_motion_spec
        self.heartbeat_spatial_spec = heartbeat_spatial_spec
        self.metabolism_spec = metabolism_spec
        self._cancel_requested = threading.Event()

    def request_cancel(self) -> None:
        self._cancel_requested.set()

    def run(self) -> None:
        mixer = None
        try:
            total_frames = int(
                self.duration_minutes * 60 * self.sample_rate
            )
            # Steam Audio effects retain convolution state across fixed
            # STEAM_SPATIAL_FRAME_SIZE blocks. Every ordinary export request
            # must therefore contain an exact whole number of spatial frames.
            # Only the final request at the true end of the file may be short.
            approximate_chunk_frames = max(
                STEAM_SPATIAL_FRAME_SIZE,
                self.sample_rate // 2,
            )
            chunk_frames = (
                approximate_chunk_frames
                // STEAM_SPATIAL_FRAME_SIZE
                * STEAM_SPATIAL_FRAME_SIZE
            )

            seed_base = int(time.time_ns() & 0x7FFFFFFF)
            mixer, _, _, _, _, _, _, _, _, _ = build_mixer(
                sample_rate=self.sample_rate,
                modes=self.modes,
                noise_spec=self.noise_spec,
                noise_evolution_spec=self.noise_evolution_spec,
                body_movement_spec=self.body_movement_spec,
                heartbeat_spec=self.heartbeat_spec,
                ambient_sample_spec=self.ambient_sample_spec,
                sound_effects_directory=self.sound_effects_directory,
                breath_spec=self.breath_spec,
                breath_evolution_spec=self.breath_evolution_spec,
                motion_spec=self.motion_spec,
                brown_motion_spec=self.brown_motion_spec,
                heartbeat_spatial_spec=self.heartbeat_spatial_spec,
                metabolism_spec=self.metabolism_spec,
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

                    remaining_frames = (
                        total_frames - frames_written
                    )

                    # Normal chunks are exact Steam Audio frame multiples.
                    # A short request occurs only once, at the true end of the
                    # complete export, so zero padding can never contaminate
                    # persistent HRTF state between ordinary chunks.
                    frame_count = (
                        chunk_frames
                        if remaining_frames > chunk_frames
                        else remaining_frames
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

        finally:
            if mixer is not None:
                mixer.close()


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
        heartbeat_state: HeartbeatState,
        ambient_sample_state: AmbientSampleState,
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
        self.heartbeat_state = heartbeat_state
        self.ambient_sample_state = ambient_sample_state
        self.breath_state = breath_state
        self.breath_evolution_state = breath_evolution_state
        self.motion_state = motion_state
        self.mixer = mixer
        self.settings_store = settings_store
        self.loaded_settings = loaded_settings
        self.dream_motif_catalog = DreamMotifCatalog(
            root_directory=SOUND_EFFECTS_DIRECTORY,
            layer_threshold_seconds=(
                DREAM_MOTIF_LAYER_THRESHOLD_SECONDS
            ),
        )
        self.motif_rng = np.random.default_rng(77123)
        self.active_motif_name = ""
        self.active_motif_asset: DreamMotifAsset | None = None
        self.previous_motif_stage = ""
        self.motif_playback_enabled = False
        self.export_worker: ExportWorker | None = None

        self.settings_save_timer = QTimer(self)
        self.settings_save_timer.setSingleShot(True)
        self.settings_save_timer.timeout.connect(self._save_settings)

        self.default_breath_spec = BreathSpec()

        self.setWindowTitle(
            "Living Brown Noise — Dream Instigator Lab — Metabolism Controller"
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

        self.base_checkbox = QCheckBox(
            "Living Brown Noise base — off mutes the existing main audio bus"
        )
        self.base_checkbox.setChecked(
            self.mode_state.get().base_enabled
        )

        self.heartbeat_checkbox = QCheckBox(
            "Heartbeat / pulse — synthesized resonant lub-dub instrument"
        )
        self.heartbeat_checkbox.setChecked(
            self.mode_state.get().heartbeat_enabled
        )

        self.soundscape_checkbox = QCheckBox(
            "Soundscape samples — fade selected WAV in and out"
        )
        self.soundscape_checkbox.setChecked(
            self.mode_state.get().soundscape_enabled
        )

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
        self.correlation_checkbox.setEnabled(True)

        self.breath_checkbox = QCheckBox(
            "Breath algorithm — gain + spectral + width modulation"
        )
        self.breath_checkbox.setChecked(
            self.mode_state.get().breath_enabled
        )

        self.motif_expand_button = QToolButton()
        self.motif_expand_button.setText("Dream motif catalogue")
        self.motif_expand_button.setCheckable(True)
        self.motif_expand_button.setChecked(True)
        self.motif_expand_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.motif_expand_button.setArrowType(
            Qt.ArrowType.RightArrow
        )
        controls_layout.addWidget(self.motif_expand_button)

        self.motif_panel = QWidget()
        motif_form = QFormLayout(self.motif_panel)
        motif_form.setContentsMargins(24, 4, 0, 8)

        self.motif_directory_label = QLabel(
            str(SOUND_EFFECTS_DIRECTORY)
        )
        self.motif_directory_label.setWordWrap(True)
        motif_form.addRow("Motif root:", self.motif_directory_label)

        self.motif_combo = QComboBox()
        motif_form.addRow("Detected motif:", self.motif_combo)

        self.motif_reload_button = QPushButton(
            "Rescan dream motifs"
        )
        motif_form.addRow("", self.motif_reload_button)

        self.motif_summary_label = QLabel(
            "Dream motifs have not been scanned yet."
        )
        self.motif_summary_label.setWordWrap(True)
        motif_form.addRow("Catalogue status:", self.motif_summary_label)

        self.motif_detail_label = QLabel("")
        self.motif_detail_label.setWordWrap(True)
        motif_form.addRow("Selected motif:", self.motif_detail_label)

        self.motif_playing_label = QLabel("No motif audio active")
        self.motif_playing_label.setWordWrap(True)
        motif_form.addRow("Motif playback:", self.motif_playing_label)

        controls_layout.addWidget(self.motif_panel)

        self.soundscape_expand_button = QToolButton()
        self.soundscape_expand_button.setText(
            "Soundscape sample test"
        )
        self.soundscape_expand_button.setCheckable(True)
        self.soundscape_expand_button.setChecked(
            bool(
                self.loaded_settings.get(
                    "soundscape_panel_expanded",
                    True,
                )
            )
        )
        self.soundscape_expand_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.soundscape_expand_button.setArrowType(
            Qt.ArrowType.RightArrow
        )
        controls_layout.addWidget(self.soundscape_expand_button)

        self.soundscape_panel = QWidget()
        soundscape_form = QFormLayout(self.soundscape_panel)
        soundscape_form.setContentsMargins(24, 4, 0, 8)
        self.soundscape_panel.setVisible(
            self.soundscape_expand_button.isChecked()
        )

        self.soundscape_panel_enable = QCheckBox(
            "Enable selected soundscape sample"
        )
        self.soundscape_panel_enable.setChecked(
            self.mode_state.get().soundscape_enabled
        )
        soundscape_form.addRow(
            "",
            self.soundscape_panel_enable,
        )

        self.soundscape_directory_label = QLabel(
            str(SOUND_EFFECTS_DIRECTORY)
        )
        self.soundscape_directory_label.setWordWrap(True)
        soundscape_form.addRow(
            "Audio directory:",
            self.soundscape_directory_label,
        )

        self.soundscape_file_combo = QComboBox()
        soundscape_form.addRow(
            "Selected audio:",
            self.soundscape_file_combo,
        )

        self.soundscape_reload_button = QPushButton(
            "Reload audio directory"
        )
        soundscape_form.addRow(
            "",
            self.soundscape_reload_button,
        )

        sample_spec, _, _, _, _ = self.ambient_sample_state.get()

        self.soundscape_fade_in_control = FloatControl(
            minimum=0.1,
            maximum=120.0,
            value=sample_spec.fade_in_seconds,
            step=0.5,
            decimals=1,
            suffix=" s",
            on_change=lambda value: self._update_soundscape(
                fade_in_seconds=value
            ),
        )
        soundscape_form.addRow(
            "Fade-in time:",
            self.soundscape_fade_in_control,
        )

        self.soundscape_fade_out_control = FloatControl(
            minimum=0.1,
            maximum=120.0,
            value=sample_spec.fade_out_seconds,
            step=0.5,
            decimals=1,
            suffix=" s",
            on_change=lambda value: self._update_soundscape(
                fade_out_seconds=value
            ),
        )
        soundscape_form.addRow(
            "Fade-out time:",
            self.soundscape_fade_out_control,
        )

        self.soundscape_duration_min_control = FloatControl(
            minimum=1.0,
            maximum=3600.0,
            value=sample_spec.duration_min_seconds,
            step=1.0,
            decimals=0,
            suffix=" s",
            on_change=lambda value: self._update_soundscape_range(
                "duration_min_seconds",
                value,
            ),
        )
        soundscape_form.addRow(
            "Audible duration minimum:",
            self.soundscape_duration_min_control,
        )

        self.soundscape_duration_max_control = FloatControl(
            minimum=1.0,
            maximum=3600.0,
            value=sample_spec.duration_max_seconds,
            step=1.0,
            decimals=0,
            suffix=" s",
            on_change=lambda value: self._update_soundscape_range(
                "duration_max_seconds",
                value,
            ),
        )
        soundscape_form.addRow(
            "Audible duration maximum:",
            self.soundscape_duration_max_control,
        )

        self.soundscape_silence_min_control = FloatControl(
            minimum=0.0,
            maximum=3600.0,
            value=sample_spec.silence_min_seconds,
            step=1.0,
            decimals=0,
            suffix=" s",
            on_change=lambda value: self._update_soundscape_range(
                "silence_min_seconds",
                value,
            ),
        )
        soundscape_form.addRow(
            "Silent interval minimum:",
            self.soundscape_silence_min_control,
        )

        self.soundscape_silence_max_control = FloatControl(
            minimum=0.0,
            maximum=3600.0,
            value=sample_spec.silence_max_seconds,
            step=1.0,
            decimals=0,
            suffix=" s",
            on_change=lambda value: self._update_soundscape_range(
                "silence_max_seconds",
                value,
            ),
        )
        soundscape_form.addRow(
            "Silent interval maximum:",
            self.soundscape_silence_max_control,
        )

        self.soundscape_volume_min_control = FloatControl(
            minimum=-60.0,
            maximum=0.0,
            value=sample_spec.volume_min_db,
            step=0.5,
            decimals=1,
            suffix=" dB",
            on_change=lambda value: self._update_soundscape_range(
                "volume_min_db",
                value,
            ),
        )
        soundscape_form.addRow(
            "Volume minimum:",
            self.soundscape_volume_min_control,
        )

        self.soundscape_volume_max_control = FloatControl(
            minimum=-60.0,
            maximum=0.0,
            value=sample_spec.volume_max_db,
            step=0.5,
            decimals=1,
            suffix=" dB",
            on_change=lambda value: self._update_soundscape_range(
                "volume_max_db",
                value,
            ),
        )
        soundscape_form.addRow(
            "Volume maximum:",
            self.soundscape_volume_max_control,
        )

        self.soundscape_volume_walk_min_control = FloatControl(
            minimum=1.0,
            maximum=1800.0,
            value=sample_spec.volume_walk_min_seconds,
            step=1.0,
            decimals=0,
            suffix=" s",
            on_change=lambda value: self._update_soundscape_range(
                "volume_walk_min_seconds",
                value,
            ),
        )
        soundscape_form.addRow(
            "Volume drift minimum:",
            self.soundscape_volume_walk_min_control,
        )

        self.soundscape_volume_walk_max_control = FloatControl(
            minimum=1.0,
            maximum=1800.0,
            value=sample_spec.volume_walk_max_seconds,
            step=1.0,
            decimals=0,
            suffix=" s",
            on_change=lambda value: self._update_soundscape_range(
                "volume_walk_max_seconds",
                value,
            ),
        )
        soundscape_form.addRow(
            "Volume drift maximum:",
            self.soundscape_volume_walk_max_control,
        )

        self.soundscape_status_label = QLabel("")
        self.soundscape_status_label.setWordWrap(True)
        soundscape_form.addRow(
            "Sample status:",
            self.soundscape_status_label,
        )

        controls_layout.addWidget(self.soundscape_panel)

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
        controls_layout.addWidget(self.base_checkbox)
        controls_layout.addWidget(self.heartbeat_checkbox)

        self.heartbeat_spatial_expand_button = QToolButton()
        self.heartbeat_spatial_expand_button.setText("Heartbeat 3D position")
        self.heartbeat_spatial_expand_button.setCheckable(True)
        self.heartbeat_spatial_expand_button.setChecked(
            bool(self.loaded_settings.get(
                "heartbeat_spatial_panel_expanded", True
            ))
        )
        self.heartbeat_spatial_expand_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.heartbeat_spatial_expand_button.setArrowType(
            Qt.ArrowType.RightArrow
        )
        controls_layout.addWidget(self.heartbeat_spatial_expand_button)

        self.heartbeat_spatial_panel = QWidget()
        heartbeat_spatial_form = QFormLayout(self.heartbeat_spatial_panel)
        heartbeat_spatial_form.setContentsMargins(24, 4, 0, 8)
        heartbeat_position = self.mixer.heartbeat_spatial_state.get()

        self.heartbeat_level_control = FloatControl(
            minimum=-24.0,
            maximum=24.0,
            value=heartbeat_position.level_db,
            step=0.5,
            decimals=1,
            suffix=" dB",
            on_change=lambda value: (
                self._update_heartbeat_position(
                    level_db=value
                )
            ),
        )
        heartbeat_spatial_form.addRow(
            "Heartbeat level:",
            self.heartbeat_level_control,
        )

        self.heartbeat_distance_control = FloatControl(
            minimum=HEARTBEAT_DISTANCE_MIN_METERS,
            maximum=HEARTBEAT_DISTANCE_MAX_METERS,
            value=heartbeat_position.distance,
            step=0.05,
            decimals=2,
            suffix=" m",
            on_change=lambda value: self._update_heartbeat_position(distance=value),
        )
        heartbeat_spatial_form.addRow(
            "Forward distance:", self.heartbeat_distance_control
        )

        self.heartbeat_horizontal_control = FloatControl(
            minimum=HEARTBEAT_HORIZONTAL_MIN_METERS,
            maximum=HEARTBEAT_HORIZONTAL_MAX_METERS,
            value=heartbeat_position.horizontal,
            step=0.05,
            decimals=2,
            suffix=" m",
            on_change=lambda value: self._update_heartbeat_position(horizontal=value),
        )
        heartbeat_spatial_form.addRow(
            "Left / right:", self.heartbeat_horizontal_control
        )

        self.heartbeat_vertical_control = FloatControl(
            minimum=HEARTBEAT_VERTICAL_MIN_METERS,
            maximum=HEARTBEAT_VERTICAL_MAX_METERS,
            value=heartbeat_position.vertical,
            step=0.05,
            decimals=2,
            suffix=" m",
            on_change=lambda value: self._update_heartbeat_position(vertical=value),
        )
        heartbeat_spatial_form.addRow(
            "Down / up:", self.heartbeat_vertical_control
        )

        self.heartbeat_position_status = QLabel("")
        self.heartbeat_position_status.setWordWrap(True)
        heartbeat_spatial_form.addRow(
            "Current position:", self.heartbeat_position_status
        )
        controls_layout.addWidget(self.heartbeat_spatial_panel)

        controls_layout.addWidget(self.soundscape_checkbox)
        controls_layout.addWidget(self.stereo_checkbox)
        controls_layout.addWidget(self.correlation_checkbox)

        self.metabolism_expand_button = QToolButton()
        self.metabolism_expand_button.setText("Metabolism")
        self.metabolism_expand_button.setCheckable(True)
        self.metabolism_expand_button.setChecked(
            bool(
                self.loaded_settings.get(
                    "metabolism_panel_expanded",
                    True,
                )
            )
        )
        self.metabolism_expand_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.metabolism_expand_button.setArrowType(
            Qt.ArrowType.RightArrow
        )
        controls_layout.addWidget(self.metabolism_expand_button)

        self.metabolism_panel = QWidget()
        metabolism_layout = QVBoxLayout(self.metabolism_panel)
        metabolism_layout.setContentsMargins(24, 4, 0, 8)
        metabolism_layout.setSpacing(4)

        metabolism_spec = self.mixer.metabolism_state.get()

        self.metabolism_enabled_checkbox = QCheckBox(
            "Enable metabolism — central controller owns all sound parameters"
        )
        self.metabolism_enabled_checkbox.setChecked(
            metabolism_spec.enabled
        )
        metabolism_layout.addWidget(
            self.metabolism_enabled_checkbox
        )

        def create_metabolism_subgroup(
            title: str,
            settings_key: str,
            default_expanded: bool,
        ) -> tuple[QToolButton, QWidget, QFormLayout]:
            button = QToolButton()
            button.setText(title)
            button.setCheckable(True)
            button.setChecked(
                bool(
                    self.loaded_settings.get(
                        settings_key,
                        default_expanded,
                    )
                )
            )
            button.setToolButtonStyle(
                Qt.ToolButtonStyle.ToolButtonTextBesideIcon
            )
            button.setArrowType(Qt.ArrowType.RightArrow)
            metabolism_layout.addWidget(button)

            panel = QWidget()
            form = QFormLayout(panel)
            form.setContentsMargins(20, 2, 0, 6)
            metabolism_layout.addWidget(panel)

            return button, panel, form

        (
            self.metabolism_rhythm_button,
            self.metabolism_rhythm_panel,
            metabolism_rhythm_form,
        ) = create_metabolism_subgroup(
            "Circadian rhythm",
            "metabolism_rhythm_expanded",
            True,
        )

        (
            self.metabolism_brown_button,
            self.metabolism_brown_panel,
            metabolism_brown_form,
        ) = create_metabolism_subgroup(
            "Brown-noise style parameters",
            "metabolism_brown_expanded",
            True,
        )

        (
            self.metabolism_breath_button,
            self.metabolism_breath_panel,
            metabolism_breath_form,
        ) = create_metabolism_subgroup(
            "Breath parameters",
            "metabolism_breath_expanded",
            False,
        )

        (
            self.metabolism_heartbeat_button,
            self.metabolism_heartbeat_panel,
            metabolism_heartbeat_form,
        ) = create_metabolism_subgroup(
            "Heartbeat / pulse",
            "metabolism_heartbeat_expanded",
            False,
        )

        (
            self.metabolism_3d_button,
            self.metabolism_3d_panel,
            metabolism_3d_form,
        ) = create_metabolism_subgroup(
            "Dual 3D brown-noise motion",
            "metabolism_3d_expanded",
            False,
        )

        def add_metabolism_control(
            form: QFormLayout,
            label: str,
            field_name: str,
            minimum: float,
            maximum: float,
            step: float,
            decimals: int,
            suffix: str = "",
        ) -> FloatControl:
            control = FloatControl(
                minimum=minimum,
                maximum=maximum,
                value=getattr(metabolism_spec, field_name),
                step=step,
                decimals=decimals,
                suffix=suffix,
                on_change=lambda value, field_name=field_name: (
                    self._update_metabolism(
                        **{field_name: value}
                    )
                ),
            )
            form.addRow(label, control)
            return control

        # Circadian rhythm.
        self.metabolism_control_0 = add_metabolism_control(
            metabolism_rhythm_form,
            "Phase minimum:",
            "phase_min_minutes",
            0.25,
            120.0,
            0.25,
            2,
            " min",
        )
        self.metabolism_control_1 = add_metabolism_control(
            metabolism_rhythm_form,
            "Phase maximum:",
            "phase_max_minutes",
            0.25,
            240.0,
            0.25,
            2,
            " min",
        )
        self.metabolism_resting_tendency_control = (
            add_metabolism_control(
                metabolism_rhythm_form,
                "Resting tendency:",
                "resting_tendency_percent",
                0.0,
                100.0,
                1.0,
                0,
                "%",
            )
        )

        # Brown-noise style parameters.
        self.metabolism_control_2 = add_metabolism_control(
            metabolism_brown_form,
            "Body minimum:",
            "brown_body_min",
            0.0,
            1.0,
            0.01,
            2,
        )
        self.metabolism_control_3 = add_metabolism_control(
            metabolism_brown_form,
            "Body maximum:",
            "brown_body_max",
            0.0,
            1.0,
            0.01,
            2,
        )
        self.metabolism_control_4 = add_metabolism_control(
            metabolism_brown_form,
            "Slope minimum:",
            "brown_slope_min",
            0.75,
            1.0,
            0.01,
            2,
        )
        self.metabolism_control_5 = add_metabolism_control(
            metabolism_brown_form,
            "Slope maximum:",
            "brown_slope_max",
            0.75,
            1.0,
            0.01,
            2,
        )
        self.metabolism_control_6 = add_metabolism_control(
            metabolism_brown_form,
            "Low-end minimum:",
            "brown_low_end_min_db",
            0.0,
            8.0,
            0.1,
            1,
            " dB",
        )
        self.metabolism_control_7 = add_metabolism_control(
            metabolism_brown_form,
            "Low-end maximum:",
            "brown_low_end_max_db",
            0.0,
            8.0,
            0.1,
            1,
            " dB",
        )
        self.metabolism_control_8 = add_metabolism_control(
            metabolism_brown_form,
            "Upper texture minimum:",
            "brown_texture_min",
            0.0,
            1.0,
            0.01,
            2,
        )
        self.metabolism_control_9 = add_metabolism_control(
            metabolism_brown_form,
            "Upper texture maximum:",
            "brown_texture_max",
            0.0,
            1.0,
            0.01,
            2,
        )

        # Breath parameters.
        self.metabolism_control_10 = add_metabolism_control(
            metabolism_breath_form,
            "Prominence minimum:",
            "breath_prominence_min",
            0.0,
            1.5,
            0.01,
            2,
        )
        self.metabolism_control_11 = add_metabolism_control(
            metabolism_breath_form,
            "Prominence maximum:",
            "breath_prominence_max",
            0.0,
            1.5,
            0.01,
            2,
        )
        self.metabolism_control_12 = add_metabolism_control(
            metabolism_breath_form,
            "Tempo minimum:",
            "breath_tempo_min",
            0.25,
            5.0,
            0.05,
            2,
            "×",
        )
        self.metabolism_control_13 = add_metabolism_control(
            metabolism_breath_form,
            "Tempo maximum:",
            "breath_tempo_max",
            0.25,
            5.0,
            0.05,
            2,
            "×",
        )
        self.metabolism_control_14 = add_metabolism_control(
            metabolism_breath_form,
            "Gain minimum:",
            "breath_gain_min_db",
            0.0,
            12.0,
            0.1,
            1,
            " dB",
        )
        self.metabolism_control_15 = add_metabolism_control(
            metabolism_breath_form,
            "Gain maximum:",
            "breath_gain_max_db",
            0.0,
            12.0,
            0.1,
            1,
            " dB",
        )
        self.metabolism_control_16 = add_metabolism_control(
            metabolism_breath_form,
            "Spectral minimum:",
            "breath_spectral_min",
            0.0,
            1.0,
            0.01,
            2,
        )
        self.metabolism_control_17 = add_metabolism_control(
            metabolism_breath_form,
            "Spectral maximum:",
            "breath_spectral_max",
            0.0,
            1.0,
            0.01,
            2,
        )
        self.metabolism_control_18 = add_metabolism_control(
            metabolism_breath_form,
            "Width minimum:",
            "breath_width_min",
            0.0,
            1.0,
            0.01,
            2,
        )
        self.metabolism_control_19 = add_metabolism_control(
            metabolism_breath_form,
            "Width maximum:",
            "breath_width_max",
            0.0,
            1.0,
            0.01,
            2,
        )

        # Heartbeat / pulse.
        self.metabolism_control_20 = add_metabolism_control(
            metabolism_heartbeat_form,
            "Distance minimum:",
            "heartbeat_distance_min",
            0.15,
            4.0,
            0.05,
            2,
            " m",
        )
        self.metabolism_control_21 = add_metabolism_control(
            metabolism_heartbeat_form,
            "Distance maximum:",
            "heartbeat_distance_max",
            0.15,
            4.0,
            0.05,
            2,
            " m",
        )
        self.metabolism_control_22 = add_metabolism_control(
            metabolism_heartbeat_form,
            "Level minimum:",
            "heartbeat_level_min_db",
            -24.0,
            24.0,
            0.5,
            1,
            " dB",
        )
        self.metabolism_control_23 = add_metabolism_control(
            metabolism_heartbeat_form,
            "Level maximum:",
            "heartbeat_level_max_db",
            -24.0,
            24.0,
            0.5,
            1,
            " dB",
        )

        # Dual 3D brown-noise motion.
        self.metabolism_control_24 = add_metabolism_control(
            metabolism_3d_form,
            "Layer amount minimum:",
            "brown_3d_amount_min",
            0.0,
            1.5,
            0.01,
            2,
        )
        self.metabolism_control_25 = add_metabolism_control(
            metabolism_3d_form,
            "Layer amount maximum:",
            "brown_3d_amount_max",
            0.0,
            1.5,
            0.01,
            2,
        )
        self.metabolism_control_26 = add_metabolism_control(
            metabolism_3d_form,
            "Sphere radius minimum:",
            "brown_radius_min",
            0.0,
            10.0,
            0.05,
            2,
            " m",
        )
        self.metabolism_control_27 = add_metabolism_control(
            metabolism_3d_form,
            "Sphere radius maximum:",
            "brown_radius_max",
            0.0,
            10.0,
            0.05,
            2,
            " m",
        )
        self.metabolism_control_28 = add_metabolism_control(
            metabolism_3d_form,
            "Center distance minimum:",
            "brown_center_distance_min",
            0.05,
            12.0,
            0.05,
            2,
            " m",
        )
        self.metabolism_control_29 = add_metabolism_control(
            metabolism_3d_form,
            "Center distance maximum:",
            "brown_center_distance_max",
            0.05,
            12.0,
            0.05,
            2,
            " m",
        )
        self.metabolism_control_30 = add_metabolism_control(
            metabolism_3d_form,
            "Evolution minimum:",
            "brown_evolution_min",
            0.0,
            1.0,
            0.01,
            2,
        )
        self.metabolism_control_31 = add_metabolism_control(
            metabolism_3d_form,
            "Evolution maximum:",
            "brown_evolution_max",
            0.0,
            1.0,
            0.01,
            2,
        )

        self.metabolism_status_label = QLabel("")
        self.metabolism_status_label.setWordWrap(True)
        metabolism_layout.addWidget(
            self.metabolism_status_label
        )

        controls_layout.addWidget(self.metabolism_panel)

        self.brown_motion_expand_button = QToolButton()
        self.brown_motion_expand_button.setText(
            "Dual 3D brown-source fluid motion"
        )
        self.brown_motion_expand_button.setCheckable(True)
        self.brown_motion_expand_button.setChecked(
            bool(
                self.loaded_settings.get(
                    "brown_motion_panel_expanded",
                    True,
                )
            )
        )
        self.brown_motion_expand_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.brown_motion_expand_button.setArrowType(
            Qt.ArrowType.RightArrow
        )
        controls_layout.addWidget(self.brown_motion_expand_button)

        self.brown_motion_panel = QWidget()
        brown_motion_form = QFormLayout(
            self.brown_motion_panel
        )
        brown_motion_form.setContentsMargins(24, 4, 0, 8)

        brown_motion_spec = self.mixer.brown_motion_state.get()

        self.brown_3d_layer_checkbox = QCheckBox(
            "Enable additive 3D position layer"
        )
        self.brown_3d_layer_checkbox.setChecked(
            brown_motion_spec.layer_enabled
        )
        brown_motion_form.addRow(
            "",
            self.brown_3d_layer_checkbox,
        )

        self.brown_3d_amount_control = FloatControl(
            minimum=0.0,
            maximum=1.5,
            value=brown_motion_spec.layer_amount,
            step=0.01,
            decimals=2,
            suffix="",
            on_change=lambda value: self._update_brown_motion(
                layer_amount=value
            ),
        )
        brown_motion_form.addRow(
            "3D layer amount:",
            self.brown_3d_amount_control,
        )

        self.brown_motion_enabled_checkbox = QCheckBox(
            "Enable continuous fluid motion"
        )
        self.brown_motion_enabled_checkbox.setChecked(
            brown_motion_spec.enabled
        )
        brown_motion_form.addRow(
            "",
            self.brown_motion_enabled_checkbox,
        )

        self.brown_motion_radius_control = FloatControl(
            minimum=0.0,
            maximum=10.0,
            value=brown_motion_spec.sphere_radius,
            step=0.05,
            decimals=2,
            suffix=" m",
            on_change=lambda value: self._update_brown_motion(
                sphere_radius=value
            ),
        )
        brown_motion_form.addRow(
            "Sphere radius:",
            self.brown_motion_radius_control,
        )

        self.brown_motion_center_control = FloatControl(
            minimum=0.05,
            maximum=12.0,
            value=brown_motion_spec.center_distance,
            step=0.05,
            decimals=2,
            suffix=" m",
            on_change=lambda value: self._update_brown_motion(
                center_distance=value
            ),
        )
        brown_motion_form.addRow(
            "Sphere-center distance:",
            self.brown_motion_center_control,
        )

        self.brown_motion_rate_control = FloatControl(
            minimum=0.0,
            maximum=1.0,
            value=brown_motion_spec.evolution_rate,
            step=0.01,
            decimals=2,
            suffix="",
            on_change=lambda value: self._update_brown_motion(
                evolution_rate=value
            ),
        )
        brown_motion_form.addRow(
            "Evolution rate:",
            self.brown_motion_rate_control,
        )

        self.brown_motion_status_label = QLabel("")
        self.brown_motion_status_label.setWordWrap(True)
        brown_motion_form.addRow(
            "Motion status:",
            self.brown_motion_status_label,
        )

        controls_layout.addWidget(self.brown_motion_panel)

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
        self.heartbeat_label = QLabel("—")

        status_form.addRow("Playback:", self.playback_label)
        status_form.addRow("Active path:", self.mode_label)
        status_form.addRow("Correlation:", self.correlation_label)
        self.pipeline_label = QLabel(
            "Correlated stereo foundation plus a soft-coupled moving 3D layer"
        )
        status_form.addRow("DSP pipeline:", self.pipeline_label)
        status_form.addRow("Breath:", self.breath_label)
        status_form.addRow("Heartbeat:", self.heartbeat_label)
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
        self.base_checkbox.toggled.connect(self._on_modes_changed)
        self.heartbeat_checkbox.toggled.connect(
            self._on_modes_changed
        )
        self.heartbeat_spatial_expand_button.toggled.connect(
            self._toggle_heartbeat_spatial_panel
        )
        self.soundscape_checkbox.toggled.connect(
            self._on_modes_changed
        )
        self.soundscape_checkbox.toggled.connect(
            self.soundscape_panel_enable.setChecked
        )
        self.soundscape_panel_enable.toggled.connect(
            self.soundscape_checkbox.setChecked
        )
        self.soundscape_file_combo.currentTextChanged.connect(
            self._on_soundscape_file_changed
        )
        self.soundscape_reload_button.clicked.connect(
            self._reload_soundscape_files
        )
        self.motif_expand_button.toggled.connect(
            self._toggle_motif_panel
        )
        self.motif_reload_button.clicked.connect(
            self._reload_dream_motifs
        )
        self.motif_combo.currentTextChanged.connect(
            self._on_motif_changed
        )
        self.soundscape_expand_button.toggled.connect(
            self._toggle_soundscape_panel
        )
        self.stereo_checkbox.toggled.connect(self._on_modes_changed)
        self.correlation_checkbox.toggled.connect(
            self._on_modes_changed
        )
        self.metabolism_expand_button.toggled.connect(
            self._toggle_metabolism_panel
        )
        self.metabolism_enabled_checkbox.toggled.connect(
            self._on_metabolism_toggled
        )
        self.metabolism_rhythm_button.toggled.connect(
            lambda expanded: self._toggle_metabolism_subgroup(
                self.metabolism_rhythm_button,
                self.metabolism_rhythm_panel,
                expanded,
            )
        )
        self.metabolism_brown_button.toggled.connect(
            lambda expanded: self._toggle_metabolism_subgroup(
                self.metabolism_brown_button,
                self.metabolism_brown_panel,
                expanded,
            )
        )
        self.metabolism_breath_button.toggled.connect(
            lambda expanded: self._toggle_metabolism_subgroup(
                self.metabolism_breath_button,
                self.metabolism_breath_panel,
                expanded,
            )
        )
        self.metabolism_heartbeat_button.toggled.connect(
            lambda expanded: self._toggle_metabolism_subgroup(
                self.metabolism_heartbeat_button,
                self.metabolism_heartbeat_panel,
                expanded,
            )
        )
        self.metabolism_3d_button.toggled.connect(
            lambda expanded: self._toggle_metabolism_subgroup(
                self.metabolism_3d_button,
                self.metabolism_3d_panel,
                expanded,
            )
        )
        self.brown_motion_expand_button.toggled.connect(
            self._toggle_brown_motion_panel
        )
        self.brown_3d_layer_checkbox.toggled.connect(
            self._on_brown_3d_layer_toggled
        )
        self.brown_motion_enabled_checkbox.toggled.connect(
            self._on_brown_motion_toggled
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

        self._toggle_motif_panel(
            self.motif_expand_button.isChecked()
        )
        self._toggle_soundscape_panel(
            self.soundscape_expand_button.isChecked()
        )
        self._reload_soundscape_files()
        self._reload_dream_motifs()
        self._toggle_noise_panel(
            self.noise_expand_button.isChecked()
        )
        self._toggle_heartbeat_spatial_panel(
            self.heartbeat_spatial_expand_button.isChecked()
        )
        self._update_heartbeat_position_status()
        self._toggle_metabolism_panel(
            self.metabolism_expand_button.isChecked()
        )
        self._toggle_metabolism_subgroup(
            self.metabolism_rhythm_button,
            self.metabolism_rhythm_panel,
            self.metabolism_rhythm_button.isChecked(),
        )
        self._toggle_metabolism_subgroup(
            self.metabolism_brown_button,
            self.metabolism_brown_panel,
            self.metabolism_brown_button.isChecked(),
        )
        self._toggle_metabolism_subgroup(
            self.metabolism_breath_button,
            self.metabolism_breath_panel,
            self.metabolism_breath_button.isChecked(),
        )
        self._toggle_metabolism_subgroup(
            self.metabolism_heartbeat_button,
            self.metabolism_heartbeat_panel,
            self.metabolism_heartbeat_button.isChecked(),
        )
        self._toggle_metabolism_subgroup(
            self.metabolism_3d_button,
            self.metabolism_3d_panel,
            self.metabolism_3d_button.isChecked(),
        )
        self._update_metabolism_status()
        self._toggle_brown_motion_panel(
            self.brown_motion_expand_button.isChecked()
        )
        self._update_brown_motion_status()
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

        saved_sound_effects_checked = self.loaded_settings.get(
            "sound_effects_checkbox_checked",
            self.mode_state.get().soundscape_enabled,
        )
        self.soundscape_checkbox.setChecked(
            bool(saved_sound_effects_checked)
        )
        self.soundscape_panel_enable.setChecked(
            bool(saved_sound_effects_checked)
        )

        self._on_modes_changed()

    def _toggle_motif_panel(self, expanded: bool) -> None:
        self.motif_panel.setVisible(expanded)
        self.motif_expand_button.setArrowType(
            Qt.ArrowType.DownArrow
            if expanded
            else Qt.ArrowType.RightArrow
        )

    def _reload_dream_motifs(self) -> None:
        self.motif_summary_label.setText(
            "Scanning dream motif subfolders…"
        )
        QApplication.processEvents()

        motifs = self.dream_motif_catalog.scan()

        self.motif_combo.blockSignals(True)
        self.motif_combo.clear()
        for motif in motifs:
            self.motif_combo.addItem(motif.name)
        self.motif_combo.blockSignals(False)

        ambient_count = sum(len(m.ambient_assets) for m in motifs)
        layered_count = sum(len(m.layered_assets) for m in motifs)

        if motifs:
            self.motif_summary_label.setText(
                f"Detected {len(motifs)} motif folder(s): "
                f"{ambient_count} long ambient file(s), "
                f"{layered_count} layered event file(s). "
                f"Layer threshold: ≤ "
                f"{DREAM_MOTIF_LAYER_THRESHOLD_SECONDS:.1f} s."
            )
            self.motif_combo.setCurrentIndex(0)
            self._on_motif_changed(self.motif_combo.currentText())
        else:
            self.motif_summary_label.setText(
                "No motif subfolders containing supported audio were "
                f"found beneath {SOUND_EFFECTS_DIRECTORY}."
            )
            self.motif_detail_label.setText("")

        if self.dream_motif_catalog.errors:
            preview = "; ".join(
                self.dream_motif_catalog.errors[:3]
            )
            extra = len(self.dream_motif_catalog.errors) - 3
            if extra > 0:
                preview += f"; plus {extra} more"
            self.motif_summary_label.setText(
                self.motif_summary_label.text()
                + f" Probe errors: {preview}"
            )

    def _choose_motif_ambient_asset(
        self,
        motif: DreamMotif,
    ) -> DreamMotifAsset | None:
        if not motif.ambient_assets:
            return None

        candidates = list(motif.ambient_assets)
        if (
            self.active_motif_asset is not None
            and len(candidates) > 1
        ):
            candidates = [
                asset
                for asset in candidates
                if asset.path != self.active_motif_asset.path
            ]

        index = int(
            self.motif_rng.integers(0, len(candidates))
        )
        return candidates[index]

    def _load_next_motif_asset(self) -> None:
        motif = self.dream_motif_catalog.find(
            self.active_motif_name
        )
        if motif is None:
            self.motif_playback_enabled = False
            self.active_motif_asset = None
            self.motif_playing_label.setText(
                "No motif audio active"
            )
            return

        asset = self._choose_motif_ambient_asset(motif)
        if asset is None:
            self.motif_playback_enabled = False
            self.active_motif_asset = None
            self.ambient_sample_state.load_file(None)
            self.motif_playing_label.setText(
                "No long ambient files in this motif"
            )
            return

        self.active_motif_asset = asset
        self.motif_playback_enabled = True

        # Automatic motif playback shares the tested ambience DSP path but
        # remains separate from the individual-file audition dropdown.
        self.ambient_sample_state.update(
            selected_filename=""
        )
        self.ambient_sample_state.load_file(asset.path)

        self.soundscape_checkbox.setChecked(True)
        self.soundscape_panel_enable.setChecked(True)

        self.motif_playing_label.setText(
            f"{asset.path.name} ({asset.duration_seconds:.1f} s)"
        )

    def _on_motif_changed(self, motif_name: str) -> None:
        motif_name = motif_name.strip()
        self.active_motif_name = motif_name
        self.previous_motif_stage = ""

        motif = self.dream_motif_catalog.find(motif_name)
        if motif is None:
            self.motif_detail_label.setText("")
            self.motif_playback_enabled = False
            self.active_motif_asset = None
            self.ambient_sample_state.load_file(None)
            self.motif_playing_label.setText(
                "No motif audio active"
            )
            return

        ambient_names = ", ".join(
            asset.path.name for asset in motif.ambient_assets[:5]
        ) or "none"
        layered_names = ", ".join(
            asset.path.name for asset in motif.layered_assets[:5]
        ) or "none"

        if len(motif.ambient_assets) > 5:
            ambient_names += ", …"
        if len(motif.layered_assets) > 5:
            layered_names += ", …"

        self.motif_detail_label.setText(
            f"{motif.name}: {len(motif.ambient_assets)} long ambient "
            f"file(s) [{ambient_names}]; "
            f"{len(motif.layered_assets)} layered event file(s) "
            f"[{layered_names}]."
        )

        self._load_next_motif_asset()
        self._schedule_settings_save()

    def _toggle_soundscape_panel(self, expanded: bool) -> None:
        self.soundscape_panel.setVisible(expanded)
        self.soundscape_expand_button.setArrowType(
            Qt.ArrowType.DownArrow
            if expanded
            else Qt.ArrowType.RightArrow
        )
        self._schedule_settings_save()

    def _reload_soundscape_files(self) -> None:
        SOUND_EFFECTS_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Individual audition mode deliberately scans only files placed in the
        # root sounds folder. Motif assets remain organized in subfolders.
        files = sorted(
            (
                path
                for path in SOUND_EFFECTS_DIRECTORY.iterdir()
                if path.is_file()
                and path.suffix.lower()
                in SUPPORTED_AUDIO_EXTENSIONS
            ),
            key=lambda path: path.name.lower(),
        )

        self.soundscape_file_combo.blockSignals(True)
        self.soundscape_file_combo.clear()
        self.soundscape_file_combo.addItem("")
        for path in files:
            self.soundscape_file_combo.addItem(path.name)
        self.soundscape_file_combo.setCurrentIndex(0)
        self.soundscape_file_combo.blockSignals(False)

        # Explicitly unload a prior audition sample, but do not disturb
        # automatic motif playback that may already be active.
        if not self.motif_playback_enabled:
            self._on_soundscape_file_changed("")

    def _on_soundscape_file_changed(self, filename: str) -> None:
        filename = filename.strip()

        if filename:
            self.motif_playback_enabled = False
            self.active_motif_asset = None
            self.motif_playing_label.setText(
                "Paused by individual audition"
            )

        self.ambient_sample_state.update(
            selected_filename=filename
        )

        if filename:
            self.soundscape_status_label.setText(
                f"Loading {filename}…"
            )
            QApplication.processEvents()

        path = (
            SOUND_EFFECTS_DIRECTORY / filename
            if filename
            else None
        )
        self.ambient_sample_state.load_file(path)

        _, audio, loaded_name, error, _ = (
            self.ambient_sample_state.get()
        )

        if error:
            self.soundscape_status_label.setText(
                f"Could not load audio: {error}"
            )
        elif audio is None:
            self.soundscape_status_label.setText(
                "No audio file selected"
            )
        else:
            duration = len(audio) / 44_100
            (
                source_db,
                normalization_db,
                normalized_db,
                normalized_peak_db,
            ) = self.ambient_sample_state.normalization_info()

            self.soundscape_status_label.setText(
                f"Loaded {loaded_name}; "
                f"{duration:.1f} s; "
                f"source typical {source_db:.1f} dBFS; "
                f"normalization {normalization_db:+.1f} dB; "
                f"normalized typical {normalized_db:.1f} dBFS; "
                f"peak {normalized_peak_db:.1f} dBFS"
            )

        self._schedule_settings_save()

    def _update_soundscape(self, **changes) -> None:
        self.ambient_sample_state.update(**changes)
        self._schedule_settings_save()

    def _update_soundscape_range(
        self,
        field_name: str,
        value: float,
    ) -> None:
        spec, _, _, _, _ = self.ambient_sample_state.get()

        changes = {field_name: value}

        if field_name == "duration_min_seconds":
            changes["duration_max_seconds"] = max(
                value,
                spec.duration_max_seconds,
            )
        elif field_name == "duration_max_seconds":
            changes["duration_min_seconds"] = min(
                value,
                spec.duration_min_seconds,
            )
        elif field_name == "silence_min_seconds":
            changes["silence_max_seconds"] = max(
                value,
                spec.silence_max_seconds,
            )
        elif field_name == "silence_max_seconds":
            changes["silence_min_seconds"] = min(
                value,
                spec.silence_min_seconds,
            )
        elif field_name == "volume_min_db":
            changes["volume_max_db"] = max(
                value,
                spec.volume_max_db,
            )
        elif field_name == "volume_max_db":
            changes["volume_min_db"] = min(
                value,
                spec.volume_min_db,
            )
        elif field_name == "volume_walk_min_seconds":
            changes["volume_walk_max_seconds"] = max(
                value,
                spec.volume_walk_max_seconds,
            )
        elif field_name == "volume_walk_max_seconds":
            changes["volume_walk_min_seconds"] = min(
                value,
                spec.volume_walk_min_seconds,
            )

        self.ambient_sample_state.update(**changes)
        self._schedule_settings_save()

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
        current_stage = self.mixer.current_soundscape_stage
        if (
            self.motif_playback_enabled
            and current_stage == "fading in"
            and self.previous_motif_stage == "silent"
        ):
            self._load_next_motif_asset()
            current_stage = self.mixer.current_soundscape_stage

        self.previous_motif_stage = current_stage

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

    def _toggle_metabolism_subgroup(
        self,
        button: QToolButton,
        panel: QWidget,
        expanded: bool,
    ) -> None:
        panel.setVisible(expanded)
        button.setArrowType(
            Qt.ArrowType.DownArrow
            if expanded
            else Qt.ArrowType.RightArrow
        )
        self._schedule_settings_save()

    def _toggle_metabolism_panel(
        self,
        expanded: bool,
    ) -> None:
        self.metabolism_panel.setVisible(expanded)
        self.metabolism_expand_button.setArrowType(
            Qt.ArrowType.DownArrow
            if expanded
            else Qt.ArrowType.RightArrow
        )
        self._schedule_settings_save()

    def _on_metabolism_toggled(
        self,
        checked: bool,
    ) -> None:
        self._update_metabolism(enabled=bool(checked))

    def _update_metabolism(self, **changes) -> None:
        self.mixer.metabolism_state.update(**changes)
        self._update_metabolism_status()
        self._schedule_settings_save()

    def _update_metabolism_status(self) -> None:
        spec = self.mixer.metabolism_state.get()
        values = self.mixer.current_metabolism_values
        if not spec.enabled or values is None:
            self.metabolism_status_label.setText(
                "off — all existing manual controls are active"
            )
            return
        self.metabolism_status_label.setText(
            f"raw state {values.activity:.3f}; "
            f"activity drive {values.activity_drive:.3f}; "
            f"body {values.brown_body:.2f}, slope {values.brown_slope:.2f}, "
            f"low end {values.brown_low_end_db:.1f} dB, "
            f"texture {values.brown_texture:.2f}; "
            f"breath {values.breath_gain_db:.1f} dB/"
            f"{values.breath_tempo:.2f}×; "
            f"heart {values.heartbeat_distance:.2f} m/"
            f"{values.heartbeat_level_db:+.1f} dB; "
            f"3D {values.brown_3d_amount:.2f}, "
            f"radius {values.brown_radius:.2f} m, "
            f"center {values.brown_center_distance:.2f} m, "
            f"evolution {values.brown_evolution:.2f}"
        )


    def _toggle_brown_motion_panel(
        self,
        expanded: bool,
    ) -> None:
        self.brown_motion_panel.setVisible(expanded)
        self.brown_motion_expand_button.setArrowType(
            Qt.ArrowType.DownArrow
            if expanded
            else Qt.ArrowType.RightArrow
        )
        self._schedule_settings_save()

    def _on_brown_3d_layer_toggled(
        self,
        checked: bool,
    ) -> None:
        self._update_brown_motion(
            layer_enabled=bool(checked)
        )

    def _on_brown_motion_toggled(
        self,
        checked: bool,
    ) -> None:
        self._update_brown_motion(enabled=bool(checked))

    def _update_brown_motion(self, **changes) -> None:
        self.mixer.brown_motion_state.update(**changes)
        self._update_brown_motion_status()
        self._schedule_settings_save()

    def _update_brown_motion_status(self) -> None:
        spec = self.mixer.brown_motion_state.get()
        left = self.mixer.current_brown_left_position
        right = self.mixer.current_brown_right_position

        state_text = "moving" if spec.enabled else "frozen"
        layer_text = (
            "3D audible"
            if spec.layer_enabled
            else "3D muted"
        )
        self.brown_motion_status_label.setText(
            f"{layer_text} @ {spec.layer_amount:.2f}; "
            f"{state_text}; separation "
            f"{self.mixer.current_brown_motion_separation:.1f}°; "
            f"L ({left.x:.2f}, {left.y:.2f}, {left.z:.2f}); "
            f"R ({right.x:.2f}, {right.y:.2f}, {right.z:.2f})"
        )

    def _toggle_heartbeat_spatial_panel(self, expanded: bool) -> None:
        self.heartbeat_spatial_panel.setVisible(expanded)
        self.heartbeat_spatial_expand_button.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self._schedule_settings_save()

    def _update_heartbeat_position(self, **changes) -> None:
        self.mixer.set_heartbeat_position(**changes)
        self._update_heartbeat_position_status()
        self._schedule_settings_save()

    def _update_heartbeat_position_status(self) -> None:
        position = self.mixer.current_heartbeat_position
        spec = self.mixer.heartbeat_spatial_state.get()
        self.heartbeat_position_status.setText(
            f"{spec.level_db:+.1f} dB; "
            f"({position.x:.2f}, {position.y:.2f}, {position.z:.2f}) m; "
            "position abruptly randomized by body movement events"
        )

    def _on_modes_changed(self) -> None:
        stereo = self.stereo_checkbox.isChecked()

        self.mode_state.set(
            base_enabled=self.base_checkbox.isChecked(),
            stereo_enabled=stereo,
            correlation_enabled=self.correlation_checkbox.isChecked(),
            breath_enabled=self.breath_checkbox.isChecked(),
            heartbeat_enabled=self.heartbeat_checkbox.isChecked(),
            soundscape_enabled=self.soundscape_checkbox.isChecked(),
        )

        if not self.base_checkbox.isChecked():
            path = "Main Living Brown Noise muted"
        elif not stereo:
            path = "Mono duplicated L/R"
        elif self.correlation_checkbox.isChecked():
            path = (
                "Stereo: shared + independent, evolving correlation"
            )
        else:
            path = "Stereo: fully independent left/right"

        if self.breath_checkbox.isChecked() and self.base_checkbox.isChecked():
            path += " + breath"
        if self.heartbeat_checkbox.isChecked():
            path += " + heartbeat"
        if self.soundscape_checkbox.isChecked():
            path += " + soundscape sample"

        self.mode_label.setText(path)
        self._schedule_settings_save()

    def _schedule_settings_save(self) -> None:
        self.settings_save_timer.start(250)

    def _save_settings(self) -> None:
        noise_spec, _ = self.noise_state.get()
        noise_evolution_spec = self.noise_evolution_state.get()
        body_movement_spec = self.body_movement_state.get()
        heartbeat_spec = self.heartbeat_state.get()
        ambient_sample_spec, _, _, _, _ = (
            self.ambient_sample_state.get()
        )
        ambient_sample_settings = asdict(ambient_sample_spec)
        ambient_sample_settings["selected_filename"] = ""
        breath_spec, _ = self.breath_state.get()
        breath_evolution_spec = self.breath_evolution_state.get()
        motion_spec = self.motion_state.get()
        brown_motion_spec = self.mixer.brown_motion_state.get()
        heartbeat_spatial_spec = self.mixer.heartbeat_spatial_state.get()
        metabolism_spec = self.mixer.metabolism_state.get()
        modes = self.mode_state.get()

        data = {
            "version": 2,
            "modes": asdict(modes),
            # Stored explicitly because soundscape catalog initialization can
            # otherwise overwrite the checkbox state during startup.
            "sound_effects_checkbox_checked": (
                self.soundscape_checkbox.isChecked()
            ),
            "brown_noise": asdict(noise_spec),
            "brown_noise_evolution": asdict(noise_evolution_spec),
            "body_movement": asdict(body_movement_spec),
            "heartbeat": asdict(heartbeat_spec),
            "ambient_sample": ambient_sample_settings,
            "soundscape_panel_expanded": (
                self.soundscape_expand_button.isChecked()
            ),
            "noise_panel_expanded": (
                self.noise_expand_button.isChecked()
            ),
            "breath": asdict(breath_spec),
            "breath_evolution": asdict(breath_evolution_spec),
            "breath_evolution_panel_expanded": (
                self.breath_evolution_expand_button.isChecked()
            ),
            "organic_motion": asdict(motion_spec),
            "dual_brown_motion": asdict(brown_motion_spec),
            "heartbeat_spatial": asdict(heartbeat_spatial_spec),
            "metabolism": asdict(metabolism_spec),
            "metabolism_panel_expanded": (
                self.metabolism_expand_button.isChecked()
            ),
            "metabolism_rhythm_expanded": (
                self.metabolism_rhythm_button.isChecked()
            ),
            "metabolism_brown_expanded": (
                self.metabolism_brown_button.isChecked()
            ),
            "metabolism_breath_expanded": (
                self.metabolism_breath_button.isChecked()
            ),
            "metabolism_heartbeat_expanded": (
                self.metabolism_heartbeat_button.isChecked()
            ),
            "metabolism_3d_expanded": (
                self.metabolism_3d_button.isChecked()
            ),
            "heartbeat_spatial_panel_expanded": (
                self.heartbeat_spatial_expand_button.isChecked()
            ),
            "brown_motion_panel_expanded": (
                self.brown_motion_expand_button.isChecked()
            ),
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
        heartbeat_spec = self.heartbeat_state.get()
        ambient_sample_spec, _, _, _, _ = (
            self.ambient_sample_state.get()
        )
        breath_spec, _ = self.breath_state.get()
        breath_evolution_spec = self.breath_evolution_state.get()
        motion_spec = self.motion_state.get()
        brown_motion_spec = self.mixer.brown_motion_state.get()
        heartbeat_spatial_spec = self.mixer.heartbeat_spatial_state.get()
        metabolism_spec = self.mixer.metabolism_state.get()

        self.export_worker = ExportWorker(
            output_path=output_path,
            duration_minutes=self.export_duration_slider.value(),
            sample_rate=44_100,
            modes=modes,
            noise_spec=noise_spec,
            noise_evolution_spec=noise_evolution_spec,
            body_movement_spec=body_movement_spec,
            heartbeat_spec=heartbeat_spec,
            ambient_sample_spec=ambient_sample_spec,
            sound_effects_directory=SOUND_EFFECTS_DIRECTORY,
            breath_spec=breath_spec,
            breath_evolution_spec=breath_evolution_spec,
            motion_spec=motion_spec,
            brown_motion_spec=brown_motion_spec,
            heartbeat_spatial_spec=heartbeat_spatial_spec,
            metabolism_spec=metabolism_spec,
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

        self._update_metabolism_status()
        self._update_brown_motion_status()
        self._update_heartbeat_position_status()

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
        self.heartbeat_label.setText(
            f"{self.mixer.heartbeat.current_rate_bpm:.1f} bpm; "
            f"prominence "
            f"{self.mixer.heartbeat.current_prominence:.3f}; "
            f"envelope {self.mixer.current_heartbeat:.3f}"
        )
        _, _, loaded_name, load_error, _ = (
            self.ambient_sample_state.get()
        )
        if load_error:
            self.soundscape_status_label.setText(
                f"Load error: {load_error}"
            )
        elif loaded_name:
            (
                _,
                normalization_db,
                normalized_db,
                _,
            ) = self.ambient_sample_state.normalization_info()

            self.soundscape_status_label.setText(
                f"{loaded_name}; "
                f"{self.mixer.current_soundscape_stage}; "
                f"evolved volume "
                f"{self.mixer.current_soundscape_gain_db:.1f} dB; "
                f"source normalization "
                f"{normalization_db:+.1f} dB "
                f"to {normalized_db:.1f} dBFS typical"
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
        self.mixer.close()
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
            base_enabled=bool(
                mode_data.get(
                    "base_enabled",
                    default_modes.base_enabled,
                )
            ),
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
            heartbeat_enabled=bool(
                mode_data.get(
                    "heartbeat_enabled",
                    default_modes.heartbeat_enabled,
                )
            ),
            soundscape_enabled=bool(
                mode_data.get(
                    "soundscape_enabled",
                    default_modes.soundscape_enabled,
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

    default_ambient_sample = AmbientSampleSpec()
    ambient_sample_data = dict(loaded.get("ambient_sample", {}))

    # Audition selection is intentionally session-only. Never restore a file
    # and begin playing it merely because it was selected in a previous run.
    ambient_sample_data["selected_filename"] = ""

    try:
        ambient_sample_spec = AmbientSampleSpec(
            **{
                field_name: ambient_sample_data.get(
                    field_name,
                    getattr(default_ambient_sample, field_name),
                )
                for field_name in asdict(default_ambient_sample)
            }
        ).validated()
    except Exception:
        ambient_sample_spec = default_ambient_sample

    default_heartbeat = HeartbeatSpec()
    heartbeat_data = loaded.get("heartbeat", {})
    try:
        heartbeat_spec = HeartbeatSpec(
            **{
                field_name: heartbeat_data.get(
                    field_name,
                    getattr(default_heartbeat, field_name),
                )
                for field_name in asdict(default_heartbeat)
            }
        ).validated()
    except Exception:
        heartbeat_spec = default_heartbeat

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

    default_brown_motion = DualBrownMotionSpec()
    brown_motion_data = loaded.get("dual_brown_motion", {})
    try:
        brown_motion_spec = DualBrownMotionSpec(
            **{
                field_name: brown_motion_data.get(
                    field_name,
                    getattr(default_brown_motion, field_name),
                )
                for field_name in asdict(default_brown_motion)
            }
        ).validated()
    except Exception:
        brown_motion_spec = default_brown_motion

    default_heartbeat_spatial = HeartbeatSpatialSpec()
    heartbeat_spatial_data = loaded.get("heartbeat_spatial", {})
    try:
        heartbeat_spatial_spec = HeartbeatSpatialSpec(
            **{
                field_name: heartbeat_spatial_data.get(
                    field_name,
                    getattr(default_heartbeat_spatial, field_name),
                )
                for field_name in asdict(default_heartbeat_spatial)
            }
        ).validated()
    except Exception:
        heartbeat_spatial_spec = default_heartbeat_spatial

    default_metabolism = MetabolismSpec()
    metabolism_data = dict(loaded.get("metabolism", {}))

    # Migrate the previous 0..1 quiet-state bias setting to the clearer
    # percentage-based resting-tendency setting.
    if (
        "resting_tendency_percent" not in metabolism_data
        and "quiet_state_bias" in metabolism_data
    ):
        try:
            metabolism_data["resting_tendency_percent"] = (
                float(metabolism_data["quiet_state_bias"]) * 100.0
            )
        except (TypeError, ValueError):
            pass

    try:
        metabolism_spec = MetabolismSpec(
            **{
                field_name: metabolism_data.get(
                    field_name,
                    getattr(default_metabolism, field_name),
                )
                for field_name in asdict(default_metabolism)
            }
        ).validated()
    except Exception:
        metabolism_spec = default_metabolism

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
        heartbeat_state,
        ambient_sample_state,
        breath_state,
        breath_evolution_state,
        motion_state,
    ) = build_mixer(
        sample_rate=sample_rate,
        modes=modes,
        noise_spec=noise_spec,
        noise_evolution_spec=noise_evolution_spec,
        body_movement_spec=body_movement_spec,
        heartbeat_spec=heartbeat_spec,
        ambient_sample_spec=ambient_sample_spec,
        sound_effects_directory=SOUND_EFFECTS_DIRECTORY,
        breath_spec=breath_spec,
        breath_evolution_spec=breath_evolution_spec,
        motion_spec=motion_spec,
        brown_motion_spec=brown_motion_spec,
        heartbeat_spatial_spec=heartbeat_spatial_spec,
        metabolism_spec=metabolism_spec,
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
        heartbeat_state=heartbeat_state,
        ambient_sample_state=ambient_sample_state,
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
