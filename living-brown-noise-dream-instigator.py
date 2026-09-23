from __future__ import annotations

import json
import logging
import queue
import math
import re
import sys
import threading
import time
import wave
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass, field, replace
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
    QLayout,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


# Runtime assets are resolved relative to this script so the complete folder
# can be moved without editing hard-coded paths. phonon.dll,
# steam_audio_renderer.py, and the sounds directory belong beside the script.
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
CEREMONIES_DIRECTORY = SCRIPT_DIRECTORY / "ceremonies"
CURATION_PATH = SCRIPT_DIRECTORY / "audio-curation.json"
EXPORT_DIRECTORY = SCRIPT_DIRECTORY / "exports"
CONDUCTOR_LOG_PATH = SCRIPT_DIRECTORY / "conductor-log.txt"
DEFAULT_STYLE_NAME = "indian"
STYLE_AMBIENTS_FOLDER = "ambients"
STYLE_VOCALS_FOLDER = "vocals"
STYLE_INSTRUMENTS_FOLDER = "instruments"
STYLE_AMBIENT_GROUPS = ("activity", "event", "space")


class AudioCuration:
    """Read-only snapshot of the curator's shared, versioned ratings file.

    Keys are POSIX paths relative to ceremonies/, never display names. Load
    outside the audio callback, at startup/style changes and export setup.
    Missing metadata means unreviewed; malformed metadata is reported instead
    of silently replacing the listener's ratings with random selection.
    """

    def __init__(self, path: Path = CURATION_PATH,
                 root: Path = CEREMONIES_DIRECTORY) -> None:
        self.path = Path(path)
        self.root = Path(root).resolve()
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            self.states: dict[str, str] = {}
            self.pending_key = None
            self.found = False
            return
        try:
            data = json.loads(text)
            if not isinstance(data, dict) or data.get("schema_version") != 1:
                raise ValueError("expected schema_version 1")
            tracks = data.get("tracks")
            if not isinstance(tracks, dict):
                raise ValueError("expected a tracks object")
            states = {}
            for key, record in tracks.items():
                if not isinstance(record, dict) or record.get("state") not in (
                    "unreviewed", "low", "medium", "high", "stash"
                ):
                    raise ValueError(f"invalid rating for {key!r}")
                states[key] = record["state"]
            pending = data.get("pending_move")
            if pending is not None and (
                not isinstance(pending, dict) or not isinstance(pending.get("key"), str)
            ):
                raise ValueError("invalid pending_move record")
            self.states = states
            self.pending_key = pending["key"] if pending else None
            self.found = True
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Cannot read curation settings from {self.path}: {exc}") from exc

    def state_for_path(self, path: Path) -> str:
        try:
            key = Path(path).resolve().relative_to(self.root).as_posix()
        except ValueError:
            # Legacy assets outside ceremonies/ have no curator rating.
            return "unreviewed"
        if key == self.pending_key:
            return "stash"  # Exclude an interrupted stash/restore until resolved.
        return self.states.get(key, "unreviewed")

    def tier_for_path(self, path: Path) -> str:
        state = self.state_for_path(path)
        return "low" if state == "unreviewed" else state


class CurationShuffleBag:
    """High -> Medium -> Low, then a fresh cycle of all three tiers.

    One instance serves exactly one category. Returning an unheard reservation
    restores its original tier. Avoiding a boundary repeat never skips a tier.
    """

    TIERS = ("high", "medium", "low")

    def __init__(self, tiers: dict[str, str], rng) -> None:
        self.tiers = dict(tiers)
        self.rng = rng
        self.last_name: str | None = None
        self.cycle = 0
        self.refill()

    def refill(self) -> None:
        self.remaining = {
            tier: [name for name, rating in self.tiers.items() if rating == tier]
            for tier in self.TIERS
        }
        for names in self.remaining.values():
            self.rng.shuffle(names)
        self.cycle += 1

    def take(self, avoid_name: str | None = None) -> str | None:
        if not self.tiers:
            return None
        if not any(self.remaining.values()):
            self.refill()
        for tier in self.TIERS:
            names = self.remaining[tier]
            if names:
                avoid = avoid_name if avoid_name in self.tiers else self.last_name
                name = next((name for name in names if name != avoid), names[0])
                names.remove(name)
                self.last_name = name
                return name
        return None

    def return_unused(self, name: str) -> None:
        tier = self.tiers.get(name)
        if tier is not None and name not in self.remaining[tier]:
            self.remaining[tier].insert(0, name)

    def take_specific(self, name: str) -> None:
        """An explicit manual audition may override automatic priority."""
        tier = self.tiers.get(name)
        if tier is not None:
            if name in self.remaining[tier]:
                self.remaining[tier].remove(name)
            self.last_name = name


def scan_ceremony_styles() -> tuple[str, ...]:
    """Return top-level folders beneath ceremonies/ as available styles."""
    CEREMONIES_DIRECTORY.mkdir(parents=True, exist_ok=True)
    return tuple(
        path.name
        for path in sorted(
            (p for p in CEREMONIES_DIRECTORY.iterdir() if p.is_dir()),
            key=lambda p: p.name.lower(),
        )
    )


def resolve_style_name(preferred: str | None = None) -> str:
    styles = scan_ceremony_styles()
    preferred = str(preferred or "").strip()
    if preferred and preferred in styles:
        return preferred
    if DEFAULT_STYLE_NAME in styles:
        return DEFAULT_STYLE_NAME
    if styles:
        return styles[0]
    # Keep the application usable before content has been authored. The folder
    # will be created by the catalog/player paths as needed.
    return preferred or DEFAULT_STYLE_NAME


def style_directory(style_name: str) -> Path:
    return CEREMONIES_DIRECTORY / str(style_name)


def style_ambients_directory(style_name: str) -> Path:
    return style_directory(style_name) / STYLE_AMBIENTS_FOLDER

SETTINGS_PATH = SCRIPT_DIRECTORY / "settings.json"
STARTUP_LOG_PATH = SCRIPT_DIRECTORY / "startup.log"


def configure_startup_logging() -> logging.Logger:
    SCRIPT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("dream_instigator")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s.%(msecs)03d [%(levelname)s] "
            "[%(threadName)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler = logging.FileHandler(
            STARTUP_LOG_PATH,
            mode="w",
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


LOGGER = configure_startup_logging()


def log_stage(message: str) -> None:
    LOGGER.info(message)
    for handler in LOGGER.handlers:
        try:
            handler.flush()
        except Exception:
            pass


def install_exception_logging() -> None:
    def log_unhandled_exception(
        exception_type,
        exception_value,
        exception_traceback,
    ) -> None:
        if issubclass(exception_type, KeyboardInterrupt):
            sys.__excepthook__(
                exception_type,
                exception_value,
                exception_traceback,
            )
            return
        LOGGER.critical(
            "Unhandled exception",
            exc_info=(
                exception_type,
                exception_value,
                exception_traceback,
            ),
        )

    sys.excepthook = log_unhandled_exception

    if hasattr(threading, "excepthook"):
        def log_thread_exception(args) -> None:
            LOGGER.critical(
                "Unhandled thread exception in %s",
                getattr(args.thread, "name", "unknown"),
                exc_info=(
                    args.exc_type,
                    args.exc_value,
                    args.exc_traceback,
                ),
            )
        threading.excepthook = log_thread_exception


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
    damping_ratio: float = 1.07
    drive_strength: float = 0.95
    drive_smoothing_seconds: float = 0.9
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

    inhale_mean_seconds: float = 3.85
    hold_mean_seconds: float = 0.07
    exhale_mean_seconds: float = 0.95
    rest_mean_seconds: float = 0.8

    timing_variation: float = 0.08
    timing_memory: float = 0.82

    depth_variation: float = 0.12
    depth_memory: float = 0.75

    deep_breath_probability: float = 0.012
    deep_breath_scale: float = 1.45

    long_rest_probability: float = 0.008
    long_rest_scale: float = 2.2

    shallow_breath_probability: float = 0.02
    shallow_breath_scale: float = 0.72

    gain_range_db: float = 11.7
    spectral_depth: float = 1.0
    width_depth: float = 0.44

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

    body: float = 0.5
    slope_strength: float = 1.0
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
        if not 0.15 <= self.body <= 1.0:
            raise ValueError("body must be between 0.15 and 1")
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
        body = float(np.clip(body, 0.15, 1.0))
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
            np.clip(body, 0.15, 1.0),
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

        self.current_body = 0.15 + 0.85 * body_n
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
    metadata_known: bool = True


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
    """
    Fast filesystem catalogue backed by a persistent JSON metadata manifest.

    Startup only enumerates filenames and reads stat information. Unchanged
    files reuse their cached duration/classification. New or changed files are
    left as metadata-unknown and are classified when the background asset
    manager first decodes them.
    """

    MANIFEST_FILENAME = '.living_brown_noise_catalog.json'

    def __init__(
        self,
        root_directory: Path,
        layer_threshold_seconds: float,
    ) -> None:
        self.root_directory = root_directory
        self.layer_threshold_seconds = float(layer_threshold_seconds)
        self.manifest_path = self.root_directory / self.MANIFEST_FILENAME
        self.motifs: tuple[DreamMotif, ...] = ()
        self.errors: tuple[str, ...] = ()

    def _load_manifest(self) -> dict:
        try:
            raw = json.loads(self.manifest_path.read_text(encoding='utf-8'))
            if isinstance(raw, dict) and isinstance(raw.get('files'), dict):
                return raw
        except Exception:
            pass
        return {
            'version': 1,
            'layer_threshold_seconds': self.layer_threshold_seconds,
            'files': {},
        }

    def scan(self) -> tuple[DreamMotif, ...]:
        self.root_directory.mkdir(parents=True, exist_ok=True)
        manifest = self._load_manifest()
        cached_files = manifest.get('files', {})
        motifs: list[DreamMotif] = []
        errors: list[str] = []

        def classify_files(name: str, directory: Path, files: list[Path]) -> DreamMotif:
            ambient: list[DreamMotifAsset] = []
            layered: list[DreamMotifAsset] = []
            for path in files:
                try:
                    stat = path.stat()
                    relative = path.relative_to(self.root_directory).as_posix()
                    cached = cached_files.get(relative, {})
                    cache_valid = (
                        isinstance(cached, dict)
                        and int(cached.get('size_bytes', -1)) == stat.st_size
                        and int(cached.get('modified_ns', -1)) == stat.st_mtime_ns
                        and float(cached.get('duration_seconds', 0.0)) > 0.0
                    )
                    if cache_valid:
                        duration = float(cached['duration_seconds'])
                        is_layered = bool(
                            cached.get(
                                'is_layered_event',
                                duration <= self.layer_threshold_seconds,
                            )
                        )
                        known = True
                    else:
                        duration = 0.0
                        is_layered = False
                        known = False

                    asset = DreamMotifAsset(
                        path=path,
                        duration_seconds=duration,
                        is_layered_event=is_layered,
                        metadata_known=known,
                    )
                    if is_layered:
                        layered.append(asset)
                    else:
                        # Unknown files live here provisionally. The background
                        # manager verifies their true type before use.
                        ambient.append(asset)
                except Exception as exc:
                    try:
                        rel = path.relative_to(self.root_directory).as_posix()
                    except Exception:
                        rel = path.name
                    errors.append(f'{rel}: {exc}')

            return DreamMotif(
                name=name,
                directory=directory,
                ambient_assets=tuple(ambient),
                layered_assets=tuple(layered),
            )

        # New authored style layout: ambients/activity, ambients/event,
        # ambients/space. These folders are organizational only. Runtime
        # behavior continues to be determined by clip duration, so flatten all
        # three folders into one style-wide pool before classifying assets.
        group_dirs = [
            self.root_directory / group
            for group in STYLE_AMBIENT_GROUPS
            if (self.root_directory / group).is_dir()
        ]
        if group_dirs:
            files = sorted(
                {
                    path
                    for directory in group_dirs
                    for path in directory.rglob('*')
                    if path.is_file()
                    and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
                },
                key=lambda p: p.relative_to(self.root_directory).as_posix().lower(),
            )
            motif = classify_files(
                self.root_directory.parent.name or 'style',
                self.root_directory,
                files,
            )
            if motif.total_assets > 0:
                motifs.append(motif)
        else:
            # Backward-compatible legacy layout: each immediate child folder
            # remains its own motif world.
            directories = sorted(
                (p for p in self.root_directory.iterdir() if p.is_dir()),
                key=lambda p: p.name.lower(),
            )
            for directory in directories:
                files = sorted(
                    (
                        p for p in directory.iterdir()
                        if p.is_file()
                        and p.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
                    ),
                    key=lambda p: p.name.lower(),
                )
                motifs.append(classify_files(directory.name, directory, files))

        self.motifs = tuple(motifs)
        self.errors = tuple(errors)
        return self.motifs

    def find(self, name: str) -> DreamMotif | None:
        for motif in self.motifs:
            if motif.name == name:
                return motif
        return None


@dataclass(frozen=True, slots=True)
class PreparedAudioAsset:
    path: Path
    mono: np.ndarray
    duration_seconds: float
    is_layered_event: bool
    byte_size: int


class AudioAssetManager:
    """
    Single-worker background decoder with a bounded LRU cache.

    The real-time engine may request assets and retrieve already-ready arrays,
    but it never opens files, decodes, resamples, waits, or performs large
    allocations. Missing assets simply remain unavailable until prepared.
    """

    PRIORITY_CRITICAL = 0
    PRIORITY_HIGH = 10
    PRIORITY_NORMAL = 20
    PRIORITY_LOW = 30

    def __init__(
        self,
        root_directory: Path,
        sample_rate: int,
        layer_threshold_seconds: float,
        maximum_cache_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        self.root_directory = root_directory
        self.sample_rate = int(sample_rate)
        self.layer_threshold_seconds = float(layer_threshold_seconds)
        self.maximum_cache_bytes = int(maximum_cache_bytes)
        self.manifest_path = (
            self.root_directory / DreamMotifCatalog.MANIFEST_FILENAME
        )

        self._lock = threading.Lock()
        self._cache: OrderedDict[str, PreparedAudioAsset] = OrderedDict()
        self._cache_bytes = 0
        self._pending: set[str] = set()
        self._failed: dict[str, str] = {}
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._sequence = 0
        self._stop_event = threading.Event()

        self._worker = threading.Thread(
            target=self._worker_loop,
            name='AudioAssetDecoder',
            daemon=True,
        )
        self._worker.start()

    @staticmethod
    def _key(path: Path) -> str:
        return str(path.resolve())

    def request(self, asset: DreamMotifAsset, priority: int) -> None:
        key = self._key(asset.path)
        with self._lock:
            if key in self._cache or key in self._pending or key in self._failed:
                return
            self._pending.add(key)
            self._sequence += 1
            sequence = self._sequence
        self._queue.put((int(priority), sequence, asset.path))

    def get_if_ready(
        self,
        asset: DreamMotifAsset | None,
    ) -> PreparedAudioAsset | None:
        if asset is None:
            return None
        key = self._key(asset.path)
        with self._lock:
            prepared = self._cache.get(key)
            if prepared is not None:
                self._cache.move_to_end(key)
            return prepared

    def error_for(self, asset: DreamMotifAsset | None) -> str:
        if asset is None:
            return ''
        with self._lock:
            return self._failed.get(self._key(asset.path), '')

    def status(self) -> tuple[int, int, int, int]:
        with self._lock:
            return (
                len(self._cache),
                len(self._pending),
                len(self._failed),
                self._cache_bytes,
            )

    def close(self) -> None:
        self._stop_event.set()
        self._queue.put((10_000, 0, None))
        self._worker.join(timeout=2.0)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                _, _, path = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if path is None:
                break

            key = self._key(path)
            try:
                prepared = self._decode(path)
                with self._lock:
                    self._pending.discard(key)
                    self._failed.pop(key, None)
                    existing = self._cache.pop(key, None)
                    if existing is not None:
                        self._cache_bytes -= existing.byte_size
                    self._cache[key] = prepared
                    self._cache_bytes += prepared.byte_size
                    self._evict_locked()
                self._update_manifest(prepared)
            except Exception as exc:
                LOGGER.exception('Background audio decode failed: %s', path)
                with self._lock:
                    self._pending.discard(key)
                    self._failed[key] = str(exc)
            finally:
                self._queue.task_done()

    def _decode(self, path: Path) -> PreparedAudioAsset:
        data, input_rate = AudioFileDecoder._decode_audio_file(path)

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
            raise ValueError('Audio file contains no usable audio')

        normalized, _, _, _, _ = (
            AudioFileDecoder._normalize_field_recording(
                data,
                self.sample_rate,
            )
        )
        mono = np.ascontiguousarray(
            np.mean(normalized.astype(np.float64), axis=1),
            dtype=np.float32,
        )
        duration = len(mono) / self.sample_rate
        return PreparedAudioAsset(
            path=path,
            mono=mono,
            duration_seconds=duration,
            is_layered_event=(
                duration <= self.layer_threshold_seconds
            ),
            byte_size=int(mono.nbytes),
        )

    def _evict_locked(self) -> None:
        while (
            self._cache_bytes > self.maximum_cache_bytes
            and len(self._cache) > 1
        ):
            _, evicted = self._cache.popitem(last=False)
            self._cache_bytes -= evicted.byte_size

    def _update_manifest(self, prepared: PreparedAudioAsset) -> None:
        try:
            stat = prepared.path.stat()
            relative = prepared.path.relative_to(
                self.root_directory
            ).as_posix()
            try:
                manifest = json.loads(
                    self.manifest_path.read_text(encoding='utf-8')
                )
                if not isinstance(manifest, dict):
                    manifest = {}
            except Exception:
                manifest = {}

            files = manifest.get('files')
            if not isinstance(files, dict):
                files = {}

            files[relative] = {
                'size_bytes': stat.st_size,
                'modified_ns': stat.st_mtime_ns,
                'duration_seconds': prepared.duration_seconds,
                'is_layered_event': prepared.is_layered_event,
                'sample_rate': self.sample_rate,
            }
            manifest = {
                'version': 1,
                'layer_threshold_seconds': self.layer_threshold_seconds,
                'files': files,
            }
            temporary = self.manifest_path.with_suffix('.tmp')
            temporary.write_text(
                json.dumps(manifest, indent=2, sort_keys=True),
                encoding='utf-8',
            )
            temporary.replace(self.manifest_path)
        except Exception:
            LOGGER.exception(
                'Could not update audio metadata manifest for %s',
                prepared.path,
            )


# =============================================================================
# Shared background audio decoding helpers
# =============================================================================

class AudioFileDecoder:
    """Decode, resample, and normalize motif assets off the audio thread."""

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
    dream_motifs_enabled: bool = True


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
        dream_motifs_enabled: bool | None = None,
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
                dream_motifs_enabled=(
                    current.dream_motifs_enabled
                    if dream_motifs_enabled is None
                    else bool(dream_motifs_enabled)
                ),
            )



@dataclass(frozen=True, slots=True)
class DualBrownMotionSpec:
    """Live controls for the two brown-noise bodies moving on a sphere."""

    layer_enabled: bool = True
    layer_amount: float = 1.5
    enabled: bool = True
    sphere_radius: float = 3.15
    center_distance: float = 0.1
    evolution_rate: float = 0.42

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
    distance: float = 0.75
    horizontal: float = 0.0
    vertical: float = -0.25
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

    enabled: bool = True
    phase_min_minutes: float = 3.0
    phase_max_minutes: float = 9.25

    # Percentage preference for resting states. Zero leaves the activity
    # drive linear; 100 strongly favors rest while preserving rare excursions.
    resting_tendency_percent: float = 38.0

    brown_body_min: float = 0.15
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
    breath_gain_min_db: float = 0.9
    breath_gain_max_db: float = 5.5
    breath_spectral_min: float = 0.05
    breath_spectral_max: float = 0.35
    breath_width_min: float = 0.03
    breath_width_max: float = 0.18

    heartbeat_distance_min: float = 0.75
    heartbeat_distance_max: float = 4.0
    heartbeat_level_min_db: float = 0.0
    heartbeat_level_max_db: float = 18.0

    brown_3d_amount_min: float = 0.16
    brown_3d_amount_max: float = 0.66
    brown_radius_min: float = 0.95
    brown_radius_max: float = 5.0
    brown_center_distance_min: float = 0.15
    brown_center_distance_max: float = 3.65
    brown_evolution_min: float = 0.07
    brown_evolution_max: float = 0.65

    def validated(self) -> "MetabolismSpec":
        if not 0.0 <= self.resting_tendency_percent <= 100.0:
            raise ValueError(
                "resting_tendency_percent must be between 0 and 100"
            )

        pairs = (
            ("phase_min_minutes", "phase_max_minutes", 0.25, 240.0),
            ("brown_body_min", "brown_body_max", 0.15, 1.0),
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



class HeartbeatProminenceLimiter:
    """
    Limits the duration of a subjectively prominent heartbeat.

    Prominence can come from either high gain or close 3D placement. A loud or
    close heartbeat is allowed to emerge naturally for a short period, then
    both level and distance are moved toward a safe background condition.

    The limiter remains in recovery until the raw metabolism request has left
    the prominent region for a sustained period. This prevents one long
    metabolism phase from repeatedly re-triggering the same loud heartbeat.
    """

    STATE_IDLE = "background"
    STATE_PROMINENT = "prominent"
    STATE_FADE_OUT = "fading out"
    STATE_COOLDOWN = "background recovery"

    def __init__(
        self,
        trigger_level_db: float = 8.0,
        trigger_distance_meters: float = 1.50,
        prominent_seconds: float = 12.0,
        fade_out_seconds: float = 14.0,
        safe_level_db: float = 4.0,
        safe_distance_meters: float = 3.20,
        rearm_seconds: float = 20.0,
    ) -> None:
        self.trigger_level_db = float(trigger_level_db)
        self.trigger_distance_meters = float(
            trigger_distance_meters
        )
        self.prominent_seconds = max(
            0.1,
            float(prominent_seconds),
        )
        self.fade_out_seconds = max(
            0.1,
            float(fade_out_seconds),
        )
        self.safe_level_db = float(safe_level_db)
        self.safe_distance_meters = float(
            safe_distance_meters
        )
        self.rearm_seconds = max(
            0.1,
            float(rearm_seconds),
        )

        self.state = self.STATE_IDLE
        self.state_elapsed = 0.0
        self.safe_request_elapsed = 0.0

        self.current_effective_level_db = (
            self.safe_level_db
        )
        self.current_effective_distance = (
            self.safe_distance_meters
        )

    @staticmethod
    def _smoothstep5(value: float) -> float:
        value = float(np.clip(value, 0.0, 1.0))
        return value ** 3 * (
            value * (value * 6.0 - 15.0) + 10.0
        )

    def _enter(self, state: str) -> None:
        self.state = state
        self.state_elapsed = 0.0

    def _is_prominent_request(
        self,
        requested_level_db: float,
        requested_distance: float,
    ) -> bool:
        return (
            requested_level_db >= self.trigger_level_db
            or requested_distance <= self.trigger_distance_meters
        )

    def advance(
        self,
        requested_level_db: float,
        requested_distance: float,
        elapsed_seconds: float,
    ) -> tuple[float, float]:
        requested_level_db = float(requested_level_db)
        requested_distance = float(requested_distance)
        elapsed_seconds = max(0.0, float(elapsed_seconds))

        prominent_request = self._is_prominent_request(
            requested_level_db,
            requested_distance,
        )

        if self.state == self.STATE_IDLE:
            if prominent_request:
                self._enter(self.STATE_PROMINENT)

            effective_level_db = requested_level_db
            effective_distance = requested_distance

        elif self.state == self.STATE_PROMINENT:
            self.state_elapsed += elapsed_seconds

            effective_level_db = requested_level_db
            effective_distance = requested_distance

            if not prominent_request:
                self._enter(self.STATE_COOLDOWN)
            elif self.state_elapsed >= self.prominent_seconds:
                self._enter(self.STATE_FADE_OUT)

        elif self.state == self.STATE_FADE_OUT:
            self.state_elapsed += elapsed_seconds

            progress = self._smoothstep5(
                self.state_elapsed / self.fade_out_seconds
            )

            target_level_db = min(
                requested_level_db,
                self.safe_level_db,
            )
            target_distance = max(
                requested_distance,
                self.safe_distance_meters,
            )

            effective_level_db = (
                requested_level_db
                + (target_level_db - requested_level_db)
                * progress
            )
            effective_distance = (
                requested_distance
                + (target_distance - requested_distance)
                * progress
            )

            if self.state_elapsed >= self.fade_out_seconds:
                self._enter(self.STATE_COOLDOWN)
                self.safe_request_elapsed = 0.0
                effective_level_db = target_level_db
                effective_distance = target_distance

        else:
            effective_level_db = min(
                requested_level_db,
                self.safe_level_db,
            )
            effective_distance = max(
                requested_distance,
                self.safe_distance_meters,
            )

            if prominent_request:
                self.safe_request_elapsed = 0.0
            else:
                self.safe_request_elapsed += elapsed_seconds
                if self.safe_request_elapsed >= self.rearm_seconds:
                    self.safe_request_elapsed = 0.0
                    self._enter(self.STATE_IDLE)

        self.current_effective_level_db = float(
            effective_level_db
        )
        self.current_effective_distance = float(
            effective_distance
        )

        return (
            self.current_effective_level_db,
            self.current_effective_distance,
        )



@dataclass(frozen=True, slots=True)
class DreamMotifSpatialSpec:
    enabled: bool = True

    # One ambient source: per-clip spatial entrance, nearest point, and exit.
    # Retired conductor fields below remain only to load older settings safely.
    far_distance_calibrated: float = 14.2
    closest_ambient_distance: float = 3.8
    ambient_approach_seconds: float = 20.0
    motif_crossfade_seconds: float = 30.0
    ambient_clip_fade_seconds: float = 4.0
    scene_duration_scale: float = 1.25

    # Live cadence and lateral motion; zero gaps allow back-to-back passages.
    # ambient_rest/radial_response/event gates are retired compatibility fields.
    startup_rest_seconds: float = 0.0
    ambient_rest_seconds: float = 0.0
    ambient_gap_min_seconds: float = 0.0
    ambient_gap_max_seconds: float = 2.0
    wander_min_seconds: float = 8.0
    wander_max_seconds: float = 16.0
    radial_response_seconds: float = 4.0
    event_grace_seconds: float = 0.0
    event_quiet_threshold: float = 0.0
    event_start_probability: float = 1.0
    busy_presence_floor: float = 0.65

    # Legacy fields retained for settings compatibility and advanced tuning.
    far_distance: float = 15.0
    near_distance: float = 2.3
    fade_in_seconds: float = 5.0
    fade_out_seconds: float = 5.0


    distant_gain_db: float = -42.0
    dominant_gain_db: float = -22.0

    # Baseline spacing between featured-effect opportunities. Actual audible
    # events remain sparser because scene, quiet-window and probability gates
    # still apply. Rejected opportunities retry sooner rather than restarting
    # an entire long interval.
    event_interval_min_seconds: float = 600.0
    event_interval_max_seconds: float = 1200.0
    event_gain_db: float = -16.0
    event_travel_seconds: float = 14.0

    # Presence sets nearest approach, Motion sets lateral excursion.
    # Other old guidance fields are retained for settings compatibility only.
    activity: float = 0.68
    presence: float = 0.59
    motion: float = 0.73
    intimacy: float = 0.91
    drama: float = 0.70
    coherence: float = 0.7
    novelty: float = 0.8

    # Testing mode removes conductor waiting while preserving real-time
    # sample playback, fades, motion, envelopes, and spatial gestures.
    testing: bool = False

    # Exclude clips <=10 seconds while retaining the longer ambient passages.
    featured_events_enabled: bool = True

    # Constant calibrated source level. Distance performs the entrance/exit.
    # busy_presence_floor limits metabolism influence on the nearest point.
    motif_calibrated_gain_db: float = -16.5
    event_calibrated_gain_db: float = -16.0

    def validated(self) -> "DreamMotifSpatialSpec":
        for name in ("startup_rest_seconds", "ambient_rest_seconds", "event_grace_seconds"):
            if not 0.0 <= getattr(self, name) <= 3600.0:
                raise ValueError(f"invalid {name}")
        for low, high, minimum in (
            ("ambient_gap_min_seconds", "ambient_gap_max_seconds", 0.0),
            ("wander_min_seconds", "wander_max_seconds", 1.0),
        ):
            if not minimum <= getattr(self, low) <= getattr(self, high) <= 3600.0:
                raise ValueError(f"invalid {low}/{high}")
        if not 0.1 <= self.radial_response_seconds <= 300.0:
            raise ValueError("invalid radial response duration")
        for name in ("event_quiet_threshold", "event_start_probability", "busy_presence_floor"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"invalid {name}")
        if not 0.1 <= self.far_distance_calibrated <= 100.0:
            raise ValueError("invalid far distance")
        if not 0.0 <= self.closest_ambient_distance <= 30.0:
            raise ValueError("invalid closest ambient distance")
        if self.closest_ambient_distance >= self.far_distance_calibrated:
            raise ValueError(
                "closest ambient distance must be less than far distance"
            )
        if not 5.0 <= self.ambient_approach_seconds <= 1800.0:
            raise ValueError("invalid ambient approach duration")
        if not 5.0 <= self.motif_crossfade_seconds <= 1800.0:
            raise ValueError("invalid motif crossfade duration")
        if not 0.5 <= self.ambient_clip_fade_seconds <= 60.0:
            raise ValueError("invalid ambient clip fade duration")
        if not 0.10 <= self.scene_duration_scale <= 2.0:
            raise ValueError("invalid conductor scene-duration scale")

        if not 1.0 <= self.far_distance <= 100.0:
            raise ValueError("invalid motif far distance")
        if not 0.15 <= self.near_distance <= 20.0:
            raise ValueError("invalid motif near distance")
        if self.near_distance >= self.far_distance:
            raise ValueError("motif near distance must be less than far distance")

        if not 1.0 <= self.fade_in_seconds <= 3600.0:
            raise ValueError("invalid motif fade-in time")
        if not 1.0 <= self.fade_out_seconds <= 3600.0:
            raise ValueError("invalid motif fade-out time")


        if not -80.0 <= self.distant_gain_db <= 0.0:
            raise ValueError("invalid distant motif gain")
        if not -80.0 <= self.dominant_gain_db <= 6.0:
            raise ValueError("invalid dominant motif gain")
        if self.distant_gain_db > self.dominant_gain_db:
            raise ValueError("distant motif cannot exceed dominant motif")

        if not (
            1.0
            <= self.event_interval_min_seconds
            <= self.event_interval_max_seconds
            <= 86_400.0
        ):
            raise ValueError("invalid motif event interval")
        if not -80.0 <= self.event_gain_db <= 12.0:
            raise ValueError("invalid motif event gain")
        if not 1.0 <= self.event_travel_seconds <= 300.0:
            raise ValueError("invalid motif event travel time")
        for field_name in (
            "activity", "presence", "motion", "intimacy",
            "drama", "coherence", "novelty",
        ):
            if not 0.0 <= getattr(self, field_name) <= 1.0:
                raise ValueError(f"invalid conductor {field_name}")
        if not isinstance(self.testing, bool):
            raise ValueError("invalid conductor testing flag")
        if not isinstance(self.featured_events_enabled, bool):
            raise ValueError("invalid featured-events flag")
        if not -80.0 <= self.motif_calibrated_gain_db <= 6.0:
            raise ValueError("invalid calibrated motif gain")
        if not -80.0 <= self.event_calibrated_gain_db <= 12.0:
            raise ValueError("invalid calibrated event gain")

        return self


class DreamMotifSpatialState:
    def __init__(self, spec: DreamMotifSpatialSpec) -> None:
        self._lock = threading.Lock()
        self._spec = spec.validated()

    def get(self) -> DreamMotifSpatialSpec:
        with self._lock:
            return self._spec

    def set(self, spec: DreamMotifSpatialSpec) -> None:
        with self._lock:
            self._spec = spec.validated()

    def update(self, **changes) -> None:
        with self._lock:
            values = asdict(self._spec)
            values.update(changes)

            if values["near_distance"] >= values["far_distance"]:
                if "near_distance" in changes:
                    values["far_distance"] = (
                        float(values["near_distance"]) + 0.1
                    )
                else:
                    values["near_distance"] = max(
                        0.15,
                        float(values["far_distance"]) - 0.1,
                    )


            if (
                values["event_interval_min_seconds"]
                > values["event_interval_max_seconds"]
            ):
                if "event_interval_min_seconds" in changes:
                    values["event_interval_max_seconds"] = values[
                        "event_interval_min_seconds"
                    ]
                else:
                    values["event_interval_min_seconds"] = values[
                        "event_interval_max_seconds"
                    ]

            for low, high in (("ambient_gap_min_seconds", "ambient_gap_max_seconds"),
                              ("wander_min_seconds", "wander_max_seconds")):
                if values[low] > values[high]:
                    if low in changes:
                        values[high] = values[low]
                    else:
                        values[low] = values[high]
            if values["closest_ambient_distance"] >= values["far_distance_calibrated"]:
                if "closest_ambient_distance" in changes:
                    values["far_distance_calibrated"] = values["closest_ambient_distance"] + 0.1
                else:
                    values["closest_ambient_distance"] = max(0.0, values["far_distance_calibrated"] - 0.1)

            self._spec = DreamMotifSpatialSpec(
                **values
            ).validated()


class DreamMotif3DEngine:
    """One non-looping ambient at a time, with its own spatial life cycle.

    Wait -> approach -> linger -> recede. The source gain stays calibrated;
    distance attenuation does the expressive fading. Only 10 ms endpoint
    ramps protect against clicks. Curation draws are consumed at playback,
    never by a speculative decode. No dominant/recessive worlds or scenes.
    """

    def __init__(self, sample_rate: int, renderer: SteamAudioRenderer,
                 root_directory: Path, state: DreamMotifSpatialState,
                 seed: int = 7712301) -> None:
        self.sample_rate = int(sample_rate)
        self.renderer = renderer
        self.root_directory = Path(root_directory)
        self.state = state
        self.rng = np.random.default_rng(seed)
        self.source = renderer.create_source(position=STEAM_DEFAULT_SOURCE_POSITION,
            spatial_blend=1.0, distance_attenuation_enabled=True)
        self._manual_lock = threading.Lock()
        self._command_lock = threading.Lock()
        self._trigger_requested = False
        self.manual_enabled = False
        self.manual_source_kind = 'ambient'
        self.manual_position = np.array([0.0, 0.0, -2.0], dtype=np.float64)
        self.manual_gain_db = -18.0
        self.manual_solo = True
        self.manual_test_motif_name = ''
        self.manual_test_asset = None
        self.manual_test_audio = None
        self.manual_test_read_position = 0
        self._manual_previous = False
        self.offline = False
        self.start_blocked = False  # Ceremony due: finish the current clip, then yield.
        self.cancel_event = None
        self.render_elapsed_seconds = 0.0
        self._event_journal = deque(maxlen=4096)
        self.current_status = 'Waiting for ambient audio'
        self.phase = 'waiting'
        self.active_asset = None
        self.active_audio = None
        self.read_position = 0
        self.current_distance = self.state.get().far_distance_calibrated
        self.current_position = np.array([0.0, 0.0, -self.current_distance])
        self._startup = True
        self._wait_limit = self.state.get().startup_rest_seconds
        self.wait_seconds = self._wait_limit
        self._load_catalog()

    def _load_catalog(self):
        self.catalog = DreamMotifCatalog(root_directory=self.root_directory,
            layer_threshold_seconds=DREAM_MOTIF_LAYER_THRESHOLD_SECONDS)
        motifs = self.catalog.scan()
        curation = AudioCuration()
        self.assets = {}
        self.asset_motifs = {}
        self.short_keys = set()
        for motif in motifs:
            for asset in (*motif.ambient_assets, *motif.layered_assets):
                key = str(asset.path)
                if curation.tier_for_path(asset.path) == 'stash':
                    continue
                self.assets[key] = asset
                self.asset_motifs[key] = motif.name
                if asset.metadata_known and asset.is_layered_event:
                    self.short_keys.add(key)
        self.ambient_priority_bag = CurationShuffleBag({
            key: curation.tier_for_path(asset.path) for key, asset in self.assets.items()
        }, self.rng)
        self.failed_keys = set()
        self.pending_key = None
        self.pending_prepared = None
        self.asset_manager = AudioAssetManager(root_directory=self.root_directory,
            sample_rate=self.sample_rate,
            layer_threshold_seconds=DREAM_MOTIF_LAYER_THRESHOLD_SECONDS)
        self.manual_test_motif_name = motifs[0].name if motifs else ''
        self._journal('AMBIENT_CATALOG', f'files={len(self.assets)}; ' + '; '.join(
            f'{tier}={len(self.ambient_priority_bag.remaining[tier])}'
            for tier in self.ambient_priority_bag.TIERS))

    @staticmethod
    def _smoothstep5(value):
        value = np.clip(value, 0.0, 1.0)
        return value ** 3 * (value * (value * 6.0 - 15.0) + 10.0)

    @staticmethod
    def _format_log_time(seconds):
        millis = max(0, int(round(seconds * 1000)))
        hours, millis = divmod(millis, 3600000)
        minutes, millis = divmod(millis, 60000)
        secs, millis = divmod(millis, 1000)
        return f'{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}'

    def _journal(self, category, message):
        self._event_journal.append((self.render_elapsed_seconds, category, message))

    def drain_event_journal(self):
        result = []
        while self._event_journal:
            result.append(self._event_journal.popleft())
        return result

    def _new_gap(self, spec):
        return float(self.rng.uniform(spec.ambient_gap_min_seconds, spec.ambient_gap_max_seconds))

    def request_trigger(self):
        with self._command_lock:
            self._trigger_requested = True

    def _peek_next(self, spec):
        """One shared priority pool for beds and effects, independent of vocals."""
        if spec.featured_events_enabled != getattr(self, '_short_enabled', None):
            self._short_enabled = spec.featured_events_enabled
            self.pending_key = self.pending_prepared = None
        eligible = {key for key in self.assets if key not in self.failed_keys
                    and (spec.featured_events_enabled or key not in self.short_keys)}
        if self.pending_key not in eligible:
            self.pending_key = self.pending_prepared = None
        if not eligible:
            return None, None
        bag = self.ambient_priority_bag
        if self.pending_key is None:
            if not any(key in eligible for names in bag.remaining.values() for key in names):
                bag.refill()
            for tier in bag.TIERS:
                names = [key for key in bag.remaining[tier] if key in eligible]
                if names:
                    self.pending_key = next((key for key in names if key != bag.last_name), names[0])
                    break
        asset = self.assets[self.pending_key]
        if self.pending_prepared is None:
            self.asset_manager.request(asset, AudioAssetManager.PRIORITY_HIGH)
            prepared = self.asset_manager.get_if_ready(asset)
            error = self.asset_manager.error_for(asset)
            if error:
                self.failed_keys.add(self.pending_key)
                self._journal('AMBIENT_LOAD_FAILED', f'{asset.path}: {error}')
                self.pending_key = None
                return None, None
            if prepared is None:
                return asset, None
            if len(prepared.mono) == 0:
                self.failed_keys.add(self.pending_key)
                self.pending_key = None
                return None, None
            if prepared.is_layered_event:
                self.short_keys.add(self.pending_key)
                if not spec.featured_events_enabled:
                    self.pending_key = None
                    return None, None
            self.pending_prepared = prepared  # Pin the one pending buffer in memory.
        return asset, self.pending_prepared

    @staticmethod
    def passage_times(duration, spec):
        # Preserve relative entrance/exit lengths when a short clip cannot fit
        # the requested times. Reserve 10% of the recording for its closest point.
        approach = max(0.01, float(spec.ambient_approach_seconds))
        recede = max(0.01, float(spec.fade_out_seconds))
        scale = min(1.0, duration * 0.90 / (approach + recede))
        return approach * scale, recede * scale

    def _start(self, asset, prepared, spec, quiet):
        self.active_asset = asset
        self.active_audio = prepared.mono
        self.read_position = 0
        self.duration_seconds = len(self.active_audio) / self.sample_rate
        self.approach_seconds, self.recede_seconds = self.passage_times(self.duration_seconds, spec)
        # Presence and metabolism modulate the nearest point through distance,
        # rather than an invisible extra gain envelope or trigger gate.
        nearness = spec.presence * (spec.busy_presence_floor + (1.0 - spec.busy_presence_floor) * quiet)
        self.passage_far = max(0.25, spec.far_distance_calibrated)
        near = max(0.25, spec.closest_ambient_distance)
        self.passage_near = near + (self.passage_far - near) * (1.0 - nearness)
        self._azimuth = float(self.rng.uniform(-0.65, 0.65))
        self._elevation = float(self.rng.uniform(-0.12, 0.2))
        self._move_elapsed = 0.0
        self._choose_motion(spec)
        self.phase = 'approach'
        self.current_distance = self.passage_far
        key = str(asset.path)
        bag = self.ambient_priority_bag
        bag.take_specific(key)
        self.pending_key = self.pending_prepared = None
        self._journal('AMBIENT_CURATION_DRAW',
            f'tier={bag.tiers[key]}; cycle={bag.cycle}; asset={asset.path}')
        self._journal('AMBIENT_PASSAGE_START',
            f'asset={asset.path}; duration={self.duration_seconds:.2f}s; '
            f'approach={self.approach_seconds:.2f}s; recede={self.recede_seconds:.2f}s; '
            f'far={self.passage_far:.2f}m; nearest={self.passage_near:.2f}m')

    def _choose_motion(self, spec):
        self._move_start = np.array([self._azimuth, self._elevation])
        span = math.radians(80.0) * spec.motion
        target = float(self.rng.uniform(-span, span)) if spec.motion > 0 else self._azimuth
        self._move_target = np.array([target,
            float(self.rng.uniform(-0.15, 0.25)) * spec.motion if spec.motion > 0 else self._elevation])
        self._move_elapsed = 0.0
        self._move_duration = float(self.rng.uniform(spec.wander_min_seconds, spec.wander_max_seconds))
        self._motion_settings = (spec.motion, spec.wander_min_seconds, spec.wander_max_seconds)

    def _position(self, elapsed, dt, spec):
        if (spec.motion, spec.wander_min_seconds, spec.wander_max_seconds) != self._motion_settings:
            self._choose_motion(spec)
        self._move_elapsed += dt
        u = float(self._smoothstep5(self._move_elapsed / self._move_duration))
        angles = self._move_start + (self._move_target - self._move_start) * u
        self._azimuth, self._elevation = map(float, angles)
        if self._move_elapsed >= self._move_duration:
            self._choose_motion(spec)
        # A slow, nonlinear emergence, reversed at the end. Interpolate distance
        # directly: attenuation rises naturally as the source comes nearer.
        approach = float(self._smoothstep5(elapsed / self.approach_seconds)) ** 2
        remaining = max(0.0, self.duration_seconds - elapsed)
        recede = float(self._smoothstep5(remaining / self.recede_seconds)) ** 2
        proximity = min(approach, recede)
        anchor_distance = self.passage_far + (self.passage_near - self.passage_far) * proximity
        # A finite Far distance can still be audible. Continue the exit beyond
        # that anchor toward practical silence at EOF. No gain fade is added.
        # The clamp keeps positions finite; even inverse-distance attenuation
        # then supplies a further 60 dB of reduction at the end.
        exit_presence = float(self._smoothstep5(remaining / self.recede_seconds))
        self.current_distance = anchor_distance / max(exit_presence, 0.001)
        self.phase = ('approach' if elapsed < self.approach_seconds else
                      'recede' if remaining <= self.recede_seconds else 'linger')
        # Spherical coordinates prevent a lateral movement cutting through the head.
        self.current_position = self.current_distance * np.array([
            math.sin(self._azimuth) * math.cos(self._elevation),
            math.sin(self._elevation),
            -math.cos(self._azimuth) * math.cos(self._elevation)])
        self.source.set_position_vector(Vector3(*map(float, self.current_position)))

    def _finish(self, spec, interrupted=False):
        if self.active_asset is not None:
            self._journal('AMBIENT_INTERRUPTED' if interrupted else 'AMBIENT_PASSAGE_END',
                          f'asset={self.active_asset.path}; played={self.read_position / self.sample_rate:.2f}s; '
                          f'duration={self.duration_seconds:.2f}s; end_distance={self.current_distance:.2f}m')
        self.active_asset = self.active_audio = None
        self.read_position = 0
        self.phase = 'waiting'
        self.wait_seconds = self._new_gap(spec)
        self._wait_limit = spec.ambient_gap_max_seconds

    def set_manual_spatial(self, *, enabled=None, source_kind=None, x=None,
                           y=None, z=None, gain_db=None, solo=None, motif_name=None):
        with self._manual_lock:
            if enabled is not None:
                self.manual_enabled = bool(enabled)
            if source_kind is not None:
                self.manual_source_kind = source_kind
            if any(v is not None for v in (x, y, z)):
                self.manual_position = np.array([
                    self.manual_position[0] if x is None else x,
                    self.manual_position[1] if y is None else y,
                    self.manual_position[2] if z is None else z], dtype=np.float64)
            if gain_db is not None:
                self.manual_gain_db = float(gain_db)
            if solo is not None:
                self.manual_solo = bool(solo)
            if motif_name is not None:
                self.manual_test_motif_name = motif_name

    def manual_snapshot(self):
        with self._manual_lock:
            return (self.manual_enabled, self.manual_source_kind, self.manual_position.copy(),
                    self.manual_gain_db, self.manual_solo, self.manual_test_motif_name)

    def _manual_audio(self, frame_count, snapshot):
        _, kind, position, gain, _, motif = snapshot
        identity = (kind, motif)
        if identity != getattr(self, '_manual_identity', None):
            self._manual_identity = identity
            self.manual_test_asset = self.manual_test_audio = None
            self.manual_test_read_position = 0
        if self.manual_test_asset is None:
            candidates = [asset for key, asset in self.assets.items()
                if key not in self.failed_keys and self.asset_motifs[key] == motif
                and ((key in self.short_keys) == (kind == 'short event'))]
            if candidates:
                self.manual_test_asset = candidates[int(self.rng.integers(len(candidates)))]
                self.asset_manager.request(self.manual_test_asset, AudioAssetManager.PRIORITY_HIGH)
        asset = self.manual_test_asset
        if asset is not None and self.manual_test_audio is None:
            prepared = self.asset_manager.get_if_ready(asset)
            if prepared is not None:
                self.manual_test_audio = prepared.mono
                if prepared.is_layered_event:
                    self.short_keys.add(str(asset.path))
                if prepared.is_layered_event != (kind == 'short event'):
                    self.manual_test_asset = self.manual_test_audio = None
            elif self.asset_manager.error_for(asset):
                self.failed_keys.add(str(asset.path))
                self.manual_test_asset = None
        if self.manual_test_audio is None or len(self.manual_test_audio) == 0:
            return np.zeros((frame_count, 2), dtype=np.float32)
        indices = (np.arange(frame_count) + self.manual_test_read_position) % len(self.manual_test_audio)
        self.manual_test_read_position += frame_count
        self.current_position = position
        self.current_distance = float(np.linalg.norm(position))
        self.source.set_position_vector(Vector3(*map(float, position)))
        self.phase = 'manual'
        return self.source.process_mono(self.manual_test_audio[indices] * 10.0 ** (gain / 20.0))

    def snapshot(self):
        return {'phase': self.phase, 'asset': str(self.active_asset.path) if self.active_asset else None,
                'elapsed_s': round(self.read_position / self.sample_rate, 3),
                'wait_s': round(self.wait_seconds, 3), 'pending': self.pending_key,
                'distance_m': round(self.current_distance, 3),
                'position_m': list(map(float, self.current_position)),
                'cycle': self.ambient_priority_bag.cycle}

    @property
    def playback_label(self):
        if self.phase == 'manual':
            return f'Manual: {self.manual_test_asset.path.name if self.manual_test_asset else "loading"}'
        return (f'{self.active_asset.path.name} — {self.phase}' if self.active_asset else
                f'{self.phase.capitalize()} — {self.wait_seconds:.1f} s')

    def _update_status(self):
        self.current_status = (f'{self.playback_label}\n'
            f'Distance {self.current_distance:.2f} m; elapsed {self.read_position / self.sample_rate:.1f} s; '
            f'next gap {self.wait_seconds:.1f} s\n'
            f'One ambient source; priority cycle {self.ambient_priority_bag.cycle}; '
            f'{len(self.failed_keys)} unavailable file(s)')

    def generate(self, frame_count, enabled, metabolism_activity=0.0):
        # Match native spatial cadence in both realtime and large export chunks.
        output = np.zeros((frame_count, 2), dtype=np.float32)
        for offset in range(0, frame_count, self.renderer.frame_size):
            count = min(self.renderer.frame_size, frame_count - offset)
            output[offset:offset + count] = self._render_piece(count, enabled, metabolism_activity)
        return output

    def _render_piece(self, frame_count, enabled, metabolism_activity=0.0):
        spec = self.state.get()
        self.render_elapsed_seconds += frame_count / self.sample_rate
        silence = np.zeros((frame_count, 2), dtype=np.float32)
        if not enabled or not spec.enabled:
            if self.active_asset is not None:
                self._finish(spec, interrupted=True)
            self.phase = 'suppressed'
            self._update_status()
            return silence
        manual = self.manual_snapshot()
        if manual[0]:
            if self.active_asset is not None:
                self._finish(spec, interrupted=True)
            self._manual_previous = True
            audio = self._manual_audio(frame_count, manual)
            self._update_status()
            return audio
        if self._manual_previous:
            self._manual_previous = False
            self.manual_test_asset = self.manual_test_audio = None
        asset, prepared = self._peek_next(spec)  # Read ahead while the current clip plays.
        if self.active_audio is None:
            if self.start_blocked:
                self.phase = 'waiting for ceremony'
                self._update_status()
                return silence
            with self._command_lock:
                force = self._trigger_requested
                self._trigger_requested = False
            limit = spec.startup_rest_seconds if self._startup else spec.ambient_gap_max_seconds
            self.wait_seconds = min(self.wait_seconds, limit)
            if spec.testing or force:
                self.wait_seconds = 0.0
            if self.wait_seconds > 0:
                self.wait_seconds = max(0.0, self.wait_seconds - frame_count / self.sample_rate)
                self.phase = 'waiting'
                self._update_status()
                return silence
            self._startup = False
            # Export may wait for decode; realtime never waits for disk/CPU.
            if self.offline and prepared is None and asset is not None:
                started = time.monotonic()
                while prepared is None and asset is not None:
                    if self.cancel_event is not None and self.cancel_event.is_set():
                        raise InterruptedError
                    if time.monotonic() - started > 120:
                        raise RuntimeError(f'Ambient decode timed out: {asset.path}')
                    time.sleep(0.01)
                    asset, prepared = self._peek_next(spec)
            if prepared is None:
                self.phase = 'loading' if asset is not None else 'no eligible clips'
                self._update_status()
                return silence
            quiet = float(np.clip(1.0 - metabolism_activity, 0.0, 1.0))
            self._start(asset, prepared, spec, quiet)
        start = self.read_position
        take = min(frame_count, len(self.active_audio) - start)
        self._position(start / self.sample_rate, take / self.sample_rate, spec)
        mono = np.zeros(frame_count, dtype=np.float32)
        indices = start + np.arange(take)
        # Click protection only. The entire perceptual entrance/exit is spatial.
        edge = max(1.0, self.sample_rate * 0.01)
        ramp = np.minimum(np.clip(indices / edge, 0, 1),
                          np.clip((len(self.active_audio) - 1 - indices) / edge, 0, 1))
        mono[:take] = self.active_audio[start:start + take] * ramp
        output = self.source.process_mono(mono * 10.0 ** (spec.motif_calibrated_gain_db / 20.0))
        self.read_position += take
        if self.read_position >= len(self.active_audio):
            self._finish(spec)
        self._update_status()
        return output

    def set_root_directory(self, root_directory):
        root_directory = Path(root_directory)
        if root_directory == self.root_directory:
            return
        self.asset_manager.close()
        self.root_directory = root_directory
        self.active_asset = self.active_audio = None
        self.read_position = 0
        self.manual_test_asset = self.manual_test_audio = None
        self.phase = 'waiting'
        self._startup = True
        self.wait_seconds = self.state.get().startup_rest_seconds
        self._load_catalog()

    def close(self):
        self.asset_manager.close()




class BaseBrownFluidStereo:
    """
    Stateful, non-spatial stereo flow for the 2D brown-noise foundation.

    Two persistent mono brown-noise personalities are treated like two heavy
    fluids sharing a stereo container. A slow under-damped primary motion gives
    the system momentum and overshoot; a smaller, quicker eddy prevents the
    redistribution from collapsing into a mathematically perfect pan.

    This class produces only gain trajectories. It does not touch Steam Audio
    or the separate moving 3D brown-noise bodies.
    """

    def __init__(self) -> None:
        self.primary_state = OrganicMotionState(
            OrganicMotionSpec(
                natural_period_seconds=30.0,
                damping_ratio=0.46,
                drive_strength=1.12,
                drive_smoothing_seconds=9.0,
                soft_limit=1.15,
            )
        )
        self.eddy_state = OrganicMotionState(
            OrganicMotionSpec(
                natural_period_seconds=10.0,
                damping_ratio=0.72,
                drive_strength=0.58,
                drive_smoothing_seconds=3.5,
                soft_limit=1.35,
            )
        )
        self.primary = OrganicMotion1D(
            self.primary_state,
            seed=246_801,
        )
        self.eddy = OrganicMotion1D(
            self.eddy_state,
            seed=246_802,
        )
        self.current_flow = 0.0
        self.current_eddy = 0.0

    @staticmethod
    def _equal_power_pan(
        position: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Map -1..+1 onto an equal-power left/right pair.

        -1 = fully left, 0 = center, +1 = fully right.
        """
        angle = (
            np.clip(position, -1.0, 1.0) + 1.0
        ) * (math.pi * 0.25)
        return np.cos(angle), np.sin(angle)

    def gains(
        self,
        frame_count: int,
        sample_rate: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        elapsed_seconds = frame_count / float(sample_rate)

        primary_start, primary_end = self.primary.advance(
            elapsed_seconds
        )
        eddy_start, eddy_end = self.eddy.advance(
            elapsed_seconds
        )

        primary = np.linspace(
            primary_start,
            primary_end,
            frame_count,
            endpoint=False,
            dtype=np.float64,
        )
        eddy = np.linspace(
            eddy_start,
            eddy_end,
            frame_count,
            endpoint=False,
            dtype=np.float64,
        )

        # The two fluids broadly displace one another, but the eddy term makes
        # their motion imperfectly reciprocal. That creates lingering residue,
        # small internal swirls, and a less synthetic "crossfade" impression.
        voice_a_position = np.clip(
            0.88 * primary + 0.20 * eddy,
            -1.0,
            1.0,
        )
        voice_b_position = np.clip(
            -0.88 * primary + 0.16 * eddy,
            -1.0,
            1.0,
        )

        a_left, a_right = self._equal_power_pan(
            voice_a_position
        )
        b_left, b_right = self._equal_power_pan(
            voice_b_position
        )

        # Normalize channel power so the perceived motion is redistribution,
        # not a slow loudness pump.
        left_norm = np.sqrt(
            np.maximum(
                1.0e-9,
                a_left * a_left + b_left * b_left,
            )
        )
        right_norm = np.sqrt(
            np.maximum(
                1.0e-9,
                a_right * a_right + b_right * b_right,
            )
        )

        a_left /= left_norm
        b_left /= left_norm
        a_right /= right_norm
        b_right /= right_norm

        self.current_flow = float(primary[-1])
        self.current_eddy = float(eddy[-1])

        return (
            a_left.astype(np.float32),
            a_right.astype(np.float32),
            b_left.astype(np.float32),
            b_right.astype(np.float32),
        )



# =============================================================================
# Recorded meditation performances
# =============================================================================

@dataclass(frozen=True, slots=True)
class MeditationSpec:
    """Shared controls for generated vocal ceremonies.

    A ceremony is assembled from multiple foreground vocal recordings selected
    from the active style. The outer ceremony duration and the rest between
    ceremonies are both user-controlled ranges. Brown-bed attenuation is
    expressed as a negative gain in dB: -6 is less reduction than -24.
    """

    enabled: bool = True

    # Short rests are intentional. The ceremony itself already dissolves into
    # and out of the Living Brown Noise bed, so long silent separators are not
    # necessary to make successive performances feel distinct.
    interval_min_minutes: float = 1.0
    interval_max_minutes: float = 3.0

    # One generated ceremony stitches as many vocal recordings as necessary to
    # reach a target drawn from this range. The final retreat/fade is included
    # inside this duration rather than added afterward.
    ceremony_min_minutes: float = 8.0
    ceremony_max_minutes: float = 12.0

    performance_level_db: float = 0.0
    brown_least_attenuation_db: float = -6.0
    brown_greatest_attenuation_db: float = -24.0
    brown_coupling_percent: float = 40.0
    spatial_near_meters: float = 2.0
    spatial_far_meters: float = 3.0
    presence_cycle_minutes: float = 6.0
    distance_modulation_percent: float = 10.0
    distance_modulation_seconds: float = 90.0
    instruments_enabled: bool = True
    instrument_selection_percent: float = 50.0
    subliminal_emergence_seconds: float = 60.0
    transition_seconds: float = 12.0

    def validated(self) -> "MeditationSpec":
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be boolean")
        if not isinstance(self.instruments_enabled, bool):
            raise ValueError("instruments_enabled must be boolean")
        bounds = {
            "interval_min_minutes": (0.25, 60.0),
            "interval_max_minutes": (0.25, 60.0),
            "ceremony_min_minutes": (3.0, 30.0),
            "ceremony_max_minutes": (3.0, 30.0),
            "performance_level_db": (-30.0, 12.0),
            "brown_least_attenuation_db": (-60.0, 0.0),
            "brown_greatest_attenuation_db": (-60.0, 0.0),
            "brown_coupling_percent": (0.0, 100.0),
            # The existing renderer has unity distance gain at two meters.
            # Never enter its near-field amplification region for recordings.
            "spatial_near_meters": (2.0, 8.0),
            "spatial_far_meters": (2.0, 8.0),
            "presence_cycle_minutes": (2.0, 30.0),
            "distance_modulation_percent": (0.0, 500.0),
            "distance_modulation_seconds": (2.0, 300.0),
            "instrument_selection_percent": (0.0, 100.0),
            "subliminal_emergence_seconds": (0.0, 180.0),
            "transition_seconds": (1.0, 60.0),
        }
        for name, (low, high) in bounds.items():
            value = float(getattr(self, name))
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be between {low} and {high}")
        if self.interval_min_minutes > self.interval_max_minutes:
            raise ValueError("minimum rest cannot exceed maximum rest")
        if self.ceremony_min_minutes > self.ceremony_max_minutes:
            raise ValueError("minimum ceremony length cannot exceed maximum ceremony length")
        if self.brown_greatest_attenuation_db > self.brown_least_attenuation_db:
            raise ValueError("greatest attenuation must be the more negative dB value")
        if self.spatial_near_meters > self.spatial_far_meters:
            raise ValueError("near distance cannot exceed far distance")
        return self

    @classmethod
    def from_settings(cls, loaded: dict) -> "MeditationSpec":
        """Read current settings and migrate the former one-MP3 ceremony UI.

        The first run of the generated-ceremony architecture deliberately uses
        its new 1-3 minute rest defaults rather than inheriting the former
        45-120 minute waits. All unrelated meditation/spatial settings survive.
        """
        raw = loaded.get("meditation", loaded.get("synthesized_meditation", {}))
        if not isinstance(raw, dict):
            raw = {}
        defaults = cls()
        values = asdict(defaults)
        for name, default in tuple(values.items()):
            if name not in raw:
                continue
            try:
                value = raw[name]
                if isinstance(default, bool):
                    if isinstance(value, bool):
                        values[name] = value
                else:
                    value = float(value)
                    if math.isfinite(value):
                        values[name] = value
            except (ValueError, TypeError, OverflowError):
                pass

        # Presence of either ceremony-length field means these settings were
        # already saved by the generated-ceremony build. Otherwise this is a
        # legacy one-recording configuration and the old long rest values no
        # longer describe the intended behavior.
        generated_architecture_saved = (
            "ceremony_min_minutes" in raw
            or "ceremony_max_minutes" in raw
        )
        if not generated_architecture_saved:
            values["interval_min_minutes"] = defaults.interval_min_minutes
            values["interval_max_minutes"] = defaults.interval_max_minutes
            values["ceremony_min_minutes"] = defaults.ceremony_min_minutes
            values["ceremony_max_minutes"] = defaults.ceremony_max_minutes

        limits = {
            "interval_min_minutes": (0.25, 60.0),
            "interval_max_minutes": (0.25, 60.0),
            "ceremony_min_minutes": (3.0, 30.0),
            "ceremony_max_minutes": (3.0, 30.0),
            "performance_level_db": (-30.0, 12.0),
            "brown_least_attenuation_db": (-60.0, 0.0),
            "brown_greatest_attenuation_db": (-60.0, 0.0),
            "brown_coupling_percent": (0.0, 100.0),
            "spatial_near_meters": (2.0, 8.0),
            "spatial_far_meters": (2.0, 8.0),
            "presence_cycle_minutes": (2.0, 30.0),
            "distance_modulation_percent": (0.0, 500.0),
            "distance_modulation_seconds": (2.0, 300.0),
            "instrument_selection_percent": (0.0, 100.0),
            "subliminal_emergence_seconds": (0.0, 180.0),
            "transition_seconds": (1.0, 60.0),
        }
        for name, (low, high) in limits.items():
            if not low <= values[name] <= high:
                values[name] = getattr(defaults, name)
        for low_name, high_name in (
            ("interval_min_minutes", "interval_max_minutes"),
            ("ceremony_min_minutes", "ceremony_max_minutes"),
            ("brown_greatest_attenuation_db", "brown_least_attenuation_db"),
            ("spatial_near_meters", "spatial_far_meters"),
        ):
            values[low_name], values[high_name] = sorted(
                (values[low_name], values[high_name])
            )
        return cls(**values).validated()


class MeditationState:
    """Thread-safe settings with normalized endpoint pairs."""

    def __init__(self, spec: MeditationSpec) -> None:
        self._lock = threading.Lock()
        self._spec = spec.validated()

    def get(self) -> MeditationSpec:
        with self._lock:
            return self._spec

    def set(self, spec: MeditationSpec) -> None:
        with self._lock:
            self._spec = spec.validated()

    def update(self, **changes) -> None:
        with self._lock:
            values = asdict(self._spec)
            values.update(changes)
            for low_name, high_name in (
                ("interval_min_minutes", "interval_max_minutes"),
                ("ceremony_min_minutes", "ceremony_max_minutes"),
                ("brown_greatest_attenuation_db", "brown_least_attenuation_db"),
                ("spatial_near_meters", "spatial_far_meters"),
            ):
                if values[low_name] > values[high_name]:
                    if low_name in changes:
                        values[high_name] = values[low_name]
                    else:
                        values[low_name] = values[high_name]
            self._spec = MeditationSpec(**values).validated()


class MeditationFieldMotion:
    """One continuous spatial presence field shared by an entire ceremony.

    Vocal file boundaries never reset this field. A newly-started ceremony
    begins at the far endpoint and immediately starts approaching; internal
    presence cycles can continue for long ceremonies, and the conductor can
    request one explicit final retreat before the ceremony ends.
    """

    STEREO_HALF_WIDTH_DEGREES = 24.0
    _PHASES = ("distant", "approach", "present", "retreat")
    _PHASE_FRACTIONS = {
        "distant": 0.20,
        "approach": 0.30,
        "present": 0.20,
        "retreat": 0.30,
    }

    def __init__(self, spec: MeditationSpec, seed: int) -> None:
        def journey(offset, initial, low, high, tmin, tmax, beta=1.4):
            return SmoothRandomJourney(
                rng=np.random.default_rng(seed + offset),
                initial_value=initial, minimum=low, maximum=high,
                duration_min_seconds=tmin, duration_max_seconds=tmax,
                beta_a=beta, beta_b=beta,
            )

        self.rng = np.random.default_rng(seed + 7)
        self.brown = journey(23, 0.45, 0.0, 1.0, 90.0, 240.0)

        # Faster than the former multi-minute directional journeys. These are
        # still slow movements, but large enough to be perceived within one
        # 2-5 minute source clip rather than only across several clips.
        self.radial_drift = journey(11, 0.50, 0.0, 1.0, 45.0, 110.0, 1.8)
        self.azimuth = journey(37, 0.0, -115.0, 115.0, 45.0, 110.0, 2.0)
        self.elevation = journey(53, 4.0, -10.0, 16.0, 60.0, 140.0, 2.0)

        self.near = spec.spatial_near_meters
        self.far = spec.spatial_far_meters
        self.least_db = spec.brown_least_attenuation_db
        self.greatest_db = spec.brown_greatest_attenuation_db
        self.coupling = spec.brown_coupling_percent / 100.0

        self.presence_phase_index = 1
        self.presence_phase = "approach"
        self.presence_elapsed = 0.0
        self.presence_start = 1.0
        self.presence_target = 0.06
        self.presence_base = 1.0
        self.presence_duration = self._phase_duration(spec, "approach")
        self.current_radial_state = 1.0

        self._final_retreat_active = False
        self._final_retreat_elapsed = 0.0
        self._final_retreat_duration = 45.0
        self._final_retreat_start = 1.0
        self._final_retreat_progress = 0.0

        self.independent_state = self.brown.current_value
        self.proximity_state = 0.0
        self.current_brown_db = self._brown_db()
        self.current_distance = self._distance()
        self.current_azimuth = self.azimuth.current_value
        self.current_elevation = self.elevation.current_value
        self.current_position = self._position(0.0)

    def _phase_duration(self, spec: MeditationSpec, phase: str) -> float:
        cycle = max(120.0, float(spec.presence_cycle_minutes) * 60.0)
        nominal = cycle * self._PHASE_FRACTIONS[phase]
        return max(20.0, nominal * float(self.rng.uniform(0.86, 1.16)))

    @staticmethod
    def _smoothstep5(value: float) -> float:
        value = float(np.clip(value, 0.0, 1.0))
        return value ** 3 * (value * (value * 6.0 - 15.0) + 10.0)

    def reset_for_ceremony(self, spec: MeditationSpec) -> None:
        """Begin an emergence immediately from the far endpoint."""
        self.presence_phase_index = 1
        self.presence_phase = "approach"
        self.presence_elapsed = 0.0
        self.presence_start = 1.0
        self.presence_target = 0.06
        self.presence_base = 1.0
        self.presence_duration = self._phase_duration(spec, "approach")
        self.current_radial_state = 1.0
        self._final_retreat_active = False
        self._final_retreat_elapsed = 0.0
        self._final_retreat_progress = 0.0
        self.proximity_state = 0.0
        self.current_distance = self._distance()
        self.current_brown_db = self._brown_db()
        self.current_position = self._position(0.0)

    def begin_final_retreat(self, duration_seconds: float) -> None:
        if self._final_retreat_active:
            return
        self._final_retreat_active = True
        self._final_retreat_elapsed = 0.0
        self._final_retreat_duration = max(1.0, float(duration_seconds))
        self._final_retreat_start = float(self.presence_base)
        self._final_retreat_progress = 0.0
        self.presence_phase = "retreat"
        self.presence_elapsed = 0.0
        self.presence_duration = self._final_retreat_duration
        self.presence_start = self._final_retreat_start
        self.presence_target = 1.0

    def _begin_next_presence_phase(self, spec: MeditationSpec) -> None:
        self.presence_phase_index = (self.presence_phase_index + 1) % len(self._PHASES)
        self.presence_phase = self._PHASES[self.presence_phase_index]
        self.presence_elapsed = 0.0
        self.presence_start = self.presence_base
        if self.presence_phase in {"approach", "present"}:
            self.presence_target = 0.06
        else:
            self.presence_target = 1.0
        self.presence_duration = self._phase_duration(spec, self.presence_phase)

    def _advance_presence(self, seconds: float, spec: MeditationSpec) -> None:
        remaining = max(0.0, float(seconds))

        if self._final_retreat_active:
            self._final_retreat_elapsed = min(
                self._final_retreat_duration,
                self._final_retreat_elapsed + remaining,
            )
            self._final_retreat_progress = self._smoothstep5(
                self._final_retreat_elapsed / self._final_retreat_duration
            )
            self.presence_base = (
                self._final_retreat_start
                + (1.0 - self._final_retreat_start) * self._final_retreat_progress
            )
            self.presence_elapsed = self._final_retreat_elapsed
            self.presence_phase = "retreat"
            return

        while remaining > 0.0:
            available = max(1.0e-9, self.presence_duration - self.presence_elapsed)
            step = min(remaining, available)
            self.presence_elapsed += step
            remaining -= step
            progress = self._smoothstep5(self.presence_elapsed / self.presence_duration)
            self.presence_base = (
                self.presence_start
                + (self.presence_target - self.presence_start) * progress
            )
            if self.presence_elapsed >= self.presence_duration - 1.0e-9:
                self.presence_base = self.presence_target
                self._begin_next_presence_phase(spec)

    def _distance(self) -> float:
        return math.exp(
            math.log(self.near)
            + self.current_radial_state * math.log(self.far / self.near)
        )

    def _brown_db(self) -> float:
        if abs(self.far - self.near) < 1.0e-9:
            self.proximity_state = 0.5
        else:
            self.proximity_state = 1.0 - self.current_radial_state
        amount = (
            (1.0 - self.coupling) * self.independent_state
            + self.coupling * self.proximity_state
        )
        return self.least_db + amount * (self.greatest_db - self.least_db)

    def _position(self, azimuth_offset: float) -> Vector3:
        az = math.radians(self.current_azimuth + azimuth_offset)
        el = math.radians(self.current_elevation)
        horizontal = self.current_distance * math.cos(el)
        return Vector3(
            horizontal * math.sin(az),
            self.current_distance * math.sin(el),
            -horizontal * math.cos(az),
        )

    def stereo_positions(self) -> tuple[Vector3, Vector3]:
        return (
            self._position(-self.STEREO_HALF_WIDTH_DEGREES),
            self._position(self.STEREO_HALF_WIDTH_DEGREES),
        )

    def advance(self, seconds: float, spec: MeditationSpec) -> tuple[float, float]:
        start_db = self.current_brown_db
        seconds = max(0.0, float(seconds))

        self._advance_presence(seconds, spec)
        drift = self.radial_drift.advance(seconds)
        self.independent_state = self.brown.advance(seconds)
        self.current_azimuth = self.azimuth.advance(seconds)
        self.current_elevation = self.elevation.advance(seconds)

        # As the explicit final retreat approaches its far endpoint, reduce the
        # small radial wander too. The ceremony therefore actually disappears
        # into the distance instead of being held a few percent forward by an
        # unrelated drift value.
        drift_scale = 1.0 - self._final_retreat_progress if self._final_retreat_active else 1.0
        drift_offset = (drift - 0.5) * 0.08 * drift_scale
        self.current_radial_state = float(np.clip(
            self.presence_base + drift_offset, 0.0, 1.0
        ))

        # Slider edits settle gradually rather than teleporting the source or
        # stepping its gain. Presence-cycle edits affect newly-entered phases.
        blend = 1.0 - math.exp(-seconds / 3.0) if seconds > 0.0 else 0.0
        self.near += (spec.spatial_near_meters - self.near) * blend
        self.far += (spec.spatial_far_meters - self.far) * blend
        self.least_db += (spec.brown_least_attenuation_db - self.least_db) * blend
        self.greatest_db += (spec.brown_greatest_attenuation_db - self.greatest_db) * blend
        self.coupling += (spec.brown_coupling_percent / 100.0 - self.coupling) * blend

        self.current_distance = self._distance()
        self.current_position = self._position(0.0)
        self.current_brown_db = self._brown_db()
        return start_db, self.current_brown_db


class CeremonyRecordingDecoder:
    """Small incremental PyAV decoder. Owned exclusively by a loader thread."""

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = int(sample_rate)
        self.container = None
        self.resampler = None
        self.decoder = None
        self._chunks = deque()
        self._offset = 0
        self._queued_frames = 0
        self._eof = True
        self.duration_seconds = 0.0
        self.elapsed_samples = 0

    @staticmethod
    def _audio_stream(container):
        streams = [s for s in container.streams if s.type == "audio"]
        if not streams:
            raise ValueError("Meditation recording contains no audio stream")
        return streams[0]

    @staticmethod
    def _duration(container, stream) -> float:
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
            if math.isfinite(duration) and duration > 0.0:
                return duration
        if container.duration is not None:
            duration = float(container.duration) / float(av.time_base)
            if math.isfinite(duration) and duration > 0.0:
                return duration
        return 0.0

    @classmethod
    def probe_duration_seconds(cls, path: Path) -> float:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Meditation recording not found: {path}")
        container = av.open(str(path))
        try:
            return cls._duration(container, cls._audio_stream(container))
        finally:
            container.close()

    def start(self, path: Path) -> None:
        self.stop()
        if not Path(path).is_file():
            raise FileNotFoundError(f"Meditation recording not found: {path}")
        try:
            self.container = av.open(str(path))
            stream = self._audio_stream(self.container)
            self.duration_seconds = self._duration(self.container, stream)
            self.resampler = av.audio.resampler.AudioResampler(
                format="fltp", layout="stereo", rate=self.sample_rate,
            )
            self.decoder = self.container.decode(stream)
            self.elapsed_samples = 0
            self._eof = False
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        if self.container is not None:
            self.container.close()
        self.container = self.resampler = self.decoder = None
        self._chunks.clear()
        self._offset = self._queued_frames = 0
        self._eof = True

    @property
    def complete(self) -> bool:
        return self._eof and self._queued_frames == 0

    def _queue_converted(self, frames) -> None:
        if frames is None:
            return
        if not isinstance(frames, list):
            frames = [frames]
        for frame in frames:
            data = np.ascontiguousarray(frame.to_ndarray().T, dtype=np.float32)
            if data.ndim != 2 or data.shape[1] != 2:
                raise ValueError(f"Unexpected resampled stereo shape: {data.shape}")
            if len(data):
                np.nan_to_num(data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                self._chunks.append(data)
                self._queued_frames += len(data)

    def render(self, frame_count: int) -> np.ndarray:
        while self._queued_frames < frame_count and not self._eof:
            try:
                frame = next(self.decoder)
            except StopIteration:
                self._queue_converted(self.resampler.resample(None))
                self._eof = True
                if self.container is not None:
                    self.container.close()
                self.container = self.resampler = self.decoder = None
            else:
                self._queue_converted(self.resampler.resample(frame))
        output = np.zeros((frame_count, 2), dtype=np.float32)
        written = 0
        while written < frame_count and self._chunks:
            chunk = self._chunks[0]
            take = min(frame_count - written, len(chunk) - self._offset)
            output[written:written + take] = chunk[self._offset:self._offset + take]
            written += take
            self._offset += take
            self._queued_frames -= take
            if self._offset == len(chunk):
                self._chunks.popleft()
                self._offset = 0
        self.elapsed_samples += written
        return output[:written]


class CeremonyRecordingPlayer:
    """Bounded read-ahead player; no file opening/decoding in the audio callback.

    Both live and offline playback consume identical decoded stereo samples.
    Live reads never wait for I/O. Offline reads may wait on the producer so
    rendering speed cannot cause dropped samples. One persistent worker owns
    every decoder/container operation, including cancellation and closing.
    """

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = int(sample_rate)
        self.path: Path | None = None
        self.duration_seconds = 0.0
        self.elapsed_samples = 0
        self.last_valid_frames = 0
        self.underrun_count = 0
        self.peak_buffered_frames = 0
        self._decode_frames = max(2048, self.sample_rate // 4)
        self._max_frames = self.sample_rate * 8
        self._prefill_frames = self.sample_rate // 2
        self._decoded_frames = 0
        self._condition = threading.Condition()
        self._chunks = deque()
        self._offset = 0
        self._queued_frames = 0
        self._revision = 0
        self._closed = False
        self._eof = True
        self._opened = False
        self._error = ""
        self._worker = threading.Thread(
            target=self._worker_loop, name="MeditationMP3Decoder", daemon=True,
        )
        self._worker.start()

    @staticmethod
    def probe_duration_seconds(path: Path) -> float:
        return CeremonyRecordingDecoder.probe_duration_seconds(path)

    @property
    def elapsed_seconds(self) -> float:
        return self.elapsed_samples / self.sample_rate

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.duration_seconds - self.elapsed_seconds)

    @property
    def complete(self) -> bool:
        with self._condition:
            return self._opened and self._eof and self._queued_frames == 0

    @property
    def error(self) -> str:
        with self._condition:
            return self._error

    @property
    def final_sample_count(self) -> int | None:
        with self._condition:
            return self._decoded_frames if self._opened and self._eof else None

    def prepare(self, path: Path) -> None:
        # Publication only. The worker performs all filesystem/PyAV work.
        with self._condition:
            if self._closed:
                raise RuntimeError("Meditation recording player is closed")
            self._revision += 1
            self.path = Path(path)
            self.duration_seconds = 0.0
            self.elapsed_samples = self.last_valid_frames = 0
            self.underrun_count = self.peak_buffered_frames = 0
            self._decoded_frames = 0
            self._chunks.clear()
            self._offset = self._queued_frames = 0
            self._eof = self._opened = False
            self._error = ""
            self._condition.notify_all()

    def stop(self) -> None:
        with self._condition:
            self._revision += 1
            self.path = None
            self._chunks.clear()
            self._offset = self._queued_frames = 0
            self._eof = True
            self._opened = False
            self._error = ""
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._worker.join(timeout=5.0)

    def _worker_loop(self) -> None:
        seen_revision = -1
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._closed or (
                        self.path is not None and self._revision != seen_revision
                    )
                )
                if self._closed:
                    return
                revision = seen_revision = self._revision
                path = self.path
            decoder = CeremonyRecordingDecoder(self.sample_rate)
            try:
                decoder.start(path)
                with self._condition:
                    if self._closed or revision != self._revision:
                        continue
                    self.duration_seconds = decoder.duration_seconds
                    self._opened = True
                    self._condition.notify_all()
                while True:
                    with self._condition:
                        self._condition.wait_for(
                            lambda: self._closed or revision != self._revision
                            or self._queued_frames <= self._max_frames - self._decode_frames
                        )
                        if self._closed or revision != self._revision:
                            break
                    chunk = decoder.render(self._decode_frames)
                    with self._condition:
                        if self._closed or revision != self._revision:
                            break
                        if len(chunk):
                            self._chunks.append(chunk)
                            self._queued_frames += len(chunk)
                            self._decoded_frames += len(chunk)
                            self.peak_buffered_frames = max(
                                self.peak_buffered_frames, self._queued_frames,
                            )
                        self._eof = decoder.complete
                        if self._eof:
                            # Actual decoded length supersedes any inaccurate
                            # container/VBR estimate at the end of the file.
                            self.duration_seconds = self._decoded_frames / self.sample_rate
                        self._condition.notify_all()
                        if self._eof:
                            break
            except Exception as exc:
                with self._condition:
                    if revision == self._revision and not self._closed:
                        self._error = f"{path}: {exc}"
                        self._eof = True
                        self._condition.notify_all()
            finally:
                try:
                    decoder.stop()
                except Exception:
                    LOGGER.exception("Could not close meditation decoder")

    def ready(self, *, wait: bool = False, cancel_event=None) -> bool:
        with self._condition:
            while True:
                if self._closed or (cancel_event is not None and cancel_event.is_set()):
                    raise InterruptedError
                if self._error:
                    raise RuntimeError(self._error)
                if self._opened and (
                    self._queued_frames >= self._prefill_frames or self._eof
                ):
                    if self._queued_frames == 0:
                        raise ValueError(f"Meditation recording decoded no audio: {self.path}")
                    return True
                if not wait:
                    return False
                self._condition.wait(timeout=0.05)

    def render(self, frame_count: int, *, wait: bool = False, cancel_event=None) -> np.ndarray:
        if frame_count > self._max_frames:
            raise ValueError("A playback read must fit within the read-ahead buffer")
        output = np.zeros((frame_count, 2), dtype=np.float32)
        self.last_valid_frames = 0
        if frame_count <= 0:
            return output
        with self._condition:
            while self._queued_frames < frame_count and not self._eof and not self._error:
                if self._closed or (cancel_event is not None and cancel_event.is_set()):
                    raise InterruptedError
                if not wait:
                    self.underrun_count += 1
                    # Do not consume half a block and discard the rest. Pause
                    # the source cursor until it can resume a complete block.
                    return output
                self._condition.wait(timeout=0.05)
            if self._error:
                raise RuntimeError(self._error)
            written = 0
            while written < frame_count and self._chunks:
                chunk = self._chunks[0]
                take = min(frame_count - written, len(chunk) - self._offset)
                output[written:written + take] = chunk[self._offset:self._offset + take]
                written += take
                self._offset += take
                self._queued_frames -= take
                if self._offset == len(chunk):
                    self._chunks.popleft()
                    self._offset = 0
            self.elapsed_samples += written
            self.last_valid_frames = written
            self._condition.notify_all()
        return output


class MeditationOrchestrator:
    """Generate ceremonies from independently shuffled vocal and instrument pools.

    Two bounded read-ahead workers handle successive passages. Matching
    categories crossfade; vocals and instruments fade out then in without
    overlapping playback. Solo instruments hold a stable front stereo position.
    Both categories share ceremony timing, emergence, exit, and brown-bed control.
    """

    RANDOM_START_TOKEN = "__generated_random__"
    CROSSFADE_MIN_SECONDS = 3.0
    CROSSFADE_MAX_SECONDS = 8.0

    def __init__(self, *, sample_rate: float, renderer: SteamAudioRenderer,
                 state: MeditationState, style_directory: Path,
                 seed: int = 8_230_601) -> None:
        self.sample_rate = int(sample_rate)
        self.renderer = renderer
        self.state = state
        self.rng = np.random.default_rng(seed)
        self.style_directory = Path(style_directory)
        self.style_name = self.style_directory.name
        self._load_curated_recordings()
        if not self.performance_registry:
            LOGGER.warning(
                "No meditation MP3 files found for style %s beneath %s",
                self.style_name, self.style_directory
            )

        # Independent categories, each with High / Medium / Low shuffle bags.
        self._refill_performance_bag()

        # Two independent bounded read-ahead workers are enough for an outgoing
        # and incoming source during crossfade. Their stereo is combined before
        # the one continuous ceremony spatial field is applied.
        self.players = [
            CeremonyRecordingPlayer(self.sample_rate),
            CeremonyRecordingPlayer(self.sample_rate),
        ]
        self._current_player_index = 0
        self._next_player_index = 1

        self.field = MeditationFieldMotion(state.get(), seed + 900)
        left, right = self.field.stereo_positions()
        self.sources = [
            renderer.create_source(position=p, spatial_blend=1.0,
                                   distance_attenuation_enabled=True)
            for p in (left, right)
        ]

        self.active_name = ""
        self.current_vocal_name = ""
        self.next_vocal_name = ""
        self.current_status = "waiting"
        self.performance_count = 0
        self.elapsed_samples = 0
        self.export_mode = False
        self.export_total_seconds = 0.0
        self._duration_cache: dict[str, float] = {}
        self._next_start_sample = math.inf
        self._event_journal = deque(maxlen=4096)
        self._command_lock = threading.Lock()
        self._manual_start_requested: str | None = None
        self._manual_stop_requested = False
        self._pending_name: str | None = None
        self._pending_manual = False
        self._pending_due_sample = math.inf
        self._replacement_name: str | None = None
        self._stopping = False
        self._scheduled_stop = False
        self._source_completed = False
        self._mix = 0.0
        self._audio_mix = 0.0
        self._level_db = state.get().performance_level_db
        # Independent RNG keeps motion choices separate from passage selection.
        self._distance_drift_rng = np.random.default_rng(seed + 2700)
        self._distance_drift_start = 0.0
        self._distance_drift_target = float(self._distance_drift_rng.uniform(-1.0, 1.0))
        self._distance_drift_progress = 0.0
        self._distance_drift_span = float(self._distance_drift_rng.uniform(0.3, 0.75))
        self._distance_drift_holding = False
        self._distance_modulation_depth = 0.0
        self._distance_modulation_period = state.get().distance_modulation_seconds
        self.current_distance_modulation_percent = 0.0
        self._modulation_bed_weight = 1.0
        self.current_brown_gain_db = 0.0
        self.current_presence_gain_db = -8.0
        self.blocks_motifs = False  # True if any part of the rendered block is occupied
        self._last_status_sample = -self.sample_rate
        self._last_field_log_sample = -30 * self.sample_rate
        self._last_error = ""
        self.cancel_event = None
        self._closed = False

        self._ceremony_target_samples = 0
        self._ceremony_elapsed_samples = 0
        self._ceremony_retreat_started = False
        self._ceremony_retreat_samples = 0

        self._crossfade_active = False
        self._crossfade_elapsed_samples = 0
        self._crossfade_total_samples = 0
        self._selection_settings = None
        self._passage_gain = 1.0
        self._handoff_out = False
        self._next_crossfade_seconds = self._choose_crossfade_seconds()
        self._current_underrun_snapshot = 0
        self._next_underrun_snapshot = 0

        self._reschedule()
        self._log_curation_plan()

    @property
    def recording_player(self) -> CeremonyRecordingPlayer:
        """Compatibility/readability alias for the currently foregrounded clip."""
        return self.players[self._current_player_index]

    @property
    def next_recording_player(self) -> CeremonyRecordingPlayer:
        return self.players[self._next_player_index]

    @staticmethod
    def _display_name_for_path(path: Path) -> str:
        stem = path.stem
        return stem.replace("_", " ").replace("-", " ").strip().title()

    @classmethod
    def _scan_recordings(cls, style_dir: Path) -> dict[str, Path]:
        """Return vocal and instrument MP3s for the currently selected style.

        The authored layout is ceremonies/<style>/vocals/. Direct MP3s in the
        style directory remain accepted for migration. Instrument MP3s live in
        instruments/<type>/ and are selected as standalone passages.
        """
        style_dir = Path(style_dir)
        vocals_dir = style_dir / STYLE_VOCALS_FOLDER
        paths: list[Path] = []
        if vocals_dir.is_dir():
            paths.extend(
                path for path in vocals_dir.iterdir()
                if path.is_file() and path.suffix.lower() == ".mp3"
            )
        if style_dir.is_dir():
            paths.extend(
                path for path in style_dir.iterdir()
                if path.is_file() and path.suffix.lower() == ".mp3"
            )
        paths = sorted(set(paths), key=lambda path: path.name.lower())

        instruments_dir = style_dir / "instruments"
        if instruments_dir.is_dir():
            paths.extend(sorted(
                (p for p in instruments_dir.glob("*/*") if p.is_file() and p.suffix.lower() == ".mp3"),
                key=lambda p: str(p).lower(),
            ))
        recordings: dict[str, Path] = {}
        for path in paths:
            base_name = cls._display_name_for_path(path)
            if path.parent.parent == instruments_dir:
                base_name = f"[Instrument: {path.parent.name}] {base_name}"
            name = base_name
            suffix = 2
            while name in recordings:
                name = f"{base_name} ({suffix})"
                suffix += 1
            recordings[name] = path
        return recordings

    def set_style_directory(self, style_dir: Path) -> None:
        """Switch the ceremony library to another authored style."""
        for player in self.players:
            player.stop()
        self.style_directory = Path(style_dir)
        self.style_name = self.style_directory.name
        self._load_curated_recordings()
        self._pending_name = None
        self._pending_manual = False
        self._replacement_name = None
        self._stopping = False
        self._scheduled_stop = False
        self.active_name = ""
        self.current_vocal_name = ""
        self.next_vocal_name = ""
        self.current_status = f"waiting — style {self.style_name}"
        self._mix = 0.0
        self._audio_mix = 0.0
        self.current_brown_gain_db = 0.0
        self.current_presence_gain_db = -8.0
        self._duration_cache.clear()
        self._crossfade_active = False
        self._ceremony_elapsed_samples = 0
        self._ceremony_target_samples = 0
        self._refill_performance_bag()
        self._log_curation_plan()
        self._reschedule()
        self._journal(
            "MEDITATION_STYLE",
            f"selected style={self.style_name}; ceremony clips={len(self.performance_registry)}",
        )

    @property
    def elapsed_seconds(self) -> float:
        return self.elapsed_samples / self.sample_rate

    @property
    def active(self) -> bool:
        return bool(self.active_name)

    @property
    def ceremony_elapsed_seconds(self) -> float:
        return self._ceremony_elapsed_samples / self.sample_rate

    @property
    def ceremony_target_seconds(self) -> float:
        return self._ceremony_target_samples / self.sample_rate

    @property
    def ceremony_remaining_seconds(self) -> float:
        return max(0.0, self.ceremony_target_seconds - self.ceremony_elapsed_seconds)

    @property
    def next_performance_seconds(self) -> float:
        if not math.isfinite(self._next_start_sample):
            return math.inf
        return max(0.0, (self._next_start_sample - self.elapsed_samples) / self.sample_rate)

    @staticmethod
    def _format_log_time(seconds: float) -> str:
        """Format an exported ceremony-log timestamp as HH:MM:SS.mmm."""
        total_ms = max(0, int(round(seconds * 1000.0)))
        hours, remainder = divmod(total_ms, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        secs, millis = divmod(remainder, 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    def _journal(self, category: str, message: str, offset: int = 0) -> None:
        self._event_journal.append((
            (self.elapsed_samples + offset) / self.sample_rate,
            str(category), str(message),
        ))

    def drain_event_journal(self) -> list[tuple[float, str, str]]:
        entries = []
        while True:
            try:
                entries.append(self._event_journal.popleft())
            except IndexError:
                return entries

    def _recording_duration_seconds(self, name: str) -> float:
        if name not in self._duration_cache:
            duration = CeremonyRecordingPlayer.probe_duration_seconds(
                self.recording_paths[name]
            )
            if duration <= 0.0:
                raise ValueError(f"Cannot determine duration for vocal clip {name}")
            self._duration_cache[name] = duration
        return self._duration_cache[name]

    def _load_curated_recordings(self) -> None:
        self.curation = AudioCuration()
        scanned = self._scan_recordings(self.style_directory)
        self.recording_paths = {
            name: path for name, path in scanned.items()
            if self.curation.state_for_path(path) != "stash"
        }
        self.performance_registry = tuple(self.recording_paths)

    def _refill_performance_bag(self) -> None:
        """Reset both categories only for a new session, style or export."""
        self._performance_bags = {
            category: CurationShuffleBag({
                name: self.curation.tier_for_path(self.recording_paths[name])
                for name in self.performance_registry
                if self._performance_category(name) == category
            }, self.rng)
            for category in ("vocals", "instruments")
        }

    @property
    def remaining_performances(self) -> list[str]:
        """Flattened view for diagnostics; selection uses the six actual bags."""
        return [name for bag in self._performance_bags.values()
                for tier in CurationShuffleBag.TIERS for name in bag.remaining[tier]]

    def _log_curation_plan(self) -> None:
        pools = []
        for category, bag in self._performance_bags.items():
            counts = ", ".join(f"{tier}={len(bag.remaining[tier])}" for tier in bag.TIERS)
            pools.append(f"{category}: {counts}")
        self._journal("MEDITATION_CURATION", (
            f"metadata={self.curation.path}; "
            f"{'loaded' if self.curation.found else 'absent; all clips unreviewed'}; "
            "priority=high -> medium -> low+unreviewed; separate category cycles; "
            + "; ".join(pools)
        ))

    @classmethod
    def _performance_category(cls, name: str) -> str:
        return "instruments" if cls._is_instrument(name) else "vocals"

    def _return_unplayed_performance(self, name: str) -> None:
        if name:
            bag = self._performance_bags[self._performance_category(name)]
            bag.return_unused(name)

    def _return_pending_vocal(self) -> None:
        """An opening clip cancelled before ceremony start was never heard."""
        if self._pending_name:
            self._return_unplayed_performance(self._pending_name)
            self.recording_player.stop()
            self._pending_name = None

    @staticmethod
    def _is_instrument(name: str) -> bool:
        return bool(name and name.startswith("[Instrument: "))

    def _take_vocal(self, avoid_name: str | None = None, *, first_passage: bool = False) -> str | None:
        """Choose category by weight, then its highest unexhausted tier."""
        spec = self.state.get()
        vocals = [n for n in self.performance_registry if not self._is_instrument(n)]
        instruments = [n for n in self.performance_registry if self._is_instrument(n)]
        if not spec.instruments_enabled:
            instruments = []
        if not vocals and not instruments:
            return None
        # avoid_name is the actual outgoing passage, not a reserved bag draw.
        # Every ceremony opens with a vocal; instruments also return to vocals.
        return_to_vocals = bool(vocals) and (first_passage or self._is_instrument(avoid_name))
        use_instrument = not return_to_vocals and bool(instruments) and (
            not vocals or self.rng.random() < spec.instrument_selection_percent / 100.0
        )
        category = "instruments" if use_instrument else "vocals"
        bag = self._performance_bags[category]
        name = bag.take(avoid_name)
        if name:
            self._journal("MEDITATION_CURATION_DRAW", (
                f"category={category}; tier={bag.tiers[name]}; cycle={bag.cycle}; "
                f"clip={name}; file={self.recording_paths[name].name}"
            ))
        return name

    def _return_reserved_vocal(self) -> None:
        """Return only unheard preload audio, preserving its original tier."""
        if self.next_vocal_name and self.next_recording_player.elapsed_samples == 0:
            self._return_unplayed_performance(self.next_vocal_name)
        self.next_recording_player.stop()
        self.next_vocal_name = ""

    def configure_export(self, total_duration_seconds: float) -> None:
        self.export_mode = True
        self.export_total_seconds = max(0.0, float(total_duration_seconds))
        self.elapsed_samples = 0
        self._duration_cache.clear()
        # Each export starts with a fresh snapshot and independent bag cycles.
        # No metadata reads happen in the render callback.
        self._load_curated_recordings()
        self._refill_performance_bag()
        self._log_curation_plan()
        spec = self.state.get()
        if not spec.enabled or not self.performance_registry:
            self._next_start_sample = math.inf
            self._journal(
                "MEDITATION_PLAN",
                "automatic generated ceremonies disabled or no ceremony clips available",
            )
            return

        # Fail fast on malformed files before a long render begins.
        for name in self.performance_registry:
            self._recording_duration_seconds(name)
        self._reschedule()
        self._journal("MEDITATION_PLAN", (
            f"style={self.style_name}; {len(self.performance_registry)} ceremony clips; "
            f"generated ceremonies {spec.ceremony_min_minutes:.1f}-"
            f"{spec.ceremony_max_minutes:.1f} min; rests "
            f"{spec.interval_min_minutes:.2f}-{spec.interval_max_minutes:.2f} min; "
            f"same-category crossfades {self.CROSSFADE_MIN_SECONDS:.0f}-"
            f"{self.CROSSFADE_MAX_SECONDS:.0f}s; category changes fade through silence; "
            "initial tier order (vocals, then instruments; category choice remains weighted): "
            + " -> ".join(self.remaining_performances)
        ))

    def _reschedule(self, from_sample: int | None = None) -> None:
        base = self.elapsed_samples if from_sample is None else from_sample
        spec = self.state.get()
        if not self.performance_registry or (self.export_mode and not spec.enabled):
            self._next_start_sample = math.inf
            return
        low = spec.interval_min_minutes * 60.0
        high = spec.interval_max_minutes * 60.0
        if abs(high - low) < 1.0e-9:
            wait = low
        else:
            wait = float(math.exp(self.rng.uniform(math.log(low), math.log(high))))
        self._next_start_sample = base + round(wait * self.sample_rate)

    def _choose_ceremony_seconds(self, spec: MeditationSpec) -> float:
        low = spec.ceremony_min_minutes * 60.0
        high = spec.ceremony_max_minutes * 60.0
        return low if abs(high - low) < 1.0e-9 else float(self.rng.uniform(low, high))

    def _choose_crossfade_seconds(self) -> float:
        return float(self.rng.uniform(
            self.CROSSFADE_MIN_SECONDS, self.CROSSFADE_MAX_SECONDS
        ))

    def _final_retreat_seconds(self, spec: MeditationSpec) -> float:
        # About one eighth of the normal presence cycle, with practical bounds.
        return float(np.clip(spec.presence_cycle_minutes * 60.0 * 0.125, 30.0, 90.0))

    @staticmethod
    def _subliminal_transition_seconds(spec: MeditationSpec) -> float:
        """Duration used when a ceremony enters or leaves awareness.

        transition_seconds remains the minimum click-safe transition for old
        settings and for a deliberately disabled subliminal stage.  Normally
        the user-tunable subliminal duration governs both ends so a ceremony
        disappears with the same patience with which it appeared.
        """
        return max(
            float(spec.transition_seconds),
            float(spec.subliminal_emergence_seconds),
        )

    def request_start(self, name: str) -> None:
        if name == self.RANDOM_START_TOKEN or name in self.recording_paths:
            self._request_start(name)
        else:
            self._journal("MEDITATION_ERROR", f"vocal clip not found: {name}")

    def _request_start(self, name: str) -> None:
        with self._command_lock:
            self._manual_start_requested = name
            self._manual_stop_requested = False

    def request_stop(self) -> None:
        with self._command_lock:
            self._manual_start_requested = None
            self._manual_stop_requested = True

    def _consume_commands(self) -> None:
        with self._command_lock:
            name = self._manual_start_requested
            stop = self._manual_stop_requested
            self._manual_start_requested = None
            self._manual_stop_requested = False

        if stop:
            self._replacement_name = None
            if self.active:
                spec = self.state.get()
                fade_seconds = self._subliminal_transition_seconds(spec)
                self._scheduled_stop = False
                self._stopping = True
                self._source_completed = False
                self.field.begin_final_retreat(fade_seconds)
                self._journal(
                    "MEDITATION_STOP_REQUEST",
                    f"{self.active_name}; subliminal fade={fade_seconds:.1f}s",
                )
            else:
                self._return_pending_vocal()
                for player in self.players:
                    player.stop()
                self._pending_name = None
                self._reschedule()
        elif name is not None:
            if self.active:
                spec = self.state.get()
                fade_seconds = self._subliminal_transition_seconds(spec)
                self._replacement_name = name
                self._scheduled_stop = False
                self._stopping = True
                self._source_completed = False
                self.field.begin_final_retreat(fade_seconds)
                self._journal(
                    "MEDITATION_SWITCH_REQUEST",
                    f"{self.active_name} -> generated ceremony starting with {name}; "
                    f"subliminal fade={fade_seconds:.1f}s first",
                )
            else:
                self._prepare(name, manual=True, due_sample=self.elapsed_samples)

    def _remove_specific_from_pool(self, name: str) -> None:
        self._performance_bags[self._performance_category(name)].take_specific(name)
        self._journal("MEDITATION_CURATION_MANUAL", f"explicit starting passage={name}")

    def _prepare(self, requested_name: str | None, *, manual: bool, due_sample: int) -> None:
        self._return_pending_vocal()
        if not self.performance_registry:
            self._last_error = f"No ceremony clips available for style {self.style_name}"
            self.current_status = self._last_error
            return

        has_vocals = any(not self._is_instrument(n) for n in self.performance_registry)
        if requested_name in (None, self.RANDOM_START_TOKEN) or (
            has_vocals and self._is_instrument(requested_name)
        ):
            name = self._take_vocal(first_passage=True)
        else:
            name = requested_name
            self._remove_specific_from_pool(name)
        if not name:
            return

        self._pending_name = name
        self._pending_manual = manual
        self._pending_due_sample = due_sample
        self._last_error = ""
        self.players[self._current_player_index].prepare(self.recording_paths[name])
        self.current_status = f"preparing generated ceremony; first passage {name}"
        self._journal(
            "MEDITATION_PREPARE",
            f"generated ceremony; first={name}; file={self.recording_paths[name].name}",
        )

    def _prepare_next_vocal(self) -> None:
        if not self.active or self.next_vocal_name or not self.performance_registry:
            return
        name = self._take_vocal(avoid_name=self.current_vocal_name)
        if not name:
            return
        self.next_vocal_name = name
        self._next_crossfade_seconds = self._choose_crossfade_seconds()
        self.next_recording_player.prepare(self.recording_paths[name])
        self._journal(
            "MEDITATION_VOCAL_PRELOAD",
            f"current={self.current_vocal_name}; next={name}; "
            f"crossfade={self._next_crossfade_seconds:.2f}s",
        )

    def _begin(self, offset: int) -> None:
        first_name = self._pending_name
        due = self._pending_due_sample
        manual = self._pending_manual
        self._pending_name = None
        self.current_vocal_name = first_name or ""
        self.active_name = f"{self.style_name} generated ceremony"
        self.performance_count += 1
        self._next_start_sample = math.inf
        self._stopping = False
        self._scheduled_stop = False
        self._source_completed = False
        self._audio_mix = 0.0
        self._mix = 0.0
        self._crossfade_active = False
        self._crossfade_elapsed_samples = 0
        self._crossfade_total_samples = 0
        self._ceremony_elapsed_samples = 0
        spec = self.state.get()
        target_seconds = self._choose_ceremony_seconds(spec)
        self._ceremony_target_samples = max(
            round(target_seconds * self.sample_rate),
            round(spec.transition_seconds * self.sample_rate) + 1,
        )
        retreat_seconds = min(
            self._final_retreat_seconds(spec),
            max(spec.transition_seconds, target_seconds * 0.35),
        )
        self._ceremony_retreat_samples = round(retreat_seconds * self.sample_rate)
        self._ceremony_retreat_started = False
        self._passage_gain = 1.0
        self._handoff_out = False
        self._selection_settings = (spec.instruments_enabled, spec.instrument_selection_percent)
        self._modulation_bed_weight = 1.0
        self.field.reset_for_ceremony(spec)
        self._current_underrun_snapshot = self.recording_player.underrun_count
        self._next_underrun_snapshot = self.next_recording_player.underrun_count
        self._journal("MEDITATION_START", (
            f"{self.active_name}; target={target_seconds / 60.0:.2f} min; "
            f"first passage={self.current_vocal_name}; "
            f"MP3={self.recording_paths[self.current_vocal_name].name}; "
            f"final retreat={retreat_seconds:.1f}s; stereo 3D pair; "
            f"subliminal emergence={spec.subliminal_emergence_seconds:.1f}s; "
            "continuous spatial field; shared evolving brown bed"
        ), offset)
        if not manual and self.elapsed_samples + offset > due + self.renderer.frame_size:
            self._journal("MEDITATION_WARNING", (
                f"generated ceremony start delayed "
                f"{(self.elapsed_samples + offset - due) / self.sample_rate:.3f}s"
            ), offset)
        self._prepare_next_vocal()
        self._last_status_sample = -self.sample_rate

    def _finish(self, offset: int) -> None:
        category = "MEDITATION_COMPLETE" if self._source_completed else "MEDITATION_STOP"
        self._journal(
            category,
            f"{self.active_name}; elapsed={self.ceremony_elapsed_seconds / 60.0:.2f} min; "
            "bed returned to normal",
            offset,
        )
        # A next vocal may have been preloaded but never entered before the
        # final retreat. Return it to the front of the pool rather than silently
        # consuming a source the listener never heard.
        self._return_reserved_vocal()
        self.recording_player.stop()
        self.active_name = ""
        self.current_vocal_name = ""
        self._stopping = False
        self._scheduled_stop = False
        self._source_completed = False
        self._crossfade_active = False
        self._audio_mix = self._mix = 0.0
        self._ceremony_elapsed_samples = 0
        self._ceremony_target_samples = 0
        self._ceremony_retreat_started = False
        self._reschedule(from_sample=self.elapsed_samples + offset)
        if self._replacement_name is not None:
            replacement = self._replacement_name
            self._replacement_name = None
            self._prepare(replacement, manual=True, due_sample=self.elapsed_samples + offset)
        self._last_status_sample = -self.sample_rate

    def _handle_first_error(self, message: str) -> None:
        self._journal("MEDITATION_ERROR", message)
        self._last_error = message
        if self.export_mode:
            raise RuntimeError(message)
        self._pending_name = None
        self.recording_player.stop()
        if self.active:
            self._stopping = True
            self._source_completed = False
        else:
            self._reschedule()
        self.current_status = f"recording error: {message}"

    def _drop_failed_next(self, message: str) -> None:
        self._journal("MEDITATION_VOCAL_ERROR", message)
        failed = self.next_vocal_name
        self.next_recording_player.stop()
        self.next_vocal_name = ""
        if failed:
            self._journal("MEDITATION_VOCAL_SKIPPED", failed)
        if self.export_mode:
            raise RuntimeError(message)
        self._prepare_next_vocal()

    def _ramp(self, current: float, target: float, count: int,
              seconds: float) -> tuple[np.ndarray, float]:
        if count == 0:
            return np.empty(0, dtype=np.float32), current
        step = 1.0 / max(1.0, seconds * self.sample_rate)
        direction = 1.0 if target > current else -1.0
        values = current + direction * step * np.arange(count, dtype=np.float64)
        end = current + direction * step * count
        if direction > 0.0:
            values = np.minimum(values, target)
            end = min(end, target)
        else:
            values = np.maximum(values, target)
            end = max(end, target)
        return values.astype(np.float32), float(end)

    def _next_ready(self) -> bool:
        if not self.next_vocal_name:
            return False
        try:
            return self.next_recording_player.ready(
                wait=self.export_mode,
                cancel_event=self.cancel_event,
            )
        except InterruptedError:
            raise
        except Exception as exc:
            self._drop_failed_next(str(exc))
            return False

    def _should_crossfade(self) -> bool:
        if self._stopping or self._crossfade_active or not self.next_vocal_name:
            return False
        if self.ceremony_remaining_seconds <= self._next_crossfade_seconds + 2.0:
            return False
        return self.recording_player.remaining_seconds <= self._next_crossfade_seconds + 0.25

    def _begin_crossfade(self) -> bool:
        if not self._next_ready():
            return False
        self._crossfade_active = True
        self._crossfade_elapsed_samples = 0
        self._crossfade_total_samples = max(
            1, round(self._next_crossfade_seconds * self.sample_rate)
        )
        self._journal(
            "MEDITATION_CROSSFADE_START",
            f"{self.current_vocal_name} -> {self.next_vocal_name}; "
            f"duration={self._next_crossfade_seconds:.2f}s; "
            f"ceremony={self.ceremony_elapsed_seconds / 60.0:.2f}/"
            f"{self.ceremony_target_seconds / 60.0:.2f}min",
        )
        return True

    def _complete_crossfade(self) -> None:
        outgoing_name = self.current_vocal_name
        incoming_name = self.next_vocal_name
        outgoing_player = self.recording_player
        outgoing_player.stop()
        self._current_player_index, self._next_player_index = (
            self._next_player_index, self._current_player_index
        )
        self.current_vocal_name = incoming_name
        self.next_vocal_name = ""
        self._crossfade_active = False
        self._crossfade_elapsed_samples = 0
        self._crossfade_total_samples = 0
        self._journal(
            "MEDITATION_CROSSFADE_COMPLETE",
            f"{outgoing_name} -> {self.current_vocal_name}; "
            f"spatial field continuous at {self.field.current_distance:.2f}m",
        )
        self._prepare_next_vocal()

    def _render_player(self, player: CeremonyRecordingPlayer, count: int) -> np.ndarray:
        return player.render(
            count,
            wait=self.export_mode,
            cancel_event=self.cancel_event,
        )

    def _render_vocals(self, count: int) -> np.ndarray:
        """Render current clip, with an equal-power handoff when due."""
        spec = self.state.get()
        selection = (spec.instruments_enabled, spec.instrument_selection_percent)
        if selection != self._selection_settings and not self._crossfade_active:
            self._return_reserved_vocal()
            self._selection_settings = selection
            self._prepare_next_vocal()

        different = bool(self.next_vocal_name) and (
            self._is_instrument(self.current_vocal_name) != self._is_instrument(self.next_vocal_name)
        )
        disabled_current = self._is_instrument(self.current_vocal_name) and not spec.instruments_enabled
        if not self._stopping and not self._crossfade_active and (
            self._should_crossfade() or disabled_current
        ):
            if different or disabled_current:
                if self._next_ready() or disabled_current:
                    self._handoff_out = True
            else:
                self._begin_crossfade()

        if self._handoff_out and not disabled_current and not self.next_vocal_name:
            self._handoff_out = False
        if self._handoff_out or self._passage_gain < 1.0:
            try:
                audio = self._render_player(self.recording_player, count)
            except InterruptedError:
                raise
            except Exception as exc:
                self._handle_first_error(str(exc))
                return np.zeros((count, 2), dtype=np.float32)
            curve, self._passage_gain = self._ramp(
                self._passage_gain, 0.0 if self._handoff_out else 1.0,
                count, 0.05 if disabled_current else 4.0,
            )
            audio *= curve[:, None]
            if self._handoff_out and self._passage_gain <= 0.0:
                if self._next_ready():
                    # Only the outgoing player was rendered in this block.
                    # Incoming playback begins next block, starting at silence.
                    self._complete_crossfade()
                    self._handoff_out = False
            return audio

        if self._crossfade_active:
            try:
                outgoing = self._render_player(self.recording_player, count)
                incoming = self._render_player(self.next_recording_player, count)
            except InterruptedError:
                raise
            except Exception as exc:
                self._handle_first_error(str(exc))
                return np.zeros((count, 2), dtype=np.float32)

            start = self._crossfade_elapsed_samples
            positions = (
                start + np.arange(count, dtype=np.float64)
            ) / max(1, self._crossfade_total_samples)
            positions = np.clip(positions, 0.0, 1.0)
            out_gain = np.cos(positions * math.pi * 0.5)
            in_gain = np.sin(positions * math.pi * 0.5)
            mixed = (
                outgoing.astype(np.float64) * out_gain[:, None]
                + incoming.astype(np.float64) * in_gain[:, None]
            ).astype(np.float32)
            self._crossfade_elapsed_samples += count

            if (
                self._crossfade_elapsed_samples >= self._crossfade_total_samples
                or self.recording_player.complete
            ):
                self._complete_crossfade()
            return mixed

        try:
            audio = self._render_player(self.recording_player, count)
        except InterruptedError:
            raise
        except Exception as exc:
            self._handle_first_error(str(exc))
            return np.zeros((count, 2), dtype=np.float32)

        if self.recording_player.complete and not self._stopping:
            # A well-preloaded source should normally have crossed before EOF.
            # If duration metadata was inaccurate, perform a graceful emergency
            # handoff instead of ending the whole ceremony.
            if self.ceremony_remaining_seconds > 2.0 and self._next_ready():
                self._journal(
                    "MEDITATION_CROSSFADE_LATE",
                    f"{self.current_vocal_name} reached EOF before planned overlap; "
                    f"continuing with {self.next_vocal_name}",
                )
                self._complete_crossfade()
                self._passage_gain = 0.0
            elif self.ceremony_remaining_seconds > 2.0:
                self._prepare_next_vocal()
        return audio

    def _update_ceremony_timeline(self, spec: MeditationSpec) -> None:
        remaining = self.ceremony_remaining_seconds
        fade_seconds = self._subliminal_transition_seconds(spec)
        if not self._ceremony_retreat_started:
            retreat_seconds = self._ceremony_retreat_samples / self.sample_rate
            if remaining <= retreat_seconds:
                self._ceremony_retreat_started = True
                self.field.begin_final_retreat(max(1.0, remaining))
                self._journal(
                    "MEDITATION_RETREAT_START",
                    f"remaining={remaining:.1f}s; vocal={self.current_vocal_name}; "
                    f"distance={self.field.current_distance:.2f}m",
                )

        if not self._stopping and remaining <= fade_seconds:
            self._stopping = True
            self._scheduled_stop = True
            self._source_completed = True
            self._journal(
                "MEDITATION_FINAL_FADE",
                f"remaining={remaining:.1f}s; subliminal fade={fade_seconds:.1f}s",
            )

    def _advance_distance_drift(self, normalized_time: float) -> float:
        """Random destinations and travel times, with smooth starts/stops and rests."""
        remaining = max(0.0, normalized_time)
        while remaining > 0.0:
            available = self._distance_drift_span - self._distance_drift_progress
            if remaining < available:
                self._distance_drift_progress += remaining
                break
            remaining -= available
            self._distance_drift_start = self._distance_drift_target
            self._distance_drift_progress = 0.0
            if not self._distance_drift_holding and self._distance_drift_rng.random() < 0.2:
                # Occasionally linger at a destination before moving again.
                self._distance_drift_holding = True
                self._distance_drift_span = float(self._distance_drift_rng.uniform(0.05, 0.2))
            else:
                self._distance_drift_holding = False
                self._distance_drift_target = float(self._distance_drift_rng.uniform(-1.0, 1.0))
                self._distance_drift_span = float(self._distance_drift_rng.uniform(0.3, 0.75))
        t = self._distance_drift_progress / self._distance_drift_span
        # Quintic easing has zero velocity and acceleration at both ends.
        eased = t * t * t * (t * (6.0 * t - 15.0) + 10.0)
        return self._distance_drift_start + (
            self._distance_drift_target - self._distance_drift_start
        ) * eased

    def _modulate_ceremony_positions(self, positions, count: int, spec: MeditationSpec):
        """Slow radial variation without changing source directions."""
        dt = count / self.sample_rate
        blend = -math.expm1(-dt / 3.0)
        self._distance_modulation_depth += (
            spec.distance_modulation_percent / 100.0 - self._distance_modulation_depth
        ) * blend
        self._distance_modulation_period += (
            spec.distance_modulation_seconds - self._distance_modulation_period
        ) * blend
        wave = self._advance_distance_drift(dt / self._distance_modulation_period)
        # Ease modulation into a new ceremony; keep its entrance/exit envelope intact.
        ease = min(1.0, self._ceremony_elapsed_samples / (30.0 * self.sample_rate))
        ease = ease * ease * (3.0 - 2.0 * ease)
        offset = self._distance_modulation_depth * wave * ease
        self.current_distance_modulation_percent = 100.0 * offset
        result = []
        for position in positions:
            radius = math.sqrt(position.x ** 2 + position.y ** 2 + position.z ** 2)
            # Never enter the renderer's near-field amplification region.
            scale = max(1.0 + offset, 2.0 / radius) if radius > 0.0 else 1.0
            result.append(Vector3(position.x * scale, position.y * scale, position.z * scale))
        return tuple(result)

    def _fill_modulated_bed(self, baseline_gains: np.ndarray) -> np.ndarray:
        """Release bed attenuation as outward modulation reduces ceremony presence."""
        outward_scale = 1.0 + max(0.0, self.current_distance_modulation_percent / 100.0)
        target_weight = 1.0 / (outward_scale * outward_scale)
        weights = np.linspace(
            self._modulation_bed_weight, target_weight,
            len(baseline_gains), endpoint=False, dtype=np.float64,
        )
        self._modulation_bed_weight = target_weight
        # Crossfade from the existing ceremony bed toward its normal gain (1).
        # Entrance/exit remain intact: an already-restored bed stays at unity.
        return (baseline_gains + (1.0 - baseline_gains) * (1.0 - weights)).astype(np.float32)

    def _render_piece(self, count: int, spec: MeditationSpec):
        self._consume_commands()
        start_offset = 0

        if not self.active:
            if self._pending_name and not self._pending_manual and not spec.enabled:
                self._return_pending_vocal()
            if self._pending_name is None and spec.enabled and self.performance_registry:
                if self._next_start_sample < self.elapsed_samples + count + 5 * self.sample_rate:
                    self._prepare(
                        self.RANDOM_START_TOKEN,
                        manual=False,
                        due_sample=int(self._next_start_sample),
                    )
            if self._pending_name is not None:
                due = self._pending_due_sample
                if due < self.elapsed_samples + count and not getattr(self, "defer_start", False):
                    try:
                        ready = self.recording_player.ready(
                            wait=self.export_mode,
                            cancel_event=self.cancel_event,
                        )
                    except InterruptedError:
                        raise
                    except Exception as exc:
                        self._handle_first_error(str(exc))
                        ready = False
                    if ready:
                        start_offset = max(0, int(due - self.elapsed_samples))
                        self._begin(start_offset)

        audio = np.zeros((count, 2), dtype=np.float32)
        mix = np.zeros(count, dtype=np.float32)
        gains = np.ones(count, dtype=np.float32)
        was_active = self.active

        if self.active:
            active_count = count - start_offset
            if active_count > 0:
                self._update_ceremony_timeline(spec)
                fade_seconds = self._subliminal_transition_seconds(spec)
                instrument_passage = self._is_instrument(self.current_vocal_name)
                vocals = self._render_vocals(active_count)
                audio[start_offset:] = vocals

                mix[start_offset:], self._mix = self._ramp(
                    self._mix,
                    0.0 if self._stopping else 1.0,
                    active_count,
                    fade_seconds if self._stopping else spec.transition_seconds,
                )

                # The first clip needs only click protection; its real emergence
                # is produced by distance, presence gain, and brown-bed coupling.
                # Final/manual fades use the full transition duration so the
                # vocal does not disappear before the spatial retreat completes.
                audio_fade_seconds = (
                    fade_seconds if self._stopping else 1.0
                )
                audio_curve, self._audio_mix = self._ramp(
                    self._audio_mix,
                    0.0 if self._stopping else 1.0,
                    active_count,
                    audio_fade_seconds,
                )
                audio[start_offset:] *= audio_curve[:, None]

                # Human hearing recognizes coherent vocals at remarkably low
                # levels.  Give every ceremony a distinct pre-emergence veil
                # before the existing spatial approach becomes fully audible.
                # The fourth-power curve deliberately spends most of the
                # selected time close to silence, then hands off smoothly at
                # unity to the normal presence journey.  Apply the same veil
                # to brown-bed attenuation below so a sudden change in the bed
                # cannot announce that the ceremony has started.
                subliminal_samples = round(
                    spec.subliminal_emergence_seconds * self.sample_rate
                )
                if subliminal_samples > 0:
                    emergence_position = (
                        self._ceremony_elapsed_samples
                        + np.arange(active_count, dtype=np.float64)
                    ) / subliminal_samples
                    emergence = np.clip(emergence_position, 0.0, 1.0)
                    emergence = emergence ** 4
                else:
                    emergence = np.ones(active_count, dtype=np.float64)
                audio[start_offset:] *= emergence[:, None]

                proximity0 = self.field.proximity_state
                db0, db1 = self.field.advance(active_count / self.sample_rate, spec)
                proximity1 = self.field.proximity_state

                attenuation = np.linspace(
                    db0, db1, active_count, endpoint=False, dtype=np.float64
                )
                effective_mix = mix[start_offset:] * emergence
                gains[start_offset:] = np.power(
                    10.0,
                    attenuation * effective_mix / 20.0,
                ).astype(np.float32)

                proximity = np.linspace(
                    proximity0, proximity1, active_count,
                    endpoint=False, dtype=np.float64,
                )
                proximity = np.clip(proximity, 0.0, 1.0)
                smooth_proximity = proximity ** 3 * (
                    proximity * (proximity * 6.0 - 15.0) + 10.0
                )
                presence_gain_db = -8.0 * (1.0 - smooth_proximity)
                presence_gain = np.power(10.0, presence_gain_db / 20.0)

                positions = self.field.stereo_positions()
                if instrument_passage:
                    # Stable front-centered stereo image for a solo instrument.
                    positions = (Vector3(-0.5, 0.0, -3.0), Vector3(0.5, 0.0, -3.0))
                    presence_gain = np.ones(active_count, dtype=np.float64)
                    presence_gain_db = np.zeros(active_count, dtype=np.float64)
                positions = self._modulate_ceremony_positions(positions, active_count, spec)
                gains[start_offset:] = self._fill_modulated_bed(gains[start_offset:])
                for source, position in zip(self.sources, positions):
                    source.set_position_vector(position)
                source_audio = audio[start_offset:]
                spatial = (
                    0.5 * self.sources[0].process_mono(
                        np.ascontiguousarray(source_audio[:, 0])
                    )
                    + 0.5 * self.sources[1].process_mono(
                        np.ascontiguousarray(source_audio[:, 1])
                    )
                )

                end_level = self._level_db + (
                    spec.performance_level_db - self._level_db
                ) * (1.0 - math.exp(-(active_count / self.sample_rate) / 0.2))
                level = np.power(
                    10.0,
                    np.linspace(
                        self._level_db, end_level, active_count, endpoint=False
                    ) / 20.0,
                )
                self._level_db = float(end_level)
                self.current_presence_gain_db = float(presence_gain_db[-1])
                audio[start_offset:] = (
                    spatial * (level * presence_gain)[:, None]
                ).astype(np.float32)

                self._ceremony_elapsed_samples += active_count

                # If a scheduled fade reached the target, or a manual fade has
                # completed, close the ceremony and only then begin the rest.
                if self._stopping and self._mix <= 0.0 and self._audio_mix <= 0.0:
                    self._finish(count)
                elif (
                    self._scheduled_stop
                    and self._ceremony_elapsed_samples >= self._ceremony_target_samples
                ):
                    # Rounding or an unusually short transition can leave a
                    # microscopic tail. Force the next block to finish cleanly.
                    self._mix = min(self._mix, 1.0e-6)
                    self._audio_mix = min(self._audio_mix, 1.0e-6)

        self.current_brown_gain_db = 20.0 * math.log10(
            max(1e-12, float(gains[-1]))
        )
        self.elapsed_samples += count
        self._refresh_status(spec)
        return audio, mix, gains, was_active or bool(np.any(mix > 0.0))

    def _refresh_status(self, spec: MeditationSpec) -> None:
        if self.elapsed_samples - self._last_status_sample < self.sample_rate:
            return
        self._last_status_sample = self.elapsed_samples

        if self.active:
            if self._stopping:
                phase = "final fade" if self._scheduled_stop else "manual fade"
            elif self._ceremony_retreat_started:
                phase = "retreating into brown bed"
            elif self._crossfade_active:
                phase = "crossfading passages"
            else:
                phase = "instrument passage" if self._is_instrument(self.current_vocal_name) else "vocal passage"

            crossfade_text = ""
            if self._crossfade_active:
                progress = min(
                    1.0,
                    self._crossfade_elapsed_samples / max(1, self._crossfade_total_samples),
                )
                crossfade_text = f"; crossfade {100.0 * progress:.0f}%"

            self.current_status = (
                f"{self.active_name}: {phase}; "
                f"{self.ceremony_elapsed_seconds / 60.0:.1f}/"
                f"{self.ceremony_target_seconds / 60.0:.1f} min; "
                f"passage {self.current_vocal_name or 'none'}; "
                f"next {self.next_vocal_name or 'none'}{crossfade_text}"
            )

            if self.elapsed_samples - self._last_field_log_sample >= 30 * self.sample_rate:
                self._last_field_log_sample = self.elapsed_samples
                p = self.field.current_position
                current_underruns = self.recording_player.underrun_count
                next_underruns = self.next_recording_player.underrun_count
                self._journal("MEDITATION_FIELD", (
                    f"{self.active_name}; vocal={self.current_vocal_name}; "
                    f"next={self.next_vocal_name or 'none'}; "
                    f"ceremony={self.ceremony_elapsed_seconds / 60.0:.2f}/"
                    f"{self.ceremony_target_seconds / 60.0:.2f}min; "
                    f"brown={self.current_brown_gain_db:.2f}dB; "
                    f"distance={self.field.current_distance:.3f}m; "
                    f"presenceGain={self.current_presence_gain_db:.2f}dB; "
                    f"xyz=({p.x:.3f},{p.y:.3f},{p.z:.3f}); "
                    f"coupling={spec.brown_coupling_percent:.0f}%; "
                    f"presence={self.field.presence_phase}; "
                    f"decoder_underruns=current:{current_underruns},next:{next_underruns}"
                ))
        elif self._pending_name:
            self.current_status = (
                f"preparing generated ceremony; first passage {self._pending_name}"
            )
        elif self._last_error:
            self.current_status = f"recording error: {self._last_error}"
        elif not spec.enabled:
            self.current_status = "automatic ceremonies disabled; manual generated ceremony available"
        elif not self.performance_registry:
            self.current_status = f"no ceremony clips found for style {self.style_name}"
        else:
            wait = self.next_performance_seconds
            if math.isfinite(wait):
                self.current_status = (
                    f"waiting; next generated ceremony in {wait / 60.0:.1f} min; "
                    f"{len(self.performance_registry)} ceremony clips in style"
                )
            else:
                self.current_status = "waiting"

    def generate(self, frame_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._closed:
            raise RuntimeError("Meditation orchestrator is closed")
        audio = np.zeros((frame_count, 2), dtype=np.float32)
        presence = np.zeros(frame_count, dtype=np.float32)
        brown_gain = np.ones(frame_count, dtype=np.float32)
        self.blocks_motifs = False
        for offset in range(0, frame_count, self.renderer.frame_size):
            count = min(self.renderer.frame_size, frame_count - offset)
            block, mix, gain, occupied = self._render_piece(count, self.state.get())
            # Preserve occupancy even when the ceremony ends within this block.
            self.blocks_motifs = self.blocks_motifs or occupied
            audio[offset:offset + count] = block
            presence[offset:offset + count] = mix
            brown_gain[offset:offset + count] = gain
        return audio, presence, brown_gain

    def close(self) -> None:
        if self._closed:
            return
        for player in self.players:
            player.close()
        for source in self.sources:
            source.close()
        self._closed = True


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
        base_voice_a: BrownNoiseInstance,
        base_voice_b: BrownNoiseInstance,
        mode_state: ModeState,
        noise_state: BrownNoiseState,
        noise_evolution_state: BrownNoiseEvolutionState,
        body_movement_state: BodyMovementState,
        heartbeat_state: HeartbeatState,
        breath_state: BreathState,
        breath_evolution_state: BreathEvolutionState,
        motion_state: OrganicMotionState,
        brown_motion_spec: DualBrownMotionSpec,
        heartbeat_spatial_spec: HeartbeatSpatialSpec,
        metabolism_spec: MetabolismSpec,
        dream_motif_spatial_spec: DreamMotifSpatialSpec,
        meditation_spec: MeditationSpec,
        sound_effects_directory: Path,
        mixer_spec: MixerSpec,
    ) -> None:
        self.sample_rate = float(sample_rate)
        self.common = common
        self.independent_left = independent_left
        self.independent_right = independent_right
        self.base_voice_a = base_voice_a
        self.base_voice_b = base_voice_b
        self.base_fluid_stereo = BaseBrownFluidStereo()
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
        self.heartbeat_prominence_limiter = (
            HeartbeatProminenceLimiter()
        )
        self.current_heartbeat_requested_level_db = (
            heartbeat_spatial_spec.level_db
        )
        self.current_heartbeat_effective_level_db = (
            heartbeat_spatial_spec.level_db
        )
        self.current_heartbeat_prominence_state = (
            HeartbeatProminenceLimiter.STATE_IDLE
        )
        self.active_ambients_directory = Path(sound_effects_directory)
        self.active_style_directory = self.active_ambients_directory.parent
        self.active_style_name = self.active_style_directory.name

        self.dream_motif_spatial_state = (
            DreamMotifSpatialState(dream_motif_spatial_spec)
        )
        self.dream_motif_3d = DreamMotif3DEngine(
            sample_rate=int(self.sample_rate),
            renderer=self.spatial_renderer,
            root_directory=self.active_ambients_directory,
            state=self.dream_motif_spatial_state,
        )

        self.meditation_state = (
            MeditationState(
                meditation_spec
            )
        )
        self.meditation = (
            MeditationOrchestrator(
                sample_rate=self.sample_rate,
                renderer=self.spatial_renderer,
                state=self.meditation_state,
                style_directory=self.active_style_directory,
                seed=self.common.seed + 8_230_601,
            )
        )
        self.current_meditation_mix = 0.0

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

        self.current_correlation = 0.536
        self.current_base_flow = 0.0
        self.current_base_eddy = 0.0
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

    def set_active_style(self, style_name: str) -> None:
        style_name = resolve_style_name(style_name)
        new_style_directory = style_directory(style_name)
        new_ambients_directory = new_style_directory / STYLE_AMBIENTS_FOLDER
        if (
            style_name == self.active_style_name
            and new_ambients_directory == self.active_ambients_directory
        ):
            return

        self.active_style_name = style_name
        self.active_style_directory = new_style_directory
        self.active_ambients_directory = new_ambients_directory
        self.dream_motif_3d.set_root_directory(new_ambients_directory)
        self.meditation.set_style_directory(new_style_directory)

    def close(self) -> None:
        self.dream_motif_3d.close()
        self.meditation.close()
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

    def _base_personality_specs(
        self,
        center: BrownNoiseSpec,
    ) -> tuple[BrownNoiseSpec, BrownNoiseSpec]:
        """Return two clearly different but still brown-noise personalities.

        Both voices inherit the slowly evolving global center. Voice A is the
        heavier/darker member; Voice B is the leaner/brighter member. The
        offsets are intentionally large enough to survive an A/B listening
        test, while remaining inside the already validated brown-noise ranges.
        """
        voice_a = replace(
            center,
            body=float(np.clip(center.body - 0.11, 0.15, 1.0)),
            slope_strength=float(
                np.clip(center.slope_strength + 0.055, 0.75, 1.0)
            ),
            low_end_emphasis_db=float(
                np.clip(center.low_end_emphasis_db + 1.8, 0.0, 8.0)
            ),
            upper_texture=float(
                np.clip(center.upper_texture - 0.18, 0.0, 1.0)
            ),
        ).validated(self.sample_rate)

        voice_b = replace(
            center,
            body=float(np.clip(center.body + 0.11, 0.15, 1.0)),
            slope_strength=float(
                np.clip(center.slope_strength - 0.055, 0.75, 1.0)
            ),
            low_end_emphasis_db=float(
                np.clip(center.low_end_emphasis_db - 1.8, 0.0, 8.0)
            ),
            upper_texture=float(
                np.clip(center.upper_texture + 0.18, 0.0, 1.0)
            ),
        ).validated(self.sample_rate)

        return voice_a, voice_b

    @staticmethod
    def _lerp(a: float, b: float, amount: float) -> float:
        return float(a + (b - a) * amount)

    def _meditation_rest_metabolism(
        self,
        values: MetabolismValues | None,
        amount: float,
    ) -> MetabolismValues:
        """
        Pull the living system toward a deliberately restful state while a
        recorded meditation is foregrounded.

        The original metabolism keeps running underneath, so when the ceremony
        ends the system smoothly rejoins wherever its long-form journey has
        reached rather than restarting.
        """
        amount = float(np.clip(amount, 0.0, 1.0))

        if values is None:
            values = MetabolismValues(
                activity=0.20,
                activity_drive=0.08,
                brown_body=0.42,
                brown_slope=0.86,
                brown_low_end_db=4.5,
                brown_texture=0.22,
                breath_prominence=0.28,
                breath_tempo=1.80,
                breath_gain_db=2.0,
                breath_spectral_depth=0.10,
                breath_width_depth=0.06,
                heartbeat_distance=3.2,
                heartbeat_level_db=2.5,
                brown_3d_amount=0.24,
                brown_radius=2.5,
                brown_center_distance=1.8,
                brown_evolution=0.12,
            )

        rest = MetabolismValues(
            activity=0.08,
            activity_drive=0.015,
            brown_body=0.34,
            brown_slope=0.84,
            brown_low_end_db=5.6,
            brown_texture=0.12,
            breath_prominence=0.20,
            breath_tempo=2.10,
            breath_gain_db=1.6,
            breath_spectral_depth=0.07,
            breath_width_depth=0.04,
            heartbeat_distance=3.55,
            heartbeat_level_db=1.5,
            brown_3d_amount=0.20,
            brown_radius=2.2,
            brown_center_distance=2.1,
            brown_evolution=0.10,
        )

        return MetabolismValues(
            activity=self._lerp(
                values.activity, rest.activity, amount
            ),
            activity_drive=self._lerp(
                values.activity_drive, rest.activity_drive, amount
            ),
            brown_body=self._lerp(
                values.brown_body, rest.brown_body, amount
            ),
            brown_slope=self._lerp(
                values.brown_slope, rest.brown_slope, amount
            ),
            brown_low_end_db=self._lerp(
                values.brown_low_end_db,
                rest.brown_low_end_db,
                amount,
            ),
            brown_texture=self._lerp(
                values.brown_texture, rest.brown_texture, amount
            ),
            breath_prominence=self._lerp(
                values.breath_prominence,
                rest.breath_prominence,
                amount,
            ),
            breath_tempo=self._lerp(
                values.breath_tempo, rest.breath_tempo, amount
            ),
            breath_gain_db=self._lerp(
                values.breath_gain_db, rest.breath_gain_db, amount
            ),
            breath_spectral_depth=self._lerp(
                values.breath_spectral_depth,
                rest.breath_spectral_depth,
                amount,
            ),
            breath_width_depth=self._lerp(
                values.breath_width_depth,
                rest.breath_width_depth,
                amount,
            ),
            heartbeat_distance=self._lerp(
                values.heartbeat_distance,
                rest.heartbeat_distance,
                amount,
            ),
            heartbeat_level_db=self._lerp(
                values.heartbeat_level_db,
                rest.heartbeat_level_db,
                amount,
            ),
            brown_3d_amount=self._lerp(
                values.brown_3d_amount,
                rest.brown_3d_amount,
                amount,
            ),
            brown_radius=self._lerp(
                values.brown_radius, rest.brown_radius, amount
            ),
            brown_center_distance=self._lerp(
                values.brown_center_distance,
                rest.brown_center_distance,
                amount,
            ),
            brown_evolution=self._lerp(
                values.brown_evolution,
                rest.brown_evolution,
                amount,
            ),
        )

    def request_start_meditation(self, name: str) -> None:
        self.meditation.request_start(name)

    def request_stop_meditation(self) -> None:
        self.meditation.request_stop()

    def generate(self, frame_count: int) -> np.ndarray:
        modes = self.mode_state.get()
        elapsed_seconds = frame_count / self.sample_rate

        # A due ceremony waits for the full ambient spatial exit. This is a
        # handoff, not an overlap or a mid-clip mute.
        self.meditation.defer_start = self.dream_motif_3d.active_audio is not None

        # One shared sample-clock path for live playback and accelerated export.
        # Recording decode runs ahead on its own bounded-memory worker.
        meditation_audio, meditation_curve, brown_ceremony_gain = (
            self.meditation.generate(frame_count)
        )
        self.dream_motif_3d.start_blocked = bool(
            self.meditation._pending_name is not None
            and self.meditation._pending_due_sample <= self.meditation.elapsed_samples
        )
        self.current_meditation_mix = float(meditation_curve[-1])
        meditation_amount = self.current_meditation_mix

        metabolism_values = self.metabolism.advance(elapsed_seconds)

        if meditation_amount > 0.0:
            metabolism_values = self._meditation_rest_metabolism(
                metabolism_values,
                meditation_amount,
            )

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

        self.base_mix = float(base_curve[-1])
        self.stereo_mix = float(stereo_curve[-1])
        self.correlation_mix = float(correlation_curve[-1])
        self.breath_mix = float(breath_curve[-1])
        self.heartbeat_mix = float(heartbeat_curve[-1])

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

        # These existing independent generators remain dedicated to the
        # already-working Steam Audio 3D bodies. Their spectra and motion are
        # deliberately left unchanged by the new 2D foundation experiment.
        left_dark, left_bright = self.independent_left.generate(
            frame_count,
            spec_snapshot=evolved_noise_spec,
        )
        right_dark, right_bright = self.independent_right.generate(
            frame_count,
            spec_snapshot=evolved_noise_spec,
        )

        base_spec_a, base_spec_b = self._base_personality_specs(
            evolved_noise_spec
        )
        base_a_dark, base_a_bright = self.base_voice_a.generate(
            frame_count,
            spec_snapshot=base_spec_a,
        )
        base_b_dark, base_b_bright = self.base_voice_b.generate(
            frame_count,
            spec_snapshot=base_spec_b,
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
        base_voice_a = self._blend(
            base_a_dark,
            base_a_bright,
            spectral_amount,
        )
        base_voice_b = self._blend(
            base_b_dark,
            base_b_bright,
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

        (
            a_left_gain,
            a_right_gain,
            b_left_gain,
            b_right_gain,
        ) = self.base_fluid_stereo.gains(
            frame_count,
            self.sample_rate,
        )
        self.current_base_flow = (
            self.base_fluid_stereo.current_flow
        )
        self.current_base_eddy = (
            self.base_fluid_stereo.current_eddy
        )

        # Two spectrally distinct mono voices now make up the independent
        # portion of the 2D foundation. Their identities physically
        # redistribute between ears instead of remaining hard-wired L/R.
        fluid_left = (
            base_voice_a * a_left_gain
            + base_voice_b * b_left_gain
        )
        fluid_right = (
            base_voice_a * a_right_gain
            + base_voice_b * b_right_gain
        )

        stereo_left = (
            common_gain * common
            + independent_gain * fluid_left
        )
        stereo_right = (
            common_gain * common
            + independent_gain * fluid_right
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

        # The same continuously evolving meditation gain applies to ALL brown
        # sources and, below, the heartbeat. No gong/bowl-specific branching.
        stereo *= brown_ceremony_gain[:, np.newaxis]

        heartbeat = self.heartbeat.generate(frame_count)
        manual_heartbeat_position_spec = (
            self.heartbeat_spatial_state.get()
        )
        if metabolism_values is None:
            heartbeat_position_spec = manual_heartbeat_position_spec
            requested_heartbeat_level_db = (
                heartbeat_position_spec.level_db
            )
            effective_heartbeat_level_db = (
                requested_heartbeat_level_db
            )
            heartbeat_prominence_state = "manual"
        else:
            requested_heartbeat_level_db = (
                metabolism_values.heartbeat_level_db
            )
            (
                effective_heartbeat_level_db,
                effective_heartbeat_distance,
            ) = self.heartbeat_prominence_limiter.advance(
                requested_heartbeat_level_db,
                metabolism_values.heartbeat_distance,
                elapsed_seconds,
            )
            heartbeat_prominence_state = (
                self.heartbeat_prominence_limiter.state
            )
            heartbeat_position_spec = replace(
                manual_heartbeat_position_spec,
                distance=effective_heartbeat_distance,
                level_db=effective_heartbeat_level_db,
            ).validated()

        self.current_heartbeat_requested_level_db = (
            requested_heartbeat_level_db
        )
        self.current_heartbeat_effective_level_db = (
            effective_heartbeat_level_db
        )
        self.current_heartbeat_prominence_state = (
            heartbeat_prominence_state
        )

        heartbeat_level = 10.0 ** (
            effective_heartbeat_level_db / 20.0
        )
        # The heartbeat is part of the underlying Living Brown Noise
        # physiological bed, so it must follow the same ceremony attenuation
        # as the brown-noise field. This keeps it from becoming unnaturally
        # prominent when a ceremony reduces or removes the brown bed.
        active_heartbeat = (
            heartbeat
            * heartbeat_curve
            * heartbeat_level
            * brown_ceremony_gain
        )

        self.current_heartbeat = float(
            self.heartbeat.current_envelope
            * heartbeat_curve[-1]
            * brown_ceremony_gain[-1]
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


        # Suppress every dream-motif path (beds, events, and manual audition)
        # throughout ceremony entrance, performance, and final disappearance.
        # Use block occupancy, not loudness or the final active state.
        stereo += self.dream_motif_3d.generate(
            frame_count,
            enabled=(modes.dream_motifs_enabled and not self.meditation.blocks_motifs),
            metabolism_activity=self.current_metabolism_activity,
        )

        # Already spatialized at the native Steam frame cadence above.
        stereo += meditation_audio


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
    """JSON settings stored beside the Python script."""

    def __init__(self) -> None:
        self.path = SETTINGS_PATH

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
    sound_effects_directory: Path,
    breath_spec: BreathSpec,
    breath_evolution_spec: BreathEvolutionSpec,
    motion_spec: OrganicMotionSpec,
    brown_motion_spec: DualBrownMotionSpec,
    heartbeat_spatial_spec: HeartbeatSpatialSpec,
    metabolism_spec: MetabolismSpec,
    dream_motif_spatial_spec: DreamMotifSpatialSpec,
    meditation_spec: MeditationSpec,
    seed_base: int,
) -> tuple[
    LivingBrownNoiseMixer,
    ModeState,
    BrownNoiseState,
    BrownNoiseEvolutionState,
    BodyMovementState,
    HeartbeatState,
    BreathState,
    BreathEvolutionState,
    OrganicMotionState,
]:
    log_stage("build_mixer: begin")
    build_started = time.perf_counter()

    noise_state = BrownNoiseState(sample_rate, noise_spec)
    noise_evolution_state = BrownNoiseEvolutionState(
        noise_evolution_spec
    )
    body_movement_state = BodyMovementState(body_movement_spec)
    heartbeat_state = HeartbeatState(heartbeat_spec)

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
    base_voice_a = BrownNoiseInstance(
        sample_rate,
        noise_state,
        seed=seed_base + 4,
    )
    base_voice_b = BrownNoiseInstance(
        sample_rate,
        noise_state,
        seed=seed_base + 5,
    )

    mode_state = ModeState(modes)
    breath_state = BreathState(breath_spec)
    breath_evolution_state = BreathEvolutionState(
        breath_evolution_spec
    )
    motion_state = OrganicMotionState(motion_spec)

    log_stage(
        "build_mixer: constructing LivingBrownNoiseMixer "
        "(Steam Audio and dream motifs initialize here)"
    )
    mixer = LivingBrownNoiseMixer(
        sample_rate=sample_rate,
        common=common,
        independent_left=independent_left,
        independent_right=independent_right,
        base_voice_a=base_voice_a,
        base_voice_b=base_voice_b,
        mode_state=mode_state,
        noise_state=noise_state,
        noise_evolution_state=noise_evolution_state,
        body_movement_state=body_movement_state,
        heartbeat_state=heartbeat_state,
        breath_state=breath_state,
        breath_evolution_state=breath_evolution_state,
        motion_state=motion_state,
        brown_motion_spec=brown_motion_spec,
        heartbeat_spatial_spec=heartbeat_spatial_spec,
        metabolism_spec=metabolism_spec,
        dream_motif_spatial_spec=dream_motif_spatial_spec,
        meditation_spec=meditation_spec,
        sound_effects_directory=sound_effects_directory,
        mixer_spec=MixerSpec(),
    )

    log_stage(
        f"build_mixer: complete; "
        f"elapsed={time.perf_counter() - build_started:.3f}s"
    )
    return (
        mixer,
        mode_state,
        noise_state,
        noise_evolution_state,
        body_movement_state,
        heartbeat_state,
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
        sound_effects_directory: Path,
        breath_spec: BreathSpec,
        breath_evolution_spec: BreathEvolutionSpec,
        motion_spec: OrganicMotionSpec,
        brown_motion_spec: DualBrownMotionSpec,
        heartbeat_spatial_spec: HeartbeatSpatialSpec,
        metabolism_spec: MetabolismSpec,
        dream_motif_spatial_spec: DreamMotifSpatialSpec,
        meditation_spec: MeditationSpec,
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
        self.sound_effects_directory = sound_effects_directory
        self.breath_spec = breath_spec
        self.breath_evolution_spec = breath_evolution_spec
        self.motion_spec = motion_spec
        self.brown_motion_spec = brown_motion_spec
        self.heartbeat_spatial_spec = heartbeat_spatial_spec
        self.metabolism_spec = metabolism_spec
        self.dream_motif_spatial_spec = dream_motif_spatial_spec
        self.meditation_spec = meditation_spec
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
            mixer, _, _, _, _, _, _, _, _ = build_mixer(
                sample_rate=self.sample_rate,
                modes=self.modes,
                noise_spec=self.noise_spec,
                noise_evolution_spec=self.noise_evolution_spec,
                body_movement_spec=self.body_movement_spec,
                heartbeat_spec=self.heartbeat_spec,
                sound_effects_directory=self.sound_effects_directory,
                breath_spec=self.breath_spec,
                breath_evolution_spec=self.breath_evolution_spec,
                motion_spec=self.motion_spec,
                brown_motion_spec=self.brown_motion_spec,
                heartbeat_spatial_spec=self.heartbeat_spatial_spec,
                metabolism_spec=self.metabolism_spec,
                dream_motif_spatial_spec=self.dream_motif_spatial_spec,
                meditation_spec=(
                    self.meditation_spec
                ),
                seed_base=seed_base,
            )

            output = Path(self.output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            log_output = output.with_suffix(".txt")

            frames_written = 0
            motif_engine = mixer.dream_motif_3d
            motif_engine.offline = True
            motif_engine.cancel_event = self._cancel_requested
            meditation_engine = mixer.meditation
            meditation_engine.cancel_event = self._cancel_requested
            meditation_engine.configure_export(
                self.duration_minutes * 60.0
            )

            # Encode the generated float stereo buffers directly to MP3.
            # This avoids creating a multi-gigabyte intermediate WAV file and
            # also avoids the classic RIFF/WAV 4 GB size limit.
            mp3_bitrate = 192_000
            container = None

            try:
                container = av.open(
                    str(output),
                    mode="w",
                    format="mp3",
                )

                # Prefer LAME when it is available in the FFmpeg build used by
                # PyAV. Fall back to the generic MP3 encoder name otherwise.
                try:
                    mp3_stream = container.add_stream(
                        "libmp3lame",
                        rate=self.sample_rate,
                    )
                except Exception:
                    mp3_stream = container.add_stream(
                        "mp3",
                        rate=self.sample_rate,
                    )

                mp3_stream.bit_rate = mp3_bitrate

                with log_output.open(
                    "w",
                    encoding="utf-8",
                ) as export_log:
                    export_log.write(
                        "Living Brown Noise — Dream Instigator export log\n"
                    )
                    export_log.write(f"Audio file: {output.name}\n")
                    export_log.write(
                        f"Duration: {self.duration_minutes} minutes\n"
                    )
                    export_log.write(
                        f"Sample rate: {self.sample_rate} Hz\n"
                    )
                    export_log.write(
                        f"Encoding: MP3 stereo, "
                        f"{mp3_bitrate // 1000} kbps\n"
                    )
                    export_log.write(f"Seed: {seed_base}\n")
                    export_log.write(
                        "Timestamps are rendered-audio positions.\n"
                    )
                    export_log.write(
                        "Format: HH:MM:SS.mmm  CATEGORY  MESSAGE\n\n"
                    )

                    while frames_written < total_frames:
                        if self._cancel_requested.is_set():
                            raise InterruptedError

                        remaining_frames = (
                            total_frames - frames_written
                        )

                        # Normal chunks are exact Steam Audio frame multiples.
                        # A short request occurs only once, at the true end of
                        # the complete export, so zero padding can never
                        # contaminate persistent HRTF state between chunks.
                        frame_count = (
                            chunk_frames
                            if remaining_frames > chunk_frames
                            else remaining_frames
                        )

                        audio = mixer.generate(frame_count)

                        for timestamp, category, message in (
                            motif_engine.drain_event_journal()
                        ):
                            export_log.write(
                                f"{motif_engine._format_log_time(timestamp)}  "
                                f"{category:<26}  {message}\n"
                            )
                        for timestamp, category, message in (
                            meditation_engine.drain_event_journal()
                        ):
                            export_log.write(
                                f"{meditation_engine._format_log_time(timestamp)}  "
                                f"{category:<26}  {message}\n"
                            )

                        # PyAV's planar float format is channels x samples.
                        # The mixer already produces float32 stereo, so no
                        # intermediate PCM16/WAV representation is needed.
                        planar = np.ascontiguousarray(
                            np.clip(audio, -1.0, 1.0).T,
                            dtype=np.float32,
                        )
                        frame = av.AudioFrame.from_ndarray(
                            planar,
                            format="fltp",
                            layout="stereo",
                        )
                        frame.sample_rate = self.sample_rate

                        for packet in mp3_stream.encode(frame):
                            container.mux(packet)

                        frames_written += frame_count
                        percent = int(
                            frames_written * 100 / total_frames
                        )
                        self.progress_changed.emit(percent)

                    # Flush delayed MP3 encoder frames.
                    for packet in mp3_stream.encode(None):
                        container.mux(packet)

                    for timestamp, category, message in (
                        motif_engine.drain_event_journal()
                    ):
                        export_log.write(
                            f"{motif_engine._format_log_time(timestamp)}  "
                            f"{category:<26}  {message}\n"
                        )
                    for timestamp, category, message in (
                        meditation_engine.drain_event_journal()
                    ):
                        export_log.write(
                            f"{meditation_engine._format_log_time(timestamp)}  "
                            f"{category:<26}  {message}\n"
                        )
                    if meditation_engine.active:
                        export_log.write(
                            f"{meditation_engine._format_log_time(self.duration_minutes * 60.0)}  "
                            "MEDITATION_EXPORT_END       "
                            f"{meditation_engine.active_name}: export ended while the "
                            "recording or its bed-return fade was still active\n"
                        )
                    export_log.write("\nEND OF EXPORT\n")

            finally:
                if container is not None:
                    container.close()

            self.progress_changed.emit(100)
            self.export_finished.emit(str(output))

        except InterruptedError:
            try:
                output = Path(self.output_path)
                output.unlink(missing_ok=True)
                output.with_suffix(".txt").unlink(missing_ok=True)
            except Exception:
                pass
            self.export_cancelled.emit()

        except Exception as exc:
            try:
                output = Path(self.output_path)
                output.unlink(missing_ok=True)
                output.with_suffix(".txt").unlink(missing_ok=True)
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
        self.breath_state = breath_state
        self.breath_evolution_state = breath_evolution_state
        self.motion_state = motion_state
        self.mixer = mixer
        self.settings_store = settings_store
        self.loaded_settings = loaded_settings
        self.dream_motif_catalog = DreamMotifCatalog(
            root_directory=self.mixer.active_ambients_directory,
            layer_threshold_seconds=(
                DREAM_MOTIF_LAYER_THRESHOLD_SECONDS
            ),
        )
        self.motif_rng = np.random.default_rng(77123)
        self.export_worker: ExportWorker | None = None

        self._conductor_log_lock = threading.Lock()
        self._conductor_log_started = time.time()
        self._last_conductor_snapshot_second = -1
        self._last_logged_callback_error = None
        try:
            CONDUCTOR_LOG_PATH.write_text(
                "Living Brown Noise — conductor diagnostic log\n"
                f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Script: {Path(__file__).resolve()}\n"
                + "=" * 78
                + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass

        self.gui_snapshot_timer = QTimer(self)
        self.gui_snapshot_timer.setSingleShot(True)
        self.gui_snapshot_timer.timeout.connect(
            lambda: self._log_gui_snapshot("settled GUI state")
        )

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

        self.dream_motif_checkbox = QCheckBox(
            "Dream motifs — asynchronous generative 3D sound worlds"
        )
        self.dream_motif_checkbox.setChecked(
            self.mode_state.get().dream_motifs_enabled
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
        self.motif_expand_button.setChecked(
            bool(
                self.loaded_settings.get(
                    "motif_panel_expanded",
                    True,
                )
            )
        )
        self.motif_expand_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.motif_expand_button.setArrowType(
            Qt.ArrowType.DownArrow
            if self.motif_expand_button.isChecked()
            else Qt.ArrowType.RightArrow
        )
        controls_layout.addWidget(self.motif_expand_button)

        self.motif_panel = QWidget()
        self.motif_panel.setVisible(
            self.motif_expand_button.isChecked()
        )
        motif_layout = QVBoxLayout(self.motif_panel)
        motif_layout.setContentsMargins(24, 4, 0, 8)
        motif_layout.setSpacing(4)

        def make_motif_subgroup(
            title: str,
            setting_key: str,
            default_expanded: bool,
        ):
            button = QToolButton()
            button.setText(title)
            button.setCheckable(True)
            button.setChecked(
                bool(
                    self.loaded_settings.get(
                        setting_key,
                        default_expanded,
                    )
                )
            )
            button.setToolButtonStyle(
                Qt.ToolButtonStyle.ToolButtonTextBesideIcon
            )
            button.setArrowType(
                Qt.ArrowType.DownArrow
                if button.isChecked()
                else Qt.ArrowType.RightArrow
            )
            panel = QWidget()
            form = QFormLayout(panel)
            form.setContentsMargins(24, 4, 0, 8)
            panel.setVisible(button.isChecked())
            motif_layout.addWidget(button)
            motif_layout.addWidget(panel)
            return button, panel, form

        (
            self.motif_catalogue_button,
            self.motif_catalogue_panel,
            motif_catalogue_form,
        ) = make_motif_subgroup(
            "Catalogue and live status",
            "motif_catalogue_group_expanded",
            False,
        )

        self.motif_directory_label = QLabel(
            str(self.mixer.active_ambients_directory)
        )
        self.motif_directory_label.setWordWrap(True)
        motif_catalogue_form.addRow("Motif root:", self.motif_directory_label)

        self.motif_combo = QComboBox()
        motif_catalogue_form.addRow("Detected motif:", self.motif_combo)

        self.motif_reload_button = QPushButton(
            "Rescan dream motifs"
        )
        motif_catalogue_form.addRow("", self.motif_reload_button)

        self.motif_summary_label = QLabel(
            "Dream motifs have not been scanned yet."
        )
        self.motif_summary_label.setWordWrap(True)
        motif_catalogue_form.addRow("Catalogue status:", self.motif_summary_label)

        self.motif_detail_label = QLabel("")
        self.motif_detail_label.setWordWrap(True)
        motif_catalogue_form.addRow("Selected motif:", self.motif_detail_label)

        self.motif_playing_label = QLabel("No motif audio active")
        self.motif_playing_label.setWordWrap(True)
        motif_catalogue_form.addRow("Motif playback:", self.motif_playing_label)

        motif_spatial_spec = self.mixer.dream_motif_spatial_state.get()

        (
            self.motif_conductor_button,
            self.motif_conductor_panel,
            motif_conductor_form,
        ) = make_motif_subgroup(
            "Automatic conductor",
            "motif_conductor_group_expanded",
            False,
        )

        self.motif_3d_enabled_checkbox = QCheckBox(
            "Enable single-source ambient playback"
        )
        self.motif_3d_enabled_checkbox.setChecked(
            motif_spatial_spec.enabled
        )
        motif_conductor_form.addRow("", self.motif_3d_enabled_checkbox)

        self.motif_force_exchange_button = QPushButton(
            "Trigger next ambient"
        )
        self.motif_force_exchange_button.setToolTip(
            "Skip the wait before the next ambient. An active clip finishes its "
            "spatial exit first; clips never overlap."
        )
        motif_conductor_form.addRow(
            "",
            self.motif_force_exchange_button,
        )

        self.motif_calibration_button = QToolButton()
        self.motif_calibration_button.setText(
            "Baseline spatial calibration"
        )
        self.motif_calibration_button.setCheckable(True)
        self.motif_calibration_button.setChecked(
            bool(
                self.loaded_settings.get(
                    "motif_calibration_group_expanded",
                    True,
                )
            )
        )
        self.motif_calibration_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.motif_calibration_button.setArrowType(
            Qt.ArrowType.DownArrow
            if self.motif_calibration_button.isChecked()
            else Qt.ArrowType.RightArrow
        )
        motif_conductor_form.addRow(
            "",
            self.motif_calibration_button,
        )

        self.motif_calibration_panel = QWidget()
        motif_calibration_form = QFormLayout(
            self.motif_calibration_panel
        )
        motif_calibration_form.setContentsMargins(24, 4, 0, 8)
        self.motif_calibration_panel.setVisible(
            self.motif_calibration_button.isChecked()
        )
        motif_conductor_form.addRow(
            "",
            self.motif_calibration_panel,
        )

        self.motif_spatial_setup_button = QToolButton()
        self.motif_spatial_setup_button.setText(
            "Spatial exit"
        )
        self.motif_spatial_setup_button.setCheckable(True)
        self.motif_spatial_setup_button.setChecked(
            bool(
                self.loaded_settings.get(
                    "motif_spatial_setup_group_expanded",
                    True,
                )
            )
        )
        self.motif_spatial_setup_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.motif_spatial_setup_button.setArrowType(
            Qt.ArrowType.DownArrow
            if self.motif_spatial_setup_button.isChecked()
            else Qt.ArrowType.RightArrow
        )
        motif_conductor_form.addRow(
            "",
            self.motif_spatial_setup_button,
        )

        self.motif_spatial_setup_panel = QWidget()
        motif_spatial_setup_form = QFormLayout(
            self.motif_spatial_setup_panel
        )
        motif_spatial_setup_form.setContentsMargins(24, 4, 0, 8)
        self.motif_spatial_setup_panel.setVisible(
            self.motif_spatial_setup_button.isChecked()
        )
        motif_conductor_form.addRow(
            "",
            self.motif_spatial_setup_panel,
        )

        self.motif_guidance_button = QToolButton()
        self.motif_guidance_button.setText(
            "Ambient presence and motion"
        )
        self.motif_guidance_button.setCheckable(True)
        self.motif_guidance_button.setChecked(
            bool(
                self.loaded_settings.get(
                    "motif_guidance_group_expanded",
                    True,
                )
            )
        )
        self.motif_guidance_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.motif_guidance_button.setArrowType(
            Qt.ArrowType.DownArrow
            if self.motif_guidance_button.isChecked()
            else Qt.ArrowType.RightArrow
        )
        motif_conductor_form.addRow(
            "",
            self.motif_guidance_button,
        )

        self.motif_guidance_panel = QWidget()
        motif_guidance_form = QFormLayout(
            self.motif_guidance_panel
        )
        motif_guidance_form.setContentsMargins(24, 4, 0, 8)
        self.motif_guidance_panel.setVisible(
            self.motif_guidance_button.isChecked()
        )
        motif_conductor_form.addRow(
            "",
            self.motif_guidance_panel,
        )

        (
            self.motif_manual_button,
            self.motif_manual_panel,
            motif_manual_form,
        ) = make_motif_subgroup(
            "Manual spatial laboratory",
            "motif_manual_group_expanded",
            False,
        )

        self.motif_manual_checkbox = QCheckBox(
            "Manual spatial tuning — pause automatic motif movement"
        )
        self.motif_manual_checkbox.setChecked(False)
        motif_manual_form.addRow("", self.motif_manual_checkbox)

        self.motif_manual_source_combo = QComboBox()
        self.motif_manual_source_combo.addItems(
            ["ambient", "short event"]
        )
        motif_manual_form.addRow(
            "Manual source:",
            self.motif_manual_source_combo,
        )

        self.motif_manual_solo_checkbox = QCheckBox(
            "Solo selected manual source"
        )
        self.motif_manual_solo_checkbox.setChecked(True)
        self.motif_manual_solo_checkbox.hide()

        self.motif_manual_x_control = FloatControl(
            minimum=-10.0,
            maximum=10.0,
            value=0.0,
            step=0.05,
            decimals=2,
            suffix=" m",
            on_change=lambda value: (
                self._update_manual_motif_spatial(x=value)
            ),
        )
        motif_manual_form.addRow(
            "Left (-) / right (+):",
            self.motif_manual_x_control,
        )

        self.motif_manual_y_control = FloatControl(
            minimum=-5.0,
            maximum=5.0,
            value=0.0,
            step=0.05,
            decimals=2,
            suffix=" m",
            on_change=lambda value: (
                self._update_manual_motif_spatial(y=value)
            ),
        )
        motif_manual_form.addRow(
            "Down (-) / up (+):",
            self.motif_manual_y_control,
        )

        self.motif_manual_z_control = FloatControl(
            minimum=-20.0,
            maximum=5.0,
            value=-2.0,
            step=0.05,
            decimals=2,
            suffix=" m",
            on_change=lambda value: (
                self._update_manual_motif_spatial(z=value)
            ),
        )
        motif_manual_form.addRow(
            "Front (-) / behind (+):",
            self.motif_manual_z_control,
        )

        self.motif_manual_gain_control = FloatControl(
            minimum=-60.0,
            maximum=6.0,
            value=-18.0,
            step=0.5,
            decimals=1,
            suffix=" dB",
            on_change=lambda value: (
                self._update_manual_motif_spatial(
                    gain_db=value
                )
            ),
        )
        motif_manual_form.addRow(
            "Manual source gain:",
            self.motif_manual_gain_control,
        )

        self.motif_manual_position_label = QLabel("")
        self.motif_manual_position_label.setWordWrap(True)
        motif_manual_form.addRow(
            "Resolved position:",
            self.motif_manual_position_label,
        )

        self._motif_parameter_controls = {}

        def add_motif_spatial_control(
            form,
            label,
            field_name,
            minimum,
            maximum,
            step,
            decimals,
            suffix="",
        ):
            control = FloatControl(
                minimum=minimum,
                maximum=maximum,
                value=getattr(motif_spatial_spec, field_name),
                step=step,
                decimals=decimals,
                suffix=suffix,
                on_change=lambda value, field_name=field_name: (
                    self._update_dream_motif_spatial(
                        **{field_name: value}
                    )
                ),
            )
            form.addRow(label, control)
            self._motif_parameter_controls[field_name] = control
            return control

        self.motif_far_distance_calibrated_control = (
            add_motif_spatial_control(
                motif_calibration_form,
                "Far distance:",
                "far_distance_calibrated",
                0.1,
                80.0,
                0.10,
                2,
                " m",
            )
        )
        self.motif_closest_ambient_control = add_motif_spatial_control(
            motif_calibration_form,
            "Closest ambient approach:",
            "closest_ambient_distance",
            0.0,
            20.0,
            0.10,
            2,
            " m",
        )
        self.motif_ambient_gain_control = add_motif_spatial_control(
            motif_calibration_form,
            "Ambient calibrated gain:",
            "motif_calibrated_gain_db",
            -60.0,
            0.0,
            0.5,
            1,
            " dB",
        )
        self.motif_approach_duration_control = add_motif_spatial_control(
            motif_calibration_form,
            "Spatial entrance duration:",
            "ambient_approach_seconds",
            5.0,
            600.0,
            1.0,
            0,
            " s",
        )


        # Explicit timing controls: saved/restored with dream_motif_spatial.
        add_motif_spatial_control(motif_calibration_form, 'Initial rest:',
                                 'startup_rest_seconds', 0, 3600, 1, 0, ' s')
        add_motif_spatial_control(motif_calibration_form, 'Ambient gap minimum:',
                                 'ambient_gap_min_seconds', 0, 600, 0.5, 1, ' s')
        add_motif_spatial_control(motif_calibration_form, 'Ambient gap maximum:',
                                 'ambient_gap_max_seconds', 0, 1800, 0.5, 1, ' s')
        add_motif_spatial_control(motif_calibration_form, '3D movement minimum:',
                                 'wander_min_seconds', 1, 600, 1, 0, ' s')
        add_motif_spatial_control(motif_calibration_form, '3D movement maximum:',
                                 'wander_max_seconds', 1, 600, 1, 0, ' s')
        add_motif_spatial_control(motif_calibration_form, 'Minimum presence during busy phases:',
                                 'busy_presence_floor', 0, 1, 0.01, 2, '')

        self.motif_approach_duration_control.setToolTip(
            "Every clip starts at Far distance. Entrance and exit times are scaled "
            "proportionally if they exceed 90% of the clip length. Applied at the next clip.")
        self.motif_fade_out_control = add_motif_spatial_control(
            motif_spatial_setup_form,
            "Spatial exit duration:",
            "fade_out_seconds",
            1.0,
            1800.0,
            1.0,
            0,
            " s",
        )


        self.motif_testing_checkbox = QCheckBox(
            "Testing — skip gaps, preserve full spatial journeys"
        )
        self.motif_testing_checkbox.setChecked(
            motif_spatial_spec.testing
        )
        motif_guidance_form.addRow(
            "",
            self.motif_testing_checkbox,
        )

        self.motif_featured_events_checkbox = QCheckBox(
            "Include short clips (10 seconds or less)"
        )
        self.motif_featured_events_checkbox.setChecked(
            motif_spatial_spec.featured_events_enabled
        )
        motif_guidance_form.addRow(
            "",
            self.motif_featured_events_checkbox,
        )
        self.motif_presence_control = add_motif_spatial_control(
            motif_guidance_form,
            "Presence:", "presence", 0.0, 1.0, 0.01, 2
        )
        self.motif_motion_control = add_motif_spatial_control(
            motif_guidance_form,
            "Lateral motion:", "motion", 0.0, 1.0, 0.01, 2
        )

        self.motif_presence_control.setToolTip(
            "Sets how close each clip comes. 1 reaches Closest ambient approach; "
            "lower values keep it farther away. Sampled at each clip's start.")
        self.motif_fade_out_control.setToolTip(
            "Recede beyond Far distance to near-silence at the actual recording end. Short clips scale "
            "entrance and exit together; no sample is cut short. Applied at the next clip.")
        self.motif_3d_status_label = QLabel("")
        self.motif_3d_status_label.setWordWrap(True)
        motif_guidance_form.addRow(
            "Ambient state:",
            self.motif_3d_status_label,
        )

        controls_layout.addWidget(self.motif_panel)

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
            minimum=0.15,
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
                "heartbeat_spatial_panel_expanded", False
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

        controls_layout.addWidget(self.dream_motif_checkbox)
        controls_layout.addWidget(self.stereo_checkbox)
        controls_layout.addWidget(self.correlation_checkbox)

        # ------------------------------------------------------------------
        # Meditation performances
        # ------------------------------------------------------------------

        self.meditation_expand_button = QToolButton()
        self.meditation_expand_button.setText(
            "Meditation performances"
        )
        self.meditation_expand_button.setCheckable(True)
        self.meditation_expand_button.setChecked(
            bool(
                self.loaded_settings.get(
                    "meditation_panel_expanded",
                    self.loaded_settings.get("synthesized_meditation_panel_expanded", False),
                )
            )
        )
        self.meditation_expand_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.meditation_expand_button.setArrowType(
            Qt.ArrowType.RightArrow
        )
        controls_layout.addWidget(self.meditation_expand_button)

        self.meditation_panel = QWidget()
        meditation_layout = QVBoxLayout(self.meditation_panel)
        meditation_layout.setContentsMargins(24, 4, 0, 8)
        meditation_layout.setSpacing(4)

        meditation_spec = (
            self.mixer.meditation_state.get()
        )

        self.meditation_enabled_checkbox = QCheckBox(
            "Enable automatic meditation performances"
        )
        self.meditation_enabled_checkbox.setChecked(
            meditation_spec.enabled
        )
        meditation_layout.addWidget(
            self.meditation_enabled_checkbox
        )

        meditation_note = QLabel(
            "Each top-level folder beneath ceremonies/ is a style. Generated ceremonies stitch "
            "vocal and instrument passages according to the instrument selection probability. "
            "Vocals and separate instruments never play together: category changes fade "
            "out then in. Instrument passages stay centered; the whole ceremony retains "
            "its gradual entrance and exit."
        )
        meditation_note.setWordWrap(True)
        meditation_layout.addWidget(meditation_note)

        style_row = QHBoxLayout()
        style_row.addWidget(QLabel("Active style:"))
        self.style_combo = QComboBox()
        available_styles = scan_ceremony_styles()
        for style_name in available_styles:
            self.style_combo.addItem(style_name, style_name)
        if self.style_combo.count() == 0:
            self.style_combo.addItem(self.mixer.active_style_name, self.mixer.active_style_name)
        style_index = self.style_combo.findData(self.mixer.active_style_name)
        if style_index >= 0:
            self.style_combo.setCurrentIndex(style_index)
        self.style_combo.setToolTip(
            "Top-level folders beneath ceremonies/. Changing style reloads both the vocal "
            "ceremony library and the themed ambient/activity/event assets."
        )
        style_row.addWidget(self.style_combo, 1)
        meditation_layout.addLayout(style_row)

        meditation_form = QFormLayout()
        meditation_layout.addLayout(meditation_form)

        self.meditation_interval_min_control = FloatControl(
            minimum=0.25,
            maximum=60.0,
            value=meditation_spec.interval_min_minutes,
            step=0.25,
            decimals=2,
            suffix=" min",
            on_change=lambda value: self._update_meditation(
                interval_min_minutes=value
            ),
        )
        meditation_form.addRow(
            "Minimum rest between ceremonies:",
            self.meditation_interval_min_control,
        )

        self.meditation_interval_max_control = FloatControl(
            minimum=0.25,
            maximum=60.0,
            value=meditation_spec.interval_max_minutes,
            step=0.25,
            decimals=2,
            suffix=" min",
            on_change=lambda value: self._update_meditation(
                interval_max_minutes=value
            ),
        )
        meditation_form.addRow(
            "Maximum rest between ceremonies:",
            self.meditation_interval_max_control,
        )

        self.meditation_ceremony_min_control = FloatControl(
            minimum=3.0,
            maximum=30.0,
            value=meditation_spec.ceremony_min_minutes,
            step=0.5,
            decimals=1,
            suffix=" min",
            on_change=lambda value: self._update_meditation(
                ceremony_min_minutes=value
            ),
        )
        meditation_form.addRow(
            "Minimum ceremony length:",
            self.meditation_ceremony_min_control,
        )

        self.meditation_ceremony_max_control = FloatControl(
            minimum=3.0,
            maximum=30.0,
            value=meditation_spec.ceremony_max_minutes,
            step=0.5,
            decimals=1,
            suffix=" min",
            on_change=lambda value: self._update_meditation(
                ceremony_max_minutes=value
            ),
        )
        meditation_form.addRow(
            "Maximum ceremony length:",
            self.meditation_ceremony_max_control,
        )

        self.meditation_level_control = FloatControl(
            minimum=-24.0,
            maximum=12.0,
            value=meditation_spec.performance_level_db,
            step=0.5,
            decimals=1,
            suffix=" dB",
            on_change=lambda value: self._update_meditation(
                performance_level_db=value
            ),
        )
        meditation_form.addRow(
            "Performance level:",
            self.meditation_level_control,
        )

        def add_meditation_control(label, field_name, minimum, maximum,
                                   step, decimals, suffix):
            control = FloatControl(
                minimum=minimum, maximum=maximum,
                value=getattr(meditation_spec, field_name),
                step=step, decimals=decimals, suffix=suffix,
                on_change=lambda value, name=field_name: self._update_meditation(
                    **{name: value}
                ),
            )
            meditation_form.addRow(label, control)
            return control

        self.meditation_least_reduction_control = add_meditation_control(
            "Brown bed — least reduction:", "brown_least_attenuation_db",
            -60.0, 0.0, 0.5, 1, " dB",
        )
        self.meditation_greatest_reduction_control = add_meditation_control(
            "Brown bed — greatest reduction:", "brown_greatest_attenuation_db",
            -60.0, 0.0, 0.5, 1, " dB",
        )
        self.meditation_least_reduction_control.setToolTip(
            "Louder end of the brown-bed range. -6 dB retains more brown "
            "noise than -24 dB. Equal endpoints give a fixed reduction."
        )
        self.meditation_greatest_reduction_control.setToolTip(
            "Quieter end of the brown-bed range; also applied to heartbeat. "
            "All meditation recordings share this range."
        )
        self.meditation_coupling_control = add_meditation_control(
            "Brown / distance coupling:", "brown_coupling_percent",
            0.0, 100.0, 1.0, 0, "%",
        )
        self.meditation_coupling_control.setToolTip(
            "0%: independent organic brown-bed evolution. 100%: near "
            "performance = more brown reduction, far = less. Changes settle "
            "smoothly. This follows spatial distance, not an audio compressor."
        )
        self.meditation_near_control = add_meditation_control(
            "Performance — nearest distance:", "spatial_near_meters",
            2.0, 8.0, 0.05, 2, " m",
        )
        self.meditation_far_control = add_meditation_control(
            "Performance — farthest distance:", "spatial_far_meters",
            2.0, 8.0, 0.05, 2, " m",
        )
        self.meditation_near_control.setToolTip(
            "Kept at least 2 m away to avoid near-field gain amplification. "
            "Equal distances freeze radial motion, not directional drift."
        )
        self.meditation_far_control.setToolTip(
            "The staged presence journey moves between these endpoints while "
            "a smaller independent radial drift remains layered on top."
        )

        self.meditation_presence_cycle_control = add_meditation_control(
            "Presence cycle:", "presence_cycle_minutes",
            2.0, 30.0, 0.5, 1, " min",
        )
        self.meditation_presence_cycle_control.setToolTip(
            "Approximate tempo of one far → approach → present → retreat "
            "journey. Individual phases vary organically around this value. "
            "Directional drift continues independently throughout."
        )

        self.meditation_modulation_depth_control = add_meditation_control(
            "Distance modulation — depth:", "distance_modulation_percent",
            0.0, 500.0, 1.0, 0, "%",
        )
        self.meditation_modulation_depth_control.setToolTip(
            "Maximum fractional distance variation around the existing ceremony position. "
            "Applies to vocals and instruments without adding directional movement. "
            "As the ceremony moves outward, brown noise fills the gap; as it returns, "
            "the bed makes room again. "
            "0% disables it; changes settle smoothly. Up to 500% for exaggerated testing. "
            "Inward motion is clamped at 2 m, including depths above 100%."
        )
        self.meditation_modulation_time_control = add_meditation_control(
            "Distance modulation — timescale:", "distance_modulation_seconds",
            2.0, 300.0, 1.0, 0, " sec",
        )
        self.meditation_modulation_time_control.setToolTip(
            "Overall pace of smooth random near/far drift, not a repeating cycle. "
            "Each journey takes roughly 30–75% of this time, with occasional short "
            "rests. At 120 seconds, journeys take about 36–90 seconds. "
            "Higher values slow the motion; 2 seconds allows rapid testing."
        )

        self.meditation_instruments_checkbox = QCheckBox("Allow instrument passages")
        self.meditation_instruments_checkbox.setChecked(meditation_spec.instruments_enabled)
        self.meditation_instruments_checkbox.setToolTip(
            "Allow instruments to be selected instead of vocals. Unchecking fades "
            "an active instrument out and returns to vocals when available. "
            "Instruments embedded in vocal recordings are unaffected."
        )
        self.meditation_instruments_checkbox.toggled.connect(
            lambda checked: self._update_meditation(instruments_enabled=checked)
        )
        meditation_form.addRow(self.meditation_instruments_checkbox)

        self.meditation_instrument_selection_control = add_meditation_control(
            "Instrument selection probability:", "instrument_selection_percent",
            0.0, 100.0, 1.0, 0, "%",
        )
        self.meditation_instrument_selection_control.setToolTip(
            "Ceremonies start with vocals. Chance of selecting an instrument after a vocal. "
            "After an instrument, always return to vocals when available. "
            "0% selects vocals only; 100% alternates vocals and instruments. "
            "Instrument-only styles fall back to another instrument. Each category "
            "uses its own shuffle bag; this is not an exact share of listening time."
        )

        self.meditation_subliminal_emergence_control = add_meditation_control(
            "Subliminal emergence:", "subliminal_emergence_seconds",
            0.0, 180.0, 5.0, 0, " sec",
        )
        self.meditation_subliminal_emergence_control.setToolTip(
            "How long a new ceremony remains beneath immediate recognition, "
            "and how gradually it disappears at the end. "
            "A strongly curved veil gradually reveals both the vocal and its "
            "effect on the brown bed, then hands off to the normal Presence "
            "cycle. The same duration governs the final vocal, brown-bed, and "
            "spatial retreat. Set to 0 seconds to use the basic transition."
        )

        meditation_buttons = QHBoxLayout()
        self.meditation_ceremony_combo = QComboBox()
        self.meditation_ceremony_combo.addItem("Select ceremony / starting passage…", None)
        self.meditation_ceremony_combo.addItem(
            "Start generated ceremony (random vocal)",
            MeditationOrchestrator.RANDOM_START_TOKEN,
        )
        registry = self.mixer.meditation.performance_registry
        has_vocals = any(not MeditationOrchestrator._is_instrument(n) for n in registry)
        for ceremony_name in registry:
            if has_vocals and MeditationOrchestrator._is_instrument(ceremony_name):
                continue
            self.meditation_ceremony_combo.addItem(ceremony_name, ceremony_name)
        self.meditation_ceremony_combo.setToolTip(
            "Start a generated ceremony. Choosing a named vocal uses it only as the first clip; "
            "the conductor then continues through the shuffled vocal bank with smooth crossfades."
        )
        self.stop_meditation_button = QPushButton(
            "Stop meditation performance"
        )
        meditation_buttons.addWidget(self.meditation_ceremony_combo, 1)
        meditation_buttons.addWidget(self.stop_meditation_button)
        meditation_layout.addLayout(meditation_buttons)

        self.meditation_status_label = QLabel("")
        self.meditation_status_label.setWordWrap(True)
        meditation_layout.addWidget(
            self.meditation_status_label
        )

        controls_layout.addWidget(self.meditation_panel)

        self.metabolism_expand_button = QToolButton()
        self.metabolism_expand_button.setText("Metabolism")
        self.metabolism_expand_button.setCheckable(True)
        self.metabolism_expand_button.setChecked(
            bool(
                self.loaded_settings.get(
                    "metabolism_panel_expanded",
                    False,
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
            False,
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
            0.15,
            1.0,
            0.01,
            2,
        )
        self.metabolism_control_3 = add_metabolism_control(
            metabolism_brown_form,
            "Body maximum:",
            "brown_body_max",
            0.15,
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
                    False,
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

        # Expanded panels grow the scroll contents rather than the window.
        # Recompute the content limits when panels are opened or closed.
        controls_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinAndMaxSize)
        self.settings_scroll_area = QScrollArea()
        self.settings_scroll_area.setWidgetResizable(True)
        self.settings_scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.settings_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.settings_scroll_area.setWidget(controls_box)
        layout.addWidget(self.settings_scroll_area, 1)

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
        self.export_duration_slider.setRange(5, 480)
        self.export_duration_slider.setSingleStep(5)
        self.export_duration_slider.setPageStep(15)
        self.export_duration_slider.setValue(
            int(self.loaded_settings.get("export_duration_minutes", 360))
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
            "Exports the current settings directly as stereo "
            "192 kbps MP3. Rendering runs faster than real time."
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
        # The settings viewport takes surplus height; transport, export, and
        # live status remain outside it and are always accessible.

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
        self.dream_motif_checkbox.toggled.connect(
            self._on_modes_changed
        )
        self.heartbeat_spatial_expand_button.toggled.connect(
            self._toggle_heartbeat_spatial_panel
        )
        self.motif_expand_button.toggled.connect(
            self._toggle_motif_panel
        )
        self.motif_catalogue_button.toggled.connect(
            lambda expanded: self._toggle_motif_subgroup(
                self.motif_catalogue_button,
                self.motif_catalogue_panel,
                expanded,
            )
        )
        self.motif_conductor_button.toggled.connect(
            lambda expanded: self._toggle_motif_subgroup(
                self.motif_conductor_button,
                self.motif_conductor_panel,
                expanded,
            )
        )
        self.motif_calibration_button.toggled.connect(
            lambda expanded: self._toggle_motif_subgroup(
                self.motif_calibration_button,
                self.motif_calibration_panel,
                expanded,
            )
        )
        self.motif_spatial_setup_button.toggled.connect(
            lambda expanded: self._toggle_motif_subgroup(
                self.motif_spatial_setup_button,
                self.motif_spatial_setup_panel,
                expanded,
            )
        )
        self.motif_guidance_button.toggled.connect(
            lambda expanded: self._toggle_motif_subgroup(
                self.motif_guidance_button,
                self.motif_guidance_panel,
                expanded,
            )
        )
        self.motif_manual_button.toggled.connect(
            lambda expanded: self._toggle_motif_subgroup(
                self.motif_manual_button,
                self.motif_manual_panel,
                expanded,
            )
        )
        self.motif_reload_button.clicked.connect(
            self._reload_dream_motifs
        )
        self.motif_3d_enabled_checkbox.toggled.connect(
            lambda checked: self._update_dream_motif_spatial(
                enabled=bool(checked)
            )
        )
        self.motif_force_exchange_button.clicked.connect(
            self._force_motif_exchange
        )
        self.motif_testing_checkbox.toggled.connect(
            lambda checked: self._update_dream_motif_spatial(
                testing=bool(checked)
            )
        )
        self.motif_featured_events_checkbox.toggled.connect(
            self._on_featured_events_toggled
        )
        self.motif_combo.currentTextChanged.connect(
            self._on_motif_changed
        )
        self.motif_manual_checkbox.toggled.connect(
            self._on_manual_motif_toggled
        )
        self.motif_manual_source_combo.currentTextChanged.connect(
            self._on_manual_motif_source_changed
        )
        self.motif_manual_solo_checkbox.toggled.connect(
            lambda checked: self._update_manual_motif_spatial(
                solo=checked
            )
        )
        self.stereo_checkbox.toggled.connect(self._on_modes_changed)
        self.correlation_checkbox.toggled.connect(
            self._on_modes_changed
        )
        self.meditation_expand_button.toggled.connect(
            self._toggle_meditation_panel
        )
        self.meditation_enabled_checkbox.toggled.connect(
            lambda checked: self._update_meditation(
                enabled=bool(checked)
            )
        )
        self.style_combo.activated.connect(
            self._on_style_changed
        )
        self.meditation_ceremony_combo.activated.connect(
            self._start_selected_meditation
        )
        self.stop_meditation_button.clicked.connect(
            self._stop_meditation
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

        QTimer.singleShot(
            0,
            lambda: self._log_gui_snapshot("startup GUI state"),
        )

        self._toggle_motif_panel(
            self.motif_expand_button.isChecked()
        )
        self._toggle_motif_subgroup(
            self.motif_catalogue_button,
            self.motif_catalogue_panel,
            self.motif_catalogue_button.isChecked(),
        )
        self._toggle_motif_subgroup(
            self.motif_conductor_button,
            self.motif_conductor_panel,
            self.motif_conductor_button.isChecked(),
        )
        self._toggle_motif_subgroup(
            self.motif_spatial_setup_button,
            self.motif_spatial_setup_panel,
            self.motif_spatial_setup_button.isChecked(),
        )
        self._toggle_motif_subgroup(
            self.motif_guidance_button,
            self.motif_guidance_panel,
            self.motif_guidance_button.isChecked(),
        )
        self._toggle_motif_subgroup(
            self.motif_manual_button,
            self.motif_manual_panel,
            self.motif_manual_button.isChecked(),
        )
        self._reload_dream_motifs()
        self._toggle_noise_panel(
            self.noise_expand_button.isChecked()
        )
        self._toggle_heartbeat_spatial_panel(
            self.heartbeat_spatial_expand_button.isChecked()
        )
        self._update_heartbeat_position_status()
        self._toggle_meditation_panel(
            self.meditation_expand_button.isChecked()
        )
        self._update_meditation_status()
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

        saved_dream_motifs_checked = self.loaded_settings.get(
            "dream_motifs_checkbox_checked",
            self.loaded_settings.get(
                "sound_effects_checkbox_checked",
                self.mode_state.get().dream_motifs_enabled,
            ),
        )
        self.dream_motif_checkbox.setChecked(
            bool(saved_dream_motifs_checked)
        )

        self._update_manual_motif_spatial(
            enabled=False,
            source_kind="ambient",
            x=0.0,
            y=0.0,
            z=-2.0,
            gain_db=-18.0,
            solo=True,
        )
        self._on_modes_changed()

    def _write_conductor_log(
        self,
        category: str,
        message: str,
    ) -> None:
        elapsed = time.time() - self._conductor_log_started
        line = (
            f"{elapsed:012.3f}  "
            f"{category:<24}  "
            f"{message}\n"
        )
        try:
            with self._conductor_log_lock:
                with CONDUCTOR_LOG_PATH.open(
                    "a",
                    encoding="utf-8",
                ) as handle:
                    handle.write(line)
        except Exception:
            pass

    def _schedule_gui_snapshot(self, reason: str) -> None:
        self._write_conductor_log("GUI_CHANGE", reason)
        self.gui_snapshot_timer.start(250)

    def _gui_snapshot_payload(self) -> dict:
        motif_spec = self.mixer.dream_motif_spatial_state.get()
        modes = self.mode_state.get()
        manual = self.mixer.dream_motif_3d.manual_snapshot()

        return {
            "modes": asdict(modes),
            "dream_motif_spatial": asdict(motif_spec),
            "controls": {
                "dream_motif_catalogue_expanded": (
                    self.motif_expand_button.isChecked()
                ),
                "catalogue_status_expanded": (
                    self.motif_catalogue_button.isChecked()
                ),
                "automatic_conductor_expanded": (
                    self.motif_conductor_button.isChecked()
                ),
                "baseline_calibration_expanded": (
                    self.motif_calibration_button.isChecked()
                ),
                "spatial_setup_expanded": (
                    self.motif_spatial_setup_button.isChecked()
                ),
                "orchestrator_guidance_expanded": (
                    self.motif_guidance_button.isChecked()
                ),
                "manual_lab_expanded": (
                    self.motif_manual_button.isChecked()
                ),
                "selected_motif": self.motif_combo.currentText(),
                "manual_source": (
                    self.motif_manual_source_combo.currentText()
                ),
                "manual_enabled": (
                    self.motif_manual_checkbox.isChecked()
                ),
                "manual_solo": (
                    self.motif_manual_solo_checkbox.isChecked()
                ),
            },
            "manual_engine_state": {
                "enabled": bool(manual[0]),
                "source_kind": str(manual[1]),
                "position": [
                    float(value) for value in manual[2]
                ],
                "gain_db": float(manual[3]),
                "solo": bool(manual[4]),
                "motif_name": str(manual[5]),
            },
        }

    def _log_gui_snapshot(self, reason: str) -> None:
        try:
            payload = self._gui_snapshot_payload()
            serialized = json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            )
            self._write_conductor_log(
                "GUI_SNAPSHOT",
                f"reason={reason}; {serialized}",
            )
        except Exception as exc:
            self._write_conductor_log(
                "GUI_SNAPSHOT_ERROR",
                str(exc),
            )

    def _log_conductor_snapshot(self) -> None:
        self._write_conductor_log("CONDUCTOR_SNAPSHOT", json.dumps(
            self.mixer.dream_motif_3d.snapshot(), sort_keys=True, separators=(",", ":")))

    def _drain_live_conductor_journal(self) -> None:
        engine = self.mixer.dream_motif_3d
        for timestamp, category, message in (
            engine.drain_event_journal()
        ):
            self._write_conductor_log(
                f"ENGINE_{category}",
                f"engine_time={timestamp:.3f}; {message}",
            )

    def _drain_live_meditation_journal(self) -> None:
        for timestamp, category, message in self.mixer.meditation.drain_event_journal():
            self._write_conductor_log(
                category, f"audio_time={timestamp:.3f}; {message}"
            )

    def _update_manual_motif_spatial(
        self,
        **changes,
    ) -> None:
        self.mixer.dream_motif_3d.set_manual_spatial(
            motif_name=self.motif_combo.currentText(),
            **changes,
        )
        self._schedule_gui_snapshot(
            "manual motif spatial changed: "
            + json.dumps(changes, sort_keys=True)
        )

        (
            _,
            source_kind,
            position,
            gain_db,
            solo,
            _,
        ) = self.mixer.dream_motif_3d.manual_snapshot()

        distance = float(np.linalg.norm(position))
        horizontal = math.degrees(
            math.atan2(position[0], -position[2])
        )
        planar = math.hypot(
            position[0],
            position[2],
        )
        elevation = math.degrees(
            math.atan2(position[1], planar)
        )
        self.motif_manual_position_label.setText(
            f"{source_kind}; distance {distance:.2f} m; "
            f"azimuth {horizontal:+.1f}°; "
            f"elevation {elevation:+.1f}°; "
            f"gain {gain_db:.1f} dB; "
            f"{'solo' if solo else 'in full mix'}"
        )

    def _on_manual_motif_toggled(
        self,
        checked: bool,
    ) -> None:
        self._update_manual_motif_spatial(
            enabled=checked
        )
        self._schedule_settings_save()

    def _on_manual_motif_source_changed(
        self,
        source_kind: str,
    ) -> None:
        self._update_manual_motif_spatial(
            source_kind=source_kind
        )
        self._schedule_settings_save()

    def _on_featured_events_toggled(self, checked: bool) -> None:
        self._write_conductor_log(
            "GUI_ACTION",
            f"featured events toggled={bool(checked)}",
        )
        self._update_dream_motif_spatial(
            featured_events_enabled=bool(checked)
        )
        self.mixer.dream_motif_3d._journal(
            "SHORT_CLIPS", f"enabled={bool(checked)}; active clip finishes naturally")

    def _update_dream_motif_spatial(self, **changes) -> None:
        self.mixer.dream_motif_spatial_state.update(**changes)
        spec = self.mixer.dream_motif_spatial_state.get()
        for name, control in self._motif_parameter_controls.items():
            control.set_value(getattr(spec, name), notify=False)
        self._schedule_gui_snapshot(
            "dream motif setting changed: "
            + json.dumps(changes, sort_keys=True)
        )
        self._schedule_settings_save()

    def _force_motif_exchange(self) -> None:
        self._write_conductor_log("GUI_ACTION", "Trigger next ambient requested")
        self.mixer.dream_motif_3d.request_trigger()
        self.motif_playing_label.setText("Next ambient requested; any active clip finishes first.")

    def _toggle_motif_panel(self, expanded: bool) -> None:
        self._schedule_gui_snapshot(
            f"Dream motif catalogue expanded={bool(expanded)}"
        )
        self.motif_panel.setVisible(expanded)
        self.motif_expand_button.setArrowType(
            Qt.ArrowType.DownArrow
            if expanded
            else Qt.ArrowType.RightArrow
        )

    def _toggle_motif_subgroup(
        self,
        button: QToolButton,
        panel: QWidget,
        expanded: bool,
    ) -> None:
        self._schedule_gui_snapshot(
            f"subgroup {button.text()!r} expanded={bool(expanded)}"
        )
        panel.setVisible(expanded)
        button.setArrowType(
            Qt.ArrowType.DownArrow
            if expanded
            else Qt.ArrowType.RightArrow
        )
        self._schedule_settings_save()

    def _reload_dream_motifs(self) -> None:
        self.motif_directory_label.setText(str(self.mixer.active_ambients_directory))
        self.motif_summary_label.setText(
            "Scanning themed ambient subfolders…"
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
                "No ambient subfolders containing supported audio were "
                f"found beneath {self.mixer.active_ambients_directory}."
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

    def _on_motif_changed(self, motif_name: str) -> None:
        motif_name = motif_name.strip()
        motif = self.dream_motif_catalog.find(motif_name)
        if motif is None:
            self.motif_detail_label.setText("")
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

        # Selection here is catalogue inspection for automatic playback and
        # chooses the source folder used by the manual layered-event test.
        self.mixer.dream_motif_3d.set_manual_spatial(
            motif_name=motif.name
        )
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


    def _toggle_meditation_panel(
        self,
        expanded: bool,
    ) -> None:
        self.meditation_panel.setVisible(expanded)
        self.meditation_expand_button.setArrowType(
            Qt.ArrowType.DownArrow
            if expanded
            else Qt.ArrowType.RightArrow
        )
        self._schedule_settings_save()

    def _update_meditation(self, **changes) -> None:
        self.mixer.meditation_state.update(**changes)

        # Keep paired interval controls visually normalized if the user crosses
        # minimum and maximum.
        spec = self.mixer.meditation_state.get()
        self.meditation_interval_min_control.set_value(
            spec.interval_min_minutes,
            notify=False,
        )
        self.meditation_interval_max_control.set_value(
            spec.interval_max_minutes,
            notify=False,
        )
        self.meditation_ceremony_min_control.set_value(
            spec.ceremony_min_minutes,
            notify=False,
        )
        self.meditation_ceremony_max_control.set_value(
            spec.ceremony_max_minutes,
            notify=False,
        )
        for control, value in (
            (self.meditation_least_reduction_control, spec.brown_least_attenuation_db),
            (self.meditation_greatest_reduction_control, spec.brown_greatest_attenuation_db),
            (self.meditation_near_control, spec.spatial_near_meters),
            (self.meditation_far_control, spec.spatial_far_meters),
        ):
            control.set_value(value, notify=False)

        self._update_meditation_status()
        self._schedule_settings_save()

    def _refresh_ceremony_combo(self) -> None:
        self.meditation_ceremony_combo.blockSignals(True)
        self.meditation_ceremony_combo.clear()
        self.meditation_ceremony_combo.addItem("Select ceremony / starting passage…", None)
        self.meditation_ceremony_combo.addItem(
            "Start generated ceremony (random vocal)",
            MeditationOrchestrator.RANDOM_START_TOKEN,
        )
        for ceremony_name in self.mixer.meditation.performance_registry:
            self.meditation_ceremony_combo.addItem(ceremony_name, ceremony_name)
        self.meditation_ceremony_combo.setCurrentIndex(0)
        self.meditation_ceremony_combo.blockSignals(False)

    def _on_style_changed(self, index: int) -> None:
        style_name = self.style_combo.itemData(index)
        if not style_name:
            return
        style_name = str(style_name)
        if style_name == self.mixer.active_style_name:
            return

        # Switching libraries mutates the background decoder/catalogue, so stop
        # realtime playback first. The user can immediately press Start again.
        was_running = self.engine.stream is not None
        if was_running:
            self._stop()

        self.mixer.set_active_style(style_name)
        self.dream_motif_catalog = DreamMotifCatalog(
            root_directory=self.mixer.active_ambients_directory,
            layer_threshold_seconds=DREAM_MOTIF_LAYER_THRESHOLD_SECONDS,
        )
        self.motif_directory_label.setText(str(self.mixer.active_ambients_directory))
        self._reload_dream_motifs()
        self._refresh_ceremony_combo()
        self._write_conductor_log(
            "STYLE_SELECTED",
            f"style={self.mixer.active_style_name}; "
            f"ambients={self.mixer.active_ambients_directory}; "
            f"vocal_clips={len(self.mixer.meditation.performance_registry)}",
        )
        self._update_meditation_status()
        self._schedule_settings_save()

    def _start_selected_meditation(self, index: int) -> None:
        name = self.meditation_ceremony_combo.itemData(index)
        if not name:
            return
        self.mixer.request_start_meditation(str(name))
        self._write_conductor_log(
            "MEDITATION_MANUAL",
            f"requested start: {name}",
        )
        # Return to the placeholder so the same ceremony can be selected again.
        self.meditation_ceremony_combo.setCurrentIndex(0)

    def _stop_meditation(self) -> None:
        self.mixer.request_stop_meditation()
        self._write_conductor_log(
            "MEDITATION_MANUAL",
            "requested stop",
        )

    def _update_meditation_status(self) -> None:
        orchestrator = self.mixer.meditation
        spec = self.mixer.meditation_state.get()
        if orchestrator.active:
            field = orchestrator.field
            instrument_passage = orchestrator._is_instrument(orchestrator.current_vocal_name)
            self.meditation_status_label.setText(
                f"{orchestrator.current_status}\n"
                f"Brown + heartbeat {orchestrator.current_brown_gain_db:+.1f} dB; "
                f"distance {3.0 if instrument_passage else field.current_distance:.2f} m; "
                f"azimuth {0.0 if instrument_passage else field.current_azimuth:+.1f}°; "
                f"elevation {0.0 if instrument_passage else field.current_elevation:+.1f}°; "
                f"coupling {spec.brown_coupling_percent:.0f}%; "
                f"presence {field.presence_phase} ({spec.presence_cycle_minutes:.1f} min cycle); "
                f"performance trim {spec.performance_level_db:+.1f} dB\n"
                f"Instrument selection: {spec.instrument_selection_percent:.0f}%; "
                f"distance modulation: {orchestrator.current_distance_modulation_percent:+.1f}%"
            )
        else:
            self.meditation_status_label.setText(orchestrator.current_status)

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
        self._schedule_gui_snapshot(
            "engine-layer checkbox changed"
        )

        self.mode_state.set(
            base_enabled=self.base_checkbox.isChecked(),
            stereo_enabled=stereo,
            correlation_enabled=self.correlation_checkbox.isChecked(),
            breath_enabled=self.breath_checkbox.isChecked(),
            heartbeat_enabled=self.heartbeat_checkbox.isChecked(),
            dream_motifs_enabled=self.dream_motif_checkbox.isChecked(),
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
        if self.dream_motif_checkbox.isChecked():
            path += " + dream motifs"

        self.mode_label.setText(path)
        self._schedule_settings_save()

    def _schedule_settings_save(self) -> None:
        self.settings_save_timer.start(250)

    def _save_settings(self) -> None:
        noise_spec, _ = self.noise_state.get()
        noise_evolution_spec = self.noise_evolution_state.get()
        body_movement_spec = self.body_movement_state.get()
        heartbeat_spec = self.heartbeat_state.get()
        breath_spec, _ = self.breath_state.get()
        breath_evolution_spec = self.breath_evolution_state.get()
        motion_spec = self.motion_state.get()
        brown_motion_spec = self.mixer.brown_motion_state.get()
        heartbeat_spatial_spec = self.mixer.heartbeat_spatial_state.get()
        metabolism_spec = self.mixer.metabolism_state.get()
        dream_motif_spatial_spec = (
            self.mixer.dream_motif_spatial_state.get()
        )
        meditation_spec = (
            self.mixer.meditation_state.get()
        )
        modes = self.mode_state.get()

        data = {
            "version": 2,
            "modes": asdict(modes),
            # Stored explicitly so catalogue initialization cannot overwrite
            # the Dream Motifs layer state during startup.
            "dream_motifs_checkbox_checked": (
                self.dream_motif_checkbox.isChecked()
            ),
            "motif_panel_expanded": (
                self.motif_expand_button.isChecked()
            ),
            "motif_catalogue_group_expanded": (
                self.motif_catalogue_button.isChecked()
            ),
            "motif_conductor_group_expanded": (
                self.motif_conductor_button.isChecked()
            ),
            "motif_calibration_group_expanded": (
                self.motif_calibration_button.isChecked()
            ),
            "motif_spatial_setup_group_expanded": (
                self.motif_spatial_setup_button.isChecked()
            ),
            "motif_guidance_group_expanded": (
                self.motif_guidance_button.isChecked()
            ),
            "motif_manual_group_expanded": (
                self.motif_manual_button.isChecked()
            ),
            "brown_noise": asdict(noise_spec),
            "brown_noise_evolution": asdict(noise_evolution_spec),
            "body_movement": asdict(body_movement_spec),
            "heartbeat": asdict(heartbeat_spec),
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
            "dream_motif_spatial": asdict(
                dream_motif_spatial_spec
            ),
            "meditation": asdict(meditation_spec),
            "selected_style": self.mixer.active_style_name,
            "meditation_panel_expanded": (
                self.meditation_expand_button.isChecked()
            ),
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
            f"{self.export_duration_slider.value()}min.mp3"
        )

        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export living brown noise",
            str(EXPORT_DIRECTORY / suggested_name),
            "MP3 audio (*.mp3)",
        )

        if not output_path:
            return

        if not output_path.lower().endswith(".mp3"):
            output_path += ".mp3"

        # Free the audio device and CPU for the renderer.
        self._stop()

        modes = self.mode_state.get()
        noise_spec, _ = self.noise_state.get()
        noise_evolution_spec = self.noise_evolution_state.get()
        body_movement_spec = self.body_movement_state.get()
        heartbeat_spec = self.heartbeat_state.get()
        breath_spec, _ = self.breath_state.get()
        breath_evolution_spec = self.breath_evolution_state.get()
        motion_spec = self.motion_state.get()
        brown_motion_spec = self.mixer.brown_motion_state.get()
        heartbeat_spatial_spec = self.mixer.heartbeat_spatial_state.get()
        metabolism_spec = self.mixer.metabolism_state.get()
        dream_motif_spatial_spec = (
            self.mixer.dream_motif_spatial_state.get()
        )

        self.export_worker = ExportWorker(
            output_path=output_path,
            duration_minutes=self.export_duration_slider.value(),
            sample_rate=44_100,
            modes=modes,
            noise_spec=noise_spec,
            noise_evolution_spec=noise_evolution_spec,
            body_movement_spec=body_movement_spec,
            heartbeat_spec=heartbeat_spec,
            sound_effects_directory=self.mixer.active_ambients_directory,
            breath_spec=breath_spec,
            breath_evolution_spec=breath_evolution_spec,
            motion_spec=motion_spec,
            brown_motion_spec=brown_motion_spec,
            heartbeat_spatial_spec=heartbeat_spatial_spec,
            metabolism_spec=metabolism_spec,
            dream_motif_spatial_spec=dream_motif_spatial_spec,
            meditation_spec=(
                self.mixer.meditation_state.get()
            ),
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
        log_path = str(Path(output_path).with_suffix(".txt"))
        self.export_status_label.setText(
            f"Export complete: {output_path}; log: {log_path}"
        )
        self._finish_export_ui()

        QMessageBox.information(
            self,
            "Export complete",
            f"Audio written to:\n{output_path}\n\n"
            f"Event log written to:\n{log_path}",
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
        self._write_conductor_log(
            "GUI_ACTION",
            "Start clicked",
        )
        self._log_gui_snapshot("playback start")
        try:
            self.engine.start()
        except Exception as exc:
            self.playback_label.setText(f"Start failed: {exc}")
            return

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.playback_label.setText("Running")

    def _stop(self) -> None:
        self._write_conductor_log(
            "GUI_ACTION",
            "Stop clicked",
        )
        self.engine.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.playback_label.setText("Stopped")

    def _refresh_status(self) -> None:
        self._drain_live_conductor_journal()
        self._drain_live_meditation_journal()

        if self.engine.callback_error is not None:
            self.playback_label.setText(
                f"Audio error: {self.engine.callback_error}"
            )
            error_text = str(self.engine.callback_error)
            if error_text != self._last_logged_callback_error:
                self._write_conductor_log(
                    "AUDIO_ERROR",
                    error_text,
                )
                self._last_logged_callback_error = error_text

        elapsed_second = int(
            time.time() - self._conductor_log_started
        )
        if elapsed_second != self._last_conductor_snapshot_second:
            self._last_conductor_snapshot_second = elapsed_second
            self._log_conductor_snapshot()

        self._update_metabolism_status()
        self._update_meditation_status()
        self.motif_3d_status_label.setText(
            self.mixer.dream_motif_3d.current_status
        )
        self.motif_playing_label.setText(
            self.mixer.dream_motif_3d.playback_label
        )
        self._update_brown_motion_status()
        self._update_heartbeat_position_status()

        self.correlation_label.setText(
            f"{self.mixer.current_correlation:.3f}; "
            f"base flow {self.mixer.current_base_flow:+.3f}; "
            f"eddy {self.mixer.current_base_eddy:+.3f}"
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
            f"envelope {self.mixer.current_heartbeat:.3f}; "
            f"level "
            f"{self.mixer.current_heartbeat_effective_level_db:+.1f} dB "
            f"(requested "
            f"{self.mixer.current_heartbeat_requested_level_db:+.1f} dB); "
            f"{self.mixer.current_heartbeat_prominence_state}"
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
        self._write_conductor_log(
            "SHUTDOWN",
            "application closing",
        )
        self._log_gui_snapshot("shutdown GUI state")
        self._drain_live_conductor_journal()
        self._drain_live_meditation_journal()
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
    log_stage("build_application: begin")
    application_started = time.perf_counter()
    sample_rate = 44_100

    log_stage("build_application: creating QApplication")
    app = QApplication(sys.argv)
    log_stage("build_application: QApplication created")

    settings_store = SettingsStore()
    log_stage(
        f"build_application: loading settings from "
        f"{settings_store.path}"
    )
    loaded = settings_store.load()
    log_stage(
        f"build_application: settings loaded; keys={len(loaded)}"
    )
    selected_style_name = resolve_style_name(loaded.get("selected_style"))
    selected_ambients_directory = style_ambients_directory(selected_style_name)
    log_stage(
        f"build_application: selected style={selected_style_name}; "
        f"ambients={selected_ambients_directory}"
    )

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
            dream_motifs_enabled=bool(
                mode_data.get(
                    "dream_motifs_enabled",
                    mode_data.get(
                        "soundscape_enabled",
                        default_modes.dream_motifs_enabled,
                    ),
                )
            ),
        )
    except Exception:
        modes = default_modes

    default_noise = BrownNoiseSpec()
    noise_data = dict(loaded.get("brown_noise", {}))
    if "body" in noise_data:
        try:
            noise_data["body"] = max(0.15, float(noise_data["body"]))
        except (TypeError, ValueError):
            pass
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

    default_dream_motif_spatial = DreamMotifSpatialSpec()
    dream_motif_spatial_data = loaded.get(
        "dream_motif_spatial",
        {},
    )
    try:
        dream_motif_spatial_spec = DreamMotifSpatialSpec(
            **{
                field_name: dream_motif_spatial_data.get(
                    field_name,
                    getattr(default_dream_motif_spatial, field_name),
                )
                for field_name in asdict(default_dream_motif_spatial)
            }
        ).validated()
    except Exception:
        dream_motif_spatial_spec = default_dream_motif_spatial

    meditation_spec = MeditationSpec.from_settings(loaded)

    default_metabolism = MetabolismSpec()
    metabolism_data = dict(loaded.get("metabolism", {}))
    for field_name in ("brown_body_min", "brown_body_max"):
        if field_name in metabolism_data:
            try:
                metabolism_data[field_name] = max(
                    0.15,
                    float(metabolism_data[field_name]),
                )
            except (TypeError, ValueError):
                pass

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
        sound_effects_directory=selected_ambients_directory,
        breath_spec=breath_spec,
        breath_evolution_spec=breath_evolution_spec,
        motion_spec=motion_spec,
        brown_motion_spec=brown_motion_spec,
        heartbeat_spatial_spec=heartbeat_spatial_spec,
        metabolism_spec=metabolism_spec,
        dream_motif_spatial_spec=dream_motif_spatial_spec,
        meditation_spec=(
            meditation_spec
        ),
        seed_base=1000,
    )

    engine = AudioEngine(
        mixer=mixer,
        sample_rate=sample_rate,
        block_size=2_048,
    )

    log_stage("build_application: constructing MainWindow")
    window_started = time.perf_counter()
    window = MainWindow(
        engine=engine,
        mode_state=mode_state,
        noise_state=noise_state,
        noise_evolution_state=noise_evolution_state,
        body_movement_state=body_movement_state,
        heartbeat_state=heartbeat_state,
        breath_state=breath_state,
        breath_evolution_state=breath_evolution_state,
        motion_state=motion_state,
        mixer=mixer,
        settings_store=settings_store,
        loaded_settings=loaded,
    )

    log_stage(
        f"build_application: MainWindow constructed; "
        f"elapsed={time.perf_counter() - window_started:.3f}s"
    )
    log_stage(
        f"build_application: complete; "
        f"elapsed={time.perf_counter() - application_started:.3f}s"
    )
    return app, window


def main() -> int:
    install_exception_logging()
    log_stage("=" * 72)
    log_stage("Dream Instigator startup")
    log_stage(f"Python executable: {sys.executable}")
    log_stage(f"Working directory: {Path.cwd()}")
    EXPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    log_stage(f"Script path: {Path(__file__).resolve()}")
    log_stage(f"Startup log: {STARTUP_LOG_PATH}")
    log_stage(f"Ceremony styles directory: {CEREMONIES_DIRECTORY}")

    try:
        app, window = build_application()
        log_stage("main: calling window.show()")
        window.show()
        log_stage("main: window.show() returned")
        QApplication.processEvents()
        log_stage(
            f"main: window visible={window.isVisible()}; "
            f"entering Qt event loop"
        )
        exit_code = app.exec()
        log_stage(f"main: Qt event loop exited; code={exit_code}")
        return exit_code
    except Exception:
        LOGGER.critical("Fatal startup exception", exc_info=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
