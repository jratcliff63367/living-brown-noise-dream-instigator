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

    def strike(
        self,
        strength: float | None = None,
        *,
        technique: str = "side",
    ) -> None:
        technique = str(technique).lower().strip()
        if technique not in {"side", "rim", "body"}:
            raise ValueError("technique must be side, rim, or body")
        if strength is not None:
            if technique == "rim":
                strength = float(strength) * 0.88
            elif technique == "body":
                strength = float(strength) * 1.05
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
    """
    High-level controls for a complete multi-bowl sound-bath performance.

    duration_minutes:
        Total arc including a quiet closing tail.

    intensity:
        Controls density and strike strength, not a rigid tempo.

    spatiality:
        Controls how boldly the practitioner moves bowls around the listener.

    rubbing:
        Controls how often rim-singing becomes the active technique.
    """

    enabled: bool = False
    duration_minutes: float = 30.0
    intensity: float = 0.62
    spatiality: float = 0.88
    rubbing: float = 0.78

    def validated(self) -> "BowlCeremonySpec":
        if not 8.0 <= self.duration_minutes <= 90.0:
            raise ValueError("duration_minutes must be between 8 and 90")
        for name in ("intensity", "spatiality", "rubbing"):
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

    def set(self, spec: BowlCeremonySpec) -> None:
        with self._lock:
            self._spec = spec.validated()

    def update(self, **changes) -> None:
        with self._lock:
            self._spec = replace(
                self._spec,
                **changes,
            ).validated()


@dataclass(frozen=True, slots=True)
class CeremonyBowlProfile:
    name: str
    fundamental_hz: float
    decay_seconds: float
    brightness: float
    inharmonicity: float
    beating: float
    body: float
    output_gain_db: float
    size_class: float


@dataclass(slots=True)
class CeremonyBowlVoice:
    profile: CeremonyBowlProfile
    state: SingingBowlState
    generator: TibetanSingingBowlGenerator

    position: np.ndarray
    move_start: np.ndarray
    move_target: np.ndarray
    move_elapsed: float
    move_duration: float

    next_strike_seconds: float = 1.0
    rub_target: float = 0.0
    current_rub: float = 0.0
    active_weight: float = 0.0
    strikes: int = 0


class BowlCeremonyController:
    """
    Complete beginning-middle-end procedural singing-bowl ceremony.

    Four differently sized bowls share one evolving performance arc. The
    controller deliberately behaves more like a practitioner than a sequencer:
    long resonances overlap; one or two bowls may be rim-sung while other bowls
    continue to ring; spatial gestures move sources around the listener's head
    and body; density rises toward an immersive middle and then recedes into
    progressively larger spaces and a final low-bowl closing.

    No beat grid is used.
    """

    PHASE_ARRIVAL = "arrival"
    PHASE_GROUNDING = "grounding"
    PHASE_OPENING = "opening"
    PHASE_IMMERSION = "immersion"
    PHASE_INTEGRATION = "integration"
    PHASE_CLOSING = "closing"
    PHASE_SILENCE = "final silence"
    PHASE_COMPLETE = "complete"

    PROFILES = (
        CeremonyBowlProfile(
            name="Large grounding bowl",
            fundamental_hz=107.3,
            decay_seconds=24.0,
            brightness=0.26,
            inharmonicity=0.46,
            beating=0.30,
            body=0.90,
            output_gain_db=-13.5,
            size_class=1.00,
        ),
        CeremonyBowlProfile(
            name="Low-mid bowl",
            fundamental_hz=143.8,
            decay_seconds=20.0,
            brightness=0.38,
            inharmonicity=0.43,
            beating=0.38,
            body=0.78,
            output_gain_db=-14.5,
            size_class=0.78,
        ),
        CeremonyBowlProfile(
            name="Middle singing bowl",
            fundamental_hz=191.6,
            decay_seconds=16.5,
            brightness=0.52,
            inharmonicity=0.40,
            beating=0.46,
            body=0.64,
            output_gain_db=-15.0,
            size_class=0.55,
        ),
        CeremonyBowlProfile(
            name="Small clear bowl",
            fundamental_hz=286.7,
            decay_seconds=12.5,
            brightness=0.68,
            inharmonicity=0.36,
            beating=0.52,
            body=0.46,
            output_gain_db=-16.0,
            size_class=0.32,
        ),
    )

    def __init__(
        self,
        sample_rate: float,
        ceremony_state: BowlCeremonyState,
        *,
        seed: int = 602_900,
    ) -> None:
        self.sample_rate = float(sample_rate)
        self.ceremony_state = ceremony_state
        self.rng = np.random.default_rng(seed)

        self.elapsed_seconds = 0.0
        self.phase = self.PHASE_ARRIVAL
        self.phase_progress = 0.0
        self.performance_progress = 0.0
        self.running = False
        self.complete = False

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

        self.voices: list[CeremonyBowlVoice] = []
        for index, profile in enumerate(self.PROFILES):
            spec = SingingBowlSpec(
                fundamental_hz=profile.fundamental_hz,
                decay_seconds=profile.decay_seconds,
                strike_strength=0.60,
                brightness=profile.brightness,
                inharmonicity=profile.inharmonicity,
                beating=profile.beating,
                body=profile.body,
                rub_level=0.0,
                rub_motion=0.38 + 0.10 * (1.0 - profile.size_class),
                output_gain_db=profile.output_gain_db,
            )
            state = SingingBowlState(spec)
            generator = TibetanSingingBowlGenerator(
                self.sample_rate,
                state,
                seed=seed + 100 + index * 17,
            )

            home = self._home_position(profile, index)
            self.voices.append(
                CeremonyBowlVoice(
                    profile=profile,
                    state=state,
                    generator=generator,
                    position=home.copy(),
                    move_start=home.copy(),
                    move_target=home.copy(),
                    move_elapsed=0.0,
                    move_duration=30.0,
                    next_strike_seconds=2.0 + index * 1.2,
                )
            )

    @staticmethod
    def _smoothstep5(value: float) -> float:
        value = float(np.clip(value, 0.0, 1.0))
        return value ** 3 * (
            value * (value * 6.0 - 15.0) + 10.0
        )

    @staticmethod
    def _home_position(
        profile: CeremonyBowlProfile,
        index: int,
    ) -> np.ndarray:
        # Large bowls live lower in the body field; smaller bowls begin closer
        # to chest/head height. All positions are listener-centered.
        homes = (
            np.array([0.0, -0.75, -2.80], dtype=np.float64),
            np.array([-1.15, -0.25, -2.20], dtype=np.float64),
            np.array([1.05, 0.25, -1.85], dtype=np.float64),
            np.array([0.45, 0.85, -1.55], dtype=np.float64),
        )
        return homes[index].copy()

    def restart(self) -> None:
        for cue_index in range(3):
            token = f"_closing_cue_{cue_index}"
            if hasattr(self, token):
                delattr(self, token)

        self.elapsed_seconds = 0.0
        self.phase = self.PHASE_ARRIVAL
        self.phase_progress = 0.0
        self.performance_progress = 0.0
        self.running = True
        self.complete = False

        for index, voice in enumerate(self.voices):
            voice.generator.clear()
            voice.state.update(rub_level=0.0)
            home = self._home_position(voice.profile, index)
            voice.position = home.copy()
            voice.move_start = home.copy()
            voice.move_target = home.copy()
            voice.move_elapsed = 0.0
            voice.move_duration = 20.0
            voice.next_strike_seconds = 1.8 + index * 1.3
            voice.current_rub = 0.0
            voice.rub_target = 0.0
            voice.active_weight = 0.0
            voice.strikes = 0

    def stop(self) -> None:
        self.running = False
        for voice in self.voices:
            voice.state.update(rub_level=0.0)

    def _phase_for_progress(
        self,
        progress: float,
    ) -> tuple[str, float]:
        # Last 4% is intentional silence after all playing has stopped.
        boundaries = (
            (0.00, 0.08, self.PHASE_ARRIVAL),
            (0.08, 0.22, self.PHASE_GROUNDING),
            (0.22, 0.42, self.PHASE_OPENING),
            (0.42, 0.72, self.PHASE_IMMERSION),
            (0.72, 0.88, self.PHASE_INTEGRATION),
            (0.88, 0.96, self.PHASE_CLOSING),
            (0.96, 1.00, self.PHASE_SILENCE),
        )
        for start, end, phase in boundaries:
            if progress < end:
                local = (progress - start) / max(1.0e-9, end - start)
                return phase, float(np.clip(local, 0.0, 1.0))
        return self.PHASE_COMPLETE, 1.0

    def _phase_energy(self, phase: str, local: float) -> float:
        local_s = self._smoothstep5(local)

        if phase == self.PHASE_ARRIVAL:
            return 0.10 + 0.16 * local_s
        if phase == self.PHASE_GROUNDING:
            return 0.26 + 0.22 * local_s
        if phase == self.PHASE_OPENING:
            return 0.48 + 0.25 * local_s
        if phase == self.PHASE_IMMERSION:
            # Broad plateau rather than a single climax.
            return 0.82 + 0.16 * math.sin(math.pi * local_s)
        if phase == self.PHASE_INTEGRATION:
            return 0.78 - 0.33 * local_s
        if phase == self.PHASE_CLOSING:
            return 0.42 - 0.30 * local_s
        return 0.0

    def _voice_weight(
        self,
        index: int,
        phase: str,
        local: float,
    ) -> float:
        # The practitioner reveals bowls gradually and removes them gradually.
        if phase == self.PHASE_ARRIVAL:
            return (1.0, 0.0, 0.0, 0.0)[index]
        if phase == self.PHASE_GROUNDING:
            return (1.0, 0.70, 0.10, 0.0)[index]
        if phase == self.PHASE_OPENING:
            return (0.95, 0.85, 0.75, 0.30 + 0.45 * local)[index]
        if phase == self.PHASE_IMMERSION:
            return (0.88, 0.95, 1.00, 0.85)[index]
        if phase == self.PHASE_INTEGRATION:
            return (0.75, 0.65, 0.55, 0.45)[index]
        if phase == self.PHASE_CLOSING:
            return (1.0, 0.22 * (1.0 - local), 0.12 * (1.0 - local), 0.0)[index]
        return 0.0

    def _strike_interval(
        self,
        voice: CeremonyBowlVoice,
        energy: float,
        weight: float,
        intensity: float,
    ) -> float:
        size = voice.profile.size_class
        effective = max(
            0.03,
            energy * weight * (0.55 + 0.65 * intensity),
        )

        # Large bowls are naturally less frequent; small bowls may answer
        # between larger resonances. Log-uniform spacing keeps it non-grid.
        minimum = 3.5 + 5.5 * size
        maximum = 10.0 + 26.0 * size
        minimum /= 0.45 + effective
        maximum /= 0.42 + effective

        minimum = float(np.clip(minimum, 2.2, 22.0))
        maximum = float(np.clip(max(maximum, minimum + 2.0), 6.0, 55.0))

        return float(
            math.exp(
                self.rng.uniform(
                    math.log(minimum),
                    math.log(maximum),
                )
            )
        )

    def _strike_strength(
        self,
        voice: CeremonyBowlVoice,
        energy: float,
        intensity: float,
        phase: str,
        local: float,
    ) -> float:
        base = 0.26 + 0.50 * energy
        base *= 0.78 + 0.32 * intensity

        # Closing strikes get progressively lighter.
        if phase == self.PHASE_CLOSING:
            base *= 0.92 - 0.48 * local

        # Big bowls tolerate slightly more physical energy.
        base *= 0.92 + 0.16 * voice.profile.size_class

        return float(
            np.clip(
                self.rng.normal(base, 0.08),
                0.14,
                0.96,
            )
        )

    def _target_rub(
        self,
        index: int,
        voice: CeremonyBowlVoice,
        phase: str,
        local: float,
        energy: float,
        rubbing: float,
        rub_shape: float,
    ) -> float:
        if phase in {
            self.PHASE_ARRIVAL,
            self.PHASE_SILENCE,
            self.PHASE_COMPLETE,
        }:
            return 0.0

        if phase == self.PHASE_GROUNDING:
            phase_amount = 0.30
        elif phase == self.PHASE_OPENING:
            phase_amount = 0.50
        elif phase == self.PHASE_IMMERSION:
            phase_amount = 0.72
        elif phase == self.PHASE_INTEGRATION:
            phase_amount = 0.44 * (1.0 - 0.45 * local)
        else:
            phase_amount = 0.24 * (1.0 - local)

        # A single practitioner usually actively rims only one or two bowls
        # while the rest continue ringing from previous excitation. The phase
        # offsets cause that "active hand" to migrate among the set.
        hand_cycle = 0.5 + 0.5 * math.sin(
            self.elapsed_seconds * (0.032 + 0.004 * index)
            + index * 1.75
        )
        selection = hand_cycle ** 2.4

        # Medium bowls sing most readily; very large/small bowls are somewhat
        # less often the sustained rubbed voice.
        size_preference = (0.72, 1.00, 0.94, 0.68)[index]

        return float(
            np.clip(
                rubbing
                * phase_amount
                * energy
                * size_preference
                * selection
                * (0.62 + 0.38 * rub_shape),
                0.0,
                0.82,
            )
        )

    def _choose_spatial_target(
        self,
        index: int,
        voice: CeremonyBowlVoice,
        phase: str,
        energy: float,
        spatiality: float,
    ) -> tuple[np.ndarray, float]:
        profile = voice.profile
        home = self._home_position(profile, index)

        # During arrival/closing, the performer largely returns bowls to
        # stable stations. The middle of the ceremony is free to become much
        # more intimate and mobile.
        if phase in {
            self.PHASE_ARRIVAL,
            self.PHASE_CLOSING,
            self.PHASE_SILENCE,
        }:
            jitter = np.array(
                [
                    self.rng.uniform(-0.20, 0.20),
                    self.rng.uniform(-0.12, 0.12),
                    self.rng.uniform(-0.18, 0.18),
                ],
                dtype=np.float64,
            )
            target = home + jitter * spatiality
            duration = self.rng.uniform(24.0, 48.0)
            return target, float(duration)

        # A palette of human-like placements around the body/head. Smaller
        # bowls are permitted closer head-level passes; large bowls spend more
        # time chest/feet/front and generally remain farther away.
        head_distance = 0.50 + 0.95 * profile.size_class
        body_distance = 0.95 + 1.10 * profile.size_class
        far_distance = 2.0 + 1.05 * profile.size_class

        targets = [
            np.array([-head_distance, 0.72, -0.58], dtype=np.float64),
            np.array([ head_distance, 0.72, -0.58], dtype=np.float64),
            np.array([0.0, 1.20, -0.72], dtype=np.float64),
            np.array([-body_distance, 0.05, -1.05], dtype=np.float64),
            np.array([ body_distance, 0.05, -1.05], dtype=np.float64),
            np.array([0.0, -0.55, -far_distance], dtype=np.float64),
            np.array([0.0, 0.20, -far_distance], dtype=np.float64),
        ]

        # Big bowls weight body/front positions. Small bowls weight head passes.
        if profile.size_class > 0.80:
            weights = np.array(
                [0.04, 0.04, 0.02, 0.14, 0.14, 0.34, 0.28],
                dtype=np.float64,
            )
        elif profile.size_class < 0.40:
            weights = np.array(
                [0.20, 0.20, 0.18, 0.10, 0.10, 0.08, 0.14],
                dtype=np.float64,
            )
        else:
            weights = np.array(
                [0.13, 0.13, 0.10, 0.16, 0.16, 0.14, 0.18],
                dtype=np.float64,
            )

        # Lower spatiality biases toward the home station.
        if self.rng.random() > spatiality * (0.45 + 0.55 * energy):
            return (
                home
                + self.rng.normal(0.0, 0.12, 3).astype(np.float64),
                float(self.rng.uniform(24.0, 45.0)),
            )

        weights /= np.sum(weights)
        target = targets[int(self.rng.choice(len(targets), p=weights))]

        # A little asymmetry avoids obviously repeated coordinate destinations.
        target = target + np.array(
            [
                self.rng.normal(0.0, 0.11),
                self.rng.normal(0.0, 0.08),
                self.rng.normal(0.0, 0.10),
            ],
            dtype=np.float64,
        )

        duration = self.rng.uniform(
            13.0 + 8.0 * profile.size_class,
            30.0 + 16.0 * profile.size_class,
        )
        return target, float(duration)

    def _update_motion(
        self,
        index: int,
        voice: CeremonyBowlVoice,
        dt: float,
        phase: str,
        energy: float,
        spatiality: float,
    ) -> None:
        voice.move_elapsed += dt

        if voice.move_elapsed >= voice.move_duration:
            target, duration = self._choose_spatial_target(
                index,
                voice,
                phase,
                energy,
                spatiality,
            )
            voice.move_start = voice.position.copy()
            voice.move_target = target
            voice.move_elapsed = 0.0
            voice.move_duration = duration

        progress = self._smoothstep5(
            voice.move_elapsed / max(1.0e-9, voice.move_duration)
        )
        voice.position = (
            voice.move_start
            + (voice.move_target - voice.move_start) * progress
        )

    def advance(self, elapsed_seconds: float) -> None:
        spec = self.ceremony_state.get()
        if not spec.enabled or not self.running or self.complete:
            return

        dt = max(0.0, float(elapsed_seconds))
        total = spec.duration_minutes * 60.0
        self.elapsed_seconds += dt

        self.performance_progress = float(
            np.clip(self.elapsed_seconds / max(1.0, total), 0.0, 1.0)
        )
        self.phase, self.phase_progress = self._phase_for_progress(
            self.performance_progress
        )

        if self.phase == self.PHASE_COMPLETE:
            self.complete = True
            self.running = False
            for voice in self.voices:
                voice.state.update(rub_level=0.0)
            return

        energy = self._phase_energy(
            self.phase,
            self.phase_progress,
        )

        dynamic = 0.5 + 0.5 * self.dynamic_wander.advance(dt)
        rub_shape = 0.5 + 0.5 * self.rub_wander.advance(dt)

        # Organic micro-variation rides on top of the deliberate long-form arc.
        energy *= 0.82 + 0.22 * dynamic
        energy = float(np.clip(energy, 0.0, 1.0))

        for index, voice in enumerate(self.voices):
            weight = self._voice_weight(
                index,
                self.phase,
                self.phase_progress,
            )
            voice.active_weight = weight

            self._update_motion(
                index,
                voice,
                dt,
                self.phase,
                energy,
                spec.spatiality,
            )

            target_rub = self._target_rub(
                index,
                voice,
                self.phase,
                self.phase_progress,
                energy,
                spec.rubbing,
                rub_shape,
            )
            voice.rub_target = target_rub

            # Rubbing fades in/out with hand-like inertia instead of switching.
            rub_time = 2.5 if target_rub > voice.current_rub else 4.5
            amount = 1.0 - math.exp(-dt / rub_time)
            voice.current_rub += (
                target_rub - voice.current_rub
            ) * amount
            voice.state.update(
                rub_level=float(
                    np.clip(voice.current_rub * weight, 0.0, 0.90)
                )
            )

            if self.phase in {
                self.PHASE_CLOSING,
                self.PHASE_SILENCE,
                self.PHASE_COMPLETE,
            }:
                # Closing is deliberately performed by the three low-bowl
                # cues below rather than by the ordinary stochastic scheduler.
                continue

            voice.next_strike_seconds -= dt
            if voice.next_strike_seconds <= 0.0:
                if weight > 0.04:
                    strength = self._strike_strength(
                        voice,
                        energy,
                        spec.intensity,
                        self.phase,
                        self.phase_progress,
                    )
                    voice.generator.strike(strength)
                    voice.strikes += 1

                voice.next_strike_seconds = self._strike_interval(
                    voice,
                    energy,
                    weight,
                    spec.intensity,
                )

        # A professional-feeling ending: increasingly sparse final low-bowl
        # punctuation followed by true silence. The normal scheduler already
        # favors the large bowl in closing; these timed cues make the endpoint
        # deliberate rather than merely probabilistic.
        if self.phase == self.PHASE_CLOSING:
            large = self.voices[0]
            cue_points = (0.10, 0.43, 0.73)
            for cue_index, cue in enumerate(cue_points):
                token = f"_closing_cue_{cue_index}"
                if (
                    self.phase_progress >= cue
                    and not hasattr(self, token)
                ):
                    setattr(self, token, True)
                    large.generator.strike(
                        (0.44, 0.34, 0.24)[cue_index]
                    )
                    large.strikes += 1

    def render_mono(self, frame_count: int) -> list[np.ndarray]:
        """
        Generate one mono block for every bowl.

        Spatial rendering remains outside this class so the same ceremony
        engine can later be integrated into Living Brown Noise without GUI or
        device dependencies.
        """
        return [
            voice.generator.generate(frame_count)
            for voice in self.voices
        ]

    @property
    def remaining_seconds(self) -> float:
        spec = self.ceremony_state.get()
        return max(
            0.0,
            spec.duration_minutes * 60.0 - self.elapsed_seconds,
        )
