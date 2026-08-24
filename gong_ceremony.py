from __future__ import annotations

import math
import threading
from dataclasses import dataclass, replace

import numpy as np

from synthesized_sound_source import OrganicWanderer1D, SmoothedValue, db_to_linear


@dataclass(frozen=True, slots=True)
class GongSpec:
    """
    Procedural large-gong / tam-tam voice.

    The generator is intentionally not a sample surrogate. It models:
      * a broad, inharmonic modal body;
      * nonlinear bloom after impact;
      * slow inter-mode beating;
      * surface/rim friction excitation;
      * pressure/contact modulation capable of producing vocal, whale-like,
        squealing, and metallic "impossible" tones.

    The friction side is deliberately first-class rather than decorative.
    """

    base_hz: float = 58.0
    size: float = 0.82
    decay_seconds: float = 34.0
    strike_strength: float = 0.58
    bloom: float = 0.72
    darkness: float = 0.64
    chaos: float = 0.52
    friction_level: float = 0.0
    friction_pressure: float = 0.46
    friction_speed: float = 0.38
    friction_brightness: float = 0.58
    friction_instability: float = 0.62
    hand_level: float = 0.0
    hand_pressure: float = 0.55
    hand_position: float = 0.68
    output_gain_db: float = -13.0

    def validated(self) -> "GongSpec":
        if not 30.0 <= self.base_hz <= 220.0:
            raise ValueError("base_hz must be between 30 and 220")
        if not 0.0 <= self.size <= 1.0:
            raise ValueError("size must be between 0 and 1")
        if not 3.0 <= self.decay_seconds <= 90.0:
            raise ValueError("decay_seconds must be between 3 and 90")
        for name in (
            "strike_strength",
            "bloom",
            "darkness",
            "chaos",
            "friction_level",
            "friction_pressure",
            "friction_speed",
            "friction_brightness",
            "friction_instability",
            "hand_level",
            "hand_pressure",
            "hand_position",
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if not -48.0 <= self.output_gain_db <= 6.0:
            raise ValueError("output_gain_db must be between -48 and +6")
        return self


class GongState:
    def __init__(self, spec: GongSpec) -> None:
        self._lock = threading.Lock()
        self._spec = spec.validated()

    def get(self) -> GongSpec:
        with self._lock:
            return self._spec

    def set(self, spec: GongSpec) -> None:
        with self._lock:
            self._spec = spec.validated()

    def update(self, **changes) -> None:
        with self._lock:
            self._spec = replace(
                self._spec,
                **changes,
            ).validated()


@dataclass(slots=True)
class _Strike:
    age_seconds: float
    strength: float
    phase: np.ndarray
    pitch_scale: float
    location: float


class ProceduralGongGenerator:
    """
    Mono procedural gong with impact and friction techniques.

    Friction is split into two families:
      1. tool friction: a synthetic friction-mallet/driver path;
      2. hand friction: palm/finger contact with nonlinear squeal/voice-like
         emergent tones.

    Both excite the same modal body, so friction sounds feel like they emerge
    from the gong rather than being layered effects.
    """

    MODE_RATIOS = np.array(
        [1.00, 1.48, 1.96, 2.63, 3.41, 4.37, 5.58, 7.12, 8.96, 11.40],
        dtype=np.float64,
    )
    DETUNE = np.array(
        [0.0, -0.015, 0.021, -0.032, 0.047, -0.058, 0.071, -0.082, 0.095, -0.11],
        dtype=np.float64,
    )
    AMPS = np.array(
        [1.0, 0.92, 0.78, 0.66, 0.54, 0.42, 0.31, 0.23, 0.16, 0.10],
        dtype=np.float64,
    )
    DECAYS = np.array(
        [1.00, 0.96, 0.90, 0.82, 0.74, 0.66, 0.57, 0.49, 0.41, 0.34],
        dtype=np.float64,
    )

    def __init__(
        self,
        sample_rate: float,
        state: GongState,
        *,
        seed: int = 904_101,
    ) -> None:
        self.sample_rate = float(sample_rate)
        self.state = state
        self.rng = np.random.default_rng(seed)
        self.strikes: list[_Strike] = []

        self.output_gain = SmoothedValue(
            self.sample_rate,
            db_to_linear(state.get().output_gain_db),
            0.18,
        )
        self.friction_smoother = SmoothedValue(
            self.sample_rate,
            state.get().friction_level,
            0.60,
        )
        self.hand_smoother = SmoothedValue(
            self.sample_rate,
            state.get().hand_level,
            0.42,
        )

        self.friction_phase = 0.0
        self.hand_phase = 0.0
        self.modal_phases = self.rng.uniform(
            0.0,
            2.0 * math.pi,
            len(self.MODE_RATIOS),
        )

        self.friction_wander = OrganicWanderer1D(
            seed=seed + 10,
            natural_period_seconds=7.0,
            damping_ratio=0.60,
            drive_strength=0.95,
            drive_smoothing_seconds=2.0,
        )
        self.pressure_wander = OrganicWanderer1D(
            seed=seed + 11,
            natural_period_seconds=11.0,
            damping_ratio=0.72,
            drive_strength=0.82,
            drive_smoothing_seconds=3.5,
        )
        self.hand_wander = OrganicWanderer1D(
            seed=seed + 12,
            natural_period_seconds=5.5,
            damping_ratio=0.52,
            drive_strength=1.10,
            drive_smoothing_seconds=1.5,
        )

        self.friction_energy = 0.0
        self.hand_energy = 0.0

    def strike(
        self,
        strength: float | None = None,
        location: float | None = None,
    ) -> None:
        spec = self.state.get()
        if strength is None:
            strength = spec.strike_strength
        if location is None:
            location = float(self.rng.uniform(0.25, 0.82))

        strength = float(np.clip(strength, 0.0, 1.4))
        location = float(np.clip(location, 0.0, 1.0))

        pitch_scale = float(
            np.clip(
                self.rng.normal(1.0, 0.004 + 0.020 * spec.chaos),
                0.94,
                1.06,
            )
        )
        phases = self.rng.uniform(
            -0.35,
            0.35,
            len(self.MODE_RATIOS),
        )
        self.strikes.append(
            _Strike(
                age_seconds=0.0,
                strength=strength,
                phase=phases,
                pitch_scale=pitch_scale,
                location=location,
            )
        )

    def clear(self) -> None:
        self.strikes.clear()
        self.friction_energy = 0.0
        self.hand_energy = 0.0

    def _frequencies(
        self,
        spec: GongSpec,
        location: float,
        pitch_scale: float,
    ) -> np.ndarray:
        location_shape = (location - 0.5) * 0.08
        ratios = self.MODE_RATIOS * (
            1.0
            + spec.chaos * self.DETUNE
            + location_shape * np.linspace(-0.25, 0.35, len(self.MODE_RATIOS))
        )
        return spec.base_hz * pitch_scale * ratios

    def _render_strike(
        self,
        event: _Strike,
        spec: GongSpec,
        frame_count: int,
    ) -> tuple[np.ndarray, float]:
        t = (
            np.arange(frame_count, dtype=np.float64) / self.sample_rate
            + event.age_seconds
        )
        freqs = self._frequencies(
            spec,
            event.location,
            event.pitch_scale,
        )

        output = np.zeros(frame_count, dtype=np.float64)

        # Gong impact "blooms" instead of exposing all modes instantly.
        bloom_times = np.linspace(
            0.04 + 0.10 * (1.0 - spec.bloom),
            0.18 + 0.55 * spec.bloom,
            len(freqs),
        )

        darkness_curve = np.linspace(
            1.15 + 0.45 * spec.darkness,
            0.85 - 0.48 * spec.darkness,
            len(freqs),
        )

        for i, freq in enumerate(freqs):
            decay = max(
                0.4,
                spec.decay_seconds * self.DECAYS[i],
            )
            attack = 1.0 - np.exp(
                -np.maximum(t, 0.0) / max(0.008, bloom_times[i])
            )
            envelope = attack * np.exp(-t / decay)

            # Slow split-mode beating.
            split_cents = (
                1.0
                + 12.0 * spec.chaos
                * (0.25 + i / max(1, len(freqs) - 1))
            )
            split = freq * 2.0 ** (split_cents / 1200.0)

            phase_a = 2.0 * math.pi * freq * t + event.phase[i]
            phase_b = 2.0 * math.pi * split * t - 0.6 * event.phase[i]

            amplitude = (
                event.strength
                * self.AMPS[i]
                * darkness_curve[i]
                * (0.88 + 0.22 * event.location)
            )

            split_mix = 0.10 + 0.28 * spec.chaos
            output += amplitude * envelope * (
                (1.0 - split_mix) * np.sin(phase_a)
                + split_mix * np.sin(phase_b)
            )

        # Initial mallet/body contact: soft broad-band excitation, not a click.
        contact_env = np.exp(-t / 0.030)
        contact_noise = self.rng.standard_normal(frame_count)
        output += (
            contact_noise
            * contact_env
            * event.strength
            * (0.010 + 0.020 * (1.0 - spec.darkness))
        )

        return output, event.age_seconds + frame_count / self.sample_rate

    def _modal_surface(
        self,
        spec: GongSpec,
        frame_count: int,
        excitation: np.ndarray,
        *,
        bright_bias: float,
        moving_position: float,
    ) -> np.ndarray:
        freqs = self._frequencies(
            spec,
            moving_position,
            1.0,
        )
        n = np.arange(frame_count, dtype=np.float64)
        output = np.zeros(frame_count, dtype=np.float64)

        for i, freq in enumerate(freqs):
            step = 2.0 * math.pi * freq / self.sample_rate
            phases = self.modal_phases[i] + step * n
            brightness = (
                0.35
                + 0.65 * bright_bias
            ) ** (i / max(1, len(freqs) - 1))
            weight = self.AMPS[i] * brightness

            output += weight * np.sin(phases)
            self.modal_phases[i] = float(
                (phases[-1] + step) % (2.0 * math.pi)
            )

        output /= max(1.0, np.sqrt(len(freqs)))
        return output * excitation

    def _render_friction(
        self,
        spec: GongSpec,
        frame_count: int,
    ) -> np.ndarray:
        level = self.friction_smoother.ramp(
            spec.friction_level,
            frame_count,
        )
        dt = frame_count / self.sample_rate

        wander = self.friction_wander.advance(
            dt * (0.4 + 2.8 * spec.friction_speed)
        )
        pressure_shape = self.pressure_wander.advance(dt)

        target = float(
            np.mean(level)
            * (0.58 + 0.42 * spec.friction_pressure)
            * (0.72 + 0.28 * (0.5 + 0.5 * pressure_shape))
        )
        catch = 0.25 + 1.6 * (1.0 - spec.friction_pressure)
        self.friction_energy += (
            target - self.friction_energy
        ) * (1.0 - math.exp(-dt / catch))

        if self.friction_energy < 1.0e-6:
            return np.zeros(frame_count, dtype=np.float64)

        n = np.arange(frame_count, dtype=np.float64)
        speed_hz = 5.0 + 24.0 * spec.friction_speed
        phase = (
            self.friction_phase
            + 2.0 * math.pi * speed_hz * n / self.sample_rate
        )

        # Stick-slip/contact pulsation. This excites the gong rather than being
        # heard as a stand-alone oscillator.
        contact = np.tanh(
            (1.6 + 5.0 * spec.friction_pressure)
            * np.sin(phase + 0.7 * wander)
        )
        self.friction_phase = float(
            (phase[-1] + 2.0 * math.pi * speed_hz / self.sample_rate)
            % (2.0 * math.pi)
        )

        contact *= self.friction_energy
        contact *= (
            0.82
            + 0.18
            * np.sin(
                2.0 * math.pi
                * (0.32 + 0.85 * spec.friction_instability)
                * n
                / self.sample_rate
                + 1.4 * wander
            )
        )

        modal = self._modal_surface(
            spec,
            frame_count,
            contact,
            bright_bias=spec.friction_brightness,
            moving_position=0.76 + 0.14 * wander,
        )

        # Bright friction-driver whistle/shooting-star component.
        squeal_hz = (
            310.0
            + 1550.0 * spec.friction_brightness
            + 260.0 * wander * spec.friction_instability
        )
        squeal = np.sin(
            2.0 * math.pi * squeal_hz * n / self.sample_rate
            + 0.25 * np.sin(
                2.0 * math.pi * 1.7 * n / self.sample_rate
            )
        )
        squeal *= (
            self.friction_energy
            * (0.015 + 0.080 * spec.friction_brightness)
        )

        rough = self.rng.standard_normal(frame_count)
        rough *= (
            self.friction_energy
            * (0.002 + 0.012 * spec.friction_instability)
        )

        return modal * 0.54 + squeal + rough

    def _render_hand(
        self,
        spec: GongSpec,
        frame_count: int,
    ) -> np.ndarray:
        level = self.hand_smoother.ramp(
            spec.hand_level,
            frame_count,
        )
        dt = frame_count / self.sample_rate

        wander = self.hand_wander.advance(dt)
        target = float(
            np.mean(level)
            * (0.50 + 0.50 * spec.hand_pressure)
        )
        self.hand_energy += (
            target - self.hand_energy
        ) * (
            1.0
            - math.exp(
                -dt / (0.18 + 0.9 * (1.0 - spec.hand_pressure))
            )
        )

        if self.hand_energy < 1.0e-6:
            return np.zeros(frame_count, dtype=np.float64)

        n = np.arange(frame_count, dtype=np.float64)

        # Hand/palm friction is intentionally expressive. The "voice-like"
        # quality comes from a moving cluster of inharmonic carrier bands with
        # pressure-dependent nonlinear modulation, not from a human vocal model.
        position = float(
            np.clip(
                spec.hand_position + 0.10 * wander,
                0.08,
                0.95,
            )
        )
        center = (
            150.0
            + 1250.0 * position
            + 420.0 * wander * spec.hand_pressure
        )
        mod_hz = 0.8 + 4.0 * spec.hand_pressure
        modulation = np.sin(
            2.0 * math.pi * mod_hz * n / self.sample_rate
            + 1.8 * wander
        )

        carriers = (
            np.sin(
                2.0 * math.pi
                * (center + 65.0 * modulation)
                * n
                / self.sample_rate
                + self.hand_phase
            )
            + 0.46
            * np.sin(
                2.0 * math.pi
                * (center * 1.73 - 90.0 * modulation)
                * n
                / self.sample_rate
                + 0.8
            )
            + 0.25
            * np.sin(
                2.0 * math.pi
                * (center * 2.44 + 130.0 * modulation)
                * n
                / self.sample_rate
                - 1.1
            )
        )
        carriers = np.tanh(
            carriers * (0.9 + 2.8 * spec.hand_pressure)
        )

        self.hand_phase = float(
            (
                self.hand_phase
                + 2.0
                * math.pi
                * center
                * frame_count
                / self.sample_rate
            )
            % (2.0 * math.pi)
        )

        # Couple hand excitation back into the gong modes.
        excitation = (
            carriers
            * self.hand_energy
            * (0.22 + 0.52 * spec.hand_pressure)
        )
        modal = self._modal_surface(
            spec,
            frame_count,
            excitation,
            bright_bias=0.72,
            moving_position=position,
        )

        # Airy skin/contact noise gives the impossible tones a physical source.
        noise = self.rng.standard_normal(frame_count)
        noise *= self.hand_energy * 0.010

        return 0.46 * excitation + 0.48 * modal + noise

    def generate(self, frame_count: int) -> np.ndarray:
        spec = self.state.get()
        frame_count = int(frame_count)
        if frame_count <= 0:
            return np.zeros(0, dtype=np.float32)

        output = np.zeros(frame_count, dtype=np.float64)
        retained: list[_Strike] = []

        for event in self.strikes:
            rendered, new_age = self._render_strike(
                event,
                spec,
                frame_count,
            )
            output += rendered
            event.age_seconds = new_age
            if event.age_seconds < spec.decay_seconds * 7.0:
                retained.append(event)

        self.strikes = retained

        output += self._render_friction(spec, frame_count)
        output += self._render_hand(spec, frame_count)

        gain = self.output_gain.ramp(
            db_to_linear(spec.output_gain_db),
            frame_count,
        )
        output *= gain

        # Gong peaks can become huge when modal families align.
        output = 0.94 * np.tanh(output * 0.70)

        return output.astype(np.float32, copy=False)


@dataclass(frozen=True, slots=True)
class GongCeremonySpec:
    enabled: bool = False
    duration_minutes: float = 30.0
    intensity: float = 0.64
    friction_presence: float = 0.82
    hand_magic: float = 0.88
    spatiality: float = 0.62

    def validated(self) -> "GongCeremonySpec":
        if not 8.0 <= self.duration_minutes <= 90.0:
            raise ValueError("duration_minutes must be between 8 and 90")
        for name in (
            "intensity",
            "friction_presence",
            "hand_magic",
            "spatiality",
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        return self


class GongCeremonyState:
    def __init__(self, spec: GongCeremonySpec) -> None:
        self._lock = threading.Lock()
        self._spec = spec.validated()

    def get(self) -> GongCeremonySpec:
        with self._lock:
            return self._spec

    def update(self, **changes) -> None:
        with self._lock:
            self._spec = replace(
                self._spec,
                **changes,
            ).validated()


@dataclass(frozen=True, slots=True)
class GongProfile:
    name: str
    base_hz: float
    size: float
    decay_seconds: float
    darkness: float
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
    friction_target: float = 0.0
    hand_target: float = 0.0
    strikes: int = 0


class GongCeremonyController:
    """
    Long-form professional-style gong bath.

    Arc:
      arrival -> grounding -> expansion -> immersion -> alchemy ->
      integration -> closing -> silence

    The performance alternates between enormous low-body blooms and smaller,
    startling friction/hand gestures. The latter are intentionally sparse
    enough to remain special.
    """

    PHASES = (
        ("arrival", 0.00, 0.08),
        ("grounding", 0.08, 0.22),
        ("expansion", 0.22, 0.42),
        ("immersion", 0.42, 0.66),
        ("alchemy", 0.66, 0.78),
        ("integration", 0.78, 0.90),
        ("closing", 0.90, 0.97),
        ("final silence", 0.97, 1.00),
    )

    PROFILES = (
        GongProfile(
            "Large tam-tam",
            48.0, 1.00, 44.0, 0.78, 0.58, -14.0,
            0.0, -0.35, -3.1,
        ),
        GongProfile(
            "Medium symphonic gong",
            72.0, 0.72, 32.0, 0.58, 0.62, -15.0,
            -1.15, 0.25, -2.4,
        ),
        GongProfile(
            "Bright friction gong",
            104.0, 0.48, 24.0, 0.38, 0.74, -16.0,
            1.05, 0.55, -2.0,
        ),
    )

    def __init__(
        self,
        sample_rate: float,
        state: GongCeremonyState,
        *,
        seed: int = 904_500,
    ) -> None:
        self.sample_rate = float(sample_rate)
        self.state = state
        self.rng = np.random.default_rng(seed)

        self.voices: list[GongVoice] = []
        for i, p in enumerate(self.PROFILES):
            gs = GongState(
                GongSpec(
                    base_hz=p.base_hz,
                    size=p.size,
                    decay_seconds=p.decay_seconds,
                    strike_strength=0.58,
                    bloom=0.74,
                    darkness=p.darkness,
                    chaos=p.chaos,
                    friction_level=0.0,
                    friction_pressure=0.48,
                    friction_speed=0.40,
                    friction_brightness=0.58,
                    friction_instability=0.62,
                    hand_level=0.0,
                    hand_pressure=0.56,
                    hand_position=0.68,
                    output_gain_db=p.output_gain_db,
                )
            )
            gen = ProceduralGongGenerator(
                self.sample_rate,
                gs,
                seed=seed + 100 + i * 29,
            )
            home = np.array([p.x, p.y, p.z], dtype=np.float64)
            self.voices.append(
                GongVoice(
                    profile=p,
                    state=gs,
                    generator=gen,
                    position=home.copy(),
                    move_start=home.copy(),
                    move_target=home.copy(),
                    move_elapsed=0.0,
                    move_duration=24.0,
                    next_strike=2.0 + i * 1.8,
                )
            )

        self.elapsed_seconds = 0.0
        self.phase = "arrival"
        self.phase_progress = 0.0
        self.performance_progress = 0.0
        self.running = False
        self.complete = False

        self._opening_cue = False
        self._closing_cues = set()

        self.energy_wander = OrganicWanderer1D(
            seed=seed + 50,
            natural_period_seconds=44.0,
            damping_ratio=0.66,
            drive_strength=0.90,
            drive_smoothing_seconds=13.0,
        )
        self.magic_wander = OrganicWanderer1D(
            seed=seed + 51,
            natural_period_seconds=17.0,
            damping_ratio=0.58,
            drive_strength=1.05,
            drive_smoothing_seconds=5.0,
        )

    def restart(self) -> None:
        self.elapsed_seconds = 0.0
        self.phase = "arrival"
        self.phase_progress = 0.0
        self.performance_progress = 0.0
        self.running = True
        self.complete = False
        self._opening_cue = False
        self._closing_cues.clear()

        for i, voice in enumerate(self.voices):
            voice.generator.clear()
            voice.state.update(
                friction_level=0.0,
                hand_level=0.0,
            )
            p = voice.profile
            home = np.array([p.x, p.y, p.z], dtype=np.float64)
            voice.position = home.copy()
            voice.move_start = home.copy()
            voice.move_target = home.copy()
            voice.move_elapsed = 0.0
            voice.move_duration = 24.0
            voice.next_strike = 2.0 + i * 1.8
            voice.friction_target = 0.0
            voice.hand_target = 0.0
            voice.strikes = 0

    def stop(self) -> None:
        self.running = False
        for voice in self.voices:
            voice.state.update(
                friction_level=0.0,
                hand_level=0.0,
            )

    def _phase_for_progress(
        self,
        progress: float,
    ) -> tuple[str, float]:
        for name, start, end in self.PHASES:
            if progress < end:
                return (
                    name,
                    float(
                        np.clip(
                            (progress - start) / (end - start),
                            0.0,
                            1.0,
                        )
                    ),
                )
        return "complete", 1.0

    def _energy(self, phase: str, local: float) -> float:
        s = local * local * (3.0 - 2.0 * local)
        if phase == "arrival":
            return 0.12 + 0.18 * s
        if phase == "grounding":
            return 0.30 + 0.22 * s
        if phase == "expansion":
            return 0.50 + 0.26 * s
        if phase == "immersion":
            return 0.80 + 0.18 * math.sin(math.pi * s)
        if phase == "alchemy":
            return 0.86 + 0.10 * math.sin(math.pi * s)
        if phase == "integration":
            return 0.72 - 0.30 * s
        if phase == "closing":
            return 0.38 - 0.26 * s
        return 0.0

    def _voice_weight(self, i: int, phase: str, local: float) -> float:
        if phase == "arrival":
            return (1.0, 0.0, 0.0)[i]
        if phase == "grounding":
            return (1.0, 0.55, 0.0)[i]
        if phase == "expansion":
            return (0.95, 0.85, 0.45 + 0.35 * local)[i]
        if phase == "immersion":
            return (0.92, 1.00, 0.85)[i]
        if phase == "alchemy":
            return (0.78, 0.82, 1.00)[i]
        if phase == "integration":
            return (0.82, 0.66, 0.45)[i]
        if phase == "closing":
            return (1.00, 0.18 * (1.0 - local), 0.0)[i]
        return 0.0

    def _schedule_strike(
        self,
        voice: GongVoice,
        energy: float,
        weight: float,
        intensity: float,
    ) -> float:
        effective = max(
            0.04,
            energy * weight * (0.58 + 0.62 * intensity),
        )
        size = voice.profile.size
        minimum = (4.0 + 8.0 * size) / (0.40 + effective)
        maximum = (11.0 + 28.0 * size) / (0.38 + effective)
        minimum = float(np.clip(minimum, 3.0, 25.0))
        maximum = float(np.clip(maximum, minimum + 2.0, 60.0))
        return float(
            math.exp(
                self.rng.uniform(
                    math.log(minimum),
                    math.log(maximum),
                )
            )
        )

    def _choose_target(
        self,
        i: int,
        voice: GongVoice,
        phase: str,
        spatiality: float,
        energy: float,
    ) -> tuple[np.ndarray, float]:
        p = voice.profile
        home = np.array([p.x, p.y, p.z], dtype=np.float64)

        if phase in {"arrival", "closing", "final silence"}:
            target = home + self.rng.normal(
                0.0,
                0.10,
                3,
            )
            return target, float(self.rng.uniform(22.0, 42.0))

        if self.rng.random() > spatiality * (0.35 + 0.65 * energy):
            return (
                home + self.rng.normal(0.0, 0.15, 3),
                float(self.rng.uniform(20.0, 38.0)),
            )

        # Unlike the hand-carried bowls, gongs remain mostly farther from the
        # listener. Spatial movement is broad placement/choreography rather
        # than constant near-head travel.
        span = 1.4 + 1.4 * spatiality
        x = float(self.rng.uniform(-span, span))
        y = float(
            self.rng.uniform(
                -0.7,
                1.0 + 0.4 * (1.0 - p.size),
            )
        )
        z = -float(
            self.rng.uniform(
                1.6 + 0.8 * p.size,
                3.6 + 1.2 * p.size,
            )
        )
        return np.array([x, y, z], dtype=np.float64), float(
            self.rng.uniform(16.0, 34.0)
        )

    def _update_motion(
        self,
        i: int,
        voice: GongVoice,
        dt: float,
        phase: str,
        spatiality: float,
        energy: float,
    ) -> None:
        voice.move_elapsed += dt
        if voice.move_elapsed >= voice.move_duration:
            target, duration = self._choose_target(
                i,
                voice,
                phase,
                spatiality,
                energy,
            )
            voice.move_start = voice.position.copy()
            voice.move_target = target
            voice.move_elapsed = 0.0
            voice.move_duration = duration

        x = float(
            np.clip(
                voice.move_elapsed / max(1.0e-9, voice.move_duration),
                0.0,
                1.0,
            )
        )
        s = x ** 3 * (x * (x * 6.0 - 15.0) + 10.0)
        voice.position = (
            voice.move_start
            + (voice.move_target - voice.move_start) * s
        )

    def _technique_targets(
        self,
        i: int,
        phase: str,
        local: float,
        energy: float,
        friction_presence: float,
        hand_magic: float,
        magic_shape: float,
    ) -> tuple[float, float]:
        if phase in {"arrival", "final silence", "complete"}:
            return 0.0, 0.0

        friction_phase = {
            "grounding": 0.20,
            "expansion": 0.46,
            "immersion": 0.68,
            "alchemy": 0.78,
            "integration": 0.40,
            "closing": 0.14 * (1.0 - local),
        }.get(phase, 0.0)

        hand_phase = {
            "grounding": 0.05,
            "expansion": 0.24,
            "immersion": 0.48,
            "alchemy": 0.88,
            "integration": 0.28,
            "closing": 0.04 * (1.0 - local),
        }.get(phase, 0.0)

        # Brightest gong is most often used for friction "magic"; larger gongs
        # contribute more body and less squeal.
        friction_pref = (0.55, 0.90, 1.00)[i]
        hand_pref = (0.35, 0.78, 1.00)[i]

        # A drifting hand-selection field keeps friction rare enough to surprise.
        selector = 0.5 + 0.5 * math.sin(
            self.elapsed_seconds * (0.021 + 0.006 * i)
            + 1.6 * i
        )

        friction = (
            friction_presence
            * friction_phase
            * energy
            * friction_pref
            * (selector ** 1.6)
        )
        hand = (
            hand_magic
            * hand_phase
            * energy
            * hand_pref
            * (0.35 + 0.65 * magic_shape)
            * (selector ** 2.5)
        )

        return (
            float(np.clip(friction, 0.0, 0.88)),
            float(np.clip(hand, 0.0, 0.92)),
        )

    def advance(self, dt: float) -> None:
        spec = self.state.get()
        if not spec.enabled or not self.running or self.complete:
            return

        dt = max(0.0, float(dt))
        total = spec.duration_minutes * 60.0
        self.elapsed_seconds += dt
        self.performance_progress = float(
            np.clip(
                self.elapsed_seconds / max(1.0, total),
                0.0,
                1.0,
            )
        )
        self.phase, self.phase_progress = self._phase_for_progress(
            self.performance_progress
        )

        if self.phase == "complete":
            self.complete = True
            self.running = False
            self.stop()
            return

        energy = self._energy(
            self.phase,
            self.phase_progress,
        )
        energy *= 0.84 + 0.20 * (
            0.5 + 0.5 * self.energy_wander.advance(dt)
        )
        energy = float(np.clip(energy, 0.0, 1.0))
        magic_shape = 0.5 + 0.5 * self.magic_wander.advance(dt)

        # Immediate unmistakable opening cue.
        if (
            self.phase == "arrival"
            and not self._opening_cue
            and self.elapsed_seconds >= 1.8
        ):
            self.voices[0].generator.strike(0.62, 0.48)
            self.voices[0].strikes += 1
            self._opening_cue = True

        for i, voice in enumerate(self.voices):
            weight = self._voice_weight(
                i,
                self.phase,
                self.phase_progress,
            )

            self._update_motion(
                i,
                voice,
                dt,
                self.phase,
                spec.spatiality,
                energy,
            )

            friction, hand = self._technique_targets(
                i,
                self.phase,
                self.phase_progress,
                energy,
                spec.friction_presence,
                spec.hand_magic,
                magic_shape,
            )
            voice.friction_target = friction * weight
            voice.hand_target = hand * weight

            # Vary pressure/position while active to create changing timbres.
            voice.state.update(
                friction_level=voice.friction_target,
                friction_pressure=float(
                    np.clip(
                        0.36
                        + 0.42 * energy
                        + 0.12 * magic_shape,
                        0.0,
                        1.0,
                    )
                ),
                friction_speed=float(
                    np.clip(
                        0.26 + 0.42 * magic_shape,
                        0.0,
                        1.0,
                    )
                ),
                friction_brightness=float(
                    np.clip(
                        0.42
                        + 0.40 * (1.0 - voice.profile.size)
                        + 0.18 * magic_shape,
                        0.0,
                        1.0,
                    )
                ),
                friction_instability=float(
                    np.clip(
                        0.46 + 0.42 * magic_shape,
                        0.0,
                        1.0,
                    )
                ),
                hand_level=voice.hand_target,
                hand_pressure=float(
                    np.clip(
                        0.40 + 0.46 * magic_shape,
                        0.0,
                        1.0,
                    )
                ),
                hand_position=float(
                    np.clip(
                        0.45
                        + 0.28 * math.sin(
                            self.elapsed_seconds * 0.031 + i
                        )
                        + 0.10 * magic_shape,
                        0.12,
                        0.92,
                    )
                ),
            )

            if self.phase in {"closing", "final silence"}:
                continue

            voice.next_strike -= dt
            if voice.next_strike <= 0.0 and weight > 0.05:
                strength = float(
                    np.clip(
                        self.rng.normal(
                            0.28 + 0.54 * energy,
                            0.10,
                        ),
                        0.14,
                        0.96,
                    )
                )
                location = float(
                    np.clip(
                        self.rng.beta(2.0, 2.0),
                        0.12,
                        0.92,
                    )
                )
                voice.generator.strike(
                    strength,
                    location,
                )
                voice.strikes += 1
                voice.next_strike = self._schedule_strike(
                    voice,
                    energy,
                    weight,
                    spec.intensity,
                )

        if self.phase == "closing":
            cues = (
                (0, 0.12, 0.46),
                (0, 0.47, 0.34),
                (0, 0.78, 0.22),
            )
            for cue_index, (voice_index, cue, strength) in enumerate(cues):
                if (
                    self.phase_progress >= cue
                    and cue_index not in self._closing_cues
                ):
                    self.voices[voice_index].generator.strike(
                        strength,
                        0.50,
                    )
                    self.voices[voice_index].strikes += 1
                    self._closing_cues.add(cue_index)

    def render_mono(self, frame_count: int) -> list[np.ndarray]:
        return [
            voice.generator.generate(frame_count)
            for voice in self.voices
        ]

    @property
    def remaining_seconds(self) -> float:
        spec = self.state.get()
        return max(
            0.0,
            spec.duration_minutes * 60.0 - self.elapsed_seconds,
        )
