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

    # Real-time safety:
    #
    # The original recovery gong retained every strike for seven complete
    # decay constants. On the 44-second large gong that meant a single strike
    # remained in the callback for 308 seconds. A ceremony therefore grew
    # progressively more expensive as old strike tails accumulated.
    #
    # Keep the perceptually important overlapping tails, but put a hard bound
    # on callback work. When the bound is reached we retain the strikes with
    # the strongest estimated remaining energy rather than blindly retaining
    # the oldest history.
    MAX_ACTIVE_STRIKES = 20
    MODE_AUDIBILITY_FLOOR = 0.0015
    CONTACT_NOISE_SECONDS = 0.22

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

    def _estimated_strike_energy(
        self,
        event: _Strike,
        spec: GongSpec,
    ) -> float:
        """Cheap estimate used only to bound the real-time strike population."""
        slowest_decay = max(0.4, spec.decay_seconds * self.DECAYS[0])
        return float(
            event.strength
            * math.exp(-event.age_seconds / slowest_decay)
        )

    def _bound_active_strikes(self, spec: GongSpec) -> None:
        if len(self.strikes) < self.MAX_ACTIVE_STRIKES:
            return

        # Preserve the tails that still carry the most energy. This is much
        # less audible than simply deleting the oldest strike during a dense
        # roll, while guaranteeing that callback cost cannot grow forever.
        ranked = sorted(
            self.strikes,
            key=lambda event: self._estimated_strike_energy(event, spec),
            reverse=True,
        )
        self.strikes = ranked[: self.MAX_ACTIVE_STRIKES - 1]

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

        self._bound_active_strikes(spec)

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
        block_start = event.age_seconds
        block_seconds = frame_count / self.sample_rate
        block_end = block_start + block_seconds

        t = (
            np.arange(frame_count, dtype=np.float64) / self.sample_rate
            + block_start
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

            amplitude = (
                event.strength
                * self.AMPS[i]
                * darkness_curve[i]
                * (0.88 + 0.22 * event.location)
            )

            # Once a particular mode is below the conservative audibility
            # floor for the entire block, do not spend trigonometric/exponential
            # work on it. High gong modes disappear from CPU load much sooner
            # than the slow low-frequency body.
            if block_start > bloom_times[i]:
                tail_at_start = amplitude * math.exp(-block_start / decay)
                if tail_at_start < self.MODE_AUDIBILITY_FLOOR:
                    continue

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

            split_mix = 0.10 + 0.28 * spec.chaos
            output += amplitude * envelope * (
                (1.0 - split_mix) * np.sin(phase_a)
                + split_mix * np.sin(phase_b)
            )

        # Initial mallet/body contact exists only around the actual impact.
        # The old code generated a fresh random-noise array for every historical
        # strike on every callback, even minutes after its contact envelope was
        # effectively zero.
        if block_start < self.CONTACT_NOISE_SECONDS:
            contact_env = np.exp(-t / 0.030)
            contact_noise = self.rng.standard_normal(frame_count)
            output += (
                contact_noise
                * contact_env
                * event.strength
                * (0.010 + 0.020 * (1.0 - spec.darkness))
            )

        return output, block_end

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

            # Retain a tail only while its slowest family still has useful
            # residual energy. The active-strike cap remains the final safety
            # bound during unusually dense passages.
            if (
                self._estimated_strike_energy(event, spec)
                >= self.MODE_AUDIBILITY_FLOOR
            ):
                retained.append(event)

        if len(retained) > self.MAX_ACTIVE_STRIKES:
            retained = sorted(
                retained,
                key=lambda event: self._estimated_strike_energy(event, spec),
                reverse=True,
            )[: self.MAX_ACTIVE_STRIKES]

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
    friction_presence: float = 0.0
    hand_magic: float = 0.0
    spatiality: float = 0.62
    dramatic_gestures: float = 0.72

    def validated(self) -> "GongCeremonySpec":
        if not 8.0 <= self.duration_minutes <= 90.0:
            raise ValueError("duration_minutes must be between 8 and 90")
        for name in (
            "intensity",
            "friction_presence",
            "hand_magic",
            "spatiality",
            "dramatic_gestures",
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
            self._spec = replace(self._spec, **changes).validated()


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
    resonance: object
    strikes: int = 0


from meditation_performer import (
    GestureClock,
    ResonanceEstimate,
    humanized_ramp,
    smoothstep,
)


class GongCeremonyController:
    """
    Human-performance-oriented gong conductor.

    This intentionally leaves ProceduralGongGenerator untouched. Complexity is
    created by performer behavior: phrases, reinforcing taps, multi-gong
    overlap, alternating-roll crescendos, accents, releases, and long periods
    where the performer does nothing because the resonant field is already the
    performance.

    Friction/whale gestures remain disabled by default until their acoustic
    model is independently validated.
    """

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

    GESTURES = (
        "establish",
        "reinforce",
        "cross_feed",
        "build",
        "alternating_roll",
        "release",
        "accent",
        "rest",
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
                    hand_level=0.0,
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
                    position=home,
                    resonance=ResonanceEstimate(
                        decay_seconds=p.decay_seconds * 0.72
                    ),
                )
            )

        self.clock = GestureClock(seed=seed + 400)
        self.elapsed_seconds = 0.0
        self.performance_progress = 0.0
        self.phase = "arrival"
        self.phase_progress = 0.0
        self.running = False
        self.complete = False

        self.primary_index = 0
        self.secondary_index = 1
        self.roll_hand = 0
        self.gesture_count = 0
        self._phrase_phase = float(self.rng.uniform(0, 2 * math.pi))

    @property
    def gesture(self) -> str:
        return self.clock.name

    @property
    def global_resonance(self) -> float:
        return float(
            np.clip(
                sum(v.resonance.normalized for v in self.voices) / 1.65,
                0.0,
                1.0,
            )
        )

    def restart(self) -> None:
        self.elapsed_seconds = 0.0
        self.performance_progress = 0.0
        self.phase = "arrival"
        self.phase_progress = 0.0
        self.running = True
        self.complete = False
        self.gesture_count = 0
        self.roll_hand = 0

        for v in self.voices:
            v.generator.clear()
            v.resonance.value = 0.0
            v.strikes = 0
            v.state.update(friction_level=0.0, hand_level=0.0)

        self._choose_next_gesture(force="establish")

    def stop(self) -> None:
        self.running = False
        for v in self.voices:
            v.state.update(friction_level=0.0, hand_level=0.0)

    def _phase_for_progress(self, p: float) -> tuple[str, float]:
        phases = (
            ("arrival", 0.00, 0.10),
            ("opening", 0.10, 0.26),
            ("development", 0.26, 0.52),
            ("deepening", 0.52, 0.72),
            ("integration", 0.72, 0.88),
            ("closing", 0.88, 0.97),
            ("final silence", 0.97, 1.00),
        )
        for name, start, end in phases:
            if p < end:
                return name, float(
                    np.clip((p - start) / (end - start), 0.0, 1.0)
                )
        return "complete", 1.0

    def _strike(
        self,
        index: int,
        strength: float,
        *,
        location: float | None = None,
    ) -> None:
        v = self.voices[index]
        strength = float(np.clip(strength, 0.08, 0.96))
        if location is None:
            location = float(self.rng.uniform(0.24, 0.82))
        v.generator.strike(strength, location)
        v.strikes += 1

        # The performer tracks the perceptual aftermath rather than changing the
        # sound engine. Softer repeated taps accumulate.
        v.resonance.excite(
            0.28 + 0.78 * strength * strength
        )

    def _choose_primary_pair(self) -> None:
        values = np.array(
            [v.resonance.normalized for v in self.voices],
            dtype=np.float64,
        )

        # Usually favor the large gong, but sometimes deliberately feed a quieter
        # secondary gong to widen the field.
        if self.rng.random() < 0.58:
            self.primary_index = 0
        else:
            quiet_bias = 1.15 - values
            quiet_bias /= quiet_bias.sum()
            self.primary_index = int(
                self.rng.choice(len(self.voices), p=quiet_bias)
            )

        candidates = [
            i for i in range(len(self.voices))
            if i != self.primary_index
        ]
        self.secondary_index = int(self.rng.choice(candidates))

    def _choose_next_gesture(self, force: str | None = None) -> None:
        spec = self.state.get()
        self._choose_primary_pair()
        g = self.global_resonance

        if force is not None:
            name = force
        elif self.phase in {"final silence"}:
            name = "rest"
        elif self.phase == "closing":
            name = self.rng.choice(
                ["release", "rest", "establish"],
                p=[0.52, 0.34, 0.14],
            )
        elif g > 0.80:
            # Important human behavior: once the field is huge, stop feeding it.
            name = self.rng.choice(
                ["release", "rest", "accent"],
                p=[0.62, 0.28, 0.10],
            )
        elif self.phase == "arrival":
            name = self.rng.choice(
                ["establish", "release", "rest"],
                p=[0.58, 0.24, 0.18],
            )
        else:
            dramatic = spec.dramatic_gestures
            choices = [
                "establish",
                "reinforce",
                "cross_feed",
                "build",
                "alternating_roll",
                "release",
                "accent",
                "rest",
            ]
            weights = np.array([
                0.11,
                0.20,
                0.17,
                0.16 + 0.08 * dramatic,
                0.04 + 0.16 * dramatic,
                0.15,
                0.07,
                0.10,
            ])
            if self.phase == "integration":
                weights *= np.array(
                    [1.0, 0.85, 0.72, 0.48, 0.28, 1.35, 0.75, 1.25]
                )
            weights /= weights.sum()
            name = str(self.rng.choice(choices, p=weights))

        if name == "establish":
            duration = self.rng.uniform(12.0, 28.0)
            first = self.rng.uniform(0.4, 1.8)
        elif name == "reinforce":
            duration = self.rng.uniform(16.0, 38.0)
            first = self.rng.uniform(0.5, 2.0)
        elif name == "cross_feed":
            duration = self.rng.uniform(18.0, 42.0)
            first = self.rng.uniform(0.3, 1.2)
        elif name == "build":
            duration = self.rng.uniform(18.0, 36.0)
            first = self.rng.uniform(0.4, 1.2)
        elif name == "alternating_roll":
            duration = self.rng.uniform(10.0, 22.0)
            first = self.rng.uniform(0.15, 0.55)
            self.roll_hand = 0
        elif name == "accent":
            duration = self.rng.uniform(8.0, 18.0)
            first = self.rng.uniform(0.2, 0.8)
        elif name == "release":
            duration = self.rng.uniform(10.0, 32.0)
            first = 9999.0
        else:
            duration = self.rng.uniform(7.0, 22.0)
            first = 9999.0

        self.clock.start(name, duration, first)
        self.gesture_count += 1
        self._phrase_phase = float(
            self.rng.uniform(0.0, 2.0 * math.pi)
        )

    def _perform_event(self) -> None:
        spec = self.state.get()
        p = self.clock.progress
        name = self.clock.name
        primary = self.primary_index
        secondary = self.secondary_index

        if name == "establish":
            strength = float(
                np.clip(self.rng.normal(0.30, 0.07), 0.16, 0.46)
            )
            self._strike(primary, strength)
            self.clock.schedule_log_uniform(4.0, 8.5)

        elif name == "reinforce":
            current = self.voices[primary].resonance.normalized
            # The more alive the gong is, the gentler the maintenance tap.
            center = 0.38 - 0.15 * current
            strength = float(
                np.clip(self.rng.normal(center, 0.06), 0.14, 0.50)
            )
            self._strike(primary, strength)
            self.clock.schedule_log_uniform(2.2, 5.8)

        elif name == "cross_feed":
            # Alternate between two objects so resonance overlaps rather than
            # merely stacking repeated hits on one gong.
            choose_primary = self.rng.random() < 0.54
            index = primary if choose_primary else secondary
            strength = float(
                np.clip(self.rng.normal(0.31, 0.08), 0.14, 0.54)
            )
            self._strike(index, strength)
            self.clock.schedule_log_uniform(1.6, 4.3)

        elif name == "build":
            # Humanized cadence compression. It is intentionally irregular rather
            # than a mathematical metronomic ramp.
            interval = humanized_ramp(
                p,
                4.8,
                1.05,
                wobble=0.16,
                phase=self._phrase_phase,
            )
            strength_center = humanized_ramp(
                p,
                0.26,
                0.44,
                wobble=0.13,
                phase=self._phrase_phase + 1.7,
            )
            strength = float(
                np.clip(
                    self.rng.normal(strength_center, 0.065),
                    0.14,
                    0.62,
                )
            )
            index = (
                primary
                if self.rng.random() < 0.72
                else secondary
            )
            self._strike(index, strength)
            self.clock.schedule_normal(
                interval,
                max(0.08, interval * 0.18),
                max(0.45, interval * 0.58),
                interval * 1.55,
            )

        elif name == "alternating_roll":
            # Reference-derived two-mallet crescendo gesture: rapid alternating
            # contacts, mostly moderate strokes, accumulating into a continuous
            # resonant field rather than a string of loud individual attacks.
            interval = humanized_ramp(
                p,
                1.45,
                0.62,
                wobble=0.10,
                phase=self._phrase_phase,
            )
            strength_center = humanized_ramp(
                p,
                0.28,
                0.46,
                wobble=0.09,
                phase=self._phrase_phase + 0.8,
            )
            strength = float(
                np.clip(
                    self.rng.normal(strength_center, 0.055),
                    0.18,
                    0.60,
                )
            )

            # Two hands strike slightly different regions.
            location = (
                0.38 + self.rng.normal(0.0, 0.035)
                if self.roll_hand == 0
                else 0.63 + self.rng.normal(0.0, 0.035)
            )
            self.roll_hand = 1 - self.roll_hand
            self._strike(primary, strength, location=float(location))

            self.clock.schedule_normal(
                interval,
                interval * 0.10,
                max(0.38, interval * 0.70),
                interval * 1.32,
            )

        elif name == "accent":
            # One meaningful stroke, then leave space around it.
            index = (
                0 if self.rng.random() < 0.72 else primary
            )
            strength = float(
                np.clip(self.rng.normal(0.62, 0.09), 0.44, 0.82)
            )
            self._strike(index, strength)
            self.clock.next_event = 9999.0

    def advance(self, dt: float) -> None:
        spec = self.state.get()
        if not spec.enabled or not self.running or self.complete:
            return

        dt = max(0.0, float(dt))
        total = spec.duration_minutes * 60.0

        self.elapsed_seconds += dt
        self.performance_progress = float(
            np.clip(self.elapsed_seconds / max(1.0, total), 0.0, 1.0)
        )
        self.phase, self.phase_progress = self._phase_for_progress(
            self.performance_progress
        )

        if self.phase == "complete":
            self.complete = True
            self.stop()
            return

        for v in self.voices:
            v.resonance.advance(dt)

        self.clock.advance(dt)

        if self.clock.next_event <= 0.0:
            self._perform_event()

        if self.clock.done:
            self._choose_next_gesture()

    def render_mono(self, frame_count: int) -> list[np.ndarray]:
        return [
            v.generator.generate(frame_count)
            for v in self.voices
        ]

    @property
    def remaining_seconds(self) -> float:
        return max(
            0.0,
            self.state.get().duration_minutes * 60.0
            - self.elapsed_seconds,
        )
