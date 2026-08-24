from __future__ import annotations

import math
import threading
from dataclasses import dataclass, replace
from typing import Generic, TypeVar

import numpy as np

from steam_audio_renderer import Vector3


T = TypeVar("T")


def db_to_linear(db: float) -> float:
    return 10.0 ** (float(db) / 20.0)


def smoothstep5(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value ** 3 * (value * (value * 6.0 - 15.0) + 10.0)


class ThreadSafeSpec(Generic[T]):
    """Small thread-safe wrapper for immutable dataclass specifications."""

    def __init__(self, spec: T) -> None:
        self._lock = threading.Lock()
        self._spec = spec

    def get(self) -> T:
        with self._lock:
            return self._spec

    def set(self, spec: T) -> None:
        with self._lock:
            self._spec = spec

    def update(self, **changes) -> None:
        with self._lock:
            self._spec = replace(self._spec, **changes)


class SmoothedValue:
    """One-pole control smoother independent of audio block size."""

    def __init__(
        self,
        sample_rate: float,
        initial: float,
        smoothing_seconds: float = 0.08,
    ) -> None:
        self.sample_rate = float(sample_rate)
        self.current = float(initial)
        self.smoothing_seconds = max(1.0e-4, float(smoothing_seconds))

    def ramp(self, target: float, frame_count: int) -> np.ndarray:
        target = float(target)
        elapsed = frame_count / self.sample_rate
        amount = 1.0 - math.exp(-elapsed / self.smoothing_seconds)
        end = self.current + (target - self.current) * amount
        values = np.linspace(
            self.current,
            end,
            frame_count,
            endpoint=False,
            dtype=np.float64,
        )
        self.current = float(end)
        return values


class OrganicWanderer1D:
    """
    Slow second-order stochastic motion.

    This owns velocity and momentum rather than generating independent random
    values, so it is useful for parameters which should hesitate, overshoot,
    reverse, and drift organically.
    """

    def __init__(
        self,
        *,
        seed: int,
        natural_period_seconds: float = 12.0,
        damping_ratio: float = 0.72,
        drive_strength: float = 1.0,
        drive_smoothing_seconds: float = 4.0,
        soft_limit: float = 1.3,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.natural_period_seconds = float(natural_period_seconds)
        self.damping_ratio = float(damping_ratio)
        self.drive_strength = float(drive_strength)
        self.drive_smoothing_seconds = float(drive_smoothing_seconds)
        self.soft_limit = float(soft_limit)

        self.position = 0.0
        self.velocity = 0.0
        self.drive = 0.0

    def advance(self, elapsed_seconds: float) -> float:
        remaining = max(0.0, float(elapsed_seconds))
        period = max(0.05, self.natural_period_seconds)
        omega = 2.0 * math.pi / period
        spring_k = omega * omega
        damping_c = 2.0 * self.damping_ratio * omega
        maximum_step = min(1.0 / 120.0, period / 80.0)

        while remaining > 0.0:
            dt = min(maximum_step, remaining)
            remaining -= dt

            decay = math.exp(
                -dt / max(0.01, self.drive_smoothing_seconds)
            )
            innovation = math.sqrt(max(0.0, 1.0 - decay * decay))
            self.drive = (
                self.drive * decay
                + float(self.rng.standard_normal()) * innovation
            )

            acceleration = (
                -spring_k * self.position
                - damping_c * self.velocity
                + self.drive_strength * spring_k * self.drive
            )

            self.velocity += acceleration * dt
            self.position += self.velocity * dt

        return float(math.tanh(self.position / self.soft_limit))


@dataclass(frozen=True, slots=True)
class SpatialMotionSpec:
    """Generic listener-centered motion for synthesized 3D sources."""

    enabled: bool = True
    distance_m: float = 1.5
    distance_wander_m: float = 0.75
    azimuth_span_degrees: float = 150.0
    elevation_span_degrees: float = 45.0
    motion_speed: float = 0.45

    def validated(self) -> "SpatialMotionSpec":
        if not 0.15 <= self.distance_m <= 20.0:
            raise ValueError("distance_m must be between 0.15 and 20")
        if not 0.0 <= self.distance_wander_m <= 10.0:
            raise ValueError("distance_wander_m must be between 0 and 10")
        if not 0.0 <= self.azimuth_span_degrees <= 360.0:
            raise ValueError("azimuth_span_degrees must be between 0 and 360")
        if not 0.0 <= self.elevation_span_degrees <= 160.0:
            raise ValueError("elevation_span_degrees must be between 0 and 160")
        if not 0.0 <= self.motion_speed <= 1.0:
            raise ValueError("motion_speed must be between 0 and 1")
        return self


class SpatialMotionState(ThreadSafeSpec[SpatialMotionSpec]):
    def __init__(self, spec: SpatialMotionSpec) -> None:
        super().__init__(spec.validated())

    def set(self, spec: SpatialMotionSpec) -> None:
        super().set(spec.validated())

    def update(self, **changes) -> None:
        with self._lock:
            self._spec = replace(self._spec, **changes).validated()


class OrganicSpatialMotion:
    """
    Reusable non-orbital 3D motion for procedural sources.

    Independent organic wanderers control azimuth, elevation, and distance.
    Different periods keep the path from collapsing into a simple orbit.
    """

    def __init__(
        self,
        state: SpatialMotionState,
        *,
        seed: int = 551_001,
    ) -> None:
        self.state = state
        self.azimuth = OrganicWanderer1D(
            seed=seed,
            natural_period_seconds=19.0,
            damping_ratio=0.58,
            drive_strength=1.08,
            drive_smoothing_seconds=5.0,
        )
        self.elevation = OrganicWanderer1D(
            seed=seed + 1,
            natural_period_seconds=31.0,
            damping_ratio=0.74,
            drive_strength=0.72,
            drive_smoothing_seconds=8.0,
        )
        self.distance = OrganicWanderer1D(
            seed=seed + 2,
            natural_period_seconds=23.0,
            damping_ratio=0.82,
            drive_strength=0.75,
            drive_smoothing_seconds=7.0,
        )

        self.current_position = Vector3(0.0, 0.0, -1.5)
        self.current_azimuth_degrees = 0.0
        self.current_elevation_degrees = 0.0
        self.current_distance_m = 1.5

    @staticmethod
    def _speed_scale(value: float) -> float:
        if value <= 0.0:
            return 0.0
        slow = 0.18
        fast = 3.0
        return math.exp(
            math.log(slow)
            + float(value) * (math.log(fast) - math.log(slow))
        )

    def advance(self, elapsed_seconds: float) -> Vector3:
        spec = self.state.get().validated()

        if not spec.enabled:
            self.current_position = Vector3(0.0, 0.0, -spec.distance_m)
            self.current_azimuth_degrees = 0.0
            self.current_elevation_degrees = 0.0
            self.current_distance_m = spec.distance_m
            return self.current_position

        scaled_dt = (
            max(0.0, float(elapsed_seconds))
            * self._speed_scale(spec.motion_speed)
        )

        az_n = self.azimuth.advance(scaled_dt)
        el_n = self.elevation.advance(scaled_dt)
        dist_n = self.distance.advance(scaled_dt)

        azimuth_deg = az_n * 0.5 * spec.azimuth_span_degrees
        elevation_deg = el_n * 0.5 * spec.elevation_span_degrees
        distance_m = max(
            0.15,
            spec.distance_m + dist_n * spec.distance_wander_m,
        )

        az = math.radians(azimuth_deg)
        el = math.radians(elevation_deg)
        horizontal = distance_m * math.cos(el)

        self.current_position = Vector3(
            horizontal * math.sin(az),
            distance_m * math.sin(el),
            -horizontal * math.cos(az),
        )
        self.current_azimuth_degrees = azimuth_deg
        self.current_elevation_degrees = elevation_deg
        self.current_distance_m = distance_m
        return self.current_position
