
from __future__ import annotations

import math

import numpy as np


def db_to_linear(db: float) -> float:
    return 10.0 ** (float(db) / 20.0)


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
        self.smoothing_seconds = max(
            1.0e-4,
            float(smoothing_seconds),
        )

    def ramp(
        self,
        target: float,
        frame_count: int,
    ) -> np.ndarray:
        target = float(target)
        elapsed = frame_count / self.sample_rate
        amount = 1.0 - math.exp(
            -elapsed / self.smoothing_seconds
        )
        end = self.current + (
            target - self.current
        ) * amount

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
    Slow second-order stochastic motion with inertia and overshoot.
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
        self.natural_period_seconds = float(
            natural_period_seconds
        )
        self.damping_ratio = float(damping_ratio)
        self.drive_strength = float(drive_strength)
        self.drive_smoothing_seconds = float(
            drive_smoothing_seconds
        )
        self.soft_limit = float(soft_limit)

        self.position = 0.0
        self.velocity = 0.0
        self.drive = 0.0

    def advance(self, elapsed_seconds: float) -> float:
        remaining = max(
            0.0,
            float(elapsed_seconds),
        )
        period = max(
            0.05,
            self.natural_period_seconds,
        )
        omega = (
            2.0 * math.pi / period
        )
        spring_k = omega * omega
        damping_c = (
            2.0
            * self.damping_ratio
            * omega
        )

        maximum_step = min(
            1.0 / 120.0,
            period / 80.0,
        )

        while remaining > 0.0:
            dt = min(
                maximum_step,
                remaining,
            )
            remaining -= dt

            decay = math.exp(
                -dt
                / max(
                    0.01,
                    self.drive_smoothing_seconds,
                )
            )
            innovation = math.sqrt(
                max(
                    0.0,
                    1.0 - decay * decay,
                )
            )
            self.drive = (
                self.drive * decay
                + float(
                    self.rng.standard_normal()
                )
                * innovation
            )

            acceleration = (
                -spring_k * self.position
                - damping_c * self.velocity
                + self.drive_strength
                * spring_k
                * self.drive
            )

            self.velocity += (
                acceleration * dt
            )
            self.position += (
                self.velocity * dt
            )

        return float(
            math.tanh(
                self.position / self.soft_limit
            )
        )
