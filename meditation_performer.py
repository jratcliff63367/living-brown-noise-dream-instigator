
from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class ResonanceEstimate:
    """
    Cheap perceptual resonance tracker for performer decisions.

    This deliberately does not alter synthesis. It estimates how "alive" an
    instrument probably is from recent excitation, allowing the virtual artist
    to decide whether to reinforce, crest, or leave it alone.
    """
    value: float = 0.0
    decay_seconds: float = 18.0

    def advance(self, dt: float) -> None:
        self.value *= math.exp(-max(0.0, dt) / max(0.2, self.decay_seconds))

    def excite(self, amount: float) -> None:
        self.value += max(0.0, float(amount))

    @property
    def normalized(self) -> float:
        return float(1.0 - math.exp(-max(0.0, self.value)))


class GestureClock:
    """
    Humanized gesture timer. A gesture has a nominal duration, but its internal
    progress is continuous and its events are intentionally irregular.
    """
    def __init__(self, *, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.name = "rest"
        self.elapsed = 0.0
        self.duration = 1.0
        self.next_event = 1.0

    @property
    def progress(self) -> float:
        return float(np.clip(self.elapsed / max(1e-9, self.duration), 0.0, 1.0))

    @property
    def done(self) -> bool:
        return self.elapsed >= self.duration

    def start(self, name: str, duration: float, first_event: float = 0.0) -> None:
        self.name = str(name)
        self.elapsed = 0.0
        self.duration = max(0.1, float(duration))
        self.next_event = max(0.0, float(first_event))

    def advance(self, dt: float) -> None:
        self.elapsed += max(0.0, float(dt))
        self.next_event -= max(0.0, float(dt))

    def schedule_log_uniform(self, low: float, high: float) -> None:
        low = max(0.03, float(low))
        high = max(low + 1e-3, float(high))
        self.next_event = float(
            math.exp(self.rng.uniform(math.log(low), math.log(high)))
        )

    def schedule_normal(self, mean: float, std: float, low: float, high: float) -> None:
        self.next_event = float(
            np.clip(self.rng.normal(mean, std), low, high)
        )


def smoothstep(value: float) -> float:
    x = float(np.clip(value, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def humanized_ramp(
    progress: float,
    start: float,
    end: float,
    *,
    wobble: float,
    phase: float,
) -> float:
    """
    Slow ramp with human-scale irregularity. Useful for cadence or strength;
    avoids machine-perfect acceleration.
    """
    p = smoothstep(progress)
    base = start + (end - start) * p
    return float(
        max(
            0.0,
            base * (
                1.0
                + wobble * math.sin(phase + 5.4 * p)
                + 0.5 * wobble * math.sin(1.7 * phase + 11.2 * p)
            ),
        )
    )
