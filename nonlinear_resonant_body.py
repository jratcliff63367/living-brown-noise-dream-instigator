
from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class ResonantBodySpec:
    cascade_threshold: float = 0.10
    cascade_rate: float = 0.30
    diffusion_rate: float = 0.05
    nonlinear_enter: float = 0.30
    nonlinear_leave: float = 0.16
    frequency_pull: float = 0.002
    sideband_amount: float = 0.16
    roughness_amount: float = 0.12
    coherence_loss: float = 0.10
    saturation: float = 0.92

    def validated(self) -> "ResonantBodySpec":
        return self


class StatefulModalNetwork:
    """
    Persistent nonlinear modal body with spectral roughness.

    Compared with the previous version, this engine adds:
      * phase decoherence that increases with nonlinear richness;
      * weak stochastic modal jitter;
      * a broadband resonant roughness field derived from current energy;
      * denser intermodal sidebands;
      * stronger qualitative changes between low-energy and high-energy states.
    """

    def __init__(
        self,
        sample_rate: float,
        frequencies_hz: np.ndarray,
        decay_seconds: np.ndarray,
        radiation: np.ndarray,
        family: np.ndarray,
        *,
        spec: ResonantBodySpec,
        seed: int,
    ) -> None:
        self.sample_rate = float(sample_rate)
        self.frequencies = np.asarray(frequencies_hz, dtype=np.float64)
        self.decays = np.asarray(decay_seconds, dtype=np.float64)
        self.radiation = np.asarray(radiation, dtype=np.float64)
        self.family = np.asarray(family, dtype=np.int32)
        self.spec = spec.validated()
        self.mode_count = len(self.frequencies)

        self.rng = np.random.default_rng(seed)
        self.energy = np.zeros(self.mode_count, dtype=np.float64)
        self.phase = self.rng.uniform(0, 2 * math.pi, self.mode_count)
        self.phase_velocity_noise = np.zeros(self.mode_count, dtype=np.float64)

        self.frequency_imperfection = self.rng.normal(
            0.0, 0.0012, self.mode_count
        )
        self.damping_imperfection = np.exp(
            self.rng.normal(0.0, 0.18, self.mode_count)
        )

        logf = np.log(np.maximum(self.frequencies, 1.0))
        distance = np.abs(logf[:, None] - logf[None, :])
        affinity = np.exp(-distance / 0.55)
        upward = (
            self.frequencies[None, :] > self.frequencies[:, None]
        ).astype(np.float64)
        family_adj = (
            np.abs(self.family[:, None] - self.family[None, :]) <= 1
        ).astype(np.float64)

        coupling = affinity * (0.28 + 0.72 * family_adj)
        coupling *= (0.25 + 0.75 * upward)
        coupling *= self.rng.uniform(0.55, 1.45, coupling.shape)
        np.fill_diagonal(coupling, 0.0)
        coupling /= np.maximum(coupling.sum(axis=1, keepdims=True), 1e-12)
        self.coupling = coupling

        self.side_count = max(12, min(32, self.mode_count // 2))
        self.side_a = self.rng.integers(0, self.mode_count, self.side_count)
        self.side_b = self.rng.integers(0, self.mode_count, self.side_count)
        self.side_kind = self.rng.integers(0, 5, self.side_count)
        self.side_phase = self.rng.uniform(0, 2 * math.pi, self.side_count)
        self.side_gain = self.rng.uniform(0.2, 1.0, self.side_count)

        self.nonlinear_active = False
        self.richness = 0.0
        self.total_energy = 0.0
        self.family_energy = np.zeros(
            max(1, int(np.max(self.family)) + 1), dtype=np.float64
        )

        self.rough_state = 0.0

    def clear(self) -> None:
        self.energy.fill(0.0)
        self.phase_velocity_noise.fill(0.0)
        self.nonlinear_active = False
        self.richness = 0.0
        self.total_energy = 0.0
        self.family_energy.fill(0.0)
        self.rough_state = 0.0

    def inject(
        self,
        weights: np.ndarray,
        amount: float,
        *,
        phase_randomization: float = 0.12,
        phase_target: np.ndarray | None = None,
        phase_lock: float = 0.0,
    ) -> None:
        weights = np.asarray(weights, dtype=np.float64)
        w = np.maximum(weights, 0.0)
        s = float(np.sum(w))
        if s <= 1e-12 or amount <= 0.0:
            return
        w /= s
        self.energy += w * float(amount)

        if phase_randomization > 0.0:
            self.phase += self.rng.normal(
                0.0, phase_randomization, self.mode_count
            ) * np.sqrt(w)

        if phase_target is not None and phase_lock > 0.0:
            target = np.asarray(phase_target, dtype=np.float64)
            delta = np.angle(np.exp(1j * (target - self.phase)))
            self.phase += np.clip(phase_lock, 0.0, 1.0) * delta * np.sqrt(w)

        self.phase %= 2 * math.pi

    def _update_richness(self, dt: float) -> None:
        self.total_energy = float(np.sum(self.energy))
        if self.nonlinear_active:
            if self.total_energy < self.spec.nonlinear_leave:
                self.nonlinear_active = False
        else:
            if self.total_energy > self.spec.nonlinear_enter:
                self.nonlinear_active = True

        target = 1.0 if self.nonlinear_active else 0.0
        tau = 0.55 if target > self.richness else 3.4
        self.richness += (target - self.richness) * (
            1.0 - math.exp(-dt / tau)
        )

    def _transfer(self, dt: float) -> None:
        excess = np.maximum(
            self.energy - self.spec.cascade_threshold, 0.0
        )
        fraction = np.clip(
            self.spec.cascade_rate
            * dt
            * (0.10 + 0.90 * self.richness),
            0.0,
            0.42,
        )
        outgoing = excess * fraction
        incoming = outgoing @ self.coupling
        self.energy -= outgoing
        self.energy += incoming * 0.945

        diffusion = np.clip(
            self.spec.diffusion_rate
            * dt
            * (0.18 + 0.82 * self.richness),
            0.0,
            0.10,
        )
        if diffusion > 0:
            local = 0.5 * (
                np.roll(self.energy, 1)
                + np.roll(self.energy, -1)
            )
            self.energy += (local - self.energy) * diffusion

        np.maximum(self.energy, 0.0, out=self.energy)

    def _sideband_freqs(self, f: np.ndarray) -> np.ndarray:
        a = f[self.side_a]
        b = f[self.side_b]
        out = np.empty_like(a)

        k = self.side_kind
        out[k == 0] = np.abs(a[k == 0] - b[k == 0])
        out[k == 1] = 0.5 * (a[k == 1] + b[k == 1])
        out[k == 2] = 0.5 * a[k == 2]
        out[k == 3] = 1.5 * a[k == 3]
        out[k == 4] = np.abs(2.0 * a[k == 4] - b[k == 4])

        return np.clip(out, 20.0, self.sample_rate * 0.44)

    def render(
        self,
        frame_count: int,
        *,
        external_drive: np.ndarray | None = None,
        external_drive_gain: float = 0.0,
    ) -> np.ndarray:
        frame_count = int(frame_count)
        if frame_count <= 0:
            return np.zeros(0, dtype=np.float32)

        dt = frame_count / self.sample_rate
        self._update_richness(dt)
        self._transfer(dt)

        start_energy = self.energy.copy()

        # Irregular damping becomes slightly less damped in the awakened state.
        decay = (
            self.decays
            * self.damping_imperfection
            * (1.0 + 0.22 * self.richness)
        )
        energy_decay = np.exp(
            -2.0 * dt / np.maximum(0.05, decay)
        )
        end_energy = start_energy * energy_decay

        e_norm = np.sqrt(
            start_energy / (0.04 + start_energy)
        )
        family_bias = np.tanh(
            (self.family.astype(np.float64) - 0.35) * 0.9
        )

        # Energy-dependent frequency pull plus stochastic micro-jitter.
        jitter_target = self.rng.normal(
            0.0,
            self.spec.coherence_loss
            * self.richness
            * 0.0018,
            self.mode_count,
        )
        self.phase_velocity_noise += (
            jitter_target - self.phase_velocity_noise
        ) * (1.0 - math.exp(-dt / 0.35))

        effective_freq = (
            self.frequencies
            * (1.0 + self.frequency_imperfection)
            * (
                1.0
                + self.spec.frequency_pull
                * self.richness
                * e_norm
                * family_bias
                + self.phase_velocity_noise
            )
        )

        n = np.arange(frame_count, dtype=np.float64)
        t = n / self.sample_rate

        a0 = np.sqrt(np.maximum(start_energy, 0.0))
        a1 = np.sqrt(np.maximum(end_energy, 0.0))
        ramp = (
            a0[:, None]
            + (a1 - a0)[:, None]
            * (n[None, :] / max(1, frame_count))
        )
        ramp *= self.radiation[:, None]

        angles = (
            self.phase[:, None]
            + 2 * math.pi * effective_freq[:, None] * t[None, :]
        )
        modal = np.sum(ramp * np.sin(angles), axis=0)

        self.phase = (
            self.phase + 2 * math.pi * effective_freq * dt
        ) % (2 * math.pi)

        side = np.zeros(frame_count, dtype=np.float64)
        if self.richness > 1e-4 and self.spec.sideband_amount > 0:
            sf = self._sideband_freqs(effective_freq)
            ea = start_energy[self.side_a]
            eb = start_energy[self.side_b]
            se = np.sqrt(np.maximum(ea * eb, 0.0))
            sa = (
                self.spec.sideband_amount
                * self.richness
                * self.side_gain
                * np.sqrt(se)
            )
            phase = (
                self.side_phase[:, None]
                + 2 * math.pi * sf[:, None] * t[None, :]
            )
            side = np.sum(sa[:, None] * np.sin(phase), axis=0)
            self.side_phase = (
                self.side_phase + 2 * math.pi * sf * dt
            ) % (2 * math.pi)

        # "Resonant roughness": filtered stochastic field whose strength follows
        # both total energy and nonlinear richness. This is what prevents the
        # instrument from reading as a clean oscillator bank.
        rough = np.zeros(frame_count, dtype=np.float64)
        rough_gain = (
            self.spec.roughness_amount
            * (0.15 + 0.85 * self.richness)
            * math.sqrt(max(0.0, self.total_energy))
        )
        if rough_gain > 1e-6:
            noise = self.rng.standard_normal(frame_count)
            alpha = math.exp(
                -2.0 * math.pi * 1800.0 / self.sample_rate
            )
            state = self.rough_state
            for i in range(frame_count):
                state = alpha * state + (1 - alpha) * noise[i]
                rough[i] = noise[i] - state
            self.rough_state = float(state)
            rough *= rough_gain

        out = modal + side + rough

        if external_drive is not None and external_drive_gain > 0:
            drive = np.asarray(external_drive, dtype=np.float64)
            out += drive * float(external_drive_gain)

        self.energy = end_energy
        self.total_energy = float(np.sum(self.energy))
        for family_index in range(len(self.family_energy)):
            self.family_energy[family_index] = float(
                np.sum(
                    self.energy[
                        self.family == family_index
                    ]
                )
            )

        out /= max(
            1.0,
            math.sqrt(self.mode_count) * 0.40,
        )
        out = self.spec.saturation * np.tanh(
            out / max(0.05, self.spec.saturation)
        )
        return out.astype(np.float32, copy=False)
