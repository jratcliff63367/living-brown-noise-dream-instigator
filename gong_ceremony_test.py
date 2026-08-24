
from __future__ import annotations

import sys
import threading

import numpy as np
import sounddevice as sd

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QGroupBox, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QSlider, QVBoxLayout, QWidget,
)

from steam_audio_renderer import DEFAULT_SAMPLE_RATE, SteamAudioRenderer, Vector3
from gong_ceremony import GongCeremonyController, GongCeremonySpec, GongCeremonyState

FRAME_SIZE = 1024


class FloatSlider(QWidget):
    def __init__(self, label, minimum, maximum, value, decimals=2, suffix=""):
        super().__init__()
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.decimals = int(decimals)
        self.suffix = suffix
        self.steps = 1000
        self.callbacks = []

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.name = QLabel(label)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self.steps)
        self.value_label = QLabel()
        self.value_label.setMinimumWidth(90)
        layout.addWidget(self.name)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.value_label)
        self.slider.valueChanged.connect(self._changed)
        self.set_value(value)

    def value(self):
        f = self.slider.value() / self.steps
        return self.minimum + f * (self.maximum - self.minimum)

    def set_value(self, value):
        f = (float(value) - self.minimum) / max(1e-9, self.maximum - self.minimum)
        self.slider.setValue(int(round(np.clip(f, 0.0, 1.0) * self.steps)))

    def _changed(self, _):
        v = self.value()
        self.value_label.setText(f"{v:.{self.decimals}f}{self.suffix}")
        for cb in self.callbacks:
            cb(v)

    def on_change(self, callback):
        self.callbacks.append(callback)


class Engine:
    def __init__(self):
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self.state = GongCeremonyState(
            GongCeremonySpec(
                enabled=False,
                duration_minutes=30.0,
                intensity=0.64,
                friction_presence=0.82,
                hand_magic=0.88,
                spatiality=0.62,
            )
        )
        self.ceremony = GongCeremonyController(
            self.sample_rate, self.state
        )
        self.renderer = SteamAudioRenderer(
            sample_rate=self.sample_rate,
            frame_size=FRAME_SIZE,
            validation_enabled=False,
            log_messages=False,
        )
        self.sources = []
        for voice in self.ceremony.voices:
            p = voice.position
            self.sources.append(
                self.renderer.create_source(
                    position=Vector3(float(p[0]), float(p[1]), float(p[2])),
                    spatial_blend=1.0,
                    distance_attenuation_enabled=True,
                )
            )
        self.stream = None
        self.lock = threading.Lock()
        self.running = False

    def callback(self, outdata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)

        dt = frames / self.sample_rate
        self.ceremony.advance(dt)
        mono_blocks = self.ceremony.render_mono(frames)
        stereo = np.zeros((frames, 2), dtype=np.float32)

        for voice, source, mono in zip(
            self.ceremony.voices, self.sources, mono_blocks
        ):
            p = voice.position
            source.set_position(float(p[0]), float(p[1]), float(p[2]))
            stereo += source.process_mono(mono)

        outdata[:] = (
            0.95 * np.tanh(stereo * 0.78)
        ).astype(np.float32)

    def start_audio(self):
        with self.lock:
            if self.stream is not None:
                return
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=2,
                dtype="float32",
                blocksize=FRAME_SIZE,
                callback=self.callback,
                latency="high",
            )
            self.stream.start()
            self.running = True

    def stop_audio(self):
        with self.lock:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            self.running = False

    def start_ceremony(self):
        self.state.update(enabled=True)
        self.ceremony.restart()

    def stop_ceremony(self):
        self.state.update(enabled=False)
        self.ceremony.stop()

    def close(self):
        self.stop_audio()
        self.renderer.close()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nonlinear Gong Ceremony Lab")
        self.resize(1040, 920)
        self.engine = Engine()

        root = QWidget()
        layout = QVBoxLayout(root)
        self.setCentralWidget(root)

        row = QHBoxLayout()
        for text, cb in (
            ("Start Audio", self.engine.start_audio),
            ("Stop Audio", self.engine.stop_audio),
            ("Start / Restart Ceremony", self.engine.start_ceremony),
            ("Stop Ceremony", self.engine.stop_ceremony),
        ):
            b = QPushButton(text)
            b.clicked.connect(cb)
            row.addWidget(b)
        layout.addLayout(row)

        desc = QLabel(
            "Third-generation gong body: 106 persistent irregular modes, dispersive metallic impact, "
            "close-mode beating, accumulated resonance, upward energy cascade, "
            "hysteresis, amplitude-dependent pitch pulling, nonlinear "
            "intermodal/subharmonic spectral filling, and friction/hand contact "
            "feeding the same resonant plate."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        group = QGroupBox("Ceremony performance")
        controls = QVBoxLayout(group)
        layout.addWidget(group)

        spec = self.engine.state.get()
        self._slider(
            controls, "Ceremony duration", 8.0, 60.0, spec.duration_minutes,
            decimals=1, suffix=" min",
            callback=lambda v: self.engine.state.update(duration_minutes=v),
        )
        self._slider(
            controls, "Performance intensity", 0.0, 1.0, spec.intensity,
            callback=lambda v: self.engine.state.update(intensity=v),
        )
        self._slider(
            controls, "Friction-tool presence", 0.0, 1.0, spec.friction_presence,
            callback=lambda v: self.engine.state.update(friction_presence=v),
        )
        self._slider(
            controls, "Hand / rim magic", 0.0, 1.0, spec.hand_magic,
            callback=lambda v: self.engine.state.update(hand_magic=v),
        )
        self._slider(
            controls, "3D spatial movement", 0.0, 1.0, spec.spatiality,
            callback=lambda v: self.engine.state.update(spatiality=v),
        )

        manual = QGroupBox("Manual resonance-building tests")
        manual_row = QHBoxLayout(manual)
        layout.addWidget(manual)

        for i, voice in enumerate(self.engine.ceremony.voices):
            b = QPushButton(f"Gentle tap — {voice.profile.name}")
            b.clicked.connect(
                lambda _checked=False, index=i:
                    self.engine.ceremony.voices[index].generator.strike(
                        0.34, 0.48, hardness=0.18
                    )
            )
            manual_row.addWidget(b)

        big = QPushButton("Assertive large-gong strike")
        big.clicked.connect(
            lambda:
                self.engine.ceremony.voices[0].generator.strike(
                    0.78, 0.52, hardness=0.36
                )
        )
        manual_row.addWidget(big)

        tech = QGroupBox("Manual friction / hand audition on bright gong")
        tech_layout = QVBoxLayout(tech)
        layout.addWidget(tech)

        bright = self.engine.ceremony.voices[2].state
        self._slider(
            tech_layout, "Friction level", 0.0, 1.0, 0.0,
            callback=lambda v: bright.update(friction_level=v),
        )
        self._slider(
            tech_layout, "Friction pressure", 0.0, 1.0, 0.52,
            callback=lambda v: bright.update(friction_pressure=v),
        )
        self._slider(
            tech_layout, "Friction speed", 0.0, 1.0, 0.42,
            callback=lambda v: bright.update(friction_speed=v),
        )
        self._slider(
            tech_layout, "Friction brightness", 0.0, 1.0, 0.62,
            callback=lambda v: bright.update(friction_brightness=v),
        )
        self._slider(
            tech_layout, "Friction instability", 0.0, 1.0, 0.68,
            callback=lambda v: bright.update(friction_instability=v),
        )
        self._slider(
            tech_layout, "Hand / rim excitation", 0.0, 1.0, 0.0,
            callback=lambda v: bright.update(hand_level=v),
        )
        self._slider(
            tech_layout, "Hand pressure", 0.0, 1.0, 0.62,
            callback=lambda v: bright.update(hand_pressure=v),
        )
        self._slider(
            tech_layout, "Hand position", 0.0, 1.0, 0.70,
            callback=lambda v: bright.update(hand_position=v),
        )

        status_group = QGroupBox("Live resonant state")
        status_layout = QVBoxLayout(status_group)
        layout.addWidget(status_group)

        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        status_layout.addWidget(self.status)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(150)
        self.refresh()

    @staticmethod
    def _slider(layout, label, minimum, maximum, value,
                decimals=2, suffix="", callback=None):
        w = FloatSlider(label, minimum, maximum, value, decimals, suffix)
        if callback:
            w.on_change(callback)
        layout.addWidget(w)
        return w

    @staticmethod
    def _time(seconds):
        seconds = max(0, int(round(seconds)))
        m, s = divmod(seconds, 60)
        return f"{m:02d}:{s:02d}"

    def refresh(self):
        c = self.engine.ceremony
        lines = [
            f"Audio: {'RUNNING' if self.engine.running else 'STOPPED'}",
            f"Ceremony: {'RUNNING' if c.running else 'STOPPED'}; "
            f"phase={c.phase}; progress={100*c.performance_progress:.1f}%; "
            f"remaining={self._time(c.remaining_seconds)}",
            "",
        ]

        for voice in c.voices:
            net = voice.generator.network
            fam = net.family_energy
            p = voice.position
            fam_text = "/".join(f"{x:.2f}" for x in fam[:3])
            lines.append(
                f"{voice.profile.name}: E={net.total_energy:.3f}; "
                f"body/metal/shimmer={fam_text}; "
                f"nonlinear={'YES' if net.nonlinear_active else 'no'}; "
                f"richness={net.richness:.2f}; strikes={voice.strikes}; "
                f"pos=({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f})m"
            )

        self.status.setText("\n".join(lines))

    def closeEvent(self, event):
        self.engine.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
