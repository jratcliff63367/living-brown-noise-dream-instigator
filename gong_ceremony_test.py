
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
        self.steps = 1000
        self.decimals = decimals
        self.suffix = suffix
        self.callbacks = []

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QLabel(label))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self.steps)
        self.value_label = QLabel()
        self.value_label.setMinimumWidth(88)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.value_label)
        self.slider.valueChanged.connect(self._changed)
        self.set_value(value)

    def value(self):
        f = self.slider.value() / self.steps
        return self.minimum + f * (self.maximum - self.minimum)

    def set_value(self, v):
        f = (float(v) - self.minimum) / max(1e-9, self.maximum - self.minimum)
        self.slider.setValue(int(round(np.clip(f, 0, 1) * self.steps)))

    def _changed(self, _):
        v = self.value()
        self.value_label.setText(f"{v:.{self.decimals}f}{self.suffix}")
        for cb in self.callbacks:
            cb(v)

    def on_change(self, cb):
        self.callbacks.append(cb)


class AudioEngine:
    def __init__(self):
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self.state = GongCeremonyState(
            GongCeremonySpec(
                enabled=False,
                duration_minutes=30.0,
                intensity=0.64,
                spatiality=0.62,
                dramatic_gestures=0.72,
                friction_presence=0.0,
                hand_magic=0.0,
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
        for v in self.ceremony.voices:
            p = v.position
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
        self.ceremony.advance(frames / self.sample_rate)
        mono = self.ceremony.render_mono(frames)
        stereo = np.zeros((frames, 2), dtype=np.float32)
        for v, src, block in zip(self.ceremony.voices, self.sources, mono):
            p = v.position
            src.set_position(float(p[0]), float(p[1]), float(p[2]))
            stereo += src.process_mono(block)
        outdata[:] = (
            0.94 * np.tanh(stereo * 0.80)
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

    def force_gesture(self, name):
        self.state.update(enabled=True)
        if not self.ceremony.running:
            self.ceremony.restart()
        self.ceremony._choose_next_gesture(force=name)

    def strike(self, index, strength=0.34):
        self.ceremony._strike(index, strength)

    def close(self):
        self.stop_audio()
        self.renderer.close()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Human Gong Performer Lab")
        self.resize(1060, 920)
        self.engine = AudioEngine()

        root = QWidget()
        layout = QVBoxLayout(root)
        self.setCentralWidget(root)

        row = QHBoxLayout()
        for label, cb in (
            ("Start Audio", self.engine.start_audio),
            ("Stop Audio", self.engine.stop_audio),
            ("Start Full Performance", self.engine.start_ceremony),
            ("Stop Performance", self.engine.stop_ceremony),
        ):
            b = QPushButton(label)
            b.clicked.connect(cb)
            row.addWidget(b)
        layout.addLayout(row)

        note = QLabel(
            "The acoustic gong core is the recovered first-pass version. "
            "This build changes performer behavior, not gong timbre. "
            "Use the gesture buttons to audition human-style phrases directly."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        cgroup = QGroupBox("Performance controls")
        cl = QVBoxLayout(cgroup)
        layout.addWidget(cgroup)

        spec = self.engine.state.get()
        self._slider(
            cl, "Ceremony duration", 8, 60, spec.duration_minutes,
            decimals=1, suffix=" min",
            callback=lambda v: self.engine.state.update(duration_minutes=v),
        )
        self._slider(
            cl, "Overall intensity", 0, 1, spec.intensity,
            callback=lambda v: self.engine.state.update(intensity=v),
        )
        self._slider(
            cl, "Dramatic gesture frequency", 0, 1, spec.dramatic_gestures,
            callback=lambda v: self.engine.state.update(dramatic_gestures=v),
        )

        ggroup = QGroupBox("Force one performer gesture")
        gl = QHBoxLayout(ggroup)
        layout.addWidget(ggroup)

        for name in (
            "establish", "reinforce", "cross_feed", "build",
            "alternating_roll", "accent", "release", "rest"
        ):
            b = QPushButton(name.replace("_", " ").title())
            b.clicked.connect(
                lambda _checked=False, n=name: self.engine.force_gesture(n)
            )
            gl.addWidget(b)

        mgroup = QGroupBox("Manual baseline strikes")
        ml = QHBoxLayout(mgroup)
        layout.addWidget(mgroup)
        for i, v in enumerate(self.engine.ceremony.voices):
            b = QPushButton(v.profile.name)
            b.clicked.connect(
                lambda _checked=False, index=i: self.engine.strike(index)
            )
            ml.addWidget(b)

        sgroup = QGroupBox("Live virtual-performer state")
        sl = QVBoxLayout(sgroup)
        layout.addWidget(sgroup)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        sl.addWidget(self.status)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(120)
        self.refresh()

    @staticmethod
    def _slider(layout, label, minimum, maximum, value,
                decimals=2, suffix="", callback=None):
        w = FloatSlider(label, minimum, maximum, value, decimals, suffix)
        if callback:
            w.on_change(callback)
        layout.addWidget(w)
        return w

    def refresh(self):
        c = self.engine.ceremony
        lines = [
            f"Audio: {'RUNNING' if self.engine.running else 'STOPPED'}",
            f"Performance: {'RUNNING' if c.running else 'STOPPED'}",
            f"Arc phase: {c.phase}  |  Gesture: {c.gesture}",
            f"Gesture progress: {100*c.clock.progress:.1f}%  |  "
            f"next stroke in {max(0,c.clock.next_event):.2f}s",
            f"Global resonance estimate: {c.global_resonance:.3f}",
            "",
        ]
        for v in c.voices:
            lines.append(
                f"{v.profile.name}: resonance={v.resonance.normalized:.3f}; "
                f"strikes={v.strikes}"
            )
        self.status.setText("\n".join(lines))

    def closeEvent(self, event):
        self.engine.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
