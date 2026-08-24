
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
from tibetan_singing_bowl import BowlCeremonyController, BowlCeremonySpec, BowlCeremonyState

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
        self.state = BowlCeremonyState(
            BowlCeremonySpec(
                enabled=False,
                duration_minutes=30.0,
                intensity=0.62,
                spatiality=0.88,
                rubbing=0.78,
            )
        )
        self.ceremony = BowlCeremonyController(
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
        outdata[:] = (0.94 * np.tanh(stereo * 0.80)).astype(np.float32)

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
        self.setWindowTitle("Nonlinear Singing Bowl Ceremony Lab")
        self.resize(980, 840)
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
            "Third-generation bowl body: measured non-harmonic shell-mode ratios plus weak satellite modes, "
            "split orthogonal mode pairs, natural beating, persistent resonant "
            "energy, intermodal coupling, and stick-slip rim excitation driving "
            "the same physical resonant body."
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
            controls, "3D movement / proximity", 0.0, 1.0, spec.spatiality,
            callback=lambda v: self.engine.state.update(spatiality=v),
        )
        self._slider(
            controls, "Rim-rubbing presence", 0.0, 1.0, spec.rubbing,
            callback=lambda v: self.engine.state.update(rubbing=v),
        )

        strike_group = QGroupBox("Manual bowl strikes / resonance build")
        strike_row = QHBoxLayout(strike_group)
        layout.addWidget(strike_group)
        for i, voice in enumerate(self.engine.ceremony.voices):
            b = QPushButton(voice.profile.name)
            b.clicked.connect(
                lambda _checked=False, index=i:
                    self.engine.ceremony.voices[index].generator.strike()
            )
            strike_row.addWidget(b)

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
            p = voice.position
            lines.append(
                f"{voice.profile.name}: energy={net.total_energy:.3f}; "
                f"nonlinear={'YES' if net.nonlinear_active else 'no'}; "
                f"richness={net.richness:.2f}; rub={voice.current_rub:.2f}; "
                f"strikes={voice.strikes}; "
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
