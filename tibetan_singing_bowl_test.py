from __future__ import annotations

import sys
import threading

import numpy as np
import sounddevice as sd

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from steam_audio_renderer import (
    DEFAULT_SAMPLE_RATE,
    SteamAudioRenderer,
    Vector3,
)
from tibetan_singing_bowl import (
    BowlCeremonyController,
    BowlCeremonySpec,
    BowlCeremonyState,
)

FRAME_SIZE = 1024


class FloatSlider(QWidget):
    def __init__(
        self,
        label,
        minimum,
        maximum,
        value,
        *,
        decimals=2,
        suffix="",
        parent=None,
    ):
        super().__init__(parent)
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.decimals = int(decimals)
        self.suffix = suffix
        self.steps = 1000

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

        self._callbacks = []
        self.slider.valueChanged.connect(self._changed)
        self.set_value(value)

    def value(self):
        f = self.slider.value() / self.steps
        return self.minimum + f * (self.maximum - self.minimum)

    def set_value(self, value):
        f = (
            (float(value) - self.minimum)
            / max(1.0e-9, self.maximum - self.minimum)
        )
        self.slider.setValue(
            int(round(np.clip(f, 0.0, 1.0) * self.steps))
        )

    def _changed(self, _):
        value = self.value()
        self.value_label.setText(
            f"{value:.{self.decimals}f}{self.suffix}"
        )
        for callback in self._callbacks:
            callback(value)

    def on_change(self, callback):
        self._callbacks.append(callback)


class BowlCeremonyAudioEngine:
    def __init__(self):
        self.sample_rate = DEFAULT_SAMPLE_RATE

        self.ceremony_state = BowlCeremonyState(
            BowlCeremonySpec(
                enabled=False,
                duration_minutes=30.0,
                intensity=0.62,
                spatiality=0.88,
                rubbing=0.78,
            )
        )
        self.ceremony = BowlCeremonyController(
            self.sample_rate,
            self.ceremony_state,
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
                    position=Vector3(
                        float(p[0]),
                        float(p[1]),
                        float(p[2]),
                    ),
                    spatial_blend=1.0,
                    distance_attenuation_enabled=True,
                )
            )

        self.stream = None
        self._lock = threading.Lock()
        self.running = False

    def callback(self, outdata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)

        dt = frames / self.sample_rate
        self.ceremony.advance(dt)
        mono_blocks = self.ceremony.render_mono(frames)

        stereo = np.zeros((frames, 2), dtype=np.float32)

        for voice, source, mono in zip(
            self.ceremony.voices,
            self.sources,
            mono_blocks,
        ):
            p = voice.position
            source.set_position(
                float(p[0]),
                float(p[1]),
                float(p[2]),
            )
            stereo += source.process_mono(mono)

        # Protect constructive multi-bowl peaks without flattening normal
        # resonance dynamics.
        outdata[:] = (
            0.94 * np.tanh(stereo * 0.82)
        ).astype(np.float32, copy=False)

    def start_audio(self):
        with self._lock:
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
        with self._lock:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            self.running = False

    def start_ceremony(self):
        self.ceremony_state.update(enabled=True)
        self.ceremony.restart()

    def stop_ceremony(self):
        self.ceremony_state.update(enabled=False)
        self.ceremony.stop()

    def close(self):
        self.stop_audio()
        self.renderer.close()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "Tibetan Singing Bowl — Full Ceremony Lab"
        )
        self.resize(940, 800)

        self.engine = BowlCeremonyAudioEngine()

        root = QWidget()
        layout = QVBoxLayout(root)
        self.setCentralWidget(root)

        row = QHBoxLayout()
        buttons = [
            ("Start Audio", self.engine.start_audio),
            ("Stop Audio", self.engine.stop_audio),
            ("Start / Restart Ceremony", self.engine.start_ceremony),
            ("Stop Ceremony", self.engine.stop_ceremony),
        ]
        for text, callback in buttons:
            button = QPushButton(text)
            button.clicked.connect(callback)
            row.addWidget(button)
        layout.addLayout(row)

        description = QLabel(
            "Automated mode performs a complete four-bowl ceremony: "
            "large-to-small bowl layering, overlapping decays, sustained "
            "rim singing, deliberate movement around the head and body, "
            "an immersive middle section, gradual integration, three soft "
            "closing low-bowl strikes, and a true silent tail."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        group = QGroupBox("Ceremony performance")
        controls = QVBoxLayout(group)
        layout.addWidget(group)

        spec = self.engine.ceremony_state.get()
        self._slider(
            controls, "Ceremony duration",
            8.0, 60.0, spec.duration_minutes,
            decimals=1, suffix=" min",
            callback=lambda v:
                self.engine.ceremony_state.update(duration_minutes=v),
        )
        self._slider(
            controls, "Performance intensity",
            0.0, 1.0, spec.intensity,
            callback=lambda v:
                self.engine.ceremony_state.update(intensity=v),
        )
        self._slider(
            controls, "3D movement / proximity",
            0.0, 1.0, spec.spatiality,
            callback=lambda v:
                self.engine.ceremony_state.update(spatiality=v),
        )
        self._slider(
            controls, "Rim-rubbing presence",
            0.0, 1.0, spec.rubbing,
            callback=lambda v:
                self.engine.ceremony_state.update(rubbing=v),
        )

        strike_group = QGroupBox(
            "Manual bowl strikes (optional during ceremony)"
        )
        strike_row = QHBoxLayout(strike_group)
        layout.addWidget(strike_group)

        for index, voice in enumerate(self.engine.ceremony.voices):
            button = QPushButton(voice.profile.name)
            button.clicked.connect(
                lambda _checked=False, i=index:
                    self.engine.ceremony.voices[i].generator.strike()
            )
            strike_row.addWidget(button)

        status_group = QGroupBox("Live performance state")
        status_layout = QVBoxLayout(status_group)
        layout.addWidget(status_group)

        self.status = QLabel()
        self.status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.status.setWordWrap(True)
        status_layout.addWidget(self.status)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(150)
        self.refresh()

    @staticmethod
    def _slider(
        layout,
        label,
        minimum,
        maximum,
        value,
        *,
        decimals=2,
        suffix="",
        callback=None,
    ):
        widget = FloatSlider(
            label,
            minimum,
            maximum,
            value,
            decimals=decimals,
            suffix=suffix,
        )
        if callback is not None:
            widget.on_change(callback)
        layout.addWidget(widget)
        return widget

    @staticmethod
    def _time(seconds):
        seconds = max(0, int(round(seconds)))
        minutes, secs = divmod(seconds, 60)
        return f"{minutes:02d}:{secs:02d}"

    def refresh(self):
        c = self.engine.ceremony
        lines = [
            f"Audio: {'RUNNING' if self.engine.running else 'STOPPED'}",
            (
                f"Ceremony: {'RUNNING' if c.running else 'STOPPED'}; "
                f"phase={c.phase}; "
                f"progress={100.0 * c.performance_progress:.1f}%; "
                f"remaining={self._time(c.remaining_seconds)}"
            ),
            "",
        ]

        for voice in c.voices:
            p = voice.position
            lines.append(
                f"{voice.profile.name}: "
                f"{voice.profile.fundamental_hz:.1f} Hz; "
                f"rub={voice.current_rub:.2f}; "
                f"weight={voice.active_weight:.2f}; "
                f"strikes={voice.strikes}; "
                f"pos=({p[0]:+.2f}, {p[1]:+.2f}, {p[2]:+.2f}) m"
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
