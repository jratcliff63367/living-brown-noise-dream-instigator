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

from steam_audio_renderer import DEFAULT_SAMPLE_RATE, SteamAudioRenderer, Vector3
from gong_ceremony import (
    GongCeremonyController,
    GongCeremonySpec,
    GongCeremonyState,
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
        self.steps = 1000
        self.decimals = int(decimals)
        self.suffix = suffix
        self.callbacks = []

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.name = QLabel(label)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self.steps)
        self.value_label = QLabel()
        self.value_label.setMinimumWidth(95)

        layout.addWidget(self.name)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.value_label)

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
        for callback in self.callbacks:
            callback(value)

    def on_change(self, callback):
        self.callbacks.append(callback)


class GongAudioEngine:
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
            self.sample_rate,
            self.state,
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

        outdata[:] = (
            0.94 * np.tanh(stereo * 0.80)
        ).astype(np.float32, copy=False)

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
        self.setWindowTitle("Procedural Gong Ceremony Lab")
        self.resize(950, 850)

        self.engine = GongAudioEngine()

        root = QWidget()
        layout = QVBoxLayout(root)
        self.setCentralWidget(root)

        row = QHBoxLayout()
        for text, callback in (
            ("Start Audio", self.engine.start_audio),
            ("Stop Audio", self.engine.stop_audio),
            ("Start / Restart Ceremony", self.engine.start_ceremony),
            ("Stop Ceremony", self.engine.stop_ceremony),
        ):
            button = QPushButton(text)
            button.clicked.connect(callback)
            row.addWidget(button)
        layout.addLayout(row)

        note = QLabel(
            "The ceremony emphasizes what makes gong work fascinating: "
            "huge low blooms, overlapping inharmonic fields, friction-mallet "
            "tones, and especially sparse hand/palm/finger-style surface "
            "excitation that can produce strange vocal, whale-like, squealing, "
            "and metallic emergent sounds."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        performance_group = QGroupBox("Ceremony performance")
        performance_layout = QVBoxLayout(performance_group)
        layout.addWidget(performance_group)

        spec = self.engine.state.get()

        self._slider(
            performance_layout,
            "Ceremony duration",
            8.0,
            60.0,
            spec.duration_minutes,
            decimals=1,
            suffix=" min",
            callback=lambda v: self.engine.state.update(
                duration_minutes=v
            ),
        )
        self._slider(
            performance_layout,
            "Performance intensity",
            0.0,
            1.0,
            spec.intensity,
            callback=lambda v: self.engine.state.update(
                intensity=v
            ),
        )
        self._slider(
            performance_layout,
            "Friction-mallet presence",
            0.0,
            1.0,
            spec.friction_presence,
            callback=lambda v: self.engine.state.update(
                friction_presence=v
            ),
        )
        self._slider(
            performance_layout,
            "Hand / rim magic",
            0.0,
            1.0,
            spec.hand_magic,
            callback=lambda v: self.engine.state.update(
                hand_magic=v
            ),
        )
        self._slider(
            performance_layout,
            "3D spatial movement",
            0.0,
            1.0,
            spec.spatiality,
            callback=lambda v: self.engine.state.update(
                spatiality=v
            ),
        )

        manual_group = QGroupBox("Manual gong tests")
        manual_layout = QHBoxLayout(manual_group)
        layout.addWidget(manual_group)

        for index, voice in enumerate(self.engine.ceremony.voices):
            button = QPushButton(f"Strike {voice.profile.name}")
            button.clicked.connect(
                lambda _checked=False, i=index:
                    self.engine.ceremony.voices[i].generator.strike()
            )
            manual_layout.addWidget(button)

        friction_group = QGroupBox("Manual technique audition")
        friction_layout = QVBoxLayout(friction_group)
        layout.addWidget(friction_group)

        self.friction_level = self._slider(
            friction_layout,
            "Bright gong friction",
            0.0,
            1.0,
            0.0,
            callback=lambda v:
                self.engine.ceremony.voices[2].state.update(
                    friction_level=v
                ),
        )
        self.hand_level = self._slider(
            friction_layout,
            "Bright gong hand/rim excitation",
            0.0,
            1.0,
            0.0,
            callback=lambda v:
                self.engine.ceremony.voices[2].state.update(
                    hand_level=v
                ),
        )
        self.hand_pressure = self._slider(
            friction_layout,
            "Hand pressure",
            0.0,
            1.0,
            0.55,
            callback=lambda v:
                self.engine.ceremony.voices[2].state.update(
                    hand_pressure=v
                ),
        )
        self.hand_position = self._slider(
            friction_layout,
            "Hand position across surface/rim",
            0.0,
            1.0,
            0.68,
            callback=lambda v:
                self.engine.ceremony.voices[2].state.update(
                    hand_position=v
                ),
        )
        self.friction_pressure = self._slider(
            friction_layout,
            "Friction pressure",
            0.0,
            1.0,
            0.48,
            callback=lambda v:
                self.engine.ceremony.voices[2].state.update(
                    friction_pressure=v
                ),
        )
        self.friction_speed = self._slider(
            friction_layout,
            "Friction speed",
            0.0,
            1.0,
            0.40,
            callback=lambda v:
                self.engine.ceremony.voices[2].state.update(
                    friction_speed=v
                ),
        )
        self.friction_brightness = self._slider(
            friction_layout,
            "Friction brightness",
            0.0,
            1.0,
            0.58,
            callback=lambda v:
                self.engine.ceremony.voices[2].state.update(
                    friction_brightness=v
                ),
        )
        self.friction_instability = self._slider(
            friction_layout,
            "Friction instability / squeal",
            0.0,
            1.0,
            0.62,
            callback=lambda v:
                self.engine.ceremony.voices[2].state.update(
                    friction_instability=v
                ),
        )

        status_group = QGroupBox("Live performance state")
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
        if callback:
            widget.on_change(callback)
        layout.addWidget(widget)
        return widget

    @staticmethod
    def _time(seconds):
        seconds = max(0, int(round(seconds)))
        m, s = divmod(seconds, 60)
        return f"{m:02d}:{s:02d}"

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
            spec = voice.state.get()
            lines.append(
                f"{voice.profile.name}: "
                f"{voice.profile.base_hz:.1f} Hz; "
                f"friction={spec.friction_level:.2f}; "
                f"hand={spec.hand_level:.2f}; "
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
