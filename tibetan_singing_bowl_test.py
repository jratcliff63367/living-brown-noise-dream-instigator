from __future__ import annotations

import sys
import threading

import numpy as np
import sounddevice as sd
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
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
from synthesized_sound_source import (
    OrganicSpatialMotion,
    SpatialMotionSpec,
    SpatialMotionState,
)
from tibetan_singing_bowl import (
    BowlCeremonyController,
    BowlCeremonySpec,
    BowlCeremonyState,
    SingingBowlSpec,
    SingingBowlState,
    TibetanSingingBowlGenerator,
)


FRAME_SIZE = 1024


class FloatSlider(QWidget):
    def __init__(
        self,
        label: str,
        minimum: float,
        maximum: float,
        value: float,
        *,
        decimals: int = 2,
        steps: int = 1000,
        suffix: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.steps = int(steps)
        self.suffix = suffix
        self.decimals = int(decimals)
        self._callbacks = []

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.name_label = QLabel(label)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self.steps)
        self.value_label = QLabel()
        self.value_label.setMinimumWidth(82)

        layout.addWidget(self.name_label)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.value_label)

        self.slider.valueChanged.connect(self._changed)
        self.set_value(value)

    def _from_slider(self, raw: int) -> float:
        f = raw / self.steps
        return self.minimum + f * (self.maximum - self.minimum)

    def _to_slider(self, value: float) -> int:
        f = (
            (float(value) - self.minimum)
            / max(1.0e-12, self.maximum - self.minimum)
        )
        return int(round(np.clip(f, 0.0, 1.0) * self.steps))

    def value(self) -> float:
        return self._from_slider(self.slider.value())

    def set_value(self, value: float) -> None:
        self.slider.setValue(self._to_slider(value))
        self._update_label(self.value())

    def _update_label(self, value: float) -> None:
        self.value_label.setText(
            f"{value:.{self.decimals}f}{self.suffix}"
        )

    def _changed(self, _raw: int) -> None:
        value = self.value()
        self._update_label(value)
        for callback in self._callbacks:
            callback(value)

    def on_change(self, callback) -> None:
        self._callbacks.append(callback)


class BowlAudioEngine:
    def __init__(self) -> None:
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self.bowl_state = SingingBowlState(SingingBowlSpec())
        self.ceremony_state = BowlCeremonyState(BowlCeremonySpec())
        self.motion_state = SpatialMotionState(
            SpatialMotionSpec(
                enabled=True,
                distance_m=1.35,
                distance_wander_m=0.85,
                azimuth_span_degrees=220.0,
                elevation_span_degrees=70.0,
                motion_speed=0.38,
            )
        )

        self.renderer = SteamAudioRenderer(
            sample_rate=self.sample_rate,
            frame_size=FRAME_SIZE,
            validation_enabled=False,
            log_messages=False,
        )
        self.source = self.renderer.create_source(
            position=Vector3(0.0, 0.0, -1.35),
            spatial_blend=1.0,
            distance_attenuation_enabled=True,
        )
        self.generator = TibetanSingingBowlGenerator(
            self.sample_rate,
            self.bowl_state,
        )
        self.ceremony = BowlCeremonyController(
            self.bowl_state,
            self.ceremony_state,
        )
        self.motion = OrganicSpatialMotion(self.motion_state)

        self.stream: sd.OutputStream | None = None
        self._stream_lock = threading.Lock()
        self.running = False
        self.manual_position = Vector3(0.0, 0.0, -1.35)

    def callback(self, outdata, frames, time_info, status) -> None:
        if status:
            print(status, file=sys.stderr)

        elapsed = frames / self.sample_rate
        self.ceremony.advance(elapsed, self.generator)
        mono = self.generator.generate(frames)

        if self.motion_state.get().enabled:
            position = self.motion.advance(elapsed)
        else:
            position = self.manual_position
        self.source.set_position_vector(position)

        outdata[:] = self.source.process_mono(mono)

    def start(self) -> None:
        with self._stream_lock:
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

    def stop(self) -> None:
        with self._stream_lock:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            self.running = False

    def close(self) -> None:
        self.stop()
        self.renderer.close()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Tibetan Singing Bowl Synthesis Lab")
        self.resize(900, 1050)
        self.engine = BowlAudioEngine()

        root = QWidget()
        layout = QVBoxLayout(root)
        self.setCentralWidget(root)

        top = QHBoxLayout()
        self.start_button = QPushButton("Start Audio")
        self.stop_button = QPushButton("Stop Audio")
        self.strike_button = QPushButton("Strike Bowl")
        self.clear_button = QPushButton("Clear Resonance")
        top.addWidget(self.start_button)
        top.addWidget(self.stop_button)
        top.addWidget(self.strike_button)
        top.addWidget(self.clear_button)
        layout.addLayout(top)

        self.start_button.clicked.connect(self.engine.start)
        self.stop_button.clicked.connect(self.engine.stop)
        self.strike_button.clicked.connect(lambda: self.engine.generator.strike())
        self.clear_button.clicked.connect(self.engine.generator.clear)

        instrument_group = QGroupBox("Bowl instrument")
        instrument_layout = QVBoxLayout(instrument_group)
        layout.addWidget(instrument_group)
        spec = self.engine.bowl_state.get()

        self._add_slider(instrument_layout, "Fundamental", 70.0, 700.0,
                         spec.fundamental_hz, 1, " Hz",
                         lambda v: self.engine.bowl_state.update(fundamental_hz=v))
        self._add_slider(instrument_layout, "Decay", 1.0, 40.0,
                         spec.decay_seconds, 1, " s",
                         lambda v: self.engine.bowl_state.update(decay_seconds=v))
        self._add_slider(instrument_layout, "Default strike strength", 0.0, 1.0,
                         spec.strike_strength, 2, "",
                         lambda v: self.engine.bowl_state.update(strike_strength=v))
        self._add_slider(instrument_layout, "Brightness", 0.0, 1.0,
                         spec.brightness, 2, "",
                         lambda v: self.engine.bowl_state.update(brightness=v))
        self._add_slider(instrument_layout, "Inharmonicity", 0.0, 1.0,
                         spec.inharmonicity, 2, "",
                         lambda v: self.engine.bowl_state.update(inharmonicity=v))
        self._add_slider(instrument_layout, "Modal beating / shimmer", 0.0, 1.0,
                         spec.beating, 2, "",
                         lambda v: self.engine.bowl_state.update(beating=v))
        self._add_slider(instrument_layout, "Body / low-mode weight", 0.0, 1.0,
                         spec.body, 2, "",
                         lambda v: self.engine.bowl_state.update(body=v))
        self._add_slider(instrument_layout, "Manual rubbing level", 0.0, 1.0,
                         spec.rub_level, 2, "",
                         lambda v: self.engine.bowl_state.update(rub_level=v))
        self._add_slider(instrument_layout, "Rubbing instability", 0.0, 1.0,
                         spec.rub_motion, 2, "",
                         lambda v: self.engine.bowl_state.update(rub_motion=v))
        self._add_slider(instrument_layout, "Output gain", -36.0, 0.0,
                         spec.output_gain_db, 1, " dB",
                         lambda v: self.engine.bowl_state.update(output_gain_db=v))

        ceremony_group = QGroupBox("Ceremony metabolism")
        ceremony_layout = QVBoxLayout(ceremony_group)
        layout.addWidget(ceremony_group)

        self.ceremony_enabled = QCheckBox("Enable automatic ceremony evolution")
        self.ceremony_enabled.toggled.connect(
            lambda checked: self.engine.ceremony_state.update(enabled=checked)
        )
        ceremony_layout.addWidget(self.ceremony_enabled)
        c = self.engine.ceremony_state.get()
        self._add_slider(ceremony_layout, "Activity / event density", 0.0, 1.0,
                         c.activity, 2, "",
                         lambda v: self.engine.ceremony_state.update(activity=v))
        self._add_slider(ceremony_layout, "Evolution speed", 0.0, 1.0,
                         c.evolution, 2, "",
                         lambda v: self.engine.ceremony_state.update(evolution=v))
        self._add_slider(ceremony_layout, "Strike tendency", 0.0, 1.0,
                         c.strike_probability, 2, "",
                         lambda v: self.engine.ceremony_state.update(strike_probability=v))
        self._add_slider(ceremony_layout, "Rubbing tendency", 0.0, 1.0,
                         c.rub_probability, 2, "",
                         lambda v: self.engine.ceremony_state.update(rub_probability=v))

        spatial_group = QGroupBox("3D spatial motion")
        spatial_layout = QVBoxLayout(spatial_group)
        layout.addWidget(spatial_group)
        m = self.engine.motion_state.get()

        self.motion_enabled = QCheckBox("Enable organic 3D motion")
        self.motion_enabled.setChecked(m.enabled)
        self.motion_enabled.toggled.connect(
            lambda checked: self.engine.motion_state.update(enabled=checked)
        )
        spatial_layout.addWidget(self.motion_enabled)
        self._add_slider(spatial_layout, "Center distance", 0.25, 6.0,
                         m.distance_m, 2, " m",
                         lambda v: self.engine.motion_state.update(distance_m=v))
        self._add_slider(spatial_layout, "Distance wander", 0.0, 4.0,
                         m.distance_wander_m, 2, " m",
                         lambda v: self.engine.motion_state.update(distance_wander_m=v))
        self._add_slider(spatial_layout, "Azimuth span", 0.0, 360.0,
                         m.azimuth_span_degrees, 0, "°",
                         lambda v: self.engine.motion_state.update(azimuth_span_degrees=v))
        self._add_slider(spatial_layout, "Elevation span", 0.0, 120.0,
                         m.elevation_span_degrees, 0, "°",
                         lambda v: self.engine.motion_state.update(elevation_span_degrees=v))
        self._add_slider(spatial_layout, "Motion speed", 0.0, 1.0,
                         m.motion_speed, 2, "",
                         lambda v: self.engine.motion_state.update(motion_speed=v))

        manual_group = QGroupBox("Manual source position (used when motion is off)")
        manual_layout = QHBoxLayout(manual_group)
        layout.addWidget(manual_group)
        self.manual_x = self._spin(-8.0, 8.0, 0.0)
        self.manual_y = self._spin(-8.0, 8.0, 0.0)
        self.manual_z = self._spin(-8.0, -0.15, -1.35)
        manual_layout.addWidget(QLabel("X"))
        manual_layout.addWidget(self.manual_x)
        manual_layout.addWidget(QLabel("Y"))
        manual_layout.addWidget(self.manual_y)
        manual_layout.addWidget(QLabel("Z"))
        manual_layout.addWidget(self.manual_z)
        for spin in (self.manual_x, self.manual_y, self.manual_z):
            spin.valueChanged.connect(self._update_manual_position)

        status_group = QGroupBox("Live state")
        status_layout = QVBoxLayout(status_group)
        layout.addWidget(status_group)
        self.status_label = QLabel()
        self.status_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        status_layout.addWidget(self.status_label)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh_status)
        self.timer.start(150)
        self._refresh_status()

    @staticmethod
    def _spin(minimum: float, maximum: float, value: float) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(minimum, maximum)
        box.setDecimals(2)
        box.setSingleStep(0.10)
        box.setValue(value)
        return box

    @staticmethod
    def _add_slider(layout, label, minimum, maximum, value,
                    decimals, suffix, callback):
        widget = FloatSlider(
            label, minimum, maximum, value,
            decimals=decimals, suffix=suffix,
        )
        widget.on_change(callback)
        layout.addWidget(widget)
        return widget

    def _update_manual_position(self) -> None:
        self.engine.manual_position = Vector3(
            self.manual_x.value(),
            self.manual_y.value(),
            self.manual_z.value(),
        )
        if not self.motion_enabled.isChecked():
            self.engine.source.set_position_vector(self.engine.manual_position)

    def _refresh_status(self) -> None:
        if self.engine.motion_state.get().enabled:
            p = self.engine.motion.current_position
            az = self.engine.motion.current_azimuth_degrees
            el = self.engine.motion.current_elevation_degrees
            dist = self.engine.motion.current_distance_m
        else:
            p = self.engine.manual_position
            az = 0.0
            el = 0.0
            dist = p.distance()

        self.status_label.setText(
            f"Audio: {'RUNNING' if self.engine.running else 'STOPPED'}\n"
            f"Position: x={p.x:+.2f} m, y={p.y:+.2f} m, z={p.z:+.2f} m\n"
            f"Azimuth: {az:+.1f}°, elevation: {el:+.1f}°, distance: {dist:.2f} m\n"
            f"Ceremony activity: {self.engine.ceremony.current_activity:.2f}; "
            f"automatic rub: {self.engine.ceremony.current_rub:.2f}; "
            f"active strike resonances: {len(self.engine.generator.events)}"
        )

    def closeEvent(self, event) -> None:
        self.engine.close()
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
