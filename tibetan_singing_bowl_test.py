
from __future__ import annotations

import sys
import threading
import numpy as np
import sounddevice as sd

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QGroupBox, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QVBoxLayout, QWidget,
)

from steam_audio_renderer import DEFAULT_SAMPLE_RATE, SteamAudioRenderer, Vector3
from tibetan_singing_bowl import (
    BowlCeremonyController, BowlCeremonySpec, BowlCeremonyState
)

FRAME_SIZE = 1024


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
            0.94 * np.tanh(stereo * 0.82)
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

    def stop_audio(self):
        with self.lock:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
                self.stream = None

    def strike(self, index, technique):
        self.ceremony.voices[index].generator.strike(
            0.56, technique=technique
        )
        self.ceremony.voices[index].strikes += 1

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
        self.setWindowTitle("Singing Bowl Strike-Technique Lab")
        self.resize(980, 720)
        self.engine = Engine()

        root = QWidget()
        layout = QVBoxLayout(root)
        self.setCentralWidget(root)

        top = QHBoxLayout()
        for label, cb in (
            ("Start Audio", self.engine.start_audio),
            ("Stop Audio", self.engine.stop_audio),
            ("Start Ceremony", self.engine.start_ceremony),
            ("Stop Ceremony", self.engine.stop_ceremony),
        ):
            b = QPushButton(label)
            b.clicked.connect(cb)
            top.addWidget(b)
        layout.addLayout(top)

        note = QLabel(
            "The accepted bowl acoustic core is preserved. This lab exposes "
            "explicit SIDE, RIM, and BODY strikes. The previous bowl version "
            "had only a generic strike and therefore did not explicitly model "
            "the common outer-wall side strike as its own technique."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        for i, v in enumerate(self.engine.ceremony.voices):
            group = QGroupBox(v.profile.name)
            row = QHBoxLayout(group)
            for technique in ("side", "rim", "body"):
                b = QPushButton(technique.title() + " strike")
                b.clicked.connect(
                    lambda _checked=False, index=i, t=technique:
                        self.engine.strike(index, t)
                )
                row.addWidget(b)
            layout.addWidget(group)

        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(150)
        self.refresh()

    def refresh(self):
        c = self.engine.ceremony
        lines = [
            f"Ceremony: {'RUNNING' if c.running else 'STOPPED'}",
            f"Phase: {c.phase}",
            "",
        ]
        for v in c.voices:
            lines.append(
                f"{v.profile.name}: strikes={v.strikes}"
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
