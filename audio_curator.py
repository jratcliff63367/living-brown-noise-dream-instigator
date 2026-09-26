#!/usr/bin/env python3
"""Living Brown Noise — Audio Curator. Python 3.10+; pip install PySide6 mutagen.

Place beside ceremonies/ and run: python audio_curator.py
Or: python audio_curator.py --project-root D:\\github\\your-project

SHARED METADATA CONTRACT (for later playback-engine integration)
  <project>/audio-curation.json, UTF-8 JSON, schema_version=1.
  tracks is a mapping keyed by POSIX path RELATIVE TO ceremonies/, e.g.
  "indian/vocals/example.mp3". Keys never change when a file is stashed.
  Each record has state: unreviewed | low | medium | high | stash.
  stash_marked=true is a review flag only; state and runtime eligibility stay
  unchanged until the explicit move operation.
  previous_state preserves the pre-stash rating. Stash mirrors the same key
  under <project>/stash/. Missing entries default to unreviewed.
  Read-only consumers: exclude stash and any pending_move.key; exhaust high,
  then medium, then low+unreviewed, separately for each playback category.
  This app does not change playback-engine selection rules or code.

  Other fields: category, updated_at. Top-level updated_at and last_selected
  are informational. Unknown metadata fields are preserved. UI preferences
  live in audio-curator-ui.ini beside this project, not in the shared ratings file. Files outside the scan
  remain in metadata; they are never silently dropped.

SAFETY
  Saves use a closed temporary file + atomic replacement (Windows compatible).
  A move is journaled before touching audio and recovered on the next launch.
  Destinations are never overwritten. Symlinks and reference directories are
  excluded. An OS-backed QLockFile prevents two curator writers per project;
  external metadata changes are detected before writing. No paid APIs used.

UI research: Qt's multimedia player example and W3C's Use of Color guidance.
https://doc.qt.io/qtforpython-6/examples/example_multimedia_player.html
https://www.w3.org/WAI/WCAG22/Understanding/use-of-color.html
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
import time
from datetime import datetime, timezone


METADATA_NAME = "audio-curation.json"
RATINGS = ("unreviewed", "low", "medium", "high")
STATES = RATINGS + ("stash",)
COLORS = {"unreviewed": "#abb7c9", "low": "#efbd66", "medium": "#7bbaff",
          "high": "#6cdbac", "stash": "#c6a4ee"}
UI_STATES = ("unreviewed", "high", "medium", "low", "marked", "stash")
STATE_ORDER = {state: rank for rank, state in enumerate(UI_STATES)}
COLORS["marked"] = "#ed9dcc"
LABELS = {s: s.capitalize() for s in UI_STATES}


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def validate_key(key):
    if not isinstance(key, str) or not key or "\\" in key or ":" in key:
        raise ValueError(f"Invalid relative audio path: {key!r}")
    p = PurePosixPath(key)
    if p.is_absolute() or any(x in (".", "..", "") for x in key.split("/")):
        raise ValueError(f"Unsafe relative audio path: {key!r}")
    return key


def category_for(key):
    parts = [s.casefold() for s in PurePosixPath(key).parts[:-1]]
    for part, category in (("vocals", "Vocals"), ("instruments", "Instruments"),
                           ("ambients", "Ambients"), ("effects", "Ambients")):
        if part in parts:
            return category
    return "Other"


def excluded_folder(name):
    n = name.casefold()
    return (n.startswith(".") or n in ("stash", "__pycache__") or
            "reference" in n or bool({"ref", "refs"} & set(re.split(r"[-_\s]+", n))))


def scan_audio(root):
    """Filesystem-only scan: no MP3 decoding, no recursive symlinks."""
    root = Path(root)
    found, errors = {}, []
    for top in ("ceremonies", "stash"):
        base = root / top
        if base.is_symlink():
            errors.append(f"Skipped linked folder: {base}")
            continue
        if not base.exists():
            continue
        def onerror(error):
            errors.append(str(error))
        for folder, dirs, names in os.walk(base, onerror=onerror, followlinks=False):
            dirs[:] = sorted(d for d in dirs if not excluded_folder(d)
                             and not (Path(folder) / d).is_symlink())
            for name in sorted(names):
                p = Path(folder) / name
                if p.suffix.casefold() != ".mp3" or p.is_symlink() or not p.is_file():
                    continue
                key = p.relative_to(base).as_posix()
                validate_key(key)
                found.setdefault(key, set()).add(top)
    return found, errors


def read_durations(root, found):
    """Read MP3 headers on the scan worker, never decode audio on the GUI thread."""
    durations, errors = {}, {}
    try:
        from mutagen.mp3 import MP3
    except ImportError:
        return {}, {key: 'Duration unavailable: install mutagen in this Python environment'
                    for key in found}
    import math
    for key, locations in found.items():
        try:
            if len(locations) != 1:
                raise ValueError('Conflicting active/stashed copies; duration not counted')
            path = Path(root) / next(iter(locations)) / key
            seconds = float(MP3(path).info.length)
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError('Invalid MP3 duration')
            durations[key] = seconds
        except Exception as exc:
            errors[key] = str(exc)
    return durations, errors


def duration_text(seconds):
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours}:{minutes:02d}:{seconds:02d}'


class Catalog:
    """Pure-Python persistence and file operations; caller serializes mutations."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.path = self.root / METADATA_NAME
        self.blocked = False
        raw = self.path.read_bytes() if self.path.exists() else None
        self.fingerprint = self._digest(raw)
        self.data = json.loads(raw.decode("utf-8")) if raw is not None else {
            "schema_version": 1, "tracks": {}, "last_selected": None}
        if not isinstance(self.data, dict) or self.data.get("schema_version") != 1:
            raise ValueError("Unsupported curation metadata version; existing data was not changed.")
        if not isinstance(self.data.get("tracks"), dict):
            raise ValueError("Curation metadata has no valid tracks map; existing data was not changed.")
        for key, record in self.data["tracks"].items():
            validate_key(key)
            if not isinstance(record, dict) or record.get("state") not in STATES:
                raise ValueError(f"Invalid state for {key}; metadata was not changed.")
            if record.get("previous_state", "unreviewed") not in RATINGS:
                raise ValueError(f"Invalid previous_state for {key}.")
        self.recover_move()

    @staticmethod
    def _digest(raw):
        return hashlib.sha256(raw).hexdigest() if raw is not None else None

    def _check_disk(self):
        if self.blocked:
            raise RuntimeError("A file operation needs recovery. Close and reopen the curator.")
        raw = self.path.read_bytes() if self.path.exists() else None
        if self._digest(raw) != self.fingerprint:
            raise RuntimeError("The ratings file changed outside this app. Close and reopen the "
                               "curator to load it. Your changes have not overwritten it.")

    def commit(self, candidate):
        self._check_disk()
        candidate = copy.deepcopy(candidate)
        candidate["updated_at"] = utc_now()
        raw = (json.dumps(candidate, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        fd, temp = tempfile.mkstemp(prefix=".audio-curation-", suffix=".tmp", dir=self.root)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            # Stream is closed before replacement: important on Windows.
            self._check_disk()
            os.replace(temp, self.path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
        self.data = candidate
        self.fingerprint = self._digest(raw)

    def path_for(self, key, location):
        validate_key(key)
        if location not in ("ceremonies", "stash"):
            raise ValueError("Invalid location")
        base = self.root / location
        path = base.joinpath(*PurePosixPath(key).parts)
        # Reject linked roots, parents and files, including broken links.
        check = path
        while check != self.root:
            if check.is_symlink():
                raise ValueError(f"Linked paths cannot be moved: {check}")
            check = check.parent
        if not path.resolve().is_relative_to(base.resolve()):
            raise ValueError("Audio path escapes its folder")
        return path

    def reconcile(self, found):
        candidate = copy.deepcopy(self.data)
        problems = {}
        for key, locations in found.items():
            record = candidate["tracks"].setdefault(key, {"state": "unreviewed"})
            record["category"] = category_for(key)
            if len(locations) != 1:
                problems[key] = "Both ceremonies and stash contain this path. Resolve the duplicate first."
                continue
            in_stash = "stash" in locations
            if in_stash and record["state"] != "stash":
                record["previous_state"] = record["state"]
                record["state"] = "stash"
            elif not in_stash and record["state"] == "stash":
                record["state"] = record.pop("previous_state", "unreviewed")
            if in_stash:
                record.pop("stash_marked", None)
            if record != self.data["tracks"].get(key):
                record["updated_at"] = utc_now()
        if candidate != self.data or not self.path.exists():
            self.commit(candidate)
        return problems

    def state(self, key):
        return self.data["tracks"].get(key, {}).get("state", "unreviewed")

    def is_marked(self, key):
        return self.state(key) != "stash" and bool(
            self.data["tracks"].get(key, {}).get("stash_marked", False))

    def display_state(self, key):
        return "marked" if self.is_marked(key) else self.state(key)

    def mark_stash(self, key, marked=True):
        if self.state(key) == "stash":
            raise ValueError("This file is already in stash. Restore it first.")
        if not self.path_for(key, "ceremonies").is_file():
            raise FileNotFoundError(key)
        previous = self.is_marked(key)
        candidate = copy.deepcopy(self.data)
        record = candidate["tracks"].setdefault(key, {"state": "unreviewed"})
        if marked:
            record["stash_marked"] = True
        else:
            record.pop("stash_marked", None)
        record["updated_at"] = utc_now()
        candidate["last_selected"] = key
        self.commit(candidate)
        return previous

    def rate(self, key, state, *, restore_mark=False):
        if state not in RATINGS:
            raise ValueError("Use move() to stash or restore a file.")
        if self.state(key) == "stash":
            raise ValueError("Restore this file before rating it.")
        if not self.path_for(key, "ceremonies").is_file():
            raise FileNotFoundError(key)
        previous = self.state(key)
        candidate = copy.deepcopy(self.data)
        record = candidate["tracks"].setdefault(key, {})
        record.update(state=state, category=category_for(key), updated_at=utc_now())
        # Selecting a rating also cancels a pending stash decision.
        if restore_mark:
            record["stash_marked"] = True
        else:
            record.pop("stash_marked", None)
        candidate["last_selected"] = key
        self.commit(candidate)
        return previous

    @staticmethod
    def _move_noreplace(source, destination):
        """Never overwrite. Windows rename refuses existing destinations.

        On POSIX, a hard link followed by unlink provides the same protection.
        These sibling folders normally live on the same filesystem. Unsupported
        cross-device moves fail safely rather than copying/deleting paid audio.
        """
        if os.path.lexists(destination):
            raise FileExistsError(f"Destination already exists: {destination}")
        if os.name == "nt":
            for attempt in range(8):
                try:
                    os.rename(source, destination)
                    return
                except PermissionError:
                    if attempt == 7:
                        raise
                    time.sleep(0.15)
        else:
            os.link(source, destination)
            os.unlink(source)

    def _finish_move(self, pending):
        candidate = copy.deepcopy(self.data)
        key = pending["key"]
        record = candidate["tracks"].setdefault(key, {})
        record.update(state=pending["target_state"], category=category_for(key), updated_at=utc_now())
        record.pop("stash_marked", None)
        if pending["target_state"] == "stash":
            record["previous_state"] = pending["old_state"]
        else:
            record.pop("previous_state", None)
        candidate.pop("pending_move", None)
        candidate["last_selected"] = key
        self.commit(candidate)

    def recover_move(self):
        pending = self.data.get("pending_move")
        if not pending:
            return
        key = pending["key"]
        if pending.get("old_state") not in STATES or pending.get("target_state") not in STATES:
            raise ValueError("Invalid pending file operation")
        if pending["target_state"] == "stash":
            if pending["old_state"] not in RATINGS:
                raise ValueError("Invalid pending stash")
            source_location, target_location = "ceremonies", "stash"
        else:
            if pending["old_state"] != "stash":
                raise ValueError("Invalid pending restore")
            source_location, target_location = "stash", "ceremonies"
        source = self.path_for(key, source_location)
        destination = self.path_for(key, target_location)
        if source.exists() and destination.exists():
            # A POSIX move may have stopped between link() and unlink().
            if os.path.samefile(source, destination):
                source.unlink()
            else:
                raise RuntimeError(f"Move recovery found two different files for {key}. "
                                   "Both have been preserved. Resolve the duplicate before reopening.")
        if destination.is_file() and not source.exists():
            self._finish_move(pending)
        elif source.is_file() and not destination.exists():
            candidate = copy.deepcopy(self.data)
            candidate.pop("pending_move", None)
            self.commit(candidate)  # The move never occurred; original rating stands.
        else:
            raise RuntimeError(f"Cannot recover move for {key}: source and destination are missing.")

    def move(self, key, restore=False):
        old_state = self.state(key)
        if restore != (old_state == "stash"):
            raise ValueError("File state changed; refresh the list first.")
        source_location, target_location = (("stash", "ceremonies") if restore
                                             else ("ceremonies", "stash"))
        source = self.path_for(key, source_location)
        destination = self.path_for(key, target_location)
        if not source.is_file():
            raise FileNotFoundError(source)
        if os.path.lexists(destination):
            raise FileExistsError(f"Destination already exists; neither file was changed:\n{destination}")
        target_state = (self.data["tracks"][key].get("previous_state", "unreviewed")
                        if restore else "stash")
        pending = {"key": key, "old_state": old_state, "target_state": target_state}
        candidate = copy.deepcopy(self.data)
        candidate["pending_move"] = pending
        self.commit(candidate)
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            self._move_noreplace(source, destination)
        except Exception:
            try:
                self.recover_move()
            except Exception:
                self.blocked = True
            raise
        try:
            self._finish_move(pending)
        except Exception:
            self.blocked = True
            raise RuntimeError("Audio was moved safely, but saving its new state failed. "
                               "Close and reopen the curator to recover the saved operation.") from None
        return old_state


def run_gui(project_root=None):
    try:
        from PySide6.QtCore import Qt, QUrl, QTimer, QThread, Signal, QSettings, QLockFile, QSize
        from PySide6.QtGui import (QColor, QPalette, QPainter, QPen, QBrush, QFont,
                                   QIcon, QPixmap, QShortcut, QKeySequence)
        from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput, QMediaDevices
        from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
            QHBoxLayout, QLabel, QPushButton, QComboBox, QLineEdit, QTreeWidget,
            QTreeWidgetItem, QSlider, QStyle, QStyleOptionSlider, QStyledItemDelegate,
            QStyleOptionViewItem, QHeaderView, QFrame, QProgressBar, QCheckBox,
            QFileDialog, QMessageBox, QMenu, QAbstractItemView)
    except ImportError as exc:
        print(f"This app needs PySide6 (including QtMultimedia).\n"
              f'Install using this Python: "{sys.executable}" -m pip install PySide6\n{exc}',
              file=sys.stderr)
        return 1

    ROLE = int(Qt.ItemDataRole.UserRole)

    class Job(QThread):
        result = Signal(object)
        failed = Signal(str)

        def __init__(self, fn, parent):
            super().__init__(parent)
            self.fn = fn

        def run(self):
            try:
                self.result.emit(self.fn())
            except Exception as exc:
                self.failed.emit(str(exc))

    class DurationJob(QThread):
        batch = Signal(object)

        def __init__(self, root, found, parent):
            super().__init__(parent)
            self.root, self.found = root, dict(found)

        def run(self):
            import math
            import time
            cache_path = self.root / "audio-curator-durations.json"
            try:
                cache = json.loads(cache_path.read_text(encoding="utf-8"))
                if not isinstance(cache, dict):
                    cache = {}
            except Exception:
                cache = {}
            results, updated = {}, {}
            last_emit = time.monotonic()
            for key, locations in self.found.items():
                if self.isInterruptionRequested():
                    break
                try:
                    if len(locations) != 1:
                        raise ValueError("Conflicting active/stashed copies")
                    path = self.root / next(iter(locations)) / key
                    stat = path.stat()
                    signature = [stat.st_size, stat.st_mtime_ns]
                    entry = cache.get(key, {})
                    seconds = entry.get("seconds") if isinstance(entry, dict) else None
                    if (not isinstance(entry, dict) or entry.get("signature") != signature
                            or not isinstance(seconds, (int, float))
                            or not math.isfinite(seconds) or seconds <= 0):
                        durations, errors = read_durations(self.root, {key: locations})
                        if key not in durations:
                            raise ValueError(errors[key])
                        seconds = durations[key]
                    updated[key] = {"signature": signature, "seconds": seconds}
                    results[key] = (seconds, None)
                except Exception as exc:
                    results[key] = (None, str(exc))
                if len(results) >= 20 or time.monotonic() - last_emit >= 0.15:
                    self.batch.emit(results)
                    results = {}
                    last_emit = time.monotonic()
            if results:
                self.batch.emit(results)
            # Cache is disposable, separate from shared ratings. Preserve entries
            # not visited if closing interrupted this scan.
            cache.update(updated)
            temp = None
            try:
                fd, temp = tempfile.mkstemp(prefix=".duration-cache-", dir=self.root)
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    json.dump(cache, stream)
                os.replace(temp, cache_path)
            except OSError:
                pass
            finally:
                if temp and os.path.exists(temp):
                    os.unlink(temp)

    class SeekSlider(QSlider):
        seekRequested = Signal(int)

        def mousePressEvent(self, event):
            if event.button() == Qt.MouseButton.LeftButton:
                option = QStyleOptionSlider()
                self.initStyleOption(option)
                handle = self.style().subControlRect(QStyle.ComplexControl.CC_Slider, option,
                                                    QStyle.SubControl.SC_SliderHandle, self)
                if not handle.contains(event.position().toPoint()):
                    value = QStyle.sliderValueFromPosition(self.minimum(), self.maximum(),
                        int(event.position().x()) - handle.width() // 2,
                        max(1, self.width() - handle.width()), option.upsideDown)
                    self.setValue(value)
                    self.seekRequested.emit(value)
                    event.accept()
                    return
            super().mousePressEvent(event)

    class SortedItem(QTreeWidgetItem):
        def __lt__(self, other):
            def key(item):
                state = item.data(1, ROLE + 1)
                return (1 if state else 0, STATE_ORDER.get(state, -1),
                        item.text(0).casefold(), item.text(0))
            return key(self) < key(other)

    class RatingDelegate(QStyledItemDelegate):
        def paint(self, painter, option, index):
            state = index.data(ROLE + 1)
            if not state:
                super().paint(painter, option, index)
                return
            opt = QStyleOptionViewItem(option)
            self.initStyleOption(opt, index)
            opt.text = ""
            opt.widget.style().drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, opt.widget)
            painter.save()
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            color = QColor(COLORS[state])
            rect = option.rect.adjusted(8, 5, -8, -5)
            rect.setWidth(min(114, rect.width()))
            painter.setBrush(QColor(color.red(), color.green(), color.blue(), 28))
            painter.setPen(QPen(color, 1))
            painter.drawRoundedRect(rect, 5, 5)
            painter.setPen(color)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, LABELS[state])
            painter.restore()

    class Window(QMainWindow):
        def __init__(self, root, lock, prefs):
            super().__init__()
            self.root, self.lock, self.prefs = root, lock, prefs
            self._ui_ready = False
            self._restoring_view = False
            self._view_initialized = False
            self.folder_items = {}
            self.duration_scope = None
            self.selected_folder = self.prefs.value("selected_folder", "")
            self.duration_job = None
            self._close_pending = False
            self.catalog = Catalog(root)
            self.found, self.problems, self.items = {}, {}, {}
            self.busy, self.loaded_key, self.selected_key = False, None, None
            self.undo_stack, self.jobs, self.shortcuts = [], [], []
            self.duration = 0
            self.durations, self.duration_errors = {}, {}
            self.audio = QAudioOutput(self)
            self.player = QMediaPlayer(self)
            self.player.setAudioOutput(self.audio)
            self.devices = QMediaDevices(self)
            self.device_ids = []
            self.setWindowTitle("Audio Curator · Living Brown Noise")
            self.resize(1180, 820)
            self.setMinimumSize(860, 620)
            self.build_ui()
            self.connect_player()
            self.restore_preferences()
            self.populate_devices()
            self.devices.audioOutputsChanged.connect(self.populate_devices)
            self.install_shortcuts()
            self.setup_ui_persistence()
            app.focusChanged.connect(self.update_shortcut_context)
            QTimer.singleShot(0, self.refresh)

        def button(self, text, callback, tooltip=""):
            button = QPushButton(text)
            button.clicked.connect(callback)
            button.setToolTip(tooltip)
            button.setMinimumHeight(34)
            return button

        def build_ui(self):
            central = QWidget()
            layout = QVBoxLayout(central)
            layout.setContentsMargins(22, 18, 22, 14)
            layout.setSpacing(12)
            self.setCentralWidget(central)
            heading = QHBoxLayout()
            titlebox = QVBoxLayout()
            title = QLabel("Audio Curator")
            title.setObjectName("title")
            titlebox.addWidget(title)
            subtitle = QLabel("Living Brown Noise   /   Listen. Rate. Keep the best close.")
            subtitle.setObjectName("muted")
            titlebox.addWidget(subtitle)
            heading.addLayout(titlebox, 1)
            self.undo_btn = self.button("Undo", self.undo, "Undo the last rating, stash or restore · Ctrl+Z")
            self.undo_btn.setEnabled(False)
            self.refresh_btn = self.button("Refresh", self.refresh, "Rescan folders · F5")
            self.execute_stash_btn = self.button("Move all marked (0)", self.execute_stash,
                "Move ALL marked files across every category to stash. Review them using the Marked filter first.")
            heading.addWidget(self.execute_stash_btn)
            heading.addWidget(self.undo_btn)
            heading.addWidget(self.refresh_btn)
            heading.addWidget(self.button("Help", self.show_help))
            layout.addLayout(heading)

            overview = QHBoxLayout()
            self.chips = {}
            for state in UI_STATES:
                chip = self.button(LABELS[state] + "  0", lambda checked=False, s=state: self.choose_state(s))
                chip.setCheckable(True)
                chip.setToolTip("Show " + LABELS[state] + " tracks across all categories; clears search")
                chip.setStyleSheet(f"QPushButton {{ color: {COLORS[state]}; border-bottom: 3px solid {COLORS[state]}; }}"
                                  f"QPushButton:checked {{ background: #304a67; border: 2px solid {COLORS[state]}; }}")
                self.chips[state] = chip
                overview.addWidget(chip, 1)
            layout.addLayout(overview)
            progress_row = QHBoxLayout()
            self.progress_label = QLabel("Scanning your audio library…")
            self.progress_label.setObjectName("muted")
            self.progress = QProgressBar()
            self.progress.setTextVisible(False)
            self.progress.setFixedHeight(6)
            progress_row.addWidget(self.progress_label)
            progress_row.addWidget(self.progress, 1)
            layout.addLayout(progress_row)

            filters = QHBoxLayout()
            self.category = QComboBox()
            for text in ("All categories", "Vocals", "Instruments", "Ambients", "Other"):
                self.category.addItem(text)
            self.category.setMinimumWidth(145)
            self.category.setAccessibleName("Category filter")
            self.state_filter = QComboBox()
            self.state_filter.setAccessibleName("Rating filter")
            for text, value in (("All active", "active"), ("All tracks (including stash)", "all"), ("Unreviewed", "unreviewed"),
                    ("High", "high"), ("Medium", "medium"), ("Low", "low"), ("Marked for stash", "marked"), ("Stash", "stash")):
                self.state_filter.addItem(text, value)
            self.state_filter.setMinimumWidth(135)
            self.search = QLineEdit()
            self.search.setPlaceholderText("Search filenames or folders…")
            self.search.setClearButtonEnabled(True)
            self.search.setAccessibleName("Search audio files")
            self.visible_label = QLabel("0 files")
            self.visible_label.setObjectName("muted")
            filters.addWidget(self.category)
            filters.addWidget(self.state_filter)
            filters.addWidget(self.search, 1)
            self.show_all_btn = self.button("Show all", self.show_all_tracks)
            self.show_all_btn.setToolTip("Clear category, rating and search filters; show active and stashed tracks")
            filters.addWidget(self.show_all_btn)
            filters.addWidget(self.visible_label)
            layout.addLayout(filters)
            self.filter_summary = QLabel()
            self.filter_summary.setObjectName("muted")
            self.filter_summary.setWordWrap(True)
            layout.addWidget(self.filter_summary)

            self.duration_heading = QLabel("Duration · current filters · includes collapsed folders")
            self.duration_heading.setObjectName("muted")
            self.duration_heading.setWordWrap(True)
            layout.addWidget(self.duration_heading)
            duration_row = QHBoxLayout()
            self.duration_labels = {}
            for state in (*UI_STATES, "total"):
                label = QLabel()
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label.setStyleSheet(f"color: {COLORS.get(state, '#edf2f8')}; padding: 5px;")
                label.setMinimumWidth(85)
                self.duration_labels[state] = label
                duration_row.addWidget(label, 1)
            layout.addLayout(duration_row)
            self.update_duration_totals([])

            self.tree = QTreeWidget()
            self.tree.setHeaderLabels(["Track / folder", "State", "Category"])
            self.tree.setColumnWidth(1, 145)
            self.tree.setColumnWidth(2, 125)
            self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
            self.tree.header().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
            self.tree.setUniformRowHeights(True)
            self.tree.header().setSectionsClickable(False)
            self.tree.setSortingEnabled(False)
            self.tree.setIndentation(22)
            self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            self.tree.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            self.tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            self.tree.setAnimated(False)
            self.tree.setAlternatingRowColors(True)
            self.tree.setItemDelegateForColumn(1, RatingDelegate(self.tree))
            self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            self.tree.setAccessibleName("Audio library folder tree")
            layout.addWidget(self.tree, 1)

            self.empty = QLabel("No tracks match these filters.")
            self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.empty.setObjectName("muted")
            self.empty.hide()
            layout.addWidget(self.empty)

            self.panel = QFrame()
            self.panel.setObjectName("playerPanel")
            panel = QVBoxLayout(self.panel)
            panel.setContentsMargins(16, 12, 16, 12)
            panel.setSpacing(9)
            selected_row = QHBoxLayout()
            self.track_title = QLabel("Select a track to review")
            self.track_title.setObjectName("trackTitle")
            self.track_title.setWordWrap(True)
            selected_row.addWidget(self.track_title, 1)
            self.state_badge = QLabel("—")
            self.state_badge.setMinimumWidth(100)
            self.state_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            selected_row.addWidget(self.state_badge)
            panel.addLayout(selected_row)
            self.track_path = QLabel("Double-click to play · Ratings apply to the selected track")
            self.track_path.setObjectName("muted")
            self.track_path.setWordWrap(True)
            panel.addWidget(self.track_path)

            seek_row = QHBoxLayout()
            self.elapsed = QLabel("0:00")
            self.elapsed.setMinimumWidth(40)
            self.seek = SeekSlider(Qt.Orientation.Horizontal)
            self.seek.setRange(0, 0)
            self.seek.setEnabled(False)
            self.seek.setAccessibleName("Track position")
            self.total_time = QLabel("0:00")
            self.total_time.setMinimumWidth(40)
            seek_row.addWidget(self.elapsed)
            seek_row.addWidget(self.seek, 1)
            seek_row.addWidget(self.total_time)
            panel.addLayout(seek_row)

            transport = QHBoxLayout()
            self.prev_btn = self.button("Previous", lambda: self.advance(-1), "Previous visible track · Ctrl+Up")
            self.play_btn = self.button("Play", self.toggle_play, "Play / pause · Space")
            self.play_btn.setObjectName("playButton")
            self.play_btn.setMinimumWidth(92)
            self.back_btn = self.button("−10s", lambda: self.skip(-10000), "Back 10 seconds · Left")
            self.forward_btn = self.button("+10s", lambda: self.skip(10000), "Forward 10 seconds · Right")
            self.next_btn = self.button("Next", lambda: self.advance(1), "Next visible track · Ctrl+Down")
            for button in (self.prev_btn, self.play_btn, self.back_btn, self.forward_btn, self.next_btn):
                transport.addWidget(button)
            transport.addStretch(1)
            self.playback_status = QLabel("Ready")
            self.playback_status.setObjectName("muted")
            transport.addWidget(self.playback_status)
            transport.addWidget(QLabel("Volume"))
            self.volume = QSlider(Qt.Orientation.Horizontal)
            self.volume.setRange(0, 100)
            self.volume.setFixedWidth(95)
            self.volume.setAccessibleName("Preview volume")
            self.volume_text = QLabel("70%")
            self.volume_text.setMinimumWidth(38)
            transport.addWidget(self.volume)
            transport.addWidget(self.volume_text)
            panel.addLayout(transport)

            rating_row = QHBoxLayout()
            self.rating_buttons = {}
            for number, state in enumerate(RATINGS):
                b = self.button(f"{number}  {LABELS[state]}", lambda checked=False, s=state: self.rate(s))
                b.setCheckable(True)
                color = COLORS[state]
                b.setStyleSheet(f"QPushButton {{color:{color};}} QPushButton:checked "
                               f"{{border:2px solid {color}; background:#253349;}}")
                self.rating_buttons[state] = b
                rating_row.addWidget(b, 1)
            rating_row.addSpacing(15)
            self.stash_btn = self.button("Mark for stash", self.stash_restore, "Flag for final review; no file is moved · S")
            self.stash_btn.setStyleSheet(f"QPushButton {{color:{COLORS['stash']};}}")
            rating_row.addWidget(self.stash_btn)
            panel.addLayout(rating_row)

            options = QHBoxLayout()
            self.auto_next = QCheckBox("Play next after rating")
            self.auto_next.setToolTip("Advance through the current filtered list after a rating or stash.")
            options.addWidget(self.auto_next)
            options.addStretch(1)
            options.addWidget(QLabel("Output"))
            self.output = QComboBox()
            self.output.setMinimumWidth(220)
            self.output.setMaximumWidth(390)
            self.output.setAccessibleName("Audio output device")
            options.addWidget(self.output)
            panel.addLayout(options)
            layout.addWidget(self.panel)
            self.footer = QLabel(str(self.root))
            self.footer.setObjectName("muted")
            self.footer.setToolTip(f"Project: {self.root}\nShared ratings: {self.catalog.path}")
            layout.addWidget(self.footer)
            self.statusBar().showMessage("Ratings save automatically")

            self.category.currentIndexChanged.connect(self.apply_filters)
            self.state_filter.currentIndexChanged.connect(self.apply_filters)
            self.search.textChanged.connect(self.apply_filters)
            self.tree.currentItemChanged.connect(self.selection_changed)
            self.tree.itemDoubleClicked.connect(self.double_click)
            self.tree.customContextMenuRequested.connect(self.context_menu)
            self.output.currentIndexChanged.connect(self.change_output)
            self.volume.valueChanged.connect(self.change_volume)
            self.update_controls()

        def connect_player(self):
            self.player.positionChanged.connect(self.position_changed)
            self.player.durationChanged.connect(self.duration_changed)
            self.player.playbackStateChanged.connect(self.playback_changed)
            self.player.mediaStatusChanged.connect(self.media_status)
            self.player.errorOccurred.connect(self.media_error)
            self.player.seekableChanged.connect(lambda yes: self.seek.setEnabled(yes and not self.busy))
            self.seek.sliderMoved.connect(lambda value: self.elapsed.setText(self.time_text(value)))
            self.seek.sliderReleased.connect(lambda: self.player.setPosition(self.seek.value()))
            self.seek.seekRequested.connect(self.player.setPosition)
            self.seek.actionTriggered.connect(lambda action: QTimer.singleShot(0, self.seek_from_keyboard)
                if not self.seek.isSliderDown() else None)

        def seek_from_keyboard(self):
            if self.seek.hasFocus() and self.player.isSeekable():
                self.player.setPosition(self.seek.value())

        def install_shortcuts(self):
            # Scope quick keys to the tree/player, leaving search text entry alone.
            actions = [(str(i), lambda s=s: self.rate(s)) for i, s in enumerate(RATINGS)]
            actions += [("Space", self.toggle_play), ("S", self.stash_restore),
                        ("Ctrl+Down", lambda: self.advance(1)), ("Ctrl+Up", lambda: self.advance(-1)),
                        ("Ctrl+Z", self.undo)]
            for parent in (self.tree, self.panel):
                for key, fn in actions:
                    shortcut = QShortcut(QKeySequence(key), parent)
                    shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
                    shortcut.activated.connect(fn)
                    self.shortcuts.append(shortcut)
            for key, fn in (("Left", lambda: self.skip(-10000)), ("Right", lambda: self.skip(10000))):
                shortcut = QShortcut(QKeySequence(key), self.tree)
                shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
                shortcut.activated.connect(fn)
                self.shortcuts.append(shortcut)
            for key, fn in (("F5", self.refresh), ("Ctrl+F", self.search.setFocus)):
                shortcut = QShortcut(QKeySequence(key), self)
                shortcut.activated.connect(fn)

        def update_shortcut_context(self, old, current):
            # Typing a device name in the output combo must never rate or stash.
            widget = current
            editing = False
            while widget is not None:
                if isinstance(widget, (QComboBox, QLineEdit)):
                    editing = True
                    break
                widget = widget.parentWidget()
            for shortcut in self.shortcuts:
                shortcut.setEnabled(not editing)

        def restore_preferences(self):
            geometry = self.prefs.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
            self.volume.setValue(int(self.prefs.value("volume", 70)))
            self.auto_next.setChecked(self.prefs.value("auto_next", False, type=bool))
            index = self.state_filter.findData(self.prefs.value("state_filter", "active"))
            self.state_filter.setCurrentIndex(max(0, index))
            index = self.category.findText(self.prefs.value("category", "All categories"))
            self.category.setCurrentIndex(max(0, index))
            self.search.setText(str(self.prefs.value("search", "")))
            header = self.prefs.value("tree_header")
            if header:
                self.tree.header().restoreState(header)
            try:
                saved = json.loads(str(self.prefs.value("folder_states", "{}")))
                self._folder_states = saved if isinstance(saved, dict) else {}
            except (ValueError, TypeError):
                self._folder_states = {}

        def setup_ui_persistence(self):
            self.ui_save_timer = QTimer(self)
            self.ui_save_timer.setSingleShot(True)
            self.ui_save_timer.setInterval(150)
            self.ui_save_timer.timeout.connect(self.save_ui_preferences)
            for signal in (self.volume.valueChanged, self.auto_next.toggled,
                    self.category.currentIndexChanged, self.state_filter.currentIndexChanged,
                    self.search.textChanged, self.output.currentIndexChanged,
                    self.tree.currentItemChanged, self.tree.itemExpanded,
                    self.tree.itemCollapsed, self.tree.verticalScrollBar().valueChanged,
                    self.tree.horizontalScrollBar().valueChanged,
                    self.tree.header().sectionResized):
                signal.connect(self.schedule_ui_save)
            self._ui_ready = True

        def schedule_ui_save(self, *args):
            if self._ui_ready and not self._restoring_view:
                self.ui_save_timer.start()

        def save_ui_preferences(self):
            if not self._ui_ready or self._restoring_view:
                return
            self.prefs.setValue("geometry", self.saveGeometry())
            self.prefs.setValue("volume", self.volume.value())
            self.prefs.setValue("auto_next", self.auto_next.isChecked())
            self.prefs.setValue("category", self.category.currentText())
            self.prefs.setValue("state_filter", self.state_filter.currentData())
            self.prefs.setValue("search", self.search.text())
            self.prefs.setValue("output_device", self.output.currentData())
            self.prefs.setValue("project_root", str(self.root))
            if self._view_initialized:
                self._folder_states.update({key: item.isExpanded() for key, item in self.folder_items.items()})
                self.prefs.setValue("folder_states", json.dumps(self._folder_states))
                self.prefs.setValue("last_selected", self.selected_key or "")
                self.prefs.setValue("selected_folder", self.selected_folder or "")
                self.prefs.setValue("scroll_y", self.tree.verticalScrollBar().value())
                self.prefs.setValue("scroll_x", self.tree.horizontalScrollBar().value())
                self.prefs.setValue("tree_header", self.tree.header().saveState())
            self.prefs.sync()
            if self.prefs.status() != QSettings.Status.NoError:
                self.statusBar().showMessage("Unable to save window preferences; check project folder permissions")

        def resizeEvent(self, event):
            super().resizeEvent(event)
            if getattr(self, "_ui_ready", False):
                self.schedule_ui_save()

        def moveEvent(self, event):
            super().moveEvent(event)
            if getattr(self, "_ui_ready", False):
                self.schedule_ui_save()

        def populate_devices(self):
            wanted = (self.output.currentData() if self.device_ids
                      else str(self.prefs.value("output_device", "")))
            self.output.blockSignals(True)
            self.output.clear()
            self.output.addItem("System default", "")
            self.device_ids = list(QMediaDevices.audioOutputs())
            for device in self.device_ids:
                self.output.addItem(device.description(), bytes(device.id()).hex())
            index = self.output.findData(wanted)
            self.output.setCurrentIndex(max(0, index))
            self.output.blockSignals(False)
            self.change_output(self.output.currentIndex())

        def change_output(self, index):
            if index <= 0:
                self.audio.setDevice(QMediaDevices.defaultAudioOutput())
            elif index <= len(self.device_ids):
                self.audio.setDevice(self.device_ids[index - 1])

        def change_volume(self, value):
            # Smooth perceptual response, with exact zero at mute.
            self.audio.setVolume((value / 100.0) ** 2)
            self.volume_text.setText(f"{value}%")

        def run_job(self, fn, success):
            self.set_busy(True)
            job = Job(fn, self)
            self.jobs.append(job)
            job.result.connect(success)
            job.failed.connect(self.job_failed)
            job.finished.connect(lambda: self.job_finished(job))
            job.start()

        def job_finished(self, job):
            self.jobs.remove(job)
            job.deleteLater()
            self.set_busy(False)
            pending = getattr(self, "play_after_job", None)
            self.play_after_job = None
            if pending and self.selected_key == pending:
                self.play_selected()

        def job_failed(self, message):
            self.statusBar().showMessage("Operation did not complete")
            QMessageBox.warning(self, "Operation did not complete", message)

        def set_busy(self, busy):
            self.busy = busy
            for widget in (self.tree, self.category, self.state_filter, self.search, self.refresh_btn, self.show_all_btn):
                widget.setEnabled(not busy)
            for chip in self.chips.values():
                chip.setEnabled(not busy)
            self.update_controls()

        def refresh(self):
            if self.busy:
                return
            preferred = (self.selected_key or self.prefs.value("last_selected", "")
                         or self.catalog.data.get("last_selected"))
            self.unload()
            self.statusBar().showMessage("Scanning folders…")
            def scan():
                found, errors = scan_audio(self.root)
                problems = self.catalog.reconcile(found)
                return found, errors, problems
            def done(result):
                self.found, errors, self.problems = result
                self.durations, self.duration_errors = {}, {}
                self.rebuild_tree(preferred)
                self.start_duration_scan()
                self.statusBar().showMessage(f"Ready · {len(self.found)} MP3 files · Ratings save automatically")
                if errors or self.problems:
                    lines = errors + [f"{k}: {v}" for k, v in self.problems.items()]
                    QMessageBox.warning(self, "Some files need attention", "\n".join(lines))
            self.run_job(scan, done)

        def start_duration_scan(self):
            if self.duration_job is not None:
                self.duration_job.requestInterruption()
                self._duration_restart = True
                return
            self._duration_restart = False
            job = DurationJob(self.root, self.found, self)
            self.duration_job = job
            job.batch.connect(self.duration_batch)
            job.finished.connect(self.duration_scan_finished)
            job.start()

        def duration_batch(self, results):
            if getattr(self, "_duration_restart", False):
                return
            for key, (seconds, error) in results.items():
                if key not in self.found:
                    continue
                if error:
                    self.duration_errors[key] = error
                    self.durations.pop(key, None)
                else:
                    self.durations[key] = seconds
                    self.duration_errors.pop(key, None)
            self.update_duration_totals(self.visible_keys())

        def duration_scan_finished(self):
            job = self.duration_job
            self.duration_job = None
            job.deleteLater()
            if self._close_pending:
                self.close()
            elif getattr(self, "_duration_restart", False):
                self.start_duration_scan()
            else:
                self.update_duration_totals(self.visible_keys())

        def state_icon(self, state):
            pixmap = QPixmap(14, 14)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            color = QColor(COLORS[state])
            painter.setPen(QPen(color, 1.8))
            painter.setBrush(Qt.BrushStyle.NoBrush if state == "unreviewed" else QBrush(color))
            painter.drawEllipse(3, 3, 8, 8)
            painter.end()
            return QIcon(pixmap)

        def rebuild_tree(self, preferred=None):
            initial = not self._view_initialized
            if not initial:
                self._folder_states.update({key: item.isExpanded() for key, item in self.folder_items.items()})
            scroll_y = int(self.prefs.value("scroll_y", 0)) if initial else self.tree.verticalScrollBar().value()
            scroll_x = int(self.prefs.value("scroll_x", 0)) if initial else self.tree.horizontalScrollBar().value()
            previous_key = self.selected_key
            self._restoring_view = True
            self.tree.blockSignals(True)
            self.tree.clear()
            self.items = {}
            folders = {}
            for key in sorted(self.found, key=str.casefold):
                state = self.catalog.display_state(key)
                location = "stash" if state == "stash" else "ceremonies"
                parts = (location,) + PurePosixPath(key).parts
                parent = None
                for depth, part in enumerate(parts[:-1]):
                    folder_key = parts[:depth + 1]
                    if folder_key not in folders:
                        folder = SortedItem([part, "", ""])
                        folder.setIcon(0, self.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon))
                        font = folder.font(0)
                        font.setBold(True)
                        folder.setFont(0, font)
                        folder.setData(0, ROLE + 2, "/".join(folder_key))
                        if parent is None:
                            self.tree.addTopLevelItem(folder)
                        else:
                            parent.addChild(folder)
                        folders[folder_key] = folder
                    parent = folders[folder_key]
                item = SortedItem([parts[-1], LABELS[state], category_for(key)])
                item.setData(0, ROLE, key)
                item.setData(1, ROLE + 1, state)
                item.setIcon(0, self.state_icon(state))
                item.setSizeHint(0, QSize(0, 35))
                item.setToolTip(0, f"{location}/{key}")
                item.setToolTip(1, LABELS[state])
                if key in self.problems:
                    item.setText(2, "Conflict")
                    item.setToolTip(2, self.problems[key])
                parent.addChild(item)
                self.items[key] = item
            self.tree.sortItems(0, Qt.SortOrder.AscendingOrder)
            self.folder_items = {"/".join(key): item for key, item in folders.items()}
            for key, item in self.folder_items.items():
                item.setExpanded(self._folder_states.get(key, True))
            self.tree.blockSignals(False)
            self.update_counts()
            self.apply_filters(preferred=preferred)
            # Selection normally expands ancestors. Restore the user's exact
            # hierarchy after selecting, including intentionally collapsed ones.
            for key, item in self.folder_items.items():
                item.setExpanded(self._folder_states.get(key, True))
            self._view_initialized = True
            def finish_view():
                self.tree.doItemsLayout()
                if initial or self.selected_key == previous_key:
                    self.tree.verticalScrollBar().setValue(scroll_y)
                    self.tree.horizontalScrollBar().setValue(scroll_x)
                self._restoring_view = False
                self.schedule_ui_save()
            QTimer.singleShot(0, finish_view)

        def reset_filters(self, state):
            # Overview counts cover the whole library. Their actions must do so too.
            widgets = (self.category, self.state_filter, self.search)
            for widget in widgets:
                widget.blockSignals(True)
            try:
                self.category.setCurrentIndex(self.category.findText("All categories"))
                self.state_filter.setCurrentIndex(self.state_filter.findData(state))
                self.search.clear()
            finally:
                for widget in widgets:
                    widget.blockSignals(False)
            self.selected_folder = None
            self.apply_filters()
            self.schedule_ui_save()

        def choose_state(self, state):
            self.reset_filters(state)

        def show_all_tracks(self, *args):
            self.reset_filters("all")

        def update_filter_summary(self):
            state = self.state_filter.currentData()
            for value, chip in self.chips.items():
                chip.setChecked(value == state)
            parts = [self.category.currentText(), self.state_filter.currentText()]
            if self.search.text().strip():
                parts.append('Search: "' + self.search.text().strip() + '"')
            self.filter_summary.setText("Showing: " + " · ".join(parts))

        def update_counts(self):
            counts = {state: 0 for state in UI_STATES}
            for key in self.found:
                counts[self.catalog.display_state(key)] += 1
            for state, chip in self.chips.items():
                chip.setText(f"{LABELS[state]}  {counts[state]}")
            reviewed = len(self.found) - counts["unreviewed"]
            self.progress_label.setText(f"{reviewed} of {len(self.found)} reviewed")
            self.progress.setRange(0, max(1, len(self.found)))
            self.progress.setValue(reviewed)

        def apply_filters(self, *args, preferred=None):
            self.update_filter_summary()
            if not self.items:
                self.empty.setText("No MP3 files found. Reference folders are excluded.")
                self.empty.show()
                self.visible_label.setText("0 files")
                self.update_duration_totals([])
                self.selection_changed(None, None)
                return
            self.empty.setText("No tracks match these filters. Click Show all to restore the full hierarchy.")
            selected = preferred or self.selected_key
            category = self.category.currentText()
            state_filter = self.state_filter.currentData()
            words = self.search.text().casefold().split()
            for key, item in self.items.items():
                state = self.catalog.display_state(key)
                visible = ((category == "All categories" or category_for(key) == category)
                    and (state_filter == "all" or ((state != "stash") if state_filter == "active" else state == state_filter))
                    and all(word in key.casefold() for word in words))
                item.setHidden(not visible)
            def prune(item):
                if item.childCount():
                    any_visible = False
                    for i in range(item.childCount()):
                        any_visible = prune(item.child(i)) or any_visible
                    item.setHidden(not any_visible)
                return not item.isHidden()
            for i in range(self.tree.topLevelItemCount()):
                prune(self.tree.topLevelItem(i))
            visible = self.visible_keys()
            self.visible_label.setText(f"{len(visible)} files")
            self.update_duration_totals(visible)
            self.empty.setVisible(not visible)
            folder = self.folder_items.get(self.selected_folder)
            if folder is not None and not folder.isHidden():
                self.tree.setCurrentItem(folder)
                self.selection_changed(folder, None)
                return
            chosen = selected if selected in visible else (visible[0] if visible else None)
            if chosen:
                self.select_key(chosen)
            else:
                self.tree.setCurrentItem(None)
                self.selection_changed(None, None)

        def update_duration_totals(self, keys):
            if self.duration_scope:
                prefix = self.duration_scope + "/"
                keys = [key for key in keys if
                        (("stash/" if self.catalog.display_state(key) == "stash" else "ceremonies/")
                         + key).startswith(prefix)]
            totals = {state: 0.0 for state in UI_STATES}
            unknown = {state: 0 for state in UI_STATES}
            for key in keys:
                state = self.catalog.display_state(key)
                if key in self.durations:
                    totals[state] += self.durations[key]
                else:
                    unknown[state] += 1
            totals["total"] = sum(totals.values())
            unknown["total"] = sum(unknown.values())
            for state, label in self.duration_labels.items():
                title = "Total" if state == "total" else LABELS[state]
                extra = f" + {unknown[state]} unknown" if unknown[state] else ""
                label.setText(f"{title}\n{duration_text(totals[state])}")
                label.setToolTip(f"{title}: {duration_text(totals[state])}{extra}. "
                    "Hours:minutes:seconds. Includes matching tracks in collapsed folders.")
            n = unknown["total"]
            pending = sum(key not in self.durations and key not in self.duration_errors for key in keys)
            note = f" · loading {pending} durations…" if pending and self.duration_job else ""
            unavailable = n - pending if self.duration_job else n
            if unavailable:
                note += f" · {unavailable} unavailable (excluded from totals)"
            self.duration_heading.setText("Duration · " + (self.duration_scope or "All folders")
                                          + " · current filters" + note)
            messages = [f"{key}: {self.duration_errors.get(key, 'Loading duration…' if self.duration_job else 'Duration unavailable')}"
                        for key in keys if key not in self.durations]
            self.duration_heading.setToolTip("\n".join(messages))

        def visible_keys(self):
            # Traverse the actual sorted hierarchy, including collapsed descendants.
            keys = []
            def visit(parent):
                for index in range(parent.childCount()):
                    item = parent.child(index)
                    if item.isHidden():
                        continue
                    key = item.data(0, ROLE)
                    if key:
                        keys.append(key)
                    else:
                        visit(item)
            visit(self.tree.invisibleRootItem())
            return keys

        def select_key(self, key):
            item = self.items.get(key)
            if item:
                parent = item.parent()
                while parent:
                    parent.setExpanded(True)
                    parent = parent.parent()
                self.tree.setCurrentItem(item)
                self.tree.scrollToItem(item)

        def selection_changed(self, current, previous):
            key = current.data(0, ROLE) if current else None
            if key != self.selected_key:
                self.unload()
            self.selected_key = key
            self.selected_folder = current.data(0, ROLE + 2) if current and not key else None
            folder = current.parent() if current and key else current
            self.duration_scope = folder.data(0, ROLE + 2) if folder else None
            self.update_duration_totals(self.visible_keys())
            self.show_selected()

        def show_selected(self):
            key = self.selected_key
            if key:
                state = self.catalog.display_state(key)
                self.track_title.setText(PurePosixPath(key).name)
                self.track_path.setText(("stash/" if state == "stash" else "ceremonies/") + key)
                self.state_badge.setText(LABELS[state])
                self.state_badge.setStyleSheet(f"color:{COLORS[state]};border:1px solid {COLORS[state]};"
                                              "border-radius:5px;padding:6px;")
            else:
                self.track_title.setText("Select a track to review")
                self.track_path.setText("Double-click to play · Ratings apply to the selected track")
                self.state_badge.setText("—")
                self.state_badge.setStyleSheet("")
            self.update_controls()

        def update_controls(self):
            key = self.selected_key
            enabled = bool(key) and not self.busy and not self.catalog.blocked and key not in self.problems
            state = self.catalog.state(key) if key else None
            for s, button in self.rating_buttons.items():
                button.setEnabled(enabled and state != "stash")
                button.setChecked(state == s)
            self.stash_btn.setEnabled(enabled)
            marked = bool(key) and self.catalog.is_marked(key)
            self.stash_btn.setText("Restore" if state == "stash" else ("Unmark stash" if marked else "Mark for stash"))
            count = sum(self.catalog.is_marked(k) for k in self.found)
            self.execute_stash_btn.setText(f"Move all marked ({count})")
            self.execute_stash_btn.setEnabled(count > 0 and not self.busy and not self.catalog.blocked)
            self.stash_btn.setToolTip("Restore to ceremonies, preserving the previous rating · S" if state == "stash"
                                     else "Toggle the final-review flag; no file is moved · S")
            self.play_btn.setEnabled(enabled)
            self.prev_btn.setEnabled(enabled)
            self.next_btn.setEnabled(enabled)
            self.back_btn.setEnabled(enabled and self.loaded_key is not None)
            self.forward_btn.setEnabled(enabled and self.loaded_key is not None)
            self.seek.setEnabled(enabled and self.player.isSeekable())
            self.undo_btn.setEnabled(bool(self.undo_stack) and not self.busy and not self.catalog.blocked)

        def unload(self):
            self.loaded_key = None
            self.player.stop()
            self.player.setSource(QUrl())
            self.duration = 0
            self.seek.setRange(0, 0)
            self.elapsed.setText("0:00")
            self.total_time.setText("0:00")
            self.playback_status.setText("Ready")

        def double_click(self, item, column):
            if item.data(0, ROLE):
                self.play_selected()

        def play_selected(self):
            key = self.selected_key
            if not key or self.busy or key in self.problems:
                return
            if self.loaded_key != key:
                location = "stash" if self.catalog.state(key) == "stash" else "ceremonies"
                try:
                    path = self.catalog.path_for(key, location)
                    if not path.is_file():
                        raise FileNotFoundError(f"This file is missing. Refresh the list.\n{path}")
                    self.loaded_key = key
                    self.player.setSource(QUrl.fromLocalFile(str(path)))
                except Exception as exc:
                    self.loaded_key = None
                    QMessageBox.warning(self, "Unable to play", str(exc))
                    return
            if self.player.mediaStatus() == QMediaPlayer.MediaStatus.EndOfMedia:
                self.player.setPosition(0)
            self.player.play()
            self.update_controls()

        def toggle_play(self):
            if self.busy:
                return
            if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                self.player.pause()
            else:
                self.play_selected()

        def playback_changed(self, state):
            playing = state == QMediaPlayer.PlaybackState.PlayingState
            self.play_btn.setText("Pause" if playing else "Play")
            self.playback_status.setText("Playing" if playing else ("Paused" if self.loaded_key else "Ready"))

        def media_status(self, status):
            if status == QMediaPlayer.MediaStatus.EndOfMedia:
                self.playback_status.setText("Finished")
                self.statusBar().showMessage("Track finished · Rate it or choose Next")
            elif status == QMediaPlayer.MediaStatus.LoadingMedia:
                self.playback_status.setText("Loading…")

        def media_error(self, error, message):
            if error != QMediaPlayer.Error.NoError and self.loaded_key:
                self.playback_status.setText("Playback error")
                self.statusBar().showMessage(f"Playback error: {message}")
                QMessageBox.warning(self, "Playback error", message + "\nThe file and rating are unchanged.")

        @staticmethod
        def time_text(milliseconds):
            seconds = max(0, milliseconds // 1000)
            return f"{seconds // 60}:{seconds % 60:02d}"

        def position_changed(self, position):
            if not self.seek.isSliderDown():
                self.seek.setValue(position)
                self.elapsed.setText(self.time_text(position))

        def duration_changed(self, duration):
            self.duration = duration
            self.seek.setRange(0, max(0, duration))
            self.total_time.setText(self.time_text(duration))

        def skip(self, delta):
            if self.loaded_key and not self.busy and self.player.isSeekable():
                self.player.setPosition(max(0, min(self.duration, self.player.position() + delta)))

        def advance(self, direction):
            if self.busy:
                return
            visible = self.visible_keys()
            if not visible:
                return
            index = visible.index(self.selected_key) if self.selected_key in visible else -1
            target = index + direction
            if not 0 <= target < len(visible):
                self.statusBar().showMessage("End of this filtered list" if direction > 0 else "First track in this list")
                return
            self.select_key(visible[target])
            self.play_selected()
            self.tree.setFocus()

        def next_after(self, key):
            visible = self.visible_keys()
            if key in visible:
                index = visible.index(key) + 1
                if index < len(visible):
                    return visible[index]
            return None

        def rate(self, state):
            key = self.selected_key
            if not key or self.busy or key in self.problems or self.catalog.state(key) == "stash":
                return
            next_key = self.next_after(key)
            was_marked = self.catalog.is_marked(key)
            try:
                old = self.catalog.rate(key, state)
            except Exception as exc:
                self.show_selected()
                QMessageBox.warning(self, "Rating was not saved", str(exc))
                return
            if old != state or was_marked:
                self.undo_stack.append(("rating", key, (old, was_marked)))
            saved_rating = LABELS[state]
            state = self.catalog.display_state(key)
            item = self.items[key]
            item.setText(1, LABELS[state])
            item.setData(1, ROLE + 1, state)
            item.setIcon(0, self.state_icon(state))
            item.setToolTip(1, LABELS[state])
            self.tree.sortItems(0, Qt.SortOrder.AscendingOrder)
            self.update_counts()
            filtered_out = self.state_filter.currentData() not in ("active", "all", state)
            self.apply_filters(preferred=next_key if (self.auto_next.isChecked() or filtered_out) and next_key else key)
            self.show_selected()
            self.statusBar().showMessage(f"Saved {saved_rating} · {PurePosixPath(key).name}")
            if self.auto_next.isChecked() and next_key:
                self.select_key(next_key)
                self.play_selected()
            elif self.auto_next.isChecked() and not next_key:
                self.unload()
                self.statusBar().showMessage("Rating saved · End of this filtered list")
            self.tree.setFocus()

        def stash_restore(self):
            key = self.selected_key
            if not key or self.busy or key in self.problems:
                return
            if self.catalog.state(key) == "stash":
                self.perform_move(key, True)
                return
            marked = not self.catalog.is_marked(key)
            next_key = self.next_after(key)
            try:
                previous = self.catalog.mark_stash(key, marked)
            except Exception as exc:
                QMessageBox.warning(self, "Flag was not saved", str(exc))
                return
            self.undo_stack.append(("mark", key, previous))
            self.rebuild_tree(next_key if marked and self.auto_next.isChecked() else key)
            self.show_selected()
            self.statusBar().showMessage(
                f"{'Marked for stash — file has not moved' if marked else 'Stash mark removed'} · {PurePosixPath(key).name}")
            if marked and self.auto_next.isChecked() and next_key:
                self.select_key(next_key)
                self.play_selected()
            self.tree.setFocus()

        def execute_stash(self):
            if self.busy or self.catalog.blocked:
                return
            keys = [key for key in self.found if self.catalog.is_marked(key)]
            if not keys:
                return
            self.unload()
            def move_marked():
                moved, errors = [], []
                for key in keys:
                    try:
                        if key in self.problems:
                            raise ValueError(self.problems[key])
                        old = self.catalog.move(key)
                        moved.append((key, old))
                    except Exception as exc:
                        errors.append(f"{key}: {exc}")
                        if self.catalog.blocked:
                            break
                return moved, errors
            def done(result):
                moved, errors = result
                for key, old in moved:
                    self.found[key] = {"stash"}
                    self.undo_stack.append(("move", key, old))
                self.rebuild_tree()
                self.show_selected()
                self.statusBar().showMessage(f"Moved {len(moved)} marked files to stash · Undo restores one at a time")
                if errors:
                    QMessageBox.warning(self, "Some marked files could not be moved", "\n".join(errors))
            self.run_job(move_marked, done)

        def perform_move(self, key, restore, undoing=False):
            next_key = self.next_after(key)
            self.unload()  # Release Qt's file handle before the background move.
            def done(old_state):
                if undoing:
                    self.undo_stack.pop()
                else:
                    self.undo_stack.append(("move", key, old_state))
                self.found[key] = {"ceremonies" if restore else "stash"}
                if undoing:
                    self.state_filter.blockSignals(True)
                    self.state_filter.setCurrentIndex(self.state_filter.findData("active" if restore else "stash"))
                    self.state_filter.blockSignals(False)
                self.rebuild_tree(key if undoing else next_key)
                self.statusBar().showMessage(f"{'Restored' if restore else 'Stashed'} · {PurePosixPath(key).name}")
                if not undoing and self.auto_next.isChecked() and next_key:
                    self.select_key(next_key)
                    # Wait for QThread.finished so playback is enabled again.
                    self.play_after_job = next_key
            self.run_job(lambda: self.catalog.move(key, restore), done)

        def undo(self):
            if self.busy or not self.undo_stack:
                return
            kind, key, previous = self.undo_stack[-1]
            if kind == "move":
                self.perform_move(key, previous != "stash", undoing=True)
                return
            try:
                if kind == "mark":
                    self.catalog.mark_stash(key, previous)
                else:
                    rating, marked = previous if isinstance(previous, tuple) else (previous, False)
                    self.catalog.rate(key, rating, restore_mark=marked)
            except Exception as exc:
                QMessageBox.warning(self, "Unable to undo", str(exc))
                return
            self.undo_stack.pop()
            self.state_filter.blockSignals(True)
            self.state_filter.setCurrentIndex(self.state_filter.findData("active"))
            self.state_filter.blockSignals(False)
            self.rebuild_tree(key)
            self.statusBar().showMessage(f"Undid {kind} · {PurePosixPath(key).name}")

        def context_menu(self, position):
            item = self.tree.itemAt(position)
            if not item or not item.data(0, ROLE) or self.busy:
                return
            self.tree.setCurrentItem(item)
            menu = QMenu(self)
            menu.addAction("Play / pause", self.toggle_play)
            menu.addSeparator()
            state = self.catalog.state(self.selected_key)
            if state != "stash":
                for i, s in enumerate(RATINGS):
                    action = menu.addAction(self.state_icon(s), f"{LABELS[s]}    {i}", lambda checked=False, s=s: self.rate(s))
                    action.setCheckable(True)
                    action.setChecked(state == s)
                menu.addSeparator()
            menu.addAction("Restore to ceremonies" if state == "stash" else ("Unmark stash" if self.catalog.is_marked(self.selected_key) else "Mark for stash"), self.stash_restore)
            menu.exec(self.tree.viewport().mapToGlobal(position))

        def show_help(self):
            QMessageBox.information(self, "Audio Curator", "Double-click a track to play. Single-click selects it and stops the previous track.\n\n"
                "0 = Unreviewed   1 = Low   2 = Medium   3 = High\n"
                "Space = Play/Pause   S = Mark/Unmark/Restore   Ctrl+Z = Undo\n"
                "Left/Right = Skip 10 seconds (in the tree)\n"
                "Ctrl+Up/Down = Previous/Next   Ctrl+F = Search   F5 = Refresh\n\n"
                "Click or drag the timeline to seek. Tracks stop at the end so you can rate them. "
                "'Play next after rating' enables a faster review workflow.\n\n"
                "Tracks sort within each folder: Unreviewed, High, Medium, Low, Marked, Stash; then filename. "
                "Duration totals follow current filters and include collapsed folders. Unknown durations are flagged. "
                "Click a colored count to filter. Reference folders are excluded. "
                "Mark for stash only flags a track. Choose Marked for stash to review it, and Unmark stash to keep it. "
                "Move all marked performs the actual moves across ALL categories and filters. "
                "Marks persist between sessions; marked files retain their runtime rating until moved. "
                "Choose Stash to listen to or restore files already moved. "
                "Choosing any rating clears its stash mark. Nothing is deleted or overwritten. "
                "Undo is available for this session. Window layout, folder expansion, filters, selection, "
                "scroll position and volume save automatically.\n\n"
                f"Shared ratings: {self.catalog.path}\n"
                "The playback app reads the ratings from this same metadata file.")

        def closeEvent(self, event):
            if self.busy:
                self.statusBar().showMessage("Finishing the current file operation; please close again in a moment.")
                event.ignore()
                return
            if self.duration_job is not None:
                self._close_pending = True
                self.duration_job.requestInterruption()
                self.hide()
                event.ignore()
                return
            self.unload()
            self.ui_save_timer.stop()
            self.save_ui_preferences()
            if self.selected_key and not self.catalog.blocked:
                candidate = copy.deepcopy(self.catalog.data)
                candidate["last_selected"] = self.selected_key
                try:
                    self.catalog.commit(candidate)
                except Exception as exc:
                    QMessageBox.warning(self, "Resume position was not saved", str(exc))
            self.lock.unlock()
            event.accept()

    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    palette = QPalette()
    for role, color in ((QPalette.ColorRole.Window, "#101720"), (QPalette.ColorRole.WindowText, "#e7edf5"),
            (QPalette.ColorRole.Base, "#141e2b"), (QPalette.ColorRole.AlternateBase, "#172231"),
            (QPalette.ColorRole.Text, "#e7edf5"), (QPalette.ColorRole.Button, "#202d3e"),
            (QPalette.ColorRole.ButtonText, "#e7edf5"), (QPalette.ColorRole.Highlight, "#314b69"),
            (QPalette.ColorRole.HighlightedText, "#ffffff"), (QPalette.ColorRole.ToolTipBase, "#253349"),
            (QPalette.ColorRole.ToolTipText, "#ffffff"), (QPalette.ColorRole.PlaceholderText, "#95a4b8")):
        palette.setColor(role, QColor(color))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor("#637187"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor("#637187"))
    app.setPalette(palette)
    app.setStyleSheet("""
        QWidget {font-size: 13px;}
        QLabel#title {font-size: 27px; font-weight: 650;}
        QLabel#muted {color: #a6b3c6;}
        QLabel#trackTitle {font-size: 16px; font-weight: 600;}
        QFrame#playerPanel {background: #182332; border: 1px solid #314156; border-radius: 10px;}
        QPushButton {background: #202d3e; border: 1px solid #3b4c62; border-radius: 6px; padding: 5px 12px;}
        QPushButton:hover {background: #2b3c52; border-color: #8eabc9;}
        QPushButton:pressed {background: #354b65;}
        QPushButton:focus {border: 2px solid #b5cfea;}
        QPushButton:disabled {background: #192332; color: #65748b; border-color: #2a3545;}
        QPushButton#playButton {background: #345276; border-color: #78a8da;}
        QComboBox, QLineEdit {background: #182332; border: 1px solid #3b4c62; border-radius: 5px; padding: 7px;}
        QComboBox:focus, QLineEdit:focus {border-color: #99c9ff;}
        QComboBox::drop-down {width: 22px; border: none;}
        QTreeWidget {border: 1px solid #314156; border-radius: 7px; outline: 0;}
        QTreeWidget::item {height: 34px; padding-left: 4px;}
        QTreeWidget::item:selected {background: #304a67; color: white;}
        QTreeWidget::item:hover:!selected {background: #223246;}
        QHeaderView::section {background: #202d3e; color: #c7d2e2; padding: 9px; border: none;}
        QSlider::groove:horizontal {height: 6px; background: #354457; border-radius: 3px;}
        QSlider::sub-page:horizontal {background: #8bbaf0; border-radius: 3px;}
        QSlider::handle:horizontal {background: #e4efff; border: 1px solid #84a9d0; width: 14px; margin: -5px 0; border-radius: 7px;}
        QProgressBar {background: #2a3849; border: none; border-radius: 3px;}
        QProgressBar::chunk {background: #6cdbac; border-radius: 3px;}
        QStatusBar {color: #a6b3c6; border-top: 1px solid #2a3849;}
        QToolTip {padding: 5px; border: 1px solid #657b98;}
    """)
    prefs = QSettings("LivingBrownNoise", "AudioCurator")
    root = Path(project_root).expanduser().resolve() if project_root else Path(__file__).resolve().parent
    if not project_root and not (root / "ceremonies").is_dir():
        remembered = prefs.value("project_root", "")
        if remembered and (Path(remembered) / "ceremonies").is_dir():
            root = Path(remembered).resolve()
        else:
            chosen = QFileDialog.getExistingDirectory(None, "Choose the project folder containing ceremonies")
            if not chosen:
                return 0
            root = Path(chosen).resolve()
            if root.name.casefold() == "ceremonies":
                root = root.parent
    if not (root / "ceremonies").is_dir():
        QMessageBox.warning(None, "Project folder not found", f"There is no ceremonies folder inside:\n{root}")
        return 1
    lock = QLockFile(str(root / ".audio-curator.lock"))
    lock.setStaleLockTime(0)
    if not lock.tryLock(0):
        QMessageBox.warning(None, "Unable to open project", "Another curator may already be open for this project, "
                            "or the folder is not writable. Close the other curator and try again.")
        return 1
    # Use one predictable project-local file, independent of interpreter or
    # registry settings. Import old preferences once for existing installations.
    local_prefs = QSettings(str(root / "audio-curator-ui.ini"), QSettings.Format.IniFormat)
    if not local_prefs.contains("preferences_version"):
        for key in prefs.allKeys():
            local_prefs.setValue(key, prefs.value(key))
        local_prefs.setValue("preferences_version", 1)
        local_prefs.sync()
    prefs.setValue("project_root", str(root))
    prefs.sync()
    try:
        window = Window(root, lock, local_prefs)
    except Exception as exc:
        lock.unlock()
        QMessageBox.critical(None, "Unable to open curation metadata", str(exc) + "\n\nExisting audio and ratings have not been replaced.")
        return 1
    window.show()
    return app.exec()


def main():
    parser = argparse.ArgumentParser(description="Review and rate Living Brown Noise MP3 samples.")
    parser.add_argument("--project-root", type=Path, help="Folder containing ceremonies/ and stash/")
    args = parser.parse_args()
    return run_gui(args.project_root)


if __name__ == "__main__":
    sys.exit(main())
