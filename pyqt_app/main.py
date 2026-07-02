"""
Laser Track — PyQt6 prototype
Workflow B: Camera → Batch Acquisition → Angle Statistics
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'shared_utils'))
sys.path.insert(0, str(Path(__file__).parent))

import json
import csv
from datetime import datetime

import numpy as np
import cv2 as cv

from pathlib import Path

_AUTOSAVE_PATH = Path.home() / '.laser_track_autosave.npz'

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter,
    QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QPushButton, QDoubleSpinBox, QSpinBox,
    QComboBox, QProgressBar, QTableWidget, QTableWidgetItem,
    QHeaderView, QSizePolicy, QMessageBox,
)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QFont, QColor

from gui.camera_widget import CameraWidget
from gui.config_panel import ConfigPanel
from gui.curve_extraction_dialog import CurveExtractionDialog
from gui.frame_review_dialog import FrameReviewDialog
from gui.homography_dialog import HomographyCalibrationDialog
from backend.acquisition_controller import (
    AcquisitionController, AcquisitionConfig,
    AcquisitionThread, FrameResult,
)
from backend.camera_config import CameraConfig
from optical_experiment import CaptureSession

_DEFAULT_CFG_PATH = Path.home() / '.laser_track_camera.json'


# ──────────────────────────────────────────────────────────────────────────────
# Session serialization helpers
# ──────────────────────────────────────────────────────────────────────────────

def _write_sessions_npz(sessions: list, path) -> None:
    """Serialize sessions (frames + metadata) to a compressed .npz file."""
    arrays = {}
    meta_sessions = []
    for i, s in enumerate(sessions):
        if s.frames:
            try:
                arrays[f'frames_{i}'] = np.stack(s.frames)
            except ValueError:
                pass  # mixed shapes — frames omitted for this session
        snake_json = [
            sr.tolist() if sr is not None else None
            for sr in (s.snake_results or [])
        ]
        meta_sessions.append({
            'session_id': s.session_id,
            'distance': s.distance,
            'unit': s.unit,
            'description': s.description,
            'timestamp': s.timestamp.isoformat(),
            'num_frames': s.num_frames,
            'angles': s.angles,
            'rejected': s.rejected,
            'snake_results': snake_json,
            'roi_params': s.roi_params,
            'snake_params': s.snake_params,
            'extraction_params': s.extraction_params,
        })
    arrays['meta'] = np.array([json.dumps({'version': 1, 'sessions': meta_sessions})])
    np.savez_compressed(path, **arrays)


def _read_sessions_npz(path) -> list:
    """Load CaptureSession objects from a .npz file written by _write_sessions_npz."""
    data = np.load(path, allow_pickle=False)
    payload = json.loads(str(data['meta'][0]))
    sessions = []
    for i, sm in enumerate(payload['sessions']):
        s = CaptureSession(
            distance=sm['distance'],
            unit=sm['unit'],
            description=sm.get('description', ''),
        )
        s.session_id = sm['session_id']
        s.timestamp = datetime.fromisoformat(sm['timestamp'])
        s.angles = sm['angles']
        s.rejected = sm.get('rejected', [False] * len(sm['angles']))
        s.num_frames = sm['num_frames']
        s.roi_params = sm.get('roi_params')
        s.snake_params = sm.get('snake_params')
        s.extraction_params = sm.get('extraction_params')
        key = f'frames_{i}'
        if key in data:
            arr = data[key]
            s.frames = [arr[j] for j in range(len(arr))]
        snake_raw = sm.get('snake_results') or []
        s.snake_results = [
            np.array(sr, dtype=np.float32) if sr is not None else None
            for sr in snake_raw
        ]
        sessions.append(s)
    return sessions


# ──────────────────────────────────────────────────────────────────────────────
# Angle histogram widget
# ──────────────────────────────────────────────────────────────────────────────

class AngleHistogram(QWidget):
    """Compact matplotlib histogram embedded in a QWidget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        fig = Figure(figsize=(3, 1.8), dpi=80, facecolor='#0d1117')
        fig.subplots_adjust(left=0.14, right=0.97, top=0.82, bottom=0.24)
        self._canvas = FigureCanvasQTAgg(fig)
        self._ax = fig.add_subplot(111)
        self._ax.set_facecolor('#0d1117')
        self._style_ax()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._canvas)

    def _style_ax(self):
        ax = self._ax
        ax.tick_params(colors='#c4e49a', labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor('#447130')
        ax.set_xlabel('Angle (°)', color='#c4e49a', fontsize=7)
        ax.set_ylabel('Count',     color='#c4e49a', fontsize=7)

    def plot(self, angles: list, title: str = '') -> None:
        ax = self._ax
        ax.clear()
        ax.set_facecolor('#0d1117')
        self._style_ax()
        if len(angles) < 2:
            ax.set_title(title or 'Waiting for data…', color='#888', fontsize=7)
            self._canvas.draw()
            return
        arr = np.array(angles)
        n_bins = min(max(int(np.sqrt(len(arr))), 5), 30)
        ax.hist(arr, bins=n_bins, color='#447130', edgecolor='#6EBA31', linewidth=0.5)
        mean = np.mean(arr)
        ax.axvline(mean, color='#FFB800', linewidth=1.5, linestyle='--',
                   label=f'μ = {mean:.2f}°')
        ax.set_title(title, color='#c4e49a', fontsize=7)
        ax.legend(fontsize=6, facecolor='#0d1117', edgecolor='#447130',
                  labelcolor='#FFB800', loc='upper right')
        self._canvas.draw()

    def clear(self) -> None:
        self._ax.clear()
        self._ax.set_facecolor('#0d1117')
        self._style_ax()
        self._canvas.draw()


# ──────────────────────────────────────────────────────────────────────────────
# Acquisition panel (right side)
# ──────────────────────────────────────────────────────────────────────────────

class AcquisitionPanel(QWidget):
    """
    Right-side panel: session config + live angle readout + session table.
    Wires to CameraWidget via start_acquisition() / stop_acquisition().
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread: AcquisitionThread | None = None
        self._camera_widget: CameraWidget | None = None
        self._sessions: list[CaptureSession] = []
        self._current_angles: list[float] = []
        self._build_ui()
        if _AUTOSAVE_PATH.exists():
            self._lbl_autosave.setText(
                f"Previous autosave found — use Load to recover:\n{_AUTOSAVE_PATH}"
            )
            self._lbl_autosave.setStyleSheet("color: #c4a84a; font-size: 10px;")

    def attach_camera(self, camera_widget: CameraWidget) -> None:
        """Connect this panel to a CameraWidget instance."""
        self._camera_widget = camera_widget
        camera_widget.frame_captured.connect(self._on_frame_captured)
        camera_widget.connected.connect(
            lambda _: self._btn_start.setEnabled(True)
        )
        camera_widget.disconnected.connect(self._on_camera_disconnected)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        # ── Session config ────────────────────────────────────────────────────
        cfg_grp = QGroupBox("Session Configuration")
        cfg_lay = QVBoxLayout(cfg_grp)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Distance:"))
        self._dist = QDoubleSpinBox()
        self._dist.setRange(0.0, 9999.0)
        self._dist.setValue(100.0)
        self._dist.setSuffix(" mm")
        self._dist.setDecimals(1)
        row1.addWidget(self._dist)

        row1.addSpacing(16)
        row1.addWidget(QLabel("Frames:"))
        self._nframes = QSpinBox()
        self._nframes.setRange(1, 500)
        self._nframes.setValue(30)
        row1.addWidget(self._nframes)
        cfg_lay.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Interval (ms):"))
        self._interval = QSpinBox()
        self._interval.setRange(50, 5000)
        self._interval.setValue(200)
        row2.addWidget(self._interval)

        row2.addSpacing(16)
        row2.addWidget(QLabel("Edge method:"))
        self._edge_method = QComboBox()
        self._edge_method.addItems(["canny", "sobel", "threshold"])
        row2.addWidget(self._edge_method)
        cfg_lay.addLayout(row2)

        root.addWidget(cfg_grp)

        # ── Controls ──────────────────────────────────────────────────────────
        ctrl_grp = QGroupBox("Acquisition Control")
        ctrl_lay = QVBoxLayout(ctrl_grp)

        btn_row = QHBoxLayout()
        self._btn_start = QPushButton("▶  Start")
        self._btn_start.setEnabled(False)
        self._btn_stop  = QPushButton("■  Stop")
        self._btn_stop.setEnabled(False)
        self._btn_start.clicked.connect(self._on_start)
        self._btn_stop.clicked.connect(self._on_stop)
        btn_row.addWidget(self._btn_start)
        btn_row.addWidget(self._btn_stop)
        ctrl_lay.addLayout(btn_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFormat("%v / %m frames")
        ctrl_lay.addWidget(self._progress)

        root.addWidget(ctrl_grp)

        # ── Live readout ──────────────────────────────────────────────────────
        live_grp = QGroupBox("Live Readout")
        live_lay = QVBoxLayout(live_grp)

        mono = QFont("Monospace", 11)

        def _metric_row(label: str) -> QLabel:
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setFixedWidth(120)
            val = QLabel("—")
            val.setFont(mono)
            val.setStyleSheet("color: #6EBA31;")
            row.addWidget(lbl)
            row.addWidget(val)
            live_lay.addLayout(row)
            return val

        self._lbl_current = _metric_row("Current angle:")
        self._lbl_mean    = _metric_row("Running mean:")
        self._lbl_std     = _metric_row("Std dev:")
        self._lbl_time    = _metric_row("Frame time:")
        self._lbl_status  = QLabel("Connect camera and press Start.")
        self._lbl_status.setWordWrap(True)
        self._lbl_status.setStyleSheet("color: #aaa; font-size: 11px;")
        live_lay.addWidget(self._lbl_status)

        self._histogram = AngleHistogram()
        self._histogram.setMinimumHeight(144)
        live_lay.addWidget(self._histogram)

        root.addWidget(live_grp)

        # ── Captured frame notice (Workflow A) ────────────────────────────────
        self._capture_notice = QLabel("")
        self._capture_notice.setStyleSheet(
            "background:#1a2a1a; color:#6EBA31; padding:4px; border-radius:4px;"
        )
        self._capture_notice.setVisible(False)
        root.addWidget(self._capture_notice)

        # ── Completed sessions table ──────────────────────────────────────────
        sess_grp = QGroupBox("Completed Sessions")
        sess_lay = QVBoxLayout(sess_grp)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["ID", "Distance", "Frames", "Mean (°)", "Std (°)"]
        )
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self._table.itemSelectionChanged.connect(self._on_table_selection_changed)
        sess_lay.addWidget(self._table)

        btn_row_sess = QHBoxLayout()
        self._btn_review = QPushButton("Review frames…")
        self._btn_review.setEnabled(False)
        self._btn_review.clicked.connect(self._on_review_session)
        btn_row_sess.addWidget(self._btn_review)

        clear_btn = QPushButton("Clear all sessions")
        clear_btn.clicked.connect(self._on_clear_sessions)
        btn_row_sess.addWidget(clear_btn)
        sess_lay.addLayout(btn_row_sess)

        btn_row_io = QHBoxLayout()
        self._btn_save = QPushButton("Save…")
        self._btn_save.clicked.connect(self._on_save_sessions)
        btn_row_io.addWidget(self._btn_save)
        self._btn_load = QPushButton("Load…")
        self._btn_load.clicked.connect(self._on_load_sessions)
        btn_row_io.addWidget(self._btn_load)
        self._btn_csv = QPushButton("Export CSV…")
        self._btn_csv.clicked.connect(self._on_export_csv)
        btn_row_io.addWidget(self._btn_csv)
        sess_lay.addLayout(btn_row_io)

        self._lbl_autosave = QLabel("")
        self._lbl_autosave.setStyleSheet("color: #556655; font-size: 10px;")
        self._lbl_autosave.setWordWrap(True)
        sess_lay.addWidget(self._lbl_autosave)

        root.addWidget(sess_grp)
        root.addStretch()

    # ── Slots ─────────────────────────────────────────────────────────────────

    @pyqtSlot()
    def _on_start(self):
        if self._camera_widget is None or not self._camera_widget.is_connected:
            QMessageBox.warning(self, "No camera", "Connect a camera first.")
            return

        config = AcquisitionConfig(
            distance          = self._dist.value(),
            unit              = "mm",
            num_frames        = self._nframes.value(),
            frame_interval_ms = self._interval.value(),
            edge_params       = {
                'method':     self._edge_method.currentText(),
                'canny_low':  50,
                'canny_high': 150,
                'blur_kernel': 3,
                'morphology': True,
            },
        )

        self._thread = AcquisitionThread(config, parent=self)
        self._camera_widget.frame_ready.connect(self._thread.on_frame)
        self._thread.frame_processed.connect(self._on_frame_processed)
        self._thread.session_complete.connect(self._on_session_complete)
        self._thread.error.connect(self._on_acq_error)
        self._thread.finished.connect(self._on_thread_finished)

        self._current_angles = []
        self._histogram.clear()
        self._table.clearSelection()

        self._progress.setMaximum(config.num_frames)
        self._progress.setValue(0)
        self._progress.setFormat(f"%v / {config.num_frames} frames")
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._lbl_status.setText("Acquiring…")

        self._thread.start()

    @pyqtSlot()
    def _on_stop(self):
        if self._thread is not None:
            session = self._thread.stop()
            if session and session.num_frames > 0:
                self._add_session(session)
        self._reset_controls()

    @pyqtSlot(object)
    def _on_frame_processed(self, result: FrameResult):
        self._lbl_current.setText(f"{result.angle_deg:+.2f}°")
        self._lbl_time.setText(f"{result.elapsed_ms:.0f} ms")

        s = result.session_stats
        if s:
            self._lbl_mean.setText(f"{s['mean']:+.2f}°")
            self._lbl_std.setText(f"{s['std']:.3f}°")

        cur, total = result.frame_index + 1, self._progress.maximum()
        self._progress.setValue(cur)
        self._lbl_status.setText(f"Frame {cur}/{total}")

        self._current_angles.append(result.angle_deg)
        self._histogram.plot(self._current_angles,
                             title=f"Live — {len(self._current_angles)}/{total} frames")

        # Show acquired frame with fitted line overlaid
        if self._camera_widget is not None:
            roi_rgb = result.diagnostics.get('roi_image')
            snake   = result.diagnostics.get('snake_optimized')
            if roi_rgb is not None and snake is not None and len(snake) >= 2:
                annotated = roi_rgb.copy()
                x0, y0 = int(snake[0, 0]),  int(snake[0, 1])
                x1, y1 = int(snake[-1, 0]), int(snake[-1, 1])
                cv.line(annotated, (x0, y0), (x1, y1), (255, 64, 64), 2)
                self._camera_widget.freeze_preview(
                    cv.cvtColor(annotated, cv.COLOR_RGB2BGR)
                )

    @pyqtSlot(object)
    def _on_session_complete(self, session: CaptureSession):
        self._add_session(session)
        self._reset_controls()
        self._lbl_status.setText(
            f"Done — {session.num_frames} frames @ "
            f"{session.distance} {session.unit}"
        )

    @pyqtSlot(str)
    def _on_acq_error(self, msg: str):
        self._lbl_status.setText(f"Error: {msg}")
        self._reset_controls()

    @pyqtSlot()
    def _on_thread_finished(self):
        self._reset_controls()

    @pyqtSlot(object)
    def _on_frame_captured(self, rgb_frame: np.ndarray):
        """Single frame from Capture button → Workflow A (profile analysis)."""
        h, w = rgb_frame.shape[:2]
        self._capture_notice.setText(
            f"📸 Frame captured ({w}×{h}) — ready for profile analysis"
        )
        self._capture_notice.setVisible(True)

    @pyqtSlot()
    def _on_camera_disconnected(self):
        self._btn_start.setEnabled(False)
        if self._thread is not None and self._thread.isRunning():
            self._on_stop()

    @pyqtSlot()
    def _on_clear_sessions(self):
        self._sessions.clear()
        self._table.setRowCount(0)
        self._histogram.clear()

    @pyqtSlot()
    def _on_table_selection_changed(self):
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            self._btn_review.setEnabled(False)
            return
        row = rows[0].row()
        self._btn_review.setEnabled(row < len(self._sessions))
        if row < len(self._sessions):
            s = self._sessions[row]
            self._histogram.plot(
                s.angles,
                title=f"Session {s.session_id} — {s.num_frames} frames @ {s.distance} {s.unit}"
            )

    @pyqtSlot()
    def _on_review_session(self):
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return
        row = rows[0].row()
        if row >= len(self._sessions):
            return
        session = self._sessions[row]
        if not session.frames:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(
                self, "No frames",
                "This session has no stored frame data to review."
            )
            return
        dlg = FrameReviewDialog(session, parent=self)
        dlg.exec()
        # Refresh table row stats in case rejections changed the filtered mean/std
        stats = session.get_filtered_statistics() or session.get_statistics()
        if stats:
            green = QColor("#6EBA31")
            self._table.item(row, 3).setText(f"{stats['mean']:.3f}")
            self._table.item(row, 3).setForeground(green)
            self._table.item(row, 4).setText(f"{stats['std']:.3f}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _add_session(self, session: CaptureSession, *, do_autosave: bool = True):
        self._sessions.append(session)
        stats = session.get_statistics()
        row = self._table.rowCount()
        self._table.insertRow(row)

        def _cell(text: str, color: QColor | None = None) -> QTableWidgetItem:
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if color:
                item.setForeground(color)
            return item

        green = QColor("#6EBA31")
        self._table.setItem(row, 0, _cell(session.session_id))
        self._table.setItem(row, 1, _cell(f"{session.distance} {session.unit}"))
        self._table.setItem(row, 2, _cell(str(session.num_frames)))
        if stats:
            self._table.setItem(row, 3, _cell(f"{stats['mean']:.3f}", green))
            self._table.setItem(row, 4, _cell(f"{stats['std']:.3f}"))
        else:
            self._table.setItem(row, 3, _cell("—"))
            self._table.setItem(row, 4, _cell("—"))

        if do_autosave:
            self._autosave()

    # ── Save / Load / Export ──────────────────────────────────────────────────

    @pyqtSlot()
    def _on_save_sessions(self):
        if not self._sessions:
            QMessageBox.information(self, "No sessions", "No sessions to save.")
            return
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Sessions",
            str(Path.home() / 'laser_sessions.npz'),
            "Session files (*.npz)",
        )
        if not path:
            return
        try:
            _write_sessions_npz(self._sessions, path)
            self._lbl_autosave.setText(
                f"Saved {len(self._sessions)} session(s) → {path}"
            )
            self._lbl_autosave.setStyleSheet("color: #6EBA31; font-size: 10px;")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))

    @pyqtSlot()
    def _on_load_sessions(self):
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Sessions",
            str(Path.home()),
            "Session files (*.npz)",
        )
        if not path:
            return
        try:
            loaded = _read_sessions_npz(path)
        except Exception as e:
            QMessageBox.critical(self, "Load failed", str(e))
            return
        for session in loaded:
            self._add_session(session, do_autosave=False)
        self._autosave()
        self._lbl_autosave.setText(
            f"Loaded {len(loaded)} session(s) from {path}"
        )
        self._lbl_autosave.setStyleSheet("color: #6EBA31; font-size: 10px;")

    @pyqtSlot()
    def _on_export_csv(self):
        if not self._sessions:
            QMessageBox.information(self, "No sessions", "No sessions to export.")
            return
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV",
            str(Path.home() / 'laser_angles.csv'),
            "CSV files (*.csv)",
        )
        if not path:
            return
        try:
            with open(path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow([
                    'session_id', 'timestamp', 'distance', 'unit',
                    'n_frames', 'n_accepted',
                    'mean_deg', 'std_deg', 'sem_deg',
                    'ci95_lower', 'ci95_upper',
                ])
                for s in self._sessions:
                    stats = s.get_filtered_statistics() or s.get_statistics()
                    if not stats:
                        continue
                    w.writerow([
                        s.session_id,
                        s.timestamp.isoformat(),
                        s.distance, s.unit,
                        s.num_frames, stats['n_frames'],
                        f"{stats['mean']:.6f}",
                        f"{stats['std']:.6f}",
                        f"{stats['sem']:.6f}",
                        f"{stats['ci_95_lower']:.6f}",
                        f"{stats['ci_95_upper']:.6f}",
                    ])
            self._lbl_autosave.setText(f"CSV exported → {path}")
            self._lbl_autosave.setStyleSheet("color: #6EBA31; font-size: 10px;")
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))

    def _autosave(self):
        try:
            _write_sessions_npz(self._sessions, _AUTOSAVE_PATH)
            self._lbl_autosave.setText(
                f"Autosaved {len(self._sessions)} session(s)"
            )
            self._lbl_autosave.setStyleSheet("color: #556655; font-size: 10px;")
        except Exception:
            pass

    def _reset_controls(self):
        connected = (
            self._camera_widget is not None
            and self._camera_widget.is_connected
        )
        self._btn_start.setEnabled(connected)
        self._btn_stop.setEnabled(False)
        if self._thread is not None:
            try:
                self._camera_widget.frame_ready.disconnect(self._thread.on_frame)
            except RuntimeError:
                pass
            self._thread = None
        if self._camera_widget is not None:
            self._camera_widget.unfreeze_preview()


# ──────────────────────────────────────────────────────────────────────────────
# Main window
# ──────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Laser Track — Prototype")
        self.resize(1100, 680)
        self._reference_curve: np.ndarray | None = None
        self._build_ui()
        self._apply_style()
        self._load_saved_config()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left pane: camera + collapsible config panel ───────────────
        left_pane = QWidget()
        left_layout = QVBoxLayout(left_pane)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        self._camera = CameraWidget()
        self._camera.setMinimumWidth(400)
        left_layout.addWidget(self._camera, stretch=1)

        cam_btn_row = QHBoxLayout()
        self._btn_settings = QPushButton("⚙  Camera Settings  ▼")
        self._btn_settings.setCheckable(True)
        self._btn_settings.setChecked(False)
        self._btn_settings.toggled.connect(self._on_settings_toggled)
        cam_btn_row.addWidget(self._btn_settings)

        self._btn_calibrate = QPushButton("Calibrate H…")
        self._btn_calibrate.setFixedWidth(110)
        self._btn_calibrate.clicked.connect(self._on_open_homography_dialog)
        cam_btn_row.addWidget(self._btn_calibrate)

        self._lbl_calib_status = QLabel("No H")
        self._lbl_calib_status.setStyleSheet("color:#556655; font-size:10px;")
        self._lbl_calib_status.setFixedWidth(60)
        cam_btn_row.addWidget(self._lbl_calib_status)

        left_layout.addLayout(cam_btn_row)

        self._config_panel = ConfigPanel()
        self._config_panel.setVisible(False)
        left_layout.addWidget(self._config_panel)

        # Wire config panel ↔ camera widget
        self._config_panel.config_changed.connect(self._camera.set_config)
        self._camera._preview.roi_drawn.connect(self._config_panel.set_roi)

        splitter.addWidget(left_pane)

        # ── Right pane: reference bar + acquisition panel ─────────────
        right_pane = QWidget()
        right_lay = QVBoxLayout(right_pane)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(0)

        ref_bar = QWidget()
        ref_bar_lay = QHBoxLayout(ref_bar)
        ref_bar_lay.setContentsMargins(8, 4, 8, 4)
        self._btn_ref = QPushButton("Reference Geometry (Cam 2)…")
        self._btn_ref.clicked.connect(self._on_open_reference_dialog)
        ref_bar_lay.addWidget(self._btn_ref)
        self._lbl_ref_status = QLabel("No reference")
        self._lbl_ref_status.setStyleSheet("color: #556655; font-size: 10px;")
        ref_bar_lay.addWidget(self._lbl_ref_status)
        ref_bar_lay.addStretch()
        right_lay.addWidget(ref_bar)

        self._acq = AcquisitionPanel()
        self._acq.setMinimumWidth(340)
        self._acq.attach_camera(self._camera)
        right_lay.addWidget(self._acq, stretch=1)

        splitter.addWidget(right_pane)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter)

    @pyqtSlot()
    def _on_open_homography_dialog(self):
        current_H = self._config_panel.get_config().homography
        initial_frame = self._camera.get_last_frame()

        def _apply_H(H: np.ndarray):
            cfg = self._config_panel.get_config()
            cfg.homography = H
            self._config_panel.set_config(cfg)
            self._camera.set_config(cfg)
            self._lbl_calib_status.setText("H ✓")
            self._lbl_calib_status.setStyleSheet("color:#6EBA31; font-size:10px;")

        dlg = HomographyCalibrationDialog(
            on_apply=_apply_H,
            initial_frame=initial_frame,
            current_H=current_H,
            parent=self,
        )
        dlg.exec()

    @pyqtSlot()
    def _on_open_reference_dialog(self):
        dlg = CurveExtractionDialog(
            last_curve=self._reference_curve, parent=self
        )
        if dlg.exec() and dlg.result is not None:
            self._reference_curve = dlg.result.mean_curve
            n = len(self._reference_curve)
            self._lbl_ref_status.setText(f"Reference: {n} pts ✓")
            self._lbl_ref_status.setStyleSheet("color: #6EBA31; font-size: 10px;")

    @pyqtSlot(bool)
    def _on_settings_toggled(self, checked: bool):
        self._config_panel.setVisible(checked)
        self._camera.set_editing_mode(checked)
        arrow = "▲" if checked else "▼"
        self._btn_settings.setText(f"⚙  Camera Settings  {arrow}")

    def _load_saved_config(self):
        if _DEFAULT_CFG_PATH.exists():
            try:
                cfg = CameraConfig.load(_DEFAULT_CFG_PATH)
                self._config_panel.set_config(cfg)
            except Exception:
                pass

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #0d1117;
                color: #c4e49a;
            }
            QGroupBox {
                border: 1px solid #447130;
                border-radius: 6px;
                margin-top: 10px;
                font-weight: bold;
                color: #6EBA31;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QPushButton {
                background: #2a3a2a;
                color: #c4e49a;
                border: 1px solid #447130;
                border-radius: 4px;
                padding: 5px 12px;
                min-height: 26px;
            }
            QPushButton:hover  { background: #3a5a2a; border-color: #6EBA31; }
            QPushButton:pressed { background: #1a2a1a; }
            QPushButton:disabled { color: #556655; border-color: #334433; }
            QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
                background: #1a2a1a;
                color: #c4e49a;
                border: 1px solid #447130;
                border-radius: 3px;
                padding: 2px 6px;
            }
            QProgressBar {
                border: 1px solid #447130;
                border-radius: 4px;
                text-align: center;
                color: #c4e49a;
                background: #1a2a1a;
            }
            QProgressBar::chunk { background: #447130; border-radius: 3px; }
            QTableWidget {
                background: #0d1117;
                gridline-color: #2a3a2a;
                color: #c4e49a;
            }
            QHeaderView::section {
                background: #1a2a1a;
                color: #6EBA31;
                border: 1px solid #447130;
                padding: 4px;
            }
            QLabel { color: #c4e49a; }
            QSplitter::handle { background: #447130; width: 2px; }
        """)

    def closeEvent(self, event):
        # Ensure threads are stopped cleanly on window close
        if self._acq._thread is not None:
            self._acq._thread.stop()
        self._camera._stop_thread()
        super().closeEvent(event)


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
