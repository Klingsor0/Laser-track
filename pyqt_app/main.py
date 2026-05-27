"""
Laser Track — PyQt6 prototype
Workflow B: Camera → Batch Acquisition → Angle Statistics
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'shared_utils'))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import cv2 as cv

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
from backend.acquisition_controller import (
    AcquisitionController, AcquisitionConfig,
    AcquisitionThread, FrameResult,
)
from optical_experiment import CaptureSession


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
        self._build_ui()

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
        sess_lay.addWidget(self._table)

        clear_btn = QPushButton("Clear all sessions")
        clear_btn.clicked.connect(self._on_clear_sessions)
        sess_lay.addWidget(clear_btn)

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

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _add_session(self, session: CaptureSession):
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


# ──────────────────────────────────────────────────────────────────────────────
# Main window
# ──────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Laser Track — Prototype")
        self.resize(1100, 680)
        self._build_ui()
        self._apply_style()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: camera widget
        self._camera = CameraWidget()
        self._camera.setMinimumWidth(400)
        splitter.addWidget(self._camera)

        # Right: acquisition panel
        self._acq = AcquisitionPanel()
        self._acq.setMinimumWidth(340)
        self._acq.attach_camera(self._camera)
        splitter.addWidget(self._acq)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter)

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
