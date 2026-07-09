"""
Curve Extraction Dialog — Workflow C.

One-time reference measurement using a second camera.
The user:
  1. Connects a second camera and freezes a reference frame.
  2. Draws an initial polygon on the frozen frame (or loads the previous result).
  3. Sets snake / acquisition parameters.
  4. Runs batch acquisition — every frame is optimized from the same initial snake.
  5. Accepts the result (mean curve + all per-frame curves).

The accepted CurveResult.mean_curve is offered as the initial snake on next open.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'shared_utils'))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2 as cv

from PyQt6.QtWidgets import (
    QDialog, QWidget, QSplitter, QScrollArea,
    QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QPushButton, QDoubleSpinBox, QSpinBox,
    QProgressBar, QDialogButtonBox, QFileDialog, QMessageBox,
    QInputDialog, QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSlot

from gui.camera_widget import CameraWidget
from gui.config_panel import ConfigPanel
from gui.homography_dialog import HomographyCalibrationDialog
from backend.curve_acquisition import CurveConfig, CurveResult, FixedFrames, CurveAcquisitionThread
from backend.camera_config import apply_acquisition_pipeline
import snake1 as snk


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class CurveExtractionDialog(QDialog):
    """
    Parameters
    ----------
    last_curve : np.ndarray (N, 2) or None
        Mean curve from the previous accepted extraction.  Offered to the
        user via "Load last curve" so they don't have to re-draw every session.
    parent : QWidget or None
    """

    def __init__(self, last_curve: np.ndarray | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Reference Geometry — Curve Extraction (Cam 2)")
        self.resize(1100, 700)

        self._last_curve: np.ndarray | None = last_curve  # from previous session
        self._frozen_bgr: np.ndarray | None = None        # reference frame
        self._initial_snake: np.ndarray | None = None     # snake ready for acquisition
        self._thread: CurveAcquisitionThread | None = None
        self.result: CurveResult | None = None            # set on Accept
        self._px_per_mm: float | None = None              # pixels per mm (None = uncalibrated)

        self._build_ui()
        self._apply_style()
        self._update_button_states()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left: camera + config ─────────────────────────────────────
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(0)

        self._camera = CameraWidget()
        self._camera.setMinimumWidth(420)
        left_lay.addWidget(self._camera, stretch=1)

        cam_btn_row = QHBoxLayout()
        self._btn_settings = QPushButton("⚙  Camera Settings  ▼")
        self._btn_settings.setCheckable(True)
        self._btn_settings.toggled.connect(self._on_settings_toggled)
        cam_btn_row.addWidget(self._btn_settings)

        self._btn_calibrate_h = QPushButton("Calibrate H…")
        self._btn_calibrate_h.setFixedWidth(110)
        self._btn_calibrate_h.clicked.connect(self._on_open_homography_dialog)
        cam_btn_row.addWidget(self._btn_calibrate_h)

        self._lbl_h_status = QLabel("No H")
        self._lbl_h_status.setStyleSheet("color:#556655; font-size:10px;")
        self._lbl_h_status.setFixedWidth(60)
        cam_btn_row.addWidget(self._lbl_h_status)

        left_lay.addLayout(cam_btn_row)

        self._config_panel = ConfigPanel()
        self._config_panel.setVisible(False)
        left_lay.addWidget(self._config_panel)

        self._config_panel.config_changed.connect(self._camera.set_config)
        self._camera._preview.roi_drawn.connect(self._config_panel.set_roi)

        splitter.addWidget(left)

        # ── Right: controls (scrollable) ──────────────────────────────
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setMinimumWidth(320)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(8, 8, 8, 8)
        right_lay.setSpacing(8)

        right_lay.addWidget(self._make_polygon_group())
        right_lay.addWidget(self._make_snake_group())
        right_lay.addWidget(self._make_acq_group())
        right_lay.addWidget(self._make_result_group())
        right_lay.addStretch()

        # ── Dialog buttons ────────────────────────────────────────────
        self._btn_box = QDialogButtonBox()
        self._btn_accept = self._btn_box.addButton(
            "Accept", QDialogButtonBox.ButtonRole.AcceptRole
        )
        self._btn_cancel = self._btn_box.addButton(
            "Cancel", QDialogButtonBox.ButtonRole.RejectRole
        )
        self._btn_accept.setEnabled(False)
        self._btn_box.accepted.connect(self._on_accept)
        self._btn_box.rejected.connect(self.reject)
        right_lay.addWidget(self._btn_box)

        right_scroll.setWidget(right)
        splitter.addWidget(right_scroll)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        root.addWidget(splitter)

    def _make_polygon_group(self) -> QGroupBox:
        grp = QGroupBox("Reference Polygon")
        lay = QVBoxLayout(grp)

        hint = QLabel(
            "1. Freeze  2. Click to add vertices  3. Close Polygon  "
            "4. Start — add more vertices and re-close anytime to update snake"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888; font-size: 10px;")
        lay.addWidget(hint)

        row1 = QHBoxLayout()
        self._btn_freeze = QPushButton("Freeze Frame")
        self._btn_freeze.setEnabled(False)
        self._btn_freeze.clicked.connect(self._on_freeze)
        row1.addWidget(self._btn_freeze)

        self._btn_unfreeze = QPushButton("Unfreeze")
        self._btn_unfreeze.setEnabled(False)
        self._btn_unfreeze.clicked.connect(self._on_unfreeze)
        row1.addWidget(self._btn_unfreeze)
        lay.addLayout(row1)

        row2 = QHBoxLayout()
        last_label = "Load last curve"
        if self._last_curve is not None:
            last_label = f"Load last curve ({len(self._last_curve)} pts)"
        self._btn_load_last = QPushButton(last_label)
        self._btn_load_last.setEnabled(False)
        self._btn_load_last.clicked.connect(self._on_load_last)
        row2.addWidget(self._btn_load_last)

        self._btn_clear_poly = QPushButton("Clear")
        self._btn_clear_poly.setEnabled(False)
        self._btn_clear_poly.clicked.connect(self._on_clear_polygon)
        row2.addWidget(self._btn_clear_poly)
        lay.addLayout(row2)

        self._btn_close_poly = QPushButton("Close Polygon")
        self._btn_close_poly.setEnabled(False)
        self._btn_close_poly.clicked.connect(self._on_close_polygon)
        lay.addWidget(self._btn_close_poly)

        self._lbl_poly_status = QLabel("No polygon drawn.")
        self._lbl_poly_status.setStyleSheet("color: #888; font-size: 10px;")
        lay.addWidget(self._lbl_poly_status)

        scale_row = QHBoxLayout()
        self._btn_set_scale = QPushButton("Set Scale…")
        self._btn_set_scale.setEnabled(False)
        self._btn_set_scale.setToolTip(
            "Click two points on a known object, then enter its real size in mm.\n"
            "All curve coordinates will be reported in mm."
        )
        self._btn_set_scale.clicked.connect(self._on_set_scale)
        scale_row.addWidget(self._btn_set_scale)

        self._lbl_scale = QLabel("No scale set")
        self._lbl_scale.setStyleSheet("color: #888; font-size: 10px;")
        scale_row.addWidget(self._lbl_scale, stretch=1)
        lay.addLayout(scale_row)

        # For live cameras: enable Freeze Frame on connect.
        # For image files: auto-freeze immediately (no "right moment" to pick).
        self._camera.connected.connect(self._on_camera_connected)
        self._camera.disconnected.connect(lambda: self._btn_freeze.setEnabled(False))

        return grp

    def _make_snake_group(self) -> QGroupBox:
        grp = QGroupBox("Snake Parameters")
        lay = QVBoxLayout(grp)

        def _row(label, widget):
            r = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setFixedWidth(110)
            r.addWidget(lbl)
            r.addWidget(widget)
            lay.addLayout(r)

        self._sp_num_points = QSpinBox()
        self._sp_num_points.setRange(10, 500)
        self._sp_num_points.setValue(100)
        _row("Points:", self._sp_num_points)

        self._sp_iterations = QSpinBox()
        self._sp_iterations.setRange(10, 500)
        self._sp_iterations.setValue(100)
        _row("Iterations:", self._sp_iterations)

        self._sp_window = QSpinBox()
        self._sp_window.setRange(1, 50)
        self._sp_window.setValue(5)
        _row("Window size:", self._sp_window)

        self._sp_alpha = QDoubleSpinBox()
        self._sp_alpha.setRange(0.0, 1.0)
        self._sp_alpha.setSingleStep(0.001)
        self._sp_alpha.setDecimals(4)
        self._sp_alpha.setValue(0.01)
        _row("α (elasticity):", self._sp_alpha)

        self._sp_beta = QDoubleSpinBox()
        self._sp_beta.setRange(-0.2, 0.2)
        self._sp_beta.setSingleStep(0.001)
        self._sp_beta.setDecimals(4)
        self._sp_beta.setValue(0.1)
        _row("β (rigidity):", self._sp_beta)

        self._sp_gamma = QDoubleSpinBox()
        self._sp_gamma.setRange(-2.0, 2.0)
        self._sp_gamma.setSingleStep(0.1)
        self._sp_gamma.setDecimals(4)
        self._sp_gamma.setValue(0.5)
        _row("γ (edge pull):", self._sp_gamma)

        self._sp_threshold = QDoubleSpinBox()
        self._sp_threshold.setRange(1e-6, 0.1)
        self._sp_threshold.setSingleStep(0.0001)
        self._sp_threshold.setDecimals(6)
        self._sp_threshold.setValue(0.0001)
        _row("Threshold:", self._sp_threshold)

        return grp

    def _make_acq_group(self) -> QGroupBox:
        grp = QGroupBox("Acquisition")
        lay = QVBoxLayout(grp)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Frames:"))
        self._sp_frames = QSpinBox()
        self._sp_frames.setRange(1, 500)
        self._sp_frames.setValue(30)
        row1.addWidget(self._sp_frames)

        row1.addSpacing(12)
        row1.addWidget(QLabel("Interval (ms):"))
        self._sp_interval = QSpinBox()
        self._sp_interval.setRange(50, 5000)
        self._sp_interval.setValue(200)
        row1.addWidget(self._sp_interval)
        lay.addLayout(row1)

        btn_row = QHBoxLayout()
        self._btn_start = QPushButton("▶  Start")
        self._btn_start.setEnabled(False)
        self._btn_start.clicked.connect(self._on_start)
        self._btn_stop = QPushButton("■  Stop")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_stop)
        btn_row.addWidget(self._btn_start)
        btn_row.addWidget(self._btn_stop)
        lay.addLayout(btn_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFormat("%v / %m frames")
        lay.addWidget(self._progress)

        self._lbl_acq_status = QLabel("Set up polygon, then press Start.")
        self._lbl_acq_status.setWordWrap(True)
        self._lbl_acq_status.setStyleSheet("color: #aaa; font-size: 10px;")
        lay.addWidget(self._lbl_acq_status)

        return grp

    def _make_result_group(self) -> QGroupBox:
        grp = QGroupBox("Result")
        lay = QVBoxLayout(grp)

        self._lbl_result = QLabel("No result yet.")
        self._lbl_result.setStyleSheet("color: #888; font-size: 10px;")
        lay.addWidget(self._lbl_result)

        export_row = QHBoxLayout()
        self._btn_export_csv = QPushButton("Export CSV")
        self._btn_export_csv.setEnabled(False)
        self._btn_export_csv.clicked.connect(lambda: self._export('csv'))
        export_row.addWidget(self._btn_export_csv)

        self._btn_export_npy = QPushButton("Export NPY")
        self._btn_export_npy.setEnabled(False)
        self._btn_export_npy.clicked.connect(lambda: self._export('npy'))
        export_row.addWidget(self._btn_export_npy)
        lay.addLayout(export_row)

        self._btn_clear_result = QPushButton("Clear Result")
        self._btn_clear_result.setEnabled(False)
        self._btn_clear_result.setToolTip(
            "Discard this result and go back to the initial snake overlay so you can re-run with different parameters."
        )
        self._btn_clear_result.clicked.connect(self._on_clear_result)
        lay.addWidget(self._btn_clear_result)

        return grp

    # ------------------------------------------------------------------
    # Slots — polygon workflow
    # ------------------------------------------------------------------

    @pyqtSlot(str)
    def _on_camera_connected(self, _source: str):
        if self._camera.source_type == 'image':
            # Static image: skip manual freeze step — just freeze immediately
            self._on_freeze()
        else:
            self._btn_freeze.setEnabled(True)

    @pyqtSlot()
    def _on_freeze(self):
        frame = self._camera.get_last_frame()
        if frame is None:
            QMessageBox.warning(self, "No frame", "Camera has not delivered a frame yet.")
            return
        self._frozen_bgr = frame.copy()
        self._camera.freeze_preview(self._frozen_bgr)
        self._camera._preview.enable_polygon_mode()
        self._btn_freeze.setEnabled(False)
        self._btn_unfreeze.setEnabled(True)
        self._btn_clear_poly.setEnabled(True)
        self._btn_close_poly.setEnabled(True)
        if self._last_curve is not None:
            self._btn_load_last.setEnabled(True)
        self._lbl_poly_status.setText("Click on the image to add polygon vertices.")
        self._update_button_states()

    @pyqtSlot()
    def _on_unfreeze(self):
        self._camera._preview.disable_polygon_mode()
        self._camera.unfreeze_preview()
        self._frozen_bgr = None
        self._initial_snake = None
        self._btn_freeze.setEnabled(self._camera.is_connected)
        self._btn_unfreeze.setEnabled(False)
        self._btn_clear_poly.setEnabled(False)
        self._btn_close_poly.setEnabled(False)
        self._btn_load_last.setEnabled(False)
        self._lbl_poly_status.setText("No polygon drawn.")
        self._update_button_states()

    @pyqtSlot()
    def _on_load_last(self):
        if self._last_curve is None:
            return
        if self._frozen_bgr is None:
            frame = self._camera.get_last_frame()
            if frame is None:
                QMessageBox.warning(self, "No frame", "Freeze a frame first.")
                return
            self._frozen_bgr = frame.copy()
            self._camera.freeze_preview(self._frozen_bgr)

        self._camera._preview.disable_polygon_mode()
        self._initial_snake = self._last_curve.copy()
        self._lbl_poly_status.setText(
            f"Loaded previous extraction: {len(self._initial_snake)} points."
        )
        self._show_snake_overlay(self._initial_snake, color=(0, 212, 255))
        self._update_button_states()

    @pyqtSlot()
    def _on_clear_polygon(self):
        self._initial_snake = None
        self._camera._preview.clear_polygon_pts()
        if self._frozen_bgr is not None:
            # Restore the clean frozen frame (remove snake overlay)
            self._camera.freeze_preview(self._frozen_bgr)
        self._lbl_poly_status.setText("Polygon cleared. Click to add vertices.")
        self._update_button_states()

    @pyqtSlot()
    def _on_close_polygon(self):
        pts = self._camera._preview.get_polygon_pts()
        if len(pts) < 3:
            QMessageBox.warning(
                self, "Too few points",
                f"Draw at least 3 vertices (currently {len(pts)})."
            )
            return
        num_pts = self._sp_num_points.value()
        self._initial_snake = snk.initialize_snake_from_polygon(pts, num_points=num_pts)
        # Hide polygon overlay but keep vertices — user can add more and re-close.
        self._camera._preview.hide_polygon_overlay()
        self._lbl_poly_status.setText(
            f"Snake ready: {len(self._initial_snake)} pts from {len(pts)} vertices. "
            "Add more vertices or click Close Polygon again to update."
        )
        self._show_snake_overlay(self._initial_snake, color=(0, 212, 255))
        self._update_button_states()

    # ------------------------------------------------------------------
    # Slots — pixel/mm scale calibration
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_set_scale(self):
        """Enter 2-point measurement mode; user clicks two points on a known object."""
        self._lbl_scale.setText("Click first point…")
        self._lbl_scale.setStyleSheet("color: #c4e49a; font-size: 10px;")
        preview = self._camera._preview
        preview.enable_measure_mode()
        preview.measurement_done.connect(self._on_measurement_done)

    @pyqtSlot(int, int, int, int)
    def _on_measurement_done(self, x1: int, y1: int, x2: int, y2: int):
        self._camera._preview.measurement_done.disconnect(self._on_measurement_done)
        px_dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if px_dist < 1:
            self._lbl_scale.setText("Points too close — try again.")
            self._lbl_scale.setStyleSheet("color: #cc4444; font-size: 10px;")
            return
        mm_val, ok = QInputDialog.getDouble(
            self, "Known distance",
            f"Real distance between the two points (pixels: {px_dist:.1f}):",
            decimals=3, min=0.001, max=100000.0, value=1.0,
        )
        if not ok or mm_val <= 0:
            self._lbl_scale.setText("Scale cancelled.")
            self._lbl_scale.setStyleSheet("color: #888; font-size: 10px;")
            return
        self._px_per_mm = px_dist / mm_val
        self._lbl_scale.setText(f"{self._px_per_mm:.3f} px/mm")
        self._lbl_scale.setStyleSheet("color: #6EBA31; font-size: 10px;")

    # ------------------------------------------------------------------
    # Slots — acquisition
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_start(self):
        if self._initial_snake is None or not self._camera.is_connected:
            return

        is_image = self._camera.source_type == 'image'
        config = CurveConfig(
            num_frames        = self._sp_frames.value(),
            # Static image has no real-time stream: disable interval throttle so all
            # seeded copies of the frame are processed without artificial delays.
            frame_interval_ms = 0 if is_image else self._sp_interval.value(),
            num_points        = self._sp_num_points.value(),
            num_iterations    = self._sp_iterations.value(),
            window_size       = self._sp_window.value(),
            alpha             = self._sp_alpha.value(),
            beta              = self._sp_beta.value(),
            gamma             = self._sp_gamma.value(),
            threshold         = self._sp_threshold.value(),
        )
        criterion = FixedFrames(config.num_frames)

        self._thread = CurveAcquisitionThread(
            self._initial_snake, config, criterion, parent=self
        )
        self._camera.frame_ready.connect(self._thread.on_frame)
        self._thread.curve_extracted.connect(self._on_curve_extracted)
        self._thread.acquisition_complete.connect(self._on_acquisition_complete)
        self._thread.progress.connect(self._on_progress)
        self._thread.iteration_update.connect(self._on_iteration_update)
        self._thread.error.connect(self._on_acq_error)

        self._progress.setMaximum(config.num_frames)
        self._progress.setValue(0)
        self._lbl_acq_status.setText("Acquiring…")
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._thread.start()

        # Image mode: no live stream — feed the frozen frame into the thread queue
        # so the snake runs num_frames times on the same image.
        if is_image and self._frozen_bgr is not None:
            processed = apply_acquisition_pipeline(self._frozen_bgr, self._camera._config)
            for _ in range(config.num_frames):
                self._thread.on_frame(processed)

    @pyqtSlot()
    def _on_stop(self):
        if self._thread is not None:
            self._thread.request_stop()
            self._btn_stop.setEnabled(False)
            # Result (or error) arrives via acquisition_complete / error signals,
            # which call _reset_acq_controls — no blocking wait here.

    @pyqtSlot(object)
    def _on_curve_extracted(self, curve: np.ndarray):
        self._show_snake_overlay(curve, color=(0, 0, 255))

    @pyqtSlot(int, int, object, float)
    def _on_iteration_update(self, frame: int, iteration: int,
                             snake: np.ndarray, improvement: float):
        total_frames = self._sp_frames.value()
        converged    = improvement < self._sp_threshold.value()
        if converged:
            status = (f"Frame {frame + 1} / {total_frames}"
                      f"  ·  converged @ iter {iteration + 1}")
        else:
            status = (f"Frame {frame + 1} / {total_frames}"
                      f"  ·  iter {iteration + 1}  Δ={improvement:.5f}")
        self._lbl_acq_status.setText(status)
        # Orange in BGR while iterating; overwritten with red when frame completes
        self._show_snake_overlay(snake, color=(0, 140, 255))

    @pyqtSlot(int, int)
    def _on_progress(self, current: int, total: int):
        self._progress.setValue(current)

    @pyqtSlot(object)
    def _on_acquisition_complete(self, result: CurveResult):
        self._store_result(result)
        self._reset_acq_controls()

    @pyqtSlot(str)
    def _on_acq_error(self, msg: str):
        self._lbl_acq_status.setText(f"Error: {msg}")
        self._reset_acq_controls()

    @pyqtSlot()
    def _on_clear_result(self):
        """Discard the acquisition result; show the initial snake so parameters can be tweaked and re-run."""
        self.result = None
        self._lbl_result.setText("Result cleared.")
        self._lbl_result.setStyleSheet("color: #888; font-size: 10px;")
        self._btn_export_csv.setEnabled(False)
        self._btn_export_npy.setEnabled(False)
        self._btn_clear_result.setEnabled(False)
        self._btn_accept.setEnabled(False)
        self._lbl_acq_status.setText("Set up polygon, then press Start.")
        if self._initial_snake is not None:
            self._show_snake_overlay(self._initial_snake, color=(0, 212, 255))
        self._update_button_states()

    # ------------------------------------------------------------------
    # Slots — dialog
    # ------------------------------------------------------------------

    @pyqtSlot(bool)
    def _on_settings_toggled(self, checked: bool):
        self._config_panel.setVisible(checked)
        self._camera.set_editing_mode(checked)
        arrow = "▲" if checked else "▼"
        self._btn_settings.setText(f"⚙  Camera Settings  {arrow}")

    @pyqtSlot()
    def _on_open_homography_dialog(self):
        current_H = self._config_panel.get_config().homography
        initial_frame = self._camera.get_last_frame()

        def _apply_H(H):
            cfg = self._config_panel.get_config()
            cfg.homography = H
            self._config_panel.set_config(cfg)
            self._camera.set_config(cfg)
            self._lbl_h_status.setText("H ✓")
            self._lbl_h_status.setStyleSheet("color:#6EBA31; font-size:10px;")

        dlg = HomographyCalibrationDialog(
            on_apply=_apply_H,
            initial_frame=initial_frame,
            current_H=current_H,
            parent=self,
        )
        dlg.exec()

    def _on_accept(self):
        if self.result is None:
            QMessageBox.warning(self, "No result", "Run an acquisition first.")
            return
        self.accept()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _show_snake_overlay(self, curve: np.ndarray, color: tuple):
        """Draw curve as a polyline on the frozen frame and freeze the preview."""
        if self._frozen_bgr is None:
            return
        annotated = self._frozen_bgr.copy()
        pts = curve.astype(int)
        for i in range(len(pts) - 1):
            x1, y1 = pts[i, 0], pts[i, 1]
            x2, y2 = pts[i + 1, 0], pts[i + 1, 1]
            cv.line(annotated, (x1, y1), (x2, y2), color, 2)
        self._camera.freeze_preview(annotated)

    def _store_result(self, result: CurveResult):
        self.result = result
        f, n = result.curves.shape[0], result.curves.shape[1]
        mean_std = result.std_per_point.mean()
        if self._px_per_mm is not None:
            mean_std_mm = mean_std / self._px_per_mm
            spread_str = f"{mean_std:.2f} px  ({mean_std_mm:.3f} mm)"
        else:
            spread_str = f"{mean_std:.2f} px"
        self._lbl_result.setText(
            f"{f} frames  ·  {n} points  ·  mean spread: {spread_str}"
        )
        self._lbl_result.setStyleSheet("color: #6EBA31; font-size: 10px;")
        self._show_snake_overlay(result.mean_curve, color=(0, 0, 255))
        self._btn_export_csv.setEnabled(True)
        self._btn_export_npy.setEnabled(True)
        self._btn_clear_result.setEnabled(True)
        self._btn_accept.setEnabled(True)
        self._update_button_states()

    def _reset_acq_controls(self):
        if self._thread is not None:
            try:
                self._camera.frame_ready.disconnect(self._thread.on_frame)
            except RuntimeError:
                pass
            self._thread = None
        self._btn_start.setEnabled(self._initial_snake is not None and self._camera.is_connected)
        self._btn_stop.setEnabled(False)

    def _update_button_states(self):
        frozen    = self._frozen_bgr is not None
        has_snake = self._initial_snake is not None
        connected = self._camera.is_connected
        running   = self._thread is not None and self._thread.isRunning()

        self._btn_freeze.setEnabled(connected and not frozen and not running)
        self._btn_start.setEnabled(has_snake and connected and not running)
        self._btn_accept.setEnabled(self.result is not None)
        self._btn_set_scale.setEnabled(frozen and not running)

    def _export(self, fmt: str):
        if self.result is None:
            return
        if fmt == 'csv':
            path, _ = QFileDialog.getSaveFileName(
                self, "Export mean curve as CSV",
                str(Path.home() / 'mean_curve.csv'),
                "CSV files (*.csv)",
            )
            if not path:
                return
            import csv
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                has_scale = self._px_per_mm is not None
                if has_scale:
                    writer.writerow(['x_px', 'y_px', 'x_mm', 'y_mm'])
                    for x, y in self.result.mean_curve.tolist():
                        writer.writerow([x, y,
                                         x / self._px_per_mm,
                                         y / self._px_per_mm])
                else:
                    writer.writerow(['x', 'y'])
                    writer.writerows(self.result.mean_curve.tolist())
        else:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export all curves as NPY",
                str(Path.home() / 'curves.npy'),
                "NumPy files (*.npy)",
            )
            if not path:
                return
            np.save(path, self.result.curves)

    # ------------------------------------------------------------------
    # Style
    # ------------------------------------------------------------------

    def _apply_style(self):
        self.setStyleSheet("""
            QDialog, QWidget {
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
            QSpinBox, QDoubleSpinBox, QLineEdit {
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
            QLabel { color: #c4e49a; }
            QScrollArea { border: none; }
            QSplitter::handle { background: #447130; width: 2px; }
        """)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if self._thread is not None and self._thread.isRunning():
            self._thread.request_stop()
            self._thread.wait(2000)
        self._camera._stop_thread()
        super().closeEvent(event)
