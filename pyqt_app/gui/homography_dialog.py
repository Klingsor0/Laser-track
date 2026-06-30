# -*- coding: utf-8 -*-
"""
Homography Calibration Dialog.

Frame sources (in priority order)
----------------------------------
1. Grabbed from the main camera (passed as initial_frame) -- no extra connection needed.
2. Secondary camera connected inside the dialog.
3. Image loaded from file.

Detection paths
---------------
Auto   : findChessboardCorners + RANSAC findHomography.
Manual : guided 4-corner click (TL -> TR -> BL -> BR).

Buttons
-------
Apply  -- sends H to the caller callback immediately; dialog stays open
          so the effect can be verified in the main window preview.
Accept -- Apply + close.
Cancel -- close without applying.
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
    QMessageBox, QFileDialog,
)
from PyQt6.QtCore import Qt, pyqtSlot

from gui.camera_widget import CameraWidget
from gui.config_panel import ConfigPanel
from backend.homography_calibration import (
    BoardConfig,
    detect_chessboard, draw_detected_corners,
    compute_homography_auto, compute_homography_manual,
    apply_homography, draw_manual_corners,
)


_CORNER_LABELS = ["top-left", "top-right", "bottom-left", "bottom-right"]


class HomographyCalibrationDialog(QDialog):
    """
    Parameters
    ----------
    on_apply : callable(np.ndarray)
        Called each time the user presses Apply or Accept.
        Typically: sets CameraConfig.homography and emits config_changed.
    initial_frame : np.ndarray (BGR) or None
        Frame grabbed from the main camera before opening the dialog.
        If provided the left pane shows it immediately -- no camera
        connection required for the primary use case.
    current_H : np.ndarray (3x3) or None
        Existing homography (shown in the result group on open).
    parent : QWidget or None
    """

    def __init__(self, on_apply, initial_frame=None,
                 current_H=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Homography Calibration — Chessboard")
        self.resize(1080, 680)

        self._on_apply_cb       = on_apply
        self._current_H         = current_H
        self._frozen_bgr: np.ndarray | None = None
        self._detected_corners  = None
        self._manual_pts: list[tuple[int, int]] = []
        self._H: np.ndarray | None = current_H

        self._build_ui()
        self._apply_style()

        # Pre-populate with the main-camera frame if supplied
        if initial_frame is not None:
            self._load_frame(initial_frame)

        self._update_states()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left: image preview ───────────────────────────────────────
        # Either shows the grabbed main-camera frame (static) or the
        # secondary camera live feed.
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(0)

        self._camera = CameraWidget()
        self._camera.setMinimumWidth(400)
        ll.addWidget(self._camera, stretch=1)

        # Secondary camera settings (collapsible; only needed when using cam 2)
        self._btn_cam2 = QPushButton("Secondary camera settings  ▼")
        self._btn_cam2.setCheckable(True)
        self._btn_cam2.toggled.connect(self._on_cam2_toggled)
        ll.addWidget(self._btn_cam2)

        self._cfg_panel = ConfigPanel()
        self._cfg_panel.setVisible(False)
        ll.addWidget(self._cfg_panel)
        self._cfg_panel.config_changed.connect(self._camera.set_config)
        self._camera._preview.roi_drawn.connect(self._cfg_panel.set_roi)

        splitter.addWidget(left)

        # ── Right: controls ───────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(300)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(8, 8, 8, 8)
        rl.setSpacing(8)

        rl.addWidget(self._make_board_group())
        rl.addWidget(self._make_frame_group())
        rl.addWidget(self._make_auto_group())
        rl.addWidget(self._make_manual_group())
        rl.addWidget(self._make_result_group())
        rl.addStretch()
        rl.addLayout(self._make_button_row())

        scroll.setWidget(right)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter)

    # ── Group builders ────────────────────────────────────────────────

    def _make_board_group(self) -> QGroupBox:
        grp = QGroupBox("Chessboard Configuration")
        lay = QVBoxLayout(grp)

        def _row(label, widget):
            r = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setFixedWidth(110)
            r.addWidget(lbl)
            r.addWidget(widget)
            lay.addLayout(r)

        self._sp_cols = QSpinBox()
        self._sp_cols.setRange(3, 20)
        self._sp_cols.setValue(9)
        self._sp_cols.setToolTip("Inner corner count horizontally")
        _row("Inner cols:", self._sp_cols)

        self._sp_rows = QSpinBox()
        self._sp_rows.setRange(3, 20)
        self._sp_rows.setValue(6)
        self._sp_rows.setToolTip("Inner corner count vertically")
        _row("Inner rows:", self._sp_rows)

        self._sp_sq = QDoubleSpinBox()
        self._sp_sq.setRange(1.0, 200.0)
        self._sp_sq.setValue(25.0)
        self._sp_sq.setSuffix(" mm")
        self._sp_sq.setDecimals(1)
        _row("Square size:", self._sp_sq)

        return grp

    def _make_frame_group(self) -> QGroupBox:
        grp = QGroupBox("Calibration Frame")
        lay = QVBoxLayout(grp)

        hint = QLabel(
            "A frame from the main camera is loaded automatically.\n"
            "You can also grab from the secondary camera or load a file."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#888; font-size:10px;")
        lay.addWidget(hint)

        row1 = QHBoxLayout()
        self._btn_freeze_cam2 = QPushButton("Grab from Cam 2")
        self._btn_freeze_cam2.setEnabled(False)
        self._btn_freeze_cam2.clicked.connect(self._on_freeze_cam2)
        row1.addWidget(self._btn_freeze_cam2)

        self._btn_load_file = QPushButton("Load image…")
        self._btn_load_file.clicked.connect(self._on_load_file)
        row1.addWidget(self._btn_load_file)
        lay.addLayout(row1)

        self._lbl_frame_status = QLabel("No frame loaded.")
        self._lbl_frame_status.setStyleSheet("color:#888; font-size:10px;")
        lay.addWidget(self._lbl_frame_status)

        # Enable "Grab from Cam 2" when secondary camera connects
        self._camera.connected.connect(
            lambda _: self._btn_freeze_cam2.setEnabled(True)
        )
        self._camera.disconnected.connect(
            lambda: self._btn_freeze_cam2.setEnabled(False)
        )

        return grp

    def _make_auto_group(self) -> QGroupBox:
        grp = QGroupBox("Auto-Detection")
        lay = QVBoxLayout(grp)

        self._btn_auto = QPushButton("Auto-detect corners")
        self._btn_auto.setEnabled(False)
        self._btn_auto.clicked.connect(self._on_auto_detect)
        lay.addWidget(self._btn_auto)

        self._lbl_auto_status = QLabel("—")
        self._lbl_auto_status.setStyleSheet("color:#888; font-size:10px;")
        self._lbl_auto_status.setWordWrap(True)
        lay.addWidget(self._lbl_auto_status)

        return grp

    def _make_manual_group(self) -> QGroupBox:
        grp = QGroupBox("Manual Fallback — Pick 4 Corners")
        lay = QVBoxLayout(grp)

        hint = QLabel(
            "Click the 4 outermost inner-corners in order:\n"
            "top-left  →  top-right  →  bottom-left  →  bottom-right"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#888; font-size:10px;")
        lay.addWidget(hint)

        row = QHBoxLayout()
        self._btn_manual = QPushButton("Pick corners")
        self._btn_manual.setEnabled(False)
        self._btn_manual.clicked.connect(self._on_start_manual)
        row.addWidget(self._btn_manual)

        self._btn_clear_manual = QPushButton("Clear")
        self._btn_clear_manual.setEnabled(False)
        self._btn_clear_manual.clicked.connect(self._on_clear_manual)
        row.addWidget(self._btn_clear_manual)
        lay.addLayout(row)

        self._lbl_manual_status = QLabel("—")
        self._lbl_manual_status.setStyleSheet("color:#888; font-size:10px;")
        lay.addWidget(self._lbl_manual_status)

        self._camera._preview.polygon_point_added.connect(self._on_corner_clicked)

        return grp

    def _make_result_group(self) -> QGroupBox:
        grp = QGroupBox("Homography")
        lay = QVBoxLayout(grp)

        self._btn_compute = QPushButton("Compute H  →  Preview rectification")
        self._btn_compute.setEnabled(False)
        self._btn_compute.clicked.connect(self._on_compute)
        lay.addWidget(self._btn_compute)

        self._lbl_H_status = QLabel(
            "H active (from previous session)."
            if self._current_H is not None
            else "No homography computed."
        )
        color = "#6EBA31" if self._current_H is not None else "#888"
        self._lbl_H_status.setStyleSheet(f"color:{color}; font-size:10px;")
        self._lbl_H_status.setWordWrap(True)
        lay.addWidget(self._lbl_H_status)

        return grp

    def _make_button_row(self) -> QHBoxLayout:
        row = QHBoxLayout()

        self._btn_apply = QPushButton("Apply H")
        self._btn_apply.setEnabled(self._current_H is not None)
        self._btn_apply.setToolTip("Apply homography to main camera; keep dialog open")
        self._btn_apply.clicked.connect(self._on_apply)
        row.addWidget(self._btn_apply)

        self._btn_accept = QPushButton("Accept")
        self._btn_accept.setEnabled(self._current_H is not None)
        self._btn_accept.setToolTip("Apply homography and close dialog")
        self._btn_accept.clicked.connect(self._on_accept)
        row.addWidget(self._btn_accept)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        row.addWidget(btn_cancel)

        return row

    # ------------------------------------------------------------------
    # Frame loading
    # ------------------------------------------------------------------

    def _load_frame(self, bgr: np.ndarray):
        """Store bgr as the frozen calibration frame and show it."""
        self._frozen_bgr = bgr.copy()
        self._detected_corners = None
        self._manual_pts = []
        self._camera.freeze_preview(self._frozen_bgr)
        h, w = bgr.shape[:2]
        self._lbl_frame_status.setText(f"Frame loaded: {w}x{h}")
        self._lbl_frame_status.setStyleSheet("color:#6EBA31; font-size:10px;")
        self._update_states()

    @pyqtSlot()
    def _on_freeze_cam2(self):
        frame = self._camera.get_last_frame()
        if frame is None:
            QMessageBox.warning(self, "No frame",
                                "Secondary camera has not delivered a frame yet.")
            return
        self._load_frame(frame)

    @pyqtSlot()
    def _on_load_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load chessboard image", str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not path:
            return
        bgr = cv.imread(path)
        if bgr is None:
            QMessageBox.warning(self, "Load failed", f"Could not read {path}")
            return
        self._load_frame(bgr)

    # ------------------------------------------------------------------
    # Auto detection
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_auto_detect(self):
        if self._frozen_bgr is None:
            return
        board = self._board_config()
        self._lbl_auto_status.setText("Detecting…")
        self._lbl_auto_status.setStyleSheet("color:#888; font-size:10px;")
        corners, _ = detect_chessboard(self._frozen_bgr, board)
        if corners is None:
            self._lbl_auto_status.setText(
                f"Not found ({board.inner_cols}x{board.inner_rows}). "
                "Check board config and lighting, or use manual fallback."
            )
            self._lbl_auto_status.setStyleSheet("color:#cc6666; font-size:10px;")
            return
        self._detected_corners = corners
        annotated = draw_detected_corners(self._frozen_bgr, corners, board)
        self._camera.freeze_preview(annotated)
        n = board.inner_cols * board.inner_rows
        self._lbl_auto_status.setText(
            f"Found {n} corners ({board.inner_cols}x{board.inner_rows}). "
            "Press Compute H."
        )
        self._lbl_auto_status.setStyleSheet("color:#6EBA31; font-size:10px;")
        self._update_states()

    # ------------------------------------------------------------------
    # Manual corner picking
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_start_manual(self):
        self._detected_corners = None
        self._manual_pts = []
        self._camera._preview.enable_polygon_mode()
        self._btn_clear_manual.setEnabled(True)
        self._lbl_manual_status.setText(
            f"Click 1/4: {_CORNER_LABELS[0]} inner corner"
        )
        self._lbl_manual_status.setStyleSheet("color:#FFB800; font-size:10px;")
        self._update_states()

    @pyqtSlot(int)
    def _on_corner_clicked(self, count: int):
        if self._frozen_bgr is None or count > 4:
            return
        pts = self._camera._preview.get_polygon_pts()
        self._manual_pts = list(pts[:4])
        annotated = draw_manual_corners(self._frozen_bgr, self._manual_pts)
        self._camera.freeze_preview(annotated)
        if count < 4:
            self._lbl_manual_status.setText(
                f"Click {count + 1}/4: {_CORNER_LABELS[count]} inner corner"
            )
        else:
            self._camera._preview.disable_polygon_mode()
            self._lbl_manual_status.setText(
                "4 corners picked. Press Compute H."
            )
            self._lbl_manual_status.setStyleSheet("color:#6EBA31; font-size:10px;")
        self._update_states()

    @pyqtSlot()
    def _on_clear_manual(self):
        self._manual_pts = []
        self._camera._preview.disable_polygon_mode()
        if self._frozen_bgr is not None:
            self._camera.freeze_preview(self._frozen_bgr)
        self._lbl_manual_status.setText("Cleared.")
        self._lbl_manual_status.setStyleSheet("color:#888; font-size:10px;")
        self._update_states()

    # ------------------------------------------------------------------
    # Homography computation
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_compute(self):
        if self._frozen_bgr is None:
            return
        board = self._board_config()
        h_img, w_img = self._frozen_bgr.shape[:2]
        try:
            if self._detected_corners is not None:
                H = compute_homography_auto(
                    self._detected_corners, board, (w_img, h_img)
                )
                method = "auto (RANSAC)"
            elif len(self._manual_pts) == 4:
                H = compute_homography_manual(
                    self._manual_pts, board, (w_img, h_img)
                )
                method = "manual (4 corners)"
            else:
                return
            self._H = H
            self._camera.freeze_preview(apply_homography(self._frozen_bgr, H))
            det = np.linalg.det(H)
            self._lbl_H_status.setText(
                f"H computed ({method})  ·  det={det:.4f}\n"
                "Preview shows rectified frame. Press Apply or Accept."
            )
            self._lbl_H_status.setStyleSheet("color:#6EBA31; font-size:10px;")
        except Exception as e:
            self._lbl_H_status.setText(f"Error: {e}")
            self._lbl_H_status.setStyleSheet("color:#cc6666; font-size:10px;")
        self._update_states()

    # ------------------------------------------------------------------
    # Apply / Accept / Cancel
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_apply(self):
        if self._H is None:
            QMessageBox.warning(self, "No homography", "Compute H first.")
            return
        self._on_apply_cb(self._H.copy())
        self._lbl_H_status.setText(
            self._lbl_H_status.text().rstrip() + "\nApplied to main camera."
        )

    @pyqtSlot()
    def _on_accept(self):
        if self._H is None:
            QMessageBox.warning(self, "No homography", "Compute H first.")
            return
        self._on_apply_cb(self._H.copy())
        self.accept()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @pyqtSlot(bool)
    def _on_cam2_toggled(self, checked: bool):
        self._cfg_panel.setVisible(checked)
        self._camera.set_editing_mode(checked)
        arrow = "▲" if checked else "▼"
        self._btn_cam2.setText(f"Secondary camera settings  {arrow}")

    def _board_config(self) -> BoardConfig:
        return BoardConfig(
            inner_cols=self._sp_cols.value(),
            inner_rows=self._sp_rows.value(),
            square_mm=self._sp_sq.value(),
        )

    def _update_states(self):
        frozen     = self._frozen_bgr is not None
        has_auto   = self._detected_corners is not None
        has_manual = len(self._manual_pts) == 4
        has_H      = self._H is not None

        self._btn_auto.setEnabled(frozen)
        self._btn_manual.setEnabled(frozen)
        self._btn_clear_manual.setEnabled(frozen and len(self._manual_pts) > 0)
        self._btn_compute.setEnabled(frozen and (has_auto or has_manual))
        self._btn_apply.setEnabled(has_H)
        self._btn_accept.setEnabled(has_H)

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
            QSpinBox, QDoubleSpinBox {
                background: #1a2a1a;
                color: #c4e49a;
                border: 1px solid #447130;
                border-radius: 3px;
                padding: 2px 6px;
            }
            QScrollArea { border: none; }
            QSplitter::handle { background: #447130; width: 2px; }
            QLabel { color: #c4e49a; }
        """)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        self._camera._preview.disable_polygon_mode()
        self._camera._stop_thread()
        super().closeEvent(event)
