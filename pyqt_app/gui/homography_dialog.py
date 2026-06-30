# -*- coding: utf-8 -*-
"""
Homography Calibration Dialog.

Computes a perspective-rectification homography H from a chessboard
visible in the camera feed.

Two paths
---------
Auto   : findChessboardCorners (all inner corners) + RANSAC findHomography.
         Shows the detected corner overlay on the frozen frame.
Manual : user clicks 4 corner inner-corners in order TL -> TR -> BL -> BR.
         A guided status label indicates which corner to click next.

On Accept, H is stored in the caller's CameraConfig via a callback.
The live preview immediately shows the rectified feed once H is applied.
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
    QDialogButtonBox, QMessageBox, QSizePolicy,
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
    on_accept : callable(np.ndarray)
        Called with the computed H when the user accepts.
        Typically sets CameraConfig.homography and emits config_changed.
    current_H : np.ndarray or None
        The existing homography (shown in the badge on open).
    parent : QWidget or None
    """

    def __init__(self, on_accept, current_H=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Homography Calibration — Chessboard")
        self.resize(1080, 680)

        self._on_accept_cb  = on_accept
        self._current_H     = current_H
        self._frozen_bgr: np.ndarray | None = None
        self._detected_corners = None    # (N,1,2) float32 from auto path
        self._manual_pts: list[tuple[int, int]] = []
        self._H: np.ndarray | None = None

        self._build_ui()
        self._apply_style()
        self._update_states()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left: camera ──────────────────────────────────────────────
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(0)

        self._camera = CameraWidget()
        self._camera.setMinimumWidth(400)
        ll.addWidget(self._camera, stretch=1)

        self._btn_settings = QPushButton("⚙  Camera Settings  ▼")
        self._btn_settings.setCheckable(True)
        self._btn_settings.toggled.connect(self._on_settings_toggled)
        ll.addWidget(self._btn_settings)

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
        rl.addWidget(self._make_freeze_group())
        rl.addWidget(self._make_auto_group())
        rl.addWidget(self._make_manual_group())
        rl.addWidget(self._make_result_group())
        rl.addStretch()

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
        rl.addWidget(self._btn_box)

        scroll.setWidget(right)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        root.addWidget(splitter)

    def _make_board_group(self) -> QGroupBox:
        grp = QGroupBox("Chessboard Configuration")
        lay = QVBoxLayout(grp)

        def _row(label, widget):
            r = QHBoxLayout()
            l = QLabel(label)
            l.setFixedWidth(110)
            r.addWidget(l)
            r.addWidget(widget)
            lay.addLayout(r)

        self._sp_cols = QSpinBox()
        self._sp_cols.setRange(3, 20)
        self._sp_cols.setValue(9)
        self._sp_cols.setToolTip("Number of inner corners horizontally")
        _row("Inner cols:", self._sp_cols)

        self._sp_rows = QSpinBox()
        self._sp_rows.setRange(3, 20)
        self._sp_rows.setValue(6)
        self._sp_rows.setToolTip("Number of inner corners vertically")
        _row("Inner rows:", self._sp_rows)

        self._sp_sq = QDoubleSpinBox()
        self._sp_sq.setRange(1.0, 200.0)
        self._sp_sq.setValue(25.0)
        self._sp_sq.setSuffix(" mm")
        self._sp_sq.setDecimals(1)
        _row("Square size:", self._sp_sq)

        return grp

    def _make_freeze_group(self) -> QGroupBox:
        grp = QGroupBox("Reference Frame")
        lay = QVBoxLayout(grp)

        row = QHBoxLayout()
        self._btn_freeze = QPushButton("Freeze Frame")
        self._btn_freeze.setEnabled(False)
        self._btn_freeze.clicked.connect(self._on_freeze)
        row.addWidget(self._btn_freeze)

        self._btn_unfreeze = QPushButton("Unfreeze")
        self._btn_unfreeze.setEnabled(False)
        self._btn_unfreeze.clicked.connect(self._on_unfreeze)
        row.addWidget(self._btn_unfreeze)
        lay.addLayout(row)

        self._camera.connected.connect(lambda _: self._btn_freeze.setEnabled(True))
        self._camera.disconnected.connect(lambda: self._btn_freeze.setEnabled(False))

        return grp

    def _make_auto_group(self) -> QGroupBox:
        grp = QGroupBox("Auto-Detection")
        lay = QVBoxLayout(grp)

        hint = QLabel("Freeze a frame, then run auto-detect.")
        hint.setStyleSheet("color:#888; font-size:10px;")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        self._btn_auto = QPushButton("Auto-detect corners")
        self._btn_auto.setEnabled(False)
        self._btn_auto.clicked.connect(self._on_auto_detect)
        lay.addWidget(self._btn_auto)

        self._lbl_auto_status = QLabel("—")
        self._lbl_auto_status.setStyleSheet("color:#888; font-size:10px;")
        lay.addWidget(self._lbl_auto_status)

        return grp

    def _make_manual_group(self) -> QGroupBox:
        grp = QGroupBox("Manual Fallback — Pick 4 Corners")
        lay = QVBoxLayout(grp)

        hint = QLabel(
            "If auto-detect fails, click the 4 outermost inner-corners "
            "in order: top-left → top-right → bottom-left → bottom-right."
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

        # Connect the point-added signal
        self._camera._preview.polygon_point_added.connect(self._on_corner_clicked)

        return grp

    def _make_result_group(self) -> QGroupBox:
        grp = QGroupBox("Homography")
        lay = QVBoxLayout(grp)

        self._btn_compute = QPushButton("Compute H  →  Preview rectification")
        self._btn_compute.setEnabled(False)
        self._btn_compute.clicked.connect(self._on_compute)
        lay.addWidget(self._btn_compute)

        self._lbl_H_status = QLabel("No homography computed.")
        self._lbl_H_status.setWordWrap(True)
        self._lbl_H_status.setStyleSheet("color:#888; font-size:10px;")
        lay.addWidget(self._lbl_H_status)

        if self._current_H is not None:
            note = QLabel("A homography is already active — Accept will replace it.")
            note.setStyleSheet("color:#FFB800; font-size:10px;")
            note.setWordWrap(True)
            lay.addWidget(note)

        return grp

    # ------------------------------------------------------------------
    # Slots — frame
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_freeze(self):
        frame = self._camera.get_last_frame()
        if frame is None:
            QMessageBox.warning(self, "No frame", "Camera has not delivered a frame yet.")
            return
        self._frozen_bgr = frame.copy()
        self._camera.freeze_preview(self._frozen_bgr)
        self._btn_freeze.setEnabled(False)
        self._btn_unfreeze.setEnabled(True)
        self._update_states()

    @pyqtSlot()
    def _on_unfreeze(self):
        self._camera._preview.disable_polygon_mode()
        self._camera.unfreeze_preview()
        self._frozen_bgr     = None
        self._detected_corners = None
        self._manual_pts     = []
        self._H              = None
        self._btn_freeze.setEnabled(self._camera.is_connected)
        self._btn_unfreeze.setEnabled(False)
        self._lbl_auto_status.setText("—")
        self._lbl_manual_status.setText("—")
        self._lbl_H_status.setText("No homography computed.")
        self._lbl_H_status.setStyleSheet("color:#888; font-size:10px;")
        self._update_states()

    # ------------------------------------------------------------------
    # Slots — auto detection
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_auto_detect(self):
        if self._frozen_bgr is None:
            return
        board = self._board_config()
        self._lbl_auto_status.setText("Detecting…")
        corners, _ = detect_chessboard(self._frozen_bgr, board)
        if corners is None:
            self._lbl_auto_status.setText(
                f"Not found. Check board size ({board.inner_cols}x{board.inner_rows}) "
                "and lighting, or use manual fallback."
            )
            self._lbl_auto_status.setStyleSheet("color:#cc6666; font-size:10px;")
            return
        self._detected_corners = corners
        annotated = draw_detected_corners(self._frozen_bgr, corners, board)
        self._camera.freeze_preview(annotated)
        n = board.inner_cols * board.inner_rows
        self._lbl_auto_status.setText(
            f"Found {n} corners ({board.inner_cols}x{board.inner_rows}). Ready to compute H."
        )
        self._lbl_auto_status.setStyleSheet("color:#6EBA31; font-size:10px;")
        self._update_states()

    # ------------------------------------------------------------------
    # Slots — manual corner picking
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_start_manual(self):
        self._detected_corners = None   # manual overrides auto
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
        if self._frozen_bgr is None:
            return
        pts = self._camera._preview.get_polygon_pts()
        # Limit to 4 — ignore extra clicks
        if count > 4:
            return
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
                "4 corners picked. Ready to compute H."
            )
            self._lbl_manual_status.setStyleSheet("color:#6EBA31; font-size:10px;")
        self._update_states()

    @pyqtSlot()
    def _on_clear_manual(self):
        self._manual_pts = []
        self._camera._preview.disable_polygon_mode()
        if self._frozen_bgr is not None:
            self._camera.freeze_preview(self._frozen_bgr)
        self._lbl_manual_status.setText("Cleared. Press 'Pick corners' to restart.")
        self._lbl_manual_status.setStyleSheet("color:#888; font-size:10px;")
        self._update_states()

    # ------------------------------------------------------------------
    # Slots — homography
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
            rectified = apply_homography(self._frozen_bgr, H)
            self._camera.freeze_preview(rectified)
            det = np.linalg.det(H)
            self._lbl_H_status.setText(
                f"H computed ({method})  ·  det={det:.4f}\n"
                "Preview shows rectified frame."
            )
            self._lbl_H_status.setStyleSheet("color:#6EBA31; font-size:10px;")

        except Exception as e:
            self._lbl_H_status.setText(f"Error: {e}")
            self._lbl_H_status.setStyleSheet("color:#cc6666; font-size:10px;")

        self._update_states()

    # ------------------------------------------------------------------
    # Slots — settings panel
    # ------------------------------------------------------------------

    @pyqtSlot(bool)
    def _on_settings_toggled(self, checked: bool):
        self._cfg_panel.setVisible(checked)
        self._camera.set_editing_mode(checked)
        arrow = "▲" if checked else "▼"
        self._btn_settings.setText(f"⚙  Camera Settings  {arrow}")

    # ------------------------------------------------------------------
    # Accept
    # ------------------------------------------------------------------

    def _on_accept(self):
        if self._H is None:
            QMessageBox.warning(self, "No homography", "Compute H first.")
            return
        self._on_accept_cb(self._H.copy())
        self.accept()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
