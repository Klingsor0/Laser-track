# -*- coding: utf-8 -*-
"""
Frame Review Dialog -- tracker mode.

Lets the user browse every frame of a completed CaptureSession,
inspect the automatic snake extraction, and mark individual frames
as rejected.  Rejected frames are excluded from the filtered statistics
but kept in the session so the decision can be undone.

Layout
------
  Top bar  : session summary
  Left     : scrollable thumbnail strip (one row per frame)
  Right    : large frame preview with snake overlay + frame stats
  Bottom   : filtered statistics + Close button
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
    QLabel, QPushButton, QDialogButtonBox,
    QSizePolicy, QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap, QColor, QPalette, QFont, QKeySequence, QShortcut

from optical_experiment import CaptureSession


# ---------------------------------------------------------------------------
# Thumbnail strip item
# ---------------------------------------------------------------------------

class _FrameThumb(QFrame):
    """Single entry in the thumbnail strip."""

    clicked = pyqtSignal(int)

    _THUMB_W = 120
    _THUMB_H = 90

    def __init__(self, idx: int, frame_bgr: np.ndarray,
                 angle: float, rejected: bool, parent=None):
        super().__init__(parent)
        self._idx = idx
        self.setFixedHeight(self._THUMB_H + 28)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(1)

        self._img_lbl = QLabel()
        self._img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_lbl.setFixedSize(self._THUMB_W, self._THUMB_H)
        lay.addWidget(self._img_lbl)

        self._txt_lbl = QLabel()
        self._txt_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._txt_lbl.setFont(QFont("Monospace", 8))
        lay.addWidget(self._txt_lbl)

        self._set_pixmap(frame_bgr)
        self.refresh(angle, rejected)

    def _set_pixmap(self, bgr: np.ndarray):
        rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if w <= 0 or h <= 0:
            return
        scale = min(self._THUMB_W / w, self._THUMB_H / h, 1.0)
        tw = max(1, int(w * scale))
        th = max(1, int(h * scale))
        rgb = cv.resize(rgb, (tw, th), interpolation=cv.INTER_AREA)
        qi = QImage(rgb.data, tw, th, tw * 3, QImage.Format.Format_RGB888)
        self._img_lbl.setPixmap(QPixmap.fromImage(qi))

    def refresh(self, angle: float, rejected: bool):
        status = "✗ rejected" if rejected else f"{angle:+.2f}°"
        self._txt_lbl.setText(f"#{self._idx + 1}  {status}")
        if rejected:
            self.setStyleSheet(
                "background:#2a0d0d; border: 1px solid #663333;"
            )
            self._txt_lbl.setStyleSheet("color: #cc6666;")
        else:
            self.setStyleSheet(
                "background:#0d1117; border: 1px solid #447130;"
            )
            self._txt_lbl.setStyleSheet("color: #6EBA31;")

    def set_selected(self, selected: bool):
        border = "#FFB800" if selected else ("#663333" if "2a0d0d" in self.styleSheet() else "#447130")
        bg     = "#1a1a00" if selected else ("2a0d0d" if "2a0d0d" in self.styleSheet() else "#0d1117")
        self.setStyleSheet(f"background:{bg}; border: 2px solid {border};")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._idx)
        super().mousePressEvent(event)


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class FrameReviewDialog(QDialog):
    """
    Parameters
    ----------
    session : CaptureSession
        The session to review.  Modifications to session.rejected are
        made in-place and persist after the dialog closes.
    parent : QWidget or None
    """

    _PREVIEW_MAX_W = 560
    _PREVIEW_MAX_H = 420

    def __init__(self, session: CaptureSession, parent=None):
        super().__init__(parent)
        self._session  = session
        self._current  = 0          # currently displayed frame index
        self._thumbs: list[_FrameThumb] = []

        # Ensure rejected list is in sync (handles sessions created before
        # the rejected field was added)
        if len(session.rejected) != session.num_frames:
            session.rejected = [False] * session.num_frames

        n = session.num_frames
        title = (
            f"Frame Review  —  Session {session.session_id}  "
            f"·  {session.distance} {session.unit}  ·  {n} frames"
        )
        self.setWindowTitle(title)
        self.resize(1050, 680)

        self._build_ui()
        self._apply_style()
        self._populate_thumbnails()
        self._show_frame(0)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Session summary bar ───────────────────────────────────────
        stats = self._session.get_statistics()
        if stats:
            summary = (
                f"Session {self._session.session_id}  |  "
                f"Distance: {self._session.distance} {self._session.unit}  |  "
                f"Frames: {self._session.num_frames}  |  "
                f"Mean: {stats['mean']:+.3f}°  |  "
                f"Std: {stats['std']:.3f}°"
            )
        else:
            summary = f"Session {self._session.session_id} — no statistics"

        top_lbl = QLabel(summary)
        top_lbl.setStyleSheet(
            "background:#1a2a1a; color:#6EBA31; padding:5px; "
            "border-radius:4px; font-size:11px;"
        )
        root.addWidget(top_lbl)

        # ── Main splitter ─────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: thumbnail strip
        self._thumb_scroll = QScrollArea()
        self._thumb_scroll.setWidgetResizable(True)
        self._thumb_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._thumb_scroll.setFixedWidth(148)

        thumb_container = QWidget()
        self._thumb_lay = QVBoxLayout(thumb_container)
        self._thumb_lay.setContentsMargins(2, 2, 2, 2)
        self._thumb_lay.setSpacing(4)
        self._thumb_lay.addStretch()
        self._thumb_scroll.setWidget(thumb_container)
        splitter.addWidget(self._thumb_scroll)

        # Right: detail view
        detail = QWidget()
        detail_lay = QVBoxLayout(detail)
        detail_lay.setContentsMargins(6, 0, 0, 0)

        # Large preview image
        self._preview_lbl = QLabel()
        self._preview_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_lbl.setMinimumSize(320, 240)
        self._preview_lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._preview_lbl.setStyleSheet("background:#111; color:#6EBA31;")
        detail_lay.addWidget(self._preview_lbl, stretch=1)

        # Frame info row
        info_row = QHBoxLayout()
        mono = QFont("Monospace", 11)

        self._lbl_index    = QLabel("Frame: —")
        self._lbl_angle    = QLabel("Angle: —")
        self._lbl_dev      = QLabel("Δmean: —")
        for lbl in (self._lbl_index, self._lbl_angle, self._lbl_dev):
            lbl.setFont(mono)
            lbl.setStyleSheet("color:#6EBA31;")
            info_row.addWidget(lbl)
        info_row.addStretch()
        detail_lay.addLayout(info_row)

        # Navigation + reject/accept row
        nav_row = QHBoxLayout()
        self._btn_prev = QPushButton("← Prev")
        self._btn_prev.setFixedWidth(80)
        self._btn_prev.clicked.connect(self._on_prev)
        nav_row.addWidget(self._btn_prev)

        self._btn_reject = QPushButton("Reject frame")
        self._btn_reject.setCheckable(True)
        self._btn_reject.setFixedWidth(140)
        self._btn_reject.clicked.connect(self._on_toggle_reject)
        nav_row.addWidget(self._btn_reject)

        self._btn_next = QPushButton("Next →")
        self._btn_next.setFixedWidth(80)
        self._btn_next.clicked.connect(self._on_next)
        nav_row.addWidget(self._btn_next)

        key_hint = QLabel("  h / l — navigate   r — reject")
        key_hint.setStyleSheet("color: #556655; font-size: 10px;")
        nav_row.addWidget(key_hint)
        nav_row.addStretch()
        detail_lay.addLayout(nav_row)

        # Filtered statistics
        stats_grp = QGroupBox("Filtered Statistics (accepted frames only)")
        stats_lay = QHBoxLayout(stats_grp)
        self._lbl_f_n    = QLabel("n: —")
        self._lbl_f_mean = QLabel("mean: —")
        self._lbl_f_std  = QLabel("std: —")
        self._lbl_f_sem  = QLabel("SEM: —")
        for lbl in (self._lbl_f_n, self._lbl_f_mean, self._lbl_f_std, self._lbl_f_sem):
            lbl.setFont(QFont("Monospace", 10))
            stats_lay.addWidget(lbl)
        stats_lay.addStretch()
        detail_lay.addWidget(stats_grp)

        splitter.addWidget(detail)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, stretch=1)

        # ── Bottom buttons ────────────────────────────────────────────
        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btn_box.rejected.connect(self.accept)
        root.addWidget(btn_box)

        # ── Keyboard shortcuts (window-level, bypass child focus) ─────
        QShortcut(QKeySequence("h"), self, activated=self._on_prev)
        QShortcut(QKeySequence("l"), self, activated=self._on_next)
        QShortcut(QKeySequence("r"), self, activated=self._on_toggle_reject)

    # ------------------------------------------------------------------
    # Populate / refresh thumbnail strip
    # ------------------------------------------------------------------

    def _populate_thumbnails(self):
        session = self._session
        for idx in range(session.num_frames):
            thumb = _FrameThumb(
                idx,
                session.frames[idx],
                session.angles[idx],
                session.rejected[idx],
            )
            thumb.clicked.connect(self._show_frame)
            # Insert before the trailing stretch
            self._thumb_lay.insertWidget(self._thumb_lay.count() - 1, thumb)
            self._thumbs.append(thumb)

    # ------------------------------------------------------------------
    # Frame display
    # ------------------------------------------------------------------

    def _show_frame(self, idx: int):
        session  = self._session
        n        = session.num_frames
        idx      = max(0, min(idx, n - 1))
        self._current = idx

        # Update thumbnail selection highlight
        for i, t in enumerate(self._thumbs):
            t.set_selected(i == idx)

        # Scroll thumbnail strip to keep selection visible
        thumb = self._thumbs[idx]
        self._thumb_scroll.ensureWidgetVisible(thumb)

        # Build annotated preview
        bgr       = session.frames[idx].copy()
        snake     = session.snake_results[idx]
        if snake is not None and len(snake) >= 2:
            pts = np.array(snake, dtype=int)
            for i in range(len(pts) - 1):
                x1, y1 = pts[i, 0],     pts[i, 1]
                x2, y2 = pts[i+1, 0],   pts[i+1, 1]
                cv.line(bgr, (x1, y1), (x2, y2), (64, 186, 49), 2)

        self._set_preview(bgr)

        # Frame info labels
        angle  = session.angles[idx]
        mean   = np.mean(session.angles)
        dev    = angle - mean
        self._lbl_index.setText(f"Frame: {idx + 1} / {n}")
        self._lbl_angle.setText(f"Angle: {angle:+.3f}°")
        self._lbl_dev.setText(f"Δmean: {dev:+.3f}°")

        # Reject button state
        rejected = session.rejected[idx]
        self._btn_reject.setChecked(rejected)
        self._btn_reject.setText("Restore frame" if rejected else "Reject frame")
        if rejected:
            self._btn_reject.setStyleSheet(
                "background:#4a1a1a; color:#cc6666; border-color:#663333;"
            )
        else:
            self._btn_reject.setStyleSheet("")

        # Navigation
        self._btn_prev.setEnabled(idx > 0)
        self._btn_next.setEnabled(idx < n - 1)

        # Filtered stats
        self._refresh_filtered_stats()

    def _set_preview(self, bgr: np.ndarray):
        rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = min(self._PREVIEW_MAX_W / w, self._PREVIEW_MAX_H / h, 1.0)
        dw, dh = int(w * scale), int(h * scale)
        if scale < 1.0:
            rgb = cv.resize(rgb, (dw, dh), interpolation=cv.INTER_AREA)
        qi = QImage(rgb.data, dw, dh, dw * 3, QImage.Format.Format_RGB888)
        self._preview_lbl.setPixmap(QPixmap.fromImage(qi))

    def _refresh_filtered_stats(self):
        stats = self._session.get_filtered_statistics()
        if stats is None:
            for lbl in (self._lbl_f_n, self._lbl_f_mean, self._lbl_f_std, self._lbl_f_sem):
                lbl.setText("—")
            return
        rej = stats['n_rejected']
        self._lbl_f_n.setText(
            f"n={stats['n_frames']}  ({rej} rejected)"
        )
        self._lbl_f_mean.setText(f"mean={stats['mean']:+.3f}°")
        self._lbl_f_std.setText(f"std={stats['std']:.3f}°")
        self._lbl_f_sem.setText(f"SEM={stats['sem']:.3f}°")

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_prev(self):
        self._show_frame(self._current - 1)

    @pyqtSlot()
    def _on_next(self):
        self._show_frame(self._current + 1)

    @pyqtSlot()
    def _on_toggle_reject(self):
        idx = self._current
        self._session.rejected[idx] = not self._session.rejected[idx]
        self._thumbs[idx].refresh(
            self._session.angles[idx], self._session.rejected[idx]
        )
        # Re-show to update button label + stats
        self._show_frame(idx)

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
                padding: 4px 10px;
                min-height: 24px;
            }
            QPushButton:hover  { background: #3a5a2a; border-color: #6EBA31; }
            QPushButton:pressed { background: #1a2a1a; }
            QPushButton:disabled { color: #556655; border-color: #334433; }
            QScrollArea { border: 1px solid #2a3a2a; }
            QSplitter::handle { background: #447130; width: 2px; }
            QLabel { color: #c4e49a; }
        """)
