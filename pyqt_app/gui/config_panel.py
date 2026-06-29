"""
Camera Config Panel — collapsible settings widget.

Groups:
  Orientation  — rotation spinbox + 90° CW/CCW quick buttons
  ROI          — x/y/w/h spinboxes, synced with preview rectangle drawing
  Filters      — Gaussian blur kernel, contrast (α), brightness (β)

Action row: Reset | Load… | Save… | Apply

Signals:
  config_changed(CameraConfig) — emitted live on every control change

Public slots / methods:
  set_roi(x, y, w, h)   — called when user draws on the camera preview
  set_config(config)    — programmatic full update (e.g. after JSON load)
  get_config()          — returns a copy of the current config
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))   # pyqt_app/

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QPushButton, QDoubleSpinBox, QSpinBox,
    QSlider, QFileDialog, QMessageBox,
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QLocale

from backend.camera_config import CameraConfig

_DEFAULT_CFG_PATH = Path.home() / '.laser_track_camera.json'


class ConfigPanel(QWidget):

    config_changed = pyqtSignal(object)   # carries a CameraConfig copy

    def __init__(self, parent=None):
        super().__init__(parent)
        self._config       = CameraConfig()
        self._saved_config = CameraConfig()   # snapshot for Reset
        self._blocking     = False            # guard against signal feedback loops
        self._build_ui()

    # ── UI construction ────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 8)
        root.setSpacing(6)

        root.addWidget(self._make_orientation_group())
        root.addWidget(self._make_roi_group())
        root.addWidget(self._make_filters_group())
        root.addLayout(self._make_action_row())

    def _make_orientation_group(self) -> QGroupBox:
        grp = QGroupBox("Orientation")
        lay = QVBoxLayout(grp)

        # ── Rotation angle ────────────────────────────────────────────
        rot_row = QHBoxLayout()
        rot_row.addWidget(QLabel("Rotation:"))
        self._rotation = QDoubleSpinBox()
        self._rotation.setRange(-180.0, 180.0)
        self._rotation.setSingleStep(0.5)
        self._rotation.setDecimals(1)
        self._rotation.setSuffix("°")
        self._rotation.setValue(0.0)
        self._rotation.valueChanged.connect(self._on_rotation_changed)
        rot_row.addWidget(self._rotation)

        rot_row.addSpacing(8)
        btn_ccw = QPushButton("↺ 90°")
        btn_ccw.setFixedWidth(64)
        btn_ccw.setToolTip("Rotate 90° counter-clockwise")
        btn_ccw.clicked.connect(lambda: self._rotate_step(-90))
        rot_row.addWidget(btn_ccw)

        btn_cw = QPushButton("↻ 90°")
        btn_cw.setFixedWidth(64)
        btn_cw.setToolTip("Rotate 90° clockwise")
        btn_cw.clicked.connect(lambda: self._rotate_step(+90))
        rot_row.addWidget(btn_cw)
        rot_row.addStretch()
        lay.addLayout(rot_row)

        # ── Axis pivot X (horizontal position in pixels) ──────────────
        ax_row = QHBoxLayout()
        ax_row.addWidget(QLabel("Axis X:"))
        self._axis_x_slider = QSlider(Qt.Orientation.Horizontal)
        self._axis_x_slider.setRange(0, 4096)
        self._axis_x_slider.setValue(320)
        self._axis_x_slider.setToolTip("Horizontal position of the vertical guideline (pixels)")
        self._axis_x_slider.valueChanged.connect(self._on_axis_x_changed)
        ax_row.addWidget(self._axis_x_slider)
        self._axis_x_spin = QSpinBox()
        self._axis_x_spin.setRange(0, 4096)
        self._axis_x_spin.setValue(320)
        self._axis_x_spin.setSuffix(" px")
        self._axis_x_spin.setFixedWidth(72)
        self._axis_x_spin.valueChanged.connect(self._on_axis_x_spin_changed)
        ax_row.addWidget(self._axis_x_spin)
        lay.addLayout(ax_row)

        # ── Axis pivot Y (height of pivot point in pixels) ────────────
        ay_row = QHBoxLayout()
        ay_row.addWidget(QLabel("Axis Y:"))
        self._axis_y_slider = QSlider(Qt.Orientation.Horizontal)
        self._axis_y_slider.setRange(0, 4096)
        self._axis_y_slider.setValue(240)
        self._axis_y_slider.setToolTip("Vertical position of the pivot point (pixels)")
        self._axis_y_slider.valueChanged.connect(self._on_axis_y_changed)
        ay_row.addWidget(self._axis_y_slider)
        self._axis_y_spin = QSpinBox()
        self._axis_y_spin.setRange(0, 4096)
        self._axis_y_spin.setValue(240)
        self._axis_y_spin.setSuffix(" px")
        self._axis_y_spin.setFixedWidth(72)
        self._axis_y_spin.valueChanged.connect(self._on_axis_y_spin_changed)
        ay_row.addWidget(self._axis_y_spin)
        lay.addLayout(ay_row)

        return grp

    def _make_roi_group(self) -> QGroupBox:
        grp = QGroupBox("ROI")
        lay = QVBoxLayout(grp)

        hint = QLabel("Draw on preview  —or—  set values below:")
        hint.setStyleSheet("color: #888; font-size: 10px;")
        lay.addWidget(hint)

        spin_row = QHBoxLayout()
        for lbl, attr, tip in [
            ("X:", "_roi_x", "Left edge (px)"),
            ("Y:", "_roi_y", "Top edge (px)"),
            ("W:", "_roi_w", "Width (px)"),
            ("H:", "_roi_h", "Height (px)"),
        ]:
            spin_row.addWidget(QLabel(lbl))
            spin = QSpinBox()
            spin.setRange(0, 9999)
            spin.setValue(0)
            spin.setToolTip(tip)
            spin.valueChanged.connect(self._on_roi_spinbox_changed)
            setattr(self, attr, spin)
            spin_row.addWidget(spin)
        lay.addLayout(spin_row)

        clear_btn = QPushButton("Clear ROI")
        clear_btn.clicked.connect(self._on_clear_roi)
        lay.addWidget(clear_btn)
        return grp

    def _make_filters_group(self) -> QGroupBox:
        grp = QGroupBox("Filters")
        lay = QVBoxLayout(grp)

        # ── Blur ──────────────────────────────────────────────────────
        blur_row = QHBoxLayout()
        blur_row.addWidget(QLabel("Blur kernel:"))
        self._blur = QSpinBox()
        self._blur.setRange(1, 31)
        self._blur.setSingleStep(2)
        self._blur.setValue(1)
        self._blur.setToolTip("Gaussian blur kernel size — 1 = disabled, must be odd")
        self._blur.valueChanged.connect(self._on_blur_changed)
        blur_row.addWidget(self._blur)
        blur_row.addWidget(QLabel("(1 = off)"))
        blur_row.addStretch()
        lay.addLayout(blur_row)

        # ── Contrast ──────────────────────────────────────────────────
        lay.addWidget(QLabel("Contrast (α):"))
        c_row = QHBoxLayout()
        self._contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self._contrast_slider.setRange(10, 300)   # maps to 0.10 – 3.00
        self._contrast_slider.setValue(100)
        self._contrast_spin = QDoubleSpinBox()
        self._contrast_spin.setRange(0.1, 3.0)
        self._contrast_spin.setSingleStep(0.05)
        self._contrast_spin.setDecimals(2)
        self._contrast_spin.setValue(1.0)
        self._contrast_slider.valueChanged.connect(self._on_contrast_slider)
        self._contrast_spin.valueChanged.connect(self._on_contrast_spin)
        c_row.addWidget(self._contrast_slider)
        c_row.addWidget(self._contrast_spin)
        lay.addLayout(c_row)

        # ── Brightness ────────────────────────────────────────────────
        lay.addWidget(QLabel("Brightness (β):"))
        b_row = QHBoxLayout()
        self._bright_slider = QSlider(Qt.Orientation.Horizontal)
        self._bright_slider.setRange(-100, 100)
        self._bright_slider.setValue(0)
        self._bright_spin = QSpinBox()
        self._bright_spin.setRange(-100, 100)
        self._bright_spin.setValue(0)
        self._bright_slider.valueChanged.connect(self._on_bright_slider)
        self._bright_spin.valueChanged.connect(self._on_bright_spin)
        b_row.addWidget(self._bright_slider)
        b_row.addWidget(self._bright_spin)
        lay.addLayout(b_row)

        return grp

    def _make_action_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        for label, slot in [
            ("Reset",  self._on_reset),
            ("Load…",  self._on_load),
            ("Save…",  self._on_save),
            ("Apply",  self._on_apply),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            row.addWidget(btn)
        return row

    # ── Control change slots ───────────────────────────────────────────

    def _on_rotation_changed(self, value: float):
        if self._blocking:
            return
        self._config.rotation = value
        self._emit()

    def _rotate_step(self, deg: int):
        new_val = (self._config.rotation + deg + 180.0) % 360.0 - 180.0
        self._blocking = True
        self._rotation.setValue(new_val)
        self._blocking = False
        self._config.rotation = new_val
        self._emit()

    def _on_axis_x_changed(self, value: int):
        if self._blocking:
            return
        self._blocking = True
        self._axis_x_spin.setValue(value)
        self._blocking = False
        self._config.axis_x = value
        self._emit()

    def _on_axis_x_spin_changed(self, value: int):
        if self._blocking:
            return
        self._blocking = True
        self._axis_x_slider.setValue(value)
        self._blocking = False
        self._config.axis_x = value
        self._emit()

    def _on_axis_y_changed(self, value: int):
        if self._blocking:
            return
        self._blocking = True
        self._axis_y_spin.setValue(value)
        self._blocking = False
        self._config.axis_y = value
        self._emit()

    def _on_axis_y_spin_changed(self, value: int):
        if self._blocking:
            return
        self._blocking = True
        self._axis_y_slider.setValue(value)
        self._blocking = False
        self._config.axis_y = value
        self._emit()

    def _on_roi_spinbox_changed(self):
        if self._blocking:
            return
        x = self._roi_x.value()
        y = self._roi_y.value()
        w = self._roi_w.value()
        h = self._roi_h.value()
        self._config.roi = (x, y, w, h) if (w > 0 and h > 0) else None
        self._emit()

    def _on_clear_roi(self):
        self._blocking = True
        for spin in (self._roi_x, self._roi_y, self._roi_w, self._roi_h):
            spin.setValue(0)
        self._blocking = False
        self._config.roi = None
        self._emit()

    def _on_blur_changed(self, value: int):
        if value > 1 and value % 2 == 0:
            self._blur.setValue(value + 1)
            return
        self._config.blur_kernel = value
        self._emit()

    def _on_contrast_slider(self, value: int):
        if self._blocking:
            return
        v = value / 100.0
        self._blocking = True
        self._contrast_spin.setValue(v)
        self._blocking = False
        self._config.contrast = v
        self._emit()

    def _on_contrast_spin(self, value: float):
        if self._blocking:
            return
        self._blocking = True
        self._contrast_slider.setValue(int(round(value * 100)))
        self._blocking = False
        self._config.contrast = value
        self._emit()

    def _on_bright_slider(self, value: int):
        if self._blocking:
            return
        self._blocking = True
        self._bright_spin.setValue(value)
        self._blocking = False
        self._config.brightness = value
        self._emit()

    def _on_bright_spin(self, value: int):
        if self._blocking:
            return
        self._blocking = True
        self._bright_slider.setValue(value)
        self._blocking = False
        self._config.brightness = value
        self._emit()

    # ── Action slots ───────────────────────────────────────────────────

    def _on_reset(self):
        self.set_config(self._saved_config.copy())

    def _on_apply(self):
        self._saved_config = self._config.copy()
        try:
            self._config.save(_DEFAULT_CFG_PATH)
        except Exception as e:
            QMessageBox.warning(self, "Save failed", str(e))

    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load camera config", str(Path.home()), "JSON files (*.json)"
        )
        if not path:
            return
        try:
            self.set_config(CameraConfig.load(path))
        except Exception as e:
            QMessageBox.warning(self, "Load failed", str(e))

    def _on_save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save camera config",
            str(Path.home() / 'camera_config.json'),
            "JSON files (*.json)",
        )
        if not path:
            return
        try:
            self._config.save(path)
        except Exception as e:
            QMessageBox.warning(self, "Save failed", str(e))

    # ── Public API ─────────────────────────────────────────────────────

    @pyqtSlot(int, int, int, int)
    def set_roi(self, x: int, y: int, w: int, h: int):
        """Receive ROI coordinates drawn by the user on the camera preview."""
        self._blocking = True
        self._roi_x.setValue(x)
        self._roi_y.setValue(y)
        self._roi_w.setValue(w)
        self._roi_h.setValue(h)
        self._blocking = False
        self._config.roi = (x, y, w, h)
        self._emit()

    def set_config(self, config: CameraConfig):
        """Update all controls to reflect config (used for load / reset)."""
        self._config = config.copy()
        self._blocking = True
        self._rotation.setValue(config.rotation)
        self._axis_x_slider.setValue(config.axis_x)
        self._axis_x_spin.setValue(config.axis_x)
        self._axis_y_slider.setValue(config.axis_y)
        self._axis_y_spin.setValue(config.axis_y)
        if config.roi:
            x, y, w, h = config.roi
            self._roi_x.setValue(x)
            self._roi_y.setValue(y)
            self._roi_w.setValue(w)
            self._roi_h.setValue(h)
        else:
            for spin in (self._roi_x, self._roi_y, self._roi_w, self._roi_h):
                spin.setValue(0)
        self._blur.setValue(config.blur_kernel)
        self._contrast_slider.setValue(int(round(config.contrast * 100)))
        self._contrast_spin.setValue(config.contrast)
        self._bright_slider.setValue(config.brightness)
        self._bright_spin.setValue(config.brightness)
        self._blocking = False
        self._emit()

    def get_config(self) -> CameraConfig:
        return self._config.copy()

    # ── Internal ───────────────────────────────────────────────────────

    def _emit(self):
        self.config_changed.emit(self._config.copy())
