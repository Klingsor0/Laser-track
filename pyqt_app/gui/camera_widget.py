"""
Camera Widget — PyQt6 live preview.

Wraps CameraManager in a QThread and provides:
  - Live preview (PreviewLabel, supports ROI drawing via mouse)
  - Connect / Disconnect controls (USB or ESP32-CAM)
  - frame_ready  signal → Workflow B (acquisition loop consumes every frame)
  - frame_captured signal → Workflow A (single-frame profile analysis)

The camera preprocessing pipeline (rotation, ROI, blur, contrast/brightness)
is applied to every frame via CameraConfig.  set_config() updates it live.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'shared_utils'))
sys.path.insert(0, str(Path(__file__).parent.parent))      # pyqt_app/

import cv2 as cv
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QLineEdit, QSpinBox, QComboBox, QGroupBox, QSizePolicy,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QPoint
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor

from backend.camera_manager import CameraManager, CameraError, SourceType
from backend.camera_config import CameraConfig, apply_preview_pipeline, apply_acquisition_pipeline


# ---------------------------------------------------------------------------
# Preview label — supports ROI rectangle drawing via mouse drag
# ---------------------------------------------------------------------------

class PreviewLabel(QLabel):
    """
    QLabel subclass that lets the user draw a ROI rectangle by click-dragging.

    Signals
    -------
    roi_drawn(x, y, w, h)
        Emitted when the user finishes drawing a rectangle (mouse release).
        Coordinates are in the original frame's pixel space.
    """

    roi_drawn = pyqtSignal(int, int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._drawing         = False
        self._drawing_enabled = False          # only True while config panel is open
        self._drag_start: QPoint | None = None
        self._drag_end:   QPoint | None = None

        # Set by _update_preview() after every rendered frame
        self._scale    = 1.0
        self._offset_x = 0
        self._offset_y = 0
        self._frame_w  = 1
        self._frame_h  = 1

        # Current ROI in frame coords (for persistent overlay after drawing)
        self._roi: tuple | None = None

        # Rotation axis guideline (visible only in editing mode)
        self._axis_visible = False
        self._axis_x       = 0.5   # fraction of frame width
        self._axis_y       = 0.5   # fraction of frame height
        self._axis_angle   = 0.0   # degrees

        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.setMouseTracking(True)

    # ── Frame info ─────────────────────────────────────────────────────

    def update_frame_info(self, scale: float, offset_x: int, offset_y: int,
                          frame_w: int, frame_h: int) -> None:
        self._scale    = scale
        self._offset_x = offset_x
        self._offset_y = offset_y
        self._frame_w  = frame_w
        self._frame_h  = frame_h

    def set_roi_overlay(self, roi: tuple | None) -> None:
        self._roi = roi
        self.update()

    def set_axis(self, x_frac: float, y_frac: float, angle_deg: float) -> None:
        self._axis_x     = x_frac
        self._axis_y     = y_frac
        self._axis_angle = angle_deg
        self.update()

    def set_axis_visible(self, visible: bool) -> None:
        self._axis_visible = visible
        self.update()

    def set_drawing_enabled(self, enabled: bool) -> None:
        self._drawing_enabled = enabled
        self.setCursor(
            Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor
        )
        if not enabled:
            self._drawing    = False
            self._drag_start = None
            self._drag_end   = None
            self.update()

    # ── Coordinate conversion ──────────────────────────────────────────

    def _to_frame(self, px: int, py: int) -> tuple[int, int]:
        if self._scale <= 0:
            return (0, 0)
        fx = int((px - self._offset_x) / self._scale)
        fy = int((py - self._offset_y) / self._scale)
        return (
            max(0, min(self._frame_w - 1, fx)),
            max(0, min(self._frame_h - 1, fy)),
        )

    def _to_preview(self, fx: int, fy: int) -> tuple[int, int]:
        return (
            int(fx * self._scale + self._offset_x),
            int(fy * self._scale + self._offset_y),
        )

    # ── Mouse events ───────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if not self._drawing_enabled:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._drawing    = True
            self._drag_start = event.position().toPoint()
            self._drag_end   = self._drag_start

    def mouseMoveEvent(self, event):
        if self._drawing:
            self._drag_end = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        if not self._drawing_enabled:
            return
        if event.button() != Qt.MouseButton.LeftButton or not self._drawing:
            return
        self._drawing  = False
        self._drag_end = event.position().toPoint()

        x1, y1 = self._to_frame(self._drag_start.x(), self._drag_start.y())
        x2, y2 = self._to_frame(self._drag_end.x(),   self._drag_end.y())
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        w, h = x2 - x1, y2 - y1

        self._drag_start = None
        self._drag_end   = None

        if w > 4 and h > 4:
            self._roi = (x1, y1, w, h)
            self.roi_drawn.emit(x1, y1, w, h)
        self.update()

    # ── Paint overlay ──────────────────────────────────────────────────

    def paintEvent(self, event):
        super().paintEvent(event)

        has_drag = self._drawing and self._drag_start and self._drag_end
        has_roi  = self._roi is not None
        has_axis = self._axis_visible

        if not (has_drag or has_roi or has_axis):
            return

        painter = QPainter(self)

        # Axis guideline (drawn first so ROI rectangle is on top)
        if has_axis:
            self._draw_axis_line(painter)

        # ROI rectangle
        if has_drag:
            pen = QPen(QColor("#6EBA31"), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            x = min(self._drag_start.x(), self._drag_end.x())
            y = min(self._drag_start.y(), self._drag_end.y())
            w = abs(self._drag_end.x() - self._drag_start.x())
            h = abs(self._drag_end.y() - self._drag_start.y())
            painter.drawRect(x, y, w, h)
        elif has_roi:
            rx, ry, rw, rh = self._roi
            px, py = self._to_preview(rx, ry)
            pw = int(rw * self._scale)
            ph = int(rh * self._scale)
            pen = QPen(QColor("#6EBA31"), 2, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawRect(px, py, pw, ph)

        painter.end()

    def _draw_axis_line(self, painter: QPainter) -> None:
        # Convert pivot (absolute frame pixels) to preview (label) coordinates
        px = int(self._axis_x * self._scale + self._offset_x)
        py = int(self._axis_y * self._scale + self._offset_y)

        # Vertical extent spans the full displayed frame area
        top_y = int(self._offset_y)
        bot_y = int(self._offset_y + self._frame_h * self._scale)

        # Vertical guideline
        painter.setPen(QPen(QColor("#FFB800"), 2, Qt.PenStyle.SolidLine))
        painter.drawLine(px, top_y, px, bot_y)

        # Pivot marker — filled circle at (axis_x, axis_y)
        painter.setPen(QPen(QColor("#FFB800"), 8,
                            Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawPoint(px, py)


# ---------------------------------------------------------------------------
# Background capture thread
# ---------------------------------------------------------------------------

class CaptureThread(QThread):
    """
    Reads frames from CameraManager in a background thread.

    Signals
    -------
    frame_ready(np.ndarray)   BGR frame — emitted for every successful read
    error(str)                Human-readable error message
    """

    frame_ready = pyqtSignal(np.ndarray)
    error       = pyqtSignal(str)

    def __init__(self, source: str, parent=None):
        """
        Parameters
        ----------
        source : str
            Either an integer string (USB index) or an MJPEG URL
            (e.g. "http://192.168.1.100:81/stream")
        """
        super().__init__(parent)
        self._source  = source
        self._running = False

    def run(self):
        cam = CameraManager()
        try:
            # Decide connection type from source string
            if self._source.startswith("http"):
                # Parse URL into components
                # Expected format: http://ip:port/path
                rest = self._source[len("http://"):]
                host_port, *path_parts = rest.split("/", 1)
                path = "/" + path_parts[0] if path_parts else "/stream"
                if ":" in host_port:
                    ip, port_str = host_port.rsplit(":", 1)
                    port = int(port_str)
                else:
                    ip, port = host_port, 81
                cam.connect_esp32(ip, port, path)
            else:
                cam.connect_usb(int(self._source))

            self._running = True
            while self._running:
                try:
                    frame = cam.read_frame()        # BGR
                    self.frame_ready.emit(frame)
                except CameraError as e:
                    self.error.emit(str(e))
                    break

        except CameraError as e:
            self.error.emit(str(e))
        finally:
            cam.disconnect()

    def stop(self):
        self._running = False
        self.wait()


# ---------------------------------------------------------------------------
# Camera widget
# ---------------------------------------------------------------------------

class CameraWidget(QWidget):
    """
    Self-contained camera preview + connection controls.

    External signals (connect these from the parent window)
    -------------------------------------------------------
    frame_ready(np.ndarray)
        Every BGR frame — for Workflow B acquisition controller.
    frame_captured(np.ndarray)
        Single RGB frame on "Capture" click — for Workflow A profile pipeline.
    connected(str)
        Emitted when camera comes online; carries a status description.
    disconnected()
        Emitted when camera goes offline.
    """

    frame_ready    = pyqtSignal(np.ndarray)
    frame_captured = pyqtSignal(np.ndarray)
    connected      = pyqtSignal(str)
    disconnected   = pyqtSignal()

    # Preview size cap — keeps the UI responsive on large streams
    MAX_PREVIEW_W = 640
    MAX_PREVIEW_H = 480

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread: CaptureThread | None = None
        self._last_frame: np.ndarray | None = None
        self._connected: bool = False
        self._config: CameraConfig = CameraConfig()
        self._editing: bool = False    # True while config panel is open
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        # --- Preview label (supports ROI drawing) ---
        self._preview = PreviewLabel("No signal")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumSize(320, 240)
        self._preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._preview.setStyleSheet("background:#111; color:#6EBA31; font-size:14px;")
        root.addWidget(self._preview)

        # --- Connection controls group ---
        grp = QGroupBox("Camera Connection")
        grp_layout = QVBoxLayout(grp)

        # Source selector
        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("Source:"))
        self._source_combo = QComboBox()
        self._source_combo.addItems(["USB Webcam", "ESP32-CAM (WiFi)"])
        self._source_combo.currentIndexChanged.connect(self._on_source_changed)
        src_row.addWidget(self._source_combo)
        grp_layout.addLayout(src_row)

        # USB controls
        self._usb_box = QWidget()
        usb_row = QHBoxLayout(self._usb_box)
        usb_row.setContentsMargins(0, 0, 0, 0)
        usb_row.addWidget(QLabel("Camera index:"))
        self._usb_index = QSpinBox()
        self._usb_index.setRange(0, 20)
        usb_row.addWidget(self._usb_index)
        usb_row.addStretch()
        grp_layout.addWidget(self._usb_box)

        # ESP32 controls
        self._esp_box = QWidget()
        esp_layout = QVBoxLayout(self._esp_box)
        esp_layout.setContentsMargins(0, 0, 0, 0)

        ip_row = QHBoxLayout()
        ip_row.addWidget(QLabel("IP:"))
        self._esp_ip = QLineEdit("192.168.1.100")
        ip_row.addWidget(self._esp_ip)

        port_row = QHBoxLayout()
        port_row.addWidget(QLabel("Port:"))
        self._esp_port = QSpinBox()
        self._esp_port.setRange(1, 65535)
        self._esp_port.setValue(81)
        port_row.addWidget(self._esp_port)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Path:"))
        self._esp_path = QLineEdit("/stream")
        path_row.addWidget(self._esp_path)

        esp_layout.addLayout(ip_row)
        esp_layout.addLayout(port_row)
        esp_layout.addLayout(path_row)
        grp_layout.addWidget(self._esp_box)

        # Default: show USB, hide ESP32
        self._esp_box.setVisible(False)

        # Connect / Disconnect / Capture row
        btn_row = QHBoxLayout()
        self._btn_connect    = QPushButton("Connect")
        self._btn_disconnect = QPushButton("Disconnect")
        self._btn_capture    = QPushButton("Capture Frame")
        self._btn_disconnect.setEnabled(False)
        self._btn_capture.setEnabled(False)

        self._btn_connect.clicked.connect(self._on_connect)
        self._btn_disconnect.clicked.connect(self._on_disconnect)
        self._btn_capture.clicked.connect(self._on_capture)

        btn_row.addWidget(self._btn_connect)
        btn_row.addWidget(self._btn_disconnect)
        btn_row.addWidget(self._btn_capture)
        grp_layout.addLayout(btn_row)

        # Status label
        self._status = QLabel("Disconnected")
        self._status.setStyleSheet("color: #cc4444;")
        grp_layout.addWidget(self._status)

        root.addWidget(grp)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_source_changed(self, index: int):
        self._usb_box.setVisible(index == 0)
        self._esp_box.setVisible(index == 1)

    def _on_connect(self):
        if self._source_combo.currentIndex() == 0:
            source = str(self._usb_index.value())
        else:
            ip   = self._esp_ip.text().strip()
            port = self._esp_port.value()
            path = self._esp_path.text().strip()
            source = f"http://{ip}:{port}{path}"

        self._thread = CaptureThread(source, parent=self)
        self._thread.frame_ready.connect(self._on_frame)
        self._thread.error.connect(self._on_error)
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.start()

        self._set_connected_state(True, source)

    def _on_disconnect(self):
        self._stop_thread()
        self._set_connected_state(False, "")

    def _on_capture(self):
        """Grab the last frame, apply full pipeline, emit as RGB for Workflow A."""
        if self._last_frame is not None:
            processed = apply_acquisition_pipeline(self._last_frame, self._config)
            rgb = cv.cvtColor(processed, cv.COLOR_BGR2RGB)
            self.frame_captured.emit(rgb)

    def _on_frame(self, bgr: np.ndarray):
        self._last_frame = bgr

        # Acquisition signal always gets the fully processed frame
        processed = apply_acquisition_pipeline(bgr, self._config)
        self.frame_ready.emit(processed)

        if self._editing:
            # While config panel is open: show full frame + ROI overlay for editing
            self._update_preview(apply_preview_pipeline(bgr, self._config),
                                 show_roi_overlay=True)
        else:
            # Normal operation: preview mirrors exactly what the algorithm receives
            self._update_preview(processed, show_roi_overlay=False)

    def _on_error(self, msg: str):
        self._status.setText(f"Error: {msg}")
        self._status.setStyleSheet("color: #cc4444;")
        self._set_connected_state(False, "")

    def _on_thread_finished(self):
        self._set_connected_state(False, "")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_preview(self, bgr: np.ndarray, show_roi_overlay: bool = False):
        rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
        frame_h, frame_w = rgb.shape[:2]

        scale = min(self.MAX_PREVIEW_W / frame_w, self.MAX_PREVIEW_H / frame_h, 1.0)
        disp_w = int(frame_w * scale)
        disp_h = int(frame_h * scale)
        if scale < 1.0:
            rgb = cv.resize(rgb, (disp_w, disp_h), interpolation=cv.INTER_AREA)

        qimg = QImage(rgb.data, disp_w, disp_h, disp_w * 3, QImage.Format.Format_RGB888)
        self._preview.setPixmap(QPixmap.fromImage(qimg))

        offset_x = max(0, (self._preview.width()  - disp_w) // 2)
        offset_y = max(0, (self._preview.height() - disp_h) // 2)
        self._preview.update_frame_info(scale, offset_x, offset_y, frame_w, frame_h)
        self._preview.set_roi_overlay(self._config.roi if show_roi_overlay else None)
        if show_roi_overlay:
            self._preview.set_axis(
                self._config.axis_x, self._config.axis_y, self._config.rotation
            )

    @pyqtSlot(object)
    def set_config(self, config: CameraConfig) -> None:
        """Apply a new CameraConfig; takes effect on the next incoming frame."""
        self._config = config

    def set_editing_mode(self, editing: bool) -> None:
        """
        Switch preview between editing mode (full frame + ROI overlay + axis line) and
        normal mode (cropped frame — identical to what the algorithm receives).
        """
        self._editing = editing
        self._preview.set_drawing_enabled(editing)
        self._preview.set_axis_visible(editing)
        if not editing:
            self._preview.set_roi_overlay(None)

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _set_connected_state(self, is_connected: bool, source: str):
        self._connected = is_connected
        self._btn_connect.setEnabled(not is_connected)
        self._btn_disconnect.setEnabled(is_connected)
        self._btn_capture.setEnabled(is_connected)
        self._source_combo.setEnabled(not is_connected)
        self._usb_box.setEnabled(not is_connected)
        self._esp_box.setEnabled(not is_connected)

        if is_connected:
            self._status.setText(f"Connected: {source}")
            self._status.setStyleSheet("color: #6EBA31;")
            self.connected.emit(source)
        else:
            self._status.setText("Disconnected")
            self._status.setStyleSheet("color: #cc4444;")
            self._preview.setText("No signal")
            self._preview.setPixmap(QPixmap())
            self._preview.set_roi_overlay(None)
            self.disconnected.emit()

    def _stop_thread(self):
        if self._thread is not None:
            self._thread.stop()
            self._thread = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        self._stop_thread()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Minimal standalone window for testing
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    from PyQt6.QtWidgets import QApplication, QMainWindow

    parser = argparse.ArgumentParser(description='Test CameraWidget')
    parser.add_argument('--usb', type=int, default=None,
                        help='Auto-connect to USB camera index')
    parser.add_argument('--esp32', metavar='IP',
                        help='Auto-connect to ESP32-CAM at this IP')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    win = QMainWindow()
    win.setWindowTitle('CameraWidget — test')

    widget = CameraWidget()

    # Wire signals to console for verification
    widget.frame_ready.connect(
        lambda f: print(f'frame_ready: shape={f.shape}', end='\r')
    )
    widget.frame_captured.connect(
        lambda f: print(f'\nframe_captured (RGB): shape={f.shape}')
    )
    widget.connected.connect(
        lambda s: print(f'\nconnected: {s}')
    )
    widget.disconnected.connect(
        lambda: print('\ndisconnected')
    )

    win.setCentralWidget(widget)
    win.resize(700, 600)
    win.show()

    # Auto-connect if args provided
    if args.usb is not None:
        widget._usb_index.setValue(args.usb)
        widget._source_combo.setCurrentIndex(0)
        widget._on_connect()
    elif args.esp32:
        widget._esp_ip.setText(args.esp32)
        widget._source_combo.setCurrentIndex(1)
        widget._on_connect()

    sys.exit(app.exec())
