"""
Camera Widget — PyQt6 live preview.

Wraps CameraManager in a QThread and provides:
  - Live preview (QLabel, 30 fps target)
  - Connect / Disconnect controls (USB or ESP32-CAM)
  - frame_ready  signal → Workflow B (acquisition loop consumes every frame)
  - frame_captured signal → Workflow A (single-frame profile analysis)
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
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

from backend.camera_manager import CameraManager, CameraError, SourceType


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
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        # --- Preview label ---
        self._preview = QLabel("No signal")
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
        """Grab the last frame and emit it as RGB for Workflow A."""
        if self._last_frame is not None:
            rgb = cv.cvtColor(self._last_frame, cv.COLOR_BGR2RGB)
            self.frame_captured.emit(rgb)

    def _on_frame(self, bgr: np.ndarray):
        self._last_frame = bgr

        # Forward to Workflow B acquisition controller
        self.frame_ready.emit(bgr)

        # Update preview (scale down if needed)
        self._update_preview(bgr)

    def _on_error(self, msg: str):
        self._status.setText(f"Error: {msg}")
        self._status.setStyleSheet("color: #cc4444;")
        self._set_connected_state(False, "")

    def _on_thread_finished(self):
        self._set_connected_state(False, "")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_preview(self, bgr: np.ndarray):
        rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        # Scale to fit preview label without distortion
        scale = min(self.MAX_PREVIEW_W / w, self.MAX_PREVIEW_H / h, 1.0)
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            rgb = cv.resize(rgb, (new_w, new_h), interpolation=cv.INTER_AREA)
            h, w = new_h, new_w

        qimg = QImage(
            rgb.data, w, h, w * 3, QImage.Format.Format_RGB888
        )
        self._preview.setPixmap(QPixmap.fromImage(qimg))

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
