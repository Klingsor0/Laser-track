"""
Camera Manager — backend, no GUI imports.

Handles connection to two source types:
  - USB webcam  (cv.VideoCapture(index))
  - ESP32-CAM   (cv.VideoCapture(mjpeg_url))

Used by both workflows:
  Workflow A (profile analysis)  → grab_frame() → single image for snake/curve pipeline
  Workflow B (angle statistics)  → read_frame() inside acquisition loop → N frames
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'shared_utils'))

import cv2 as cv
import numpy as np
from enum import Enum, auto


class SourceType(Enum):
    USB    = auto()
    ESP32  = auto()


class CameraError(Exception):
    pass


class CameraManager:
    """
    Manages a single camera connection (USB webcam or ESP32-CAM MJPEG stream).

    Designed to be owned by a QThread in the GUI layer; all methods are
    blocking and GUI-free — the caller decides how to schedule them.
    """

    def __init__(self):
        self._cap: cv.VideoCapture | None = None
        self._source_type: SourceType | None = None
        self._url: str | None = None          # last successful URL (ESP32)
        self._index: int | None = None        # last successful index (USB)

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect_usb(self, index: int = 0) -> None:
        """
        Open a USB webcam.

        Args:
            index: OpenCV camera index (0 = default webcam)

        Raises:
            CameraError: If the device cannot be opened or no frames arrive
        """
        self.disconnect()

        cap = cv.VideoCapture(index)
        if not cap.isOpened():
            raise CameraError(f"Cannot open USB camera at index {index}")

        ret, _ = cap.read()
        if not ret:
            cap.release()
            raise CameraError(f"USB camera {index} opened but returned no frames")

        self._cap = cap
        self._source_type = SourceType.USB
        self._index = index

    def connect_esp32(self, ip: str, port: int = 81, path: str = "/stream") -> None:
        """
        Open an ESP32-CAM MJPEG network stream.

        Args:
            ip:   IP address of the ESP32-CAM
            port: HTTP port (default 81)
            path: Stream endpoint (default /stream)

        Raises:
            CameraError: If the URL cannot be opened or no frames arrive
        """
        self.disconnect()

        url = f"http://{ip}:{port}{path}"
        cap = cv.VideoCapture(url)
        if not cap.isOpened():
            raise CameraError(f"Cannot open ESP32-CAM stream at {url}")

        ret, _ = cap.read()
        if not ret:
            cap.release()
            raise CameraError(f"Connected to {url} but received no frames")

        self._cap = cap
        self._source_type = SourceType.ESP32
        self._url = url

    def disconnect(self) -> None:
        """Release the current camera, if any."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._source_type = None

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def read_frame(self) -> np.ndarray:
        """
        Read the next frame (BGR).

        Returns:
            numpy.ndarray: BGR image

        Raises:
            CameraError: If not connected or the read fails
        """
        if self._cap is None:
            raise CameraError("Camera is not connected")

        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise CameraError("Failed to read frame — stream may have dropped")

        return frame

    def grab_frame(self) -> np.ndarray:
        """
        Grab a single frame for Workflow A (profile analysis).

        Converts to RGB so it can be fed directly into the image
        processing pipeline that expects RGB input.

        Returns:
            numpy.ndarray: RGB image, same resolution as the camera stream
        """
        bgr = self.read_frame()
        return cv.cvtColor(bgr, cv.COLOR_BGR2RGB)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def source_type(self) -> SourceType | None:
        return self._source_type

    @property
    def resolution(self) -> tuple[int, int] | None:
        """Return (width, height) or None if not connected."""
        if self._cap is None:
            return None
        w = int(self._cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    @property
    def fps(self) -> float | None:
        """Reported FPS from the camera (may be 0 for network streams)."""
        if self._cap is None:
            return None
        return self._cap.get(cv.CAP_PROP_FPS)

    def __repr__(self) -> str:
        if not self.is_connected:
            return "CameraManager(disconnected)"
        src = self._url if self._source_type == SourceType.ESP32 else f"USB:{self._index}"
        return f"CameraManager({src}, {self.resolution}, {self.fps:.1f}fps)"

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.disconnect()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test CameraManager without GUI')
    sub = parser.add_subparsers(dest='source', required=True)

    p_usb = sub.add_parser('usb', help='Test USB webcam')
    p_usb.add_argument('--index', type=int, default=0)
    p_usb.add_argument('--frames', type=int, default=5)

    p_esp = sub.add_parser('esp32', help='Test ESP32-CAM stream')
    p_esp.add_argument('--ip', required=True)
    p_esp.add_argument('--port', type=int, default=81)
    p_esp.add_argument('--path', default='/stream')
    p_esp.add_argument('--frames', type=int, default=5)

    args = parser.parse_args()

    cam = CameraManager()

    try:
        if args.source == 'usb':
            print(f'Connecting to USB camera {args.index}...')
            cam.connect_usb(args.index)
        else:
            print(f'Connecting to ESP32-CAM at {args.ip}:{args.port}{args.path}...')
            cam.connect_esp32(args.ip, args.port, args.path)

        print(f'Connected: {cam}')
        print(f'  is_connected = {cam.is_connected}')
        print(f'  source_type  = {cam.source_type}')
        print(f'  resolution   = {cam.resolution}')
        print(f'  fps          = {cam.fps}')

        for i in range(args.frames):
            frame = cam.read_frame()
            print(f'  Frame {i+1}: shape={frame.shape}  dtype={frame.dtype}')

        # grab_frame (RGB for Workflow A)
        rgb = cam.grab_frame()
        assert rgb.shape[2] == 3
        print(f'grab_frame (RGB): shape={rgb.shape}  ✓')

    except CameraError as e:
        print(f'CameraError: {e}')
    finally:
        cam.disconnect()
        print('Disconnected.')
