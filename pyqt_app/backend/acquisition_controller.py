"""
Acquisition Controller — Workflow B backend, no GUI imports.

Stateful controller for a single angle-measurement session.
Receives BGR frames one at a time (from CameraWidget.frame_ready or
a pre-collected list) and feeds them through the extraction pipeline.

Designed to be driven by AcquisitionThread (see below) in the PyQt6
layer, or from a plain loop in a headless test.

Workflow B data flow
--------------------
CameraWidget.frame_ready(bgr)
    → AcquisitionThread.on_frame(bgr)          [QThread, GUI layer]
        → AcquisitionController.process_frame(bgr)   [this file]
            → extract_angle_from_frame()              [shared_utils]
            → CaptureSession.add_frame()              [shared_utils]
        ← FrameResult(angle, diagnostics, progress)
    → emit progress/frame signals to UI widgets
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'shared_utils'))

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Callable

from optical_experiment import CaptureSession
from angle_extraction import extract_angle_from_frame


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class AcquisitionConfig:
    """All parameters for one acquisition session."""
    distance:         float = 100.0
    unit:             str   = 'mm'
    num_frames:       int   = 60
    frame_interval_ms: int  = 100      # minimum ms between processed frames
    top_margin:       int   = 10
    bottom_margin:    int   = 10
    snake_params: dict = field(default_factory=lambda: {
        'num_points':    50,
        'num_iterations': 100,
        'window_size':   5,
        'alpha':         0.01,
        'beta':          10.0,   # high → straight-line constraint
        'gamma':         0.5,
        'threshold':     0.0001,
    })
    edge_params: dict = field(default_factory=lambda: {
        'method':     'canny',
        'canny_low':  50,
        'canny_high': 150,
        'blur_kernel': 3,
        'morphology': True,
    })


# ---------------------------------------------------------------------------
# Per-frame result
# ---------------------------------------------------------------------------

@dataclass
class FrameResult:
    frame_index:   int
    angle_deg:     float
    diagnostics:   dict        # full extract_angle_from_frame diagnostics dict
    session_stats: dict | None  # CaptureSession.get_statistics() after this frame
    elapsed_ms:    float


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class AcquisitionError(Exception):
    pass


class AcquisitionController:
    """
    Processes frames for a single measurement session.

    Usage (live, driven by QThread)
    --------------------------------
        cfg = AcquisitionConfig(distance=150.0, num_frames=60)
        ctrl = AcquisitionController()
        ctrl.start(cfg)
        # … for each frame arriving from CameraWidget.frame_ready:
        result = ctrl.process_frame(bgr_frame)
        if ctrl.is_complete:
            session = ctrl.finish()

    Usage (batch / headless)
    -------------------------
        session = AcquisitionController.run_batch(frames, cfg)
    """

    def __init__(self):
        self._session:  CaptureSession | None = None
        self._config:   AcquisitionConfig | None = None
        self._active:   bool = False
        self._last_frame_time: float = 0.0

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start(self, config: AcquisitionConfig) -> None:
        """Begin a new session. Resets any previous state."""
        if self._active:
            raise AcquisitionError(
                "A session is already active — call stop() or finish() first"
            )
        self._config  = config
        self._session = CaptureSession(
            distance=config.distance,
            unit=config.unit,
        )
        self._session.snake_params     = config.snake_params
        self._session.extraction_params = {
            'top_margin':    config.top_margin,
            'bottom_margin': config.bottom_margin,
            'edge_params':   config.edge_params,
        }
        self._active = True
        self._last_frame_time = 0.0

    def process_frame(self, bgr_frame: np.ndarray) -> FrameResult | None:
        """
        Process one BGR frame from the camera.

        Skips the frame if it arrives before frame_interval_ms has elapsed
        (rate limiting without blocking the caller).

        Returns a FrameResult, or None if the frame was rate-limited.

        Raises AcquisitionError if called outside an active session.
        """
        if not self._active:
            raise AcquisitionError("No active session — call start() first")

        # Rate limiting
        now = time.monotonic() * 1000
        if now - self._last_frame_time < self._config.frame_interval_ms:
            return None
        self._last_frame_time = now

        cfg = self._config
        t0  = time.monotonic()

        # Convert BGR → RGB (extract_angle_from_frame expects RGB)
        import cv2 as cv
        rgb = cv.cvtColor(bgr_frame, cv.COLOR_BGR2RGB)

        result = extract_angle_from_frame(
            rgb,
            roi_bbox=None,
            rotation=0.0,
            top_margin=cfg.top_margin,
            bottom_margin=cfg.bottom_margin,
            snake_params=cfg.snake_params,
            edge_params=cfg.edge_params,
            return_diagnostics=True,
        )

        self._session.add_frame(
            bgr_frame,
            result['angle_deg'],
            result.get('snake_optimized'),
        )

        elapsed_ms = (time.monotonic() - t0) * 1000

        return FrameResult(
            frame_index   = self._session.num_frames - 1,
            angle_deg     = result['angle_deg'],
            diagnostics   = result,
            session_stats = self._session.get_statistics(),
            elapsed_ms    = elapsed_ms,
        )

    @property
    def is_complete(self) -> bool:
        """True when num_frames target has been reached."""
        if self._session is None or self._config is None:
            return False
        return self._session.num_frames >= self._config.num_frames

    def finish(self) -> CaptureSession:
        """
        Mark session complete and return it.
        Call when is_complete is True or the user stops early.
        """
        if self._session is None:
            raise AcquisitionError("No session to finish")
        self._active = False
        return self._session

    def stop(self) -> CaptureSession:
        """Early stop — same as finish() but signals intent."""
        return self.finish()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def progress(self) -> tuple[int, int]:
        """(frames_captured, target). Both 0 if no session."""
        if self._session is None or self._config is None:
            return (0, 0)
        return (self._session.num_frames, self._config.num_frames)

    @property
    def current_session(self) -> CaptureSession | None:
        return self._session

    # ------------------------------------------------------------------
    # Batch / headless mode
    # ------------------------------------------------------------------

    @classmethod
    def run_batch(
        cls,
        frames: list[np.ndarray],
        config: AcquisitionConfig | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> CaptureSession:
        """
        Process a pre-collected list of BGR frames in one call.

        Useful for headless testing and for Workflow A (single image
        analysed as a batch of one).

        Args:
            frames:            List of BGR numpy arrays
            config:            AcquisitionConfig (defaults used if None)
            progress_callback: Optional fn(current, total) for CLI progress

        Returns:
            Completed CaptureSession
        """
        if config is None:
            config = AcquisitionConfig(num_frames=len(frames))

        ctrl = cls()
        # Bypass rate limiting for batch mode
        config_no_rate = AcquisitionConfig(**{
            **config.__dict__,
            'frame_interval_ms': 0,
            'num_frames': len(frames),
        })
        ctrl.start(config_no_rate)

        for i, frame in enumerate(frames):
            try:
                ctrl.process_frame(frame)
            except Exception as e:
                print(f"Warning: frame {i} failed — {e}")

            if progress_callback:
                progress_callback(i + 1, len(frames))

        return ctrl.finish()


# ---------------------------------------------------------------------------
# PyQt6 acquisition thread (thin wrapper — keep in this file so
# the backend is self-contained; the GUI layer just imports it)
# ---------------------------------------------------------------------------

try:
    import queue as _queue
    from PyQt6.QtCore import QThread, pyqtSignal as Signal

    class AcquisitionThread(QThread):
        """
        Drives AcquisitionController from the CameraWidget.frame_ready signal.

        Connect
        -------
            camera_widget.frame_ready.connect(acq_thread.on_frame)
            acq_thread.frame_processed.connect(my_widget.update_ui)
            acq_thread.session_complete.connect(my_widget.on_session_done)
        """

        frame_processed  = Signal(object)   # FrameResult
        session_complete = Signal(object)   # CaptureSession
        error            = Signal(str)

        def __init__(self, config: AcquisitionConfig, parent=None):
            super().__init__(parent)
            self._config  = config
            self._ctrl    = AcquisitionController()
            self._queue:  _queue.Queue = _queue.Queue()
            self._running = False

        def on_frame(self, bgr: np.ndarray):
            """Slot — connect to CameraWidget.frame_ready (thread-safe)."""
            if self._running:
                self._queue.put(bgr)

        def run(self):
            self._ctrl.start(self._config)
            self._running = True

            while self._running:
                try:
                    frame = self._queue.get(timeout=0.05)
                    result = self._ctrl.process_frame(frame)
                    if result is not None:
                        self.frame_processed.emit(result)
                except _queue.Empty:
                    pass
                except Exception as e:
                    self.error.emit(str(e))

                if self._ctrl.is_complete:
                    self._running = False
                    session = self._ctrl.finish()
                    self.session_complete.emit(session)

        def stop(self):
            self._running = False
            self.wait()
            if self._ctrl.is_active:
                return self._ctrl.stop()

except ImportError:
    pass   # PyQt6 not available — AcquisitionThread simply not defined


# ---------------------------------------------------------------------------
# Standalone test (no camera needed — synthetic frames)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    import cv2 as cv

    parser = argparse.ArgumentParser(
        description='Test AcquisitionController with synthetic frames'
    )
    parser.add_argument('--frames', type=int, default=10,
                        help='Number of synthetic frames to process')
    parser.add_argument('--image', default=None,
                        help='Real image to use instead of synthetic noise')
    args = parser.parse_args()

    # Build test frames
    if args.image:
        base = cv.imread(args.image)
        if base is None:
            print(f'ERROR: cannot load {args.image}')
            raise SystemExit(1)
        frames = [base + np.random.randint(-5, 5, base.shape, dtype=np.int16)
                  .clip(0, 255).astype(np.uint8)
                  for _ in range(args.frames)]
        print(f'Using {args.image} as base, {args.frames} slightly-noised copies')
    else:
        h, w = 120, 80
        frames = []
        for _ in range(args.frames):
            f = np.zeros((h, w, 3), dtype=np.uint8)
            # Vertical bright line slightly off-centre (simulates laser)
            cx = w // 2 + np.random.randint(-3, 3)
            f[:, cx-1:cx+2, :] = 220
            frames.append(f)
        print(f'Using {args.frames} synthetic laser-line frames ({h}×{w})')

    # --- batch mode ---
    cfg = AcquisitionConfig(
        distance=100.0,
        unit='mm',
        num_frames=len(frames),
        frame_interval_ms=0,
    )

    processed = 0
    def show_progress(cur, tot):
        global processed
        processed = cur
        print(f'  [{cur:3d}/{tot}]', end='\r')

    session = AcquisitionController.run_batch(frames, cfg, show_progress)
    print()

    stats = session.get_statistics()
    print(f'Session: {session}')
    if stats:
        print(f'  mean  = {stats["mean"]:.3f}°')
        print(f'  std   = {stats["std"]:.3f}°')
        print(f'  n     = {stats["n_frames"]}')
        print(f'  range = {stats["min"]:.3f}° – {stats["max"]:.3f}°')
    else:
        print('  (no valid angles extracted)')

    print('All tests passed.')
