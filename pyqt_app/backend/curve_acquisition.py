# -*- coding: utf-8 -*-
"""
Curve acquisition backend - Workflow C.

Batch snake-curve extraction from a live camera feed.
Every frame is optimized starting from the same user-defined initial snake,
so small frame-to-frame variations are captured independently.

Data flow
---------
CameraWidget.frame_ready(bgr)
    → CurveAcquisitionThread.on_frame(bgr)
        → green-channel edge image
        → snk.optimize_snake_greedy(edge, initial_snake, ...)
        → emit curve_extracted(np.ndarray (N, 2))
    → when criterion satisfied: emit acquisition_complete(CurveResult)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'shared_utils'))

import time
import queue
import numpy as np
from dataclasses import dataclass

import cv2 as cv
import snake1 as snk


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CurveConfig:
    num_frames:        int   = 30
    frame_interval_ms: int   = 200
    num_points:        int   = 100
    num_iterations:    int   = 100
    window_size:       int   = 5
    alpha:             float = 0.01
    beta:              float = 0.1
    gamma:             float = 0.5
    threshold:         float = 0.0001


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class CurveResult:
    curves:        np.ndarray   # (F, N, 2) — all optimized curves
    mean_curve:    np.ndarray   # (N, 2)
    std_per_point: np.ndarray   # (N,) — per-point L2 spread across frames


def _build_result(curves: list) -> CurveResult:
    arr  = np.array(curves)                              # (F, N, 2)
    mean = arr.mean(axis=0)                              # (N, 2)
    diff = arr - mean[None]                              # (F, N, 2)
    std  = np.sqrt((diff ** 2).sum(axis=2)).mean(axis=0) # (N,)
    return CurveResult(curves=arr, mean_curve=mean, std_per_point=std)


# ---------------------------------------------------------------------------
# Stopping criteria
# ---------------------------------------------------------------------------

class FixedFrames:
    """Stop after collecting exactly n frames."""
    def __init__(self, n: int):
        self.n = n

    def should_stop(self, curves: list) -> bool:
        return len(curves) >= self.n


# ---------------------------------------------------------------------------
# PyQt6 acquisition thread
# ---------------------------------------------------------------------------

try:
    from PyQt6.QtCore import QThread, pyqtSignal as Signal

    class CurveAcquisitionThread(QThread):
        """
        Receives BGR frames from CameraWidget.frame_ready and extracts
        one optimized snake curve per frame.

        Connect
        -------
            camera.frame_ready.connect(thread.on_frame)
            thread.curve_extracted.connect(dialog.on_curve)
            thread.acquisition_complete.connect(dialog.on_complete)
            thread.progress.connect(dialog.on_progress)
        """

        curve_extracted      = Signal(object)         # np.ndarray (N, 2)
        acquisition_complete = Signal(object)         # CurveResult
        progress             = Signal(int, int)       # (current, total)
        iteration_update     = Signal(int, int, object, float)  # (frame, iter, snake, improvement)
        error                = Signal(str)

        def __init__(self, initial_snake: np.ndarray,
                     config: CurveConfig,
                     criterion: FixedFrames,
                     parent=None):
            super().__init__(parent)
            self._initial_snake = initial_snake.copy()
            self._config        = config
            self._criterion     = criterion
            self._queue: queue.Queue = queue.Queue()
            self._running       = True   # True from init so pre-start on_frame() calls queue correctly
            self._curves: list[np.ndarray] = []

        def on_frame(self, bgr: np.ndarray):
            """Slot — connect to CameraWidget.frame_ready (thread-safe)."""
            if self._running:
                self._queue.put(bgr)

        def request_stop(self):
            """Non-blocking stop. Result delivered via acquisition_complete or error signal."""
            self._running = False

        def run(self):
            self._curves  = []
            cfg           = self._config
            last_time     = 0.0

            while self._running:
                try:
                    bgr = self._queue.get(timeout=0.05)
                except queue.Empty:
                    continue

                now = time.monotonic() * 1000
                if now - last_time < cfg.frame_interval_ms:
                    continue
                last_time = now

                try:
                    rgb  = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
                    edge = rgb[:, :, 1].astype(np.uint8)

                    frame_num = len(self._curves)

                    def _iter_cb(iteration, snake, improvement,
                                 _fn=frame_num, _thr=cfg.threshold):
                        converged = improvement < _thr
                        if converged or iteration % 5 == 0:
                            self.iteration_update.emit(
                                _fn, iteration,
                                np.array(snake, dtype=float),
                                float(improvement),
                            )

                    optimized, _ = snk.optimize_snake_greedy(
                        edge,
                        self._initial_snake,
                        num_iterations=cfg.num_iterations,
                        window_size=cfg.window_size,
                        alpha=cfg.alpha,
                        beta=cfg.beta,
                        gamma=cfg.gamma,
                        threshold=cfg.threshold,
                        callback=_iter_cb,
                    )
                    curve = np.array(optimized, dtype=float)
                    self._curves.append(curve)
                    self.curve_extracted.emit(curve)

                    total = self._criterion.n if hasattr(self._criterion, 'n') else -1
                    self.progress.emit(len(self._curves), total)

                except Exception as e:
                    self.error.emit(str(e))
                    continue

                if self._criterion.should_stop(self._curves):
                    self._running = False

            if self._curves:
                self.acquisition_complete.emit(_build_result(self._curves))
            else:
                self.error.emit("Stopped before any frames were processed.")

        def stop(self) -> 'CurveResult | None':
            """Blocking stop for cleanup (e.g. closeEvent). Prefer request_stop() from UI."""
            self._running = False
            self.wait()
            return _build_result(self._curves) if self._curves else None

except ImportError:
    pass
