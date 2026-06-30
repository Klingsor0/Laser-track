"""
Camera configuration dataclass + per-frame preprocessing pipelines.

Two pipelines are provided:
  apply_preview_pipeline    — rotation + filters only (no ROI crop).
                              Used by CameraWidget to update the live preview.
                              The ROI is drawn as an overlay rectangle instead.
  apply_acquisition_pipeline — rotation → ROI crop → filters.
                               Used for the frame_ready signal consumed by
                               Workflow B (AcquisitionThread).
"""

import sys
from dataclasses import dataclass
from pathlib import Path
import json

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'shared_utils'))
from image_processing import rotacion, gaussian_blur
from preprocessing import adjust_contrast


@dataclass
class CameraConfig:
    rotation:    float = 0.0           # degrees; applied around the axis pivot point
    axis_x:      int   = 320           # pivot x in absolute frame pixels
    axis_y:      int   = 240           # pivot y in absolute frame pixels
    roi:         tuple | None = None   # (x, y, w, h) in frame pixels; None = full frame
    blur_kernel: int   = 1             # Gaussian blur kernel size; 1 = disabled (must be odd)
    contrast:    float = 1.0           # alpha for convertScaleAbs
    brightness:  int   = 0             # beta  for convertScaleAbs
    homography:  np.ndarray | None = None  # 3x3 perspective rectification matrix

    def copy(self) -> 'CameraConfig':
        return CameraConfig(
            rotation=self.rotation,
            axis_x=self.axis_x,
            axis_y=self.axis_y,
            roi=tuple(self.roi) if self.roi else None,
            blur_kernel=self.blur_kernel,
            contrast=self.contrast,
            brightness=self.brightness,
            homography=self.homography.copy() if self.homography is not None else None,
        )

    def to_dict(self) -> dict:
        return {
            'rotation':    self.rotation,
            'axis_x':      self.axis_x,
            'axis_y':      self.axis_y,
            'roi':         list(self.roi) if self.roi else None,
            'blur_kernel': self.blur_kernel,
            'contrast':    self.contrast,
            'brightness':  self.brightness,
            'homography':  self.homography.tolist() if self.homography is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'CameraConfig':
        roi = d.get('roi')
        h   = d.get('homography')
        return cls(
            rotation=float(d.get('rotation', 0.0)),
            axis_x=int(d.get('axis_x', 320)),
            axis_y=int(d.get('axis_y', 240)),
            roi=tuple(int(v) for v in roi) if roi else None,
            blur_kernel=int(d.get('blur_kernel', 1)),
            contrast=float(d.get('contrast', 1.0)),
            brightness=int(d.get('brightness', 0)),
            homography=np.array(h, dtype=np.float64) if h is not None else None,
        )

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path) -> 'CameraConfig':
        return cls.from_dict(json.loads(Path(path).read_text()))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_rotation(img: np.ndarray, rotation: float,
                    axis_x: int = 320, axis_y: int = 240) -> np.ndarray:
    if rotation == 0.0:
        return img
    h, w = img.shape[:2]
    M = cv.getRotationMatrix2D((float(axis_x), float(axis_y)), rotation, 1)
    return cv.warpAffine(img, M, (w, h))


def _apply_filters(img: np.ndarray, blur_kernel: int,
                   contrast: float, brightness: int) -> np.ndarray:
    if blur_kernel > 1:
        k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        img = gaussian_blur(img, k)
    if contrast != 1.0 or brightness != 0:
        img = adjust_contrast(img, contrast, brightness)
    return img


# ---------------------------------------------------------------------------
# Public pipelines
# ---------------------------------------------------------------------------

def _apply_homography(img: np.ndarray, H: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return cv.warpPerspective(img, H, (w, h))


def apply_preview_pipeline(bgr: np.ndarray, config: CameraConfig) -> np.ndarray:
    """
    Homography → rotation → filters on the full frame.
    ROI is intentionally NOT cropped — shown as an overlay rectangle on the preview.
    """
    img = bgr.copy()
    if config.homography is not None:
        img = _apply_homography(img, config.homography)
    img = _apply_rotation(img, config.rotation, config.axis_x, config.axis_y)
    img = _apply_filters(img, config.blur_kernel, config.contrast, config.brightness)
    return img


def apply_acquisition_pipeline(bgr: np.ndarray, config: CameraConfig) -> np.ndarray:
    """
    Homography → rotation → ROI crop → filters.
    Emitted via frame_ready for the acquisition controller.
    """
    img = bgr.copy()
    if config.homography is not None:
        img = _apply_homography(img, config.homography)
    img = _apply_rotation(img, config.rotation, config.axis_x, config.axis_y)

    if config.roi is not None:
        x, y, w, h = config.roi
        ih, iw = img.shape[:2]
        x = max(0, min(x, iw - 1))
        y = max(0, min(y, ih - 1))
        w = max(1, min(w, iw - x))
        h = max(1, min(h, ih - y))
        img = img[y:y + h, x:x + w]

    img = _apply_filters(img, config.blur_kernel, config.contrast, config.brightness)
    return img
