# -*- coding: utf-8 -*-
"""
Homography calibration backend.

Computes a perspective-rectification homography H from a chessboard
pattern, with a 4-point manual fallback when auto-detection fails.

Two paths:
  Auto  -- findChessboardCorners (all inner corners) + findHomography (RANSAC)
  Manual -- user picks 4 outer inner-corners in order TL, TR, BL, BR
             + getPerspectiveTransform (exact)

In both cases H maps the raw camera frame to a rectified view where
the grid lines are horizontal and vertical and the squares are square.
"""

from dataclasses import dataclass
import numpy as np
import cv2 as cv


# ---------------------------------------------------------------------------
# Board description
# ---------------------------------------------------------------------------

@dataclass
class BoardConfig:
    inner_cols: int   = 9      # inner corner count horizontally
    inner_rows: int   = 6      # inner corner count vertically
    square_mm:  float = 25.0   # physical square side length


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------

def detect_chessboard(
    bgr: np.ndarray,
    board: BoardConfig,
) -> tuple[np.ndarray | None, np.ndarray]:
    """
    Run findChessboardCorners + sub-pixel refinement.

    Returns
    -------
    corners : (N, 1, 2) float32  or  None if not found
    gray    : grayscale version of the input (for display)
    """
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    flags = cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv.findChessboardCorners(
        gray, (board.inner_cols, board.inner_rows), flags
    )
    if not found:
        return None, gray
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners, gray


def draw_detected_corners(
    bgr: np.ndarray,
    corners: np.ndarray,
    board: BoardConfig,
) -> np.ndarray:
    """Return a copy of bgr with detected corners drawn."""
    out = bgr.copy()
    cv.drawChessboardCorners(
        out, (board.inner_cols, board.inner_rows), corners, True
    )
    return out


# ---------------------------------------------------------------------------
# Homography computation
# ---------------------------------------------------------------------------

def _ideal_dst(board: BoardConfig, output_wh: tuple[int, int],
               n_cols: int, n_rows: int) -> np.ndarray:
    """
    Build an ideal destination grid centred in output_wh.

    n_cols, n_rows : number of points in each direction
                     (inner_cols / inner_rows for auto,
                      2 / 2 = just TL TR BL BR for manual)
    """
    out_w, out_h = output_wh
    margin = 0.10
    avail_w = out_w * (1 - 2 * margin)
    avail_h = out_h * (1 - 2 * margin)
    sq_px = min(avail_w / max(n_cols - 1, 1), avail_h / max(n_rows - 1, 1))
    ox = (out_w - sq_px * (n_cols - 1)) / 2
    oy = (out_h - sq_px * (n_rows - 1)) / 2
    pts = []
    for r in range(n_rows):
        for c in range(n_cols):
            pts.append([ox + c * sq_px, oy + r * sq_px])
    return np.array(pts, dtype=np.float64)


def compute_homography_auto(
    corners: np.ndarray,
    board: BoardConfig,
    output_wh: tuple[int, int],
) -> np.ndarray:
    """
    Compute H from all detected inner corners (RANSAC).

    corners    : (N, 1, 2) float32 from detect_chessboard
    output_wh  : (width, height) of the output frame
    Returns    : H (3x3 float64)
    """
    src = corners.reshape(-1, 2).astype(np.float64)
    dst = _ideal_dst(board, output_wh, board.inner_cols, board.inner_rows)
    H, _ = cv.findHomography(src, dst, cv.RANSAC, 5.0)
    return H


def compute_homography_manual(
    pts_tl_tr_bl_br: list[tuple[int, int]],
    board: BoardConfig,
    output_wh: tuple[int, int],
) -> np.ndarray:
    """
    Compute H from 4 manually picked corner inner-corners
    in order: top-left, top-right, bottom-left, bottom-right.

    Uses getPerspectiveTransform (exact, no RANSAC).
    Returns : H (3x3 float64)
    """
    nc = board.inner_cols - 1   # number of spans between outer inner-corners
    nr = board.inner_rows - 1
    dst4 = _ideal_dst(board, output_wh, 2, 2)   # just the 4 corners

    src = np.array(pts_tl_tr_bl_br, dtype=np.float32)
    dst = dst4.astype(np.float32)
    H = cv.getPerspectiveTransform(src, dst)
    return H.astype(np.float64)


# ---------------------------------------------------------------------------
# Application helpers
# ---------------------------------------------------------------------------

def apply_homography(bgr: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Warp bgr by H; output size equals input size."""
    h, w = bgr.shape[:2]
    return cv.warpPerspective(bgr, H, (w, h))


def draw_manual_corners(
    bgr: np.ndarray,
    pts: list[tuple[int, int]],
) -> np.ndarray:
    """Annotate a frozen frame with the manually picked corner points."""
    labels = ["TL", "TR", "BL", "BR"]
    colors = [(0, 212, 255), (0, 212, 255), (0, 212, 255), (0, 212, 255)]
    out = bgr.copy()
    for i, (x, y) in enumerate(pts):
        cv.drawMarker(out, (x, y), colors[i],
                      cv.MARKER_CROSS, 20, 2, cv.LINE_AA)
        cv.putText(out, labels[i], (x + 8, y - 8),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2, cv.LINE_AA)
    return out
