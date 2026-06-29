"""
Calibration Utilities
Scale calculation and pixel-to-unit conversion — GUI-independent.
Extracted from streamlit_app/tabs/tab1_setup.py
"""

import numpy as np


def calculate_scale_from_line_endpoints(pt1, pt2, real_distance, unit='mm'):
    """
    Calculate scale factor from two endpoints of a reference line.

    Args:
        pt1:           (x, y) start point in pixels
        pt2:           (x, y) end point in pixels
        real_distance: Known physical length of the line
        unit:          Physical unit string ('mm', 'cm', 'm', 'µm', 'inches')

    Returns:
        dict with keys:
            scale          – physical units per pixel
            unit           – unit string
            reference_value – real_distance as supplied
            pixel_distance – Euclidean distance between pt1 and pt2
            line_coords    – (x1, y1, x2, y2) tuple

    Raises:
        ValueError: If the two points are identical (zero pixel distance)
    """
    x1, y1 = pt1
    x2, y2 = pt2
    pixel_distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    if pixel_distance == 0:
        raise ValueError('pt1 and pt2 are identical — cannot calculate scale')

    return {
        'scale': real_distance / pixel_distance,
        'unit': unit,
        'reference_value': real_distance,
        'pixel_distance': pixel_distance,
        'line_coords': (x1, y1, x2, y2),
    }


def apply_calibration(pixel_value, calibration):
    """
    Convert a pixel measurement to physical units using a calibration dict.

    Args:
        pixel_value:  Distance or length in pixels (scalar or array)
        calibration:  Dict returned by calculate_scale_from_line_endpoints

    Returns:
        Physical measurement in calibration['unit'] (same type as pixel_value)
    """
    return pixel_value * calibration['scale']


def calibration_summary(calibration):
    """
    Return a human-readable string summarising the calibration.

    Args:
        calibration: Dict returned by calculate_scale_from_line_endpoints

    Returns:
        str
    """
    c = calibration
    return (
        f"Scale: {c['scale']:.6f} {c['unit']}/px  |  "
        f"Reference: {c['reference_value']:.3f} {c['unit']} = {c['pixel_distance']:.1f} px"
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test calibration utilities')
    parser.add_argument('--x1', type=float, default=50.0)
    parser.add_argument('--y1', type=float, default=100.0)
    parser.add_argument('--x2', type=float, default=150.0)
    parser.add_argument('--y2', type=float, default=100.0)
    parser.add_argument('--distance', type=float, default=10.0,
                        help='Known physical length of the reference line')
    parser.add_argument('--unit', default='mm')
    args = parser.parse_args()

    # --- calculate_scale_from_line_endpoints ---
    calib = calculate_scale_from_line_endpoints(
        (args.x1, args.y1), (args.x2, args.y2),
        real_distance=args.distance, unit=args.unit
    )
    print(calibration_summary(calib))
    print(f'  scale         = {calib["scale"]:.6f} {calib["unit"]}/px')
    print(f'  pixel_distance= {calib["pixel_distance"]:.2f} px')

    # Verify: re-converting the pixel distance must recover real_distance
    recovered = apply_calibration(calib['pixel_distance'], calib)
    assert abs(recovered - args.distance) < 1e-9, \
        f'Round-trip failed: got {recovered}, expected {args.distance}'
    print(f'  round-trip    = {recovered:.6f} {calib["unit"]}  ✓')

    # --- apply_calibration on a range of pixel values ---
    pixel_samples = np.array([0, 25, 50, 100, 200])
    physical = apply_calibration(pixel_samples, calib)
    print('\nPixel -> physical conversion:')
    for px, ph in zip(pixel_samples, physical):
        print(f'  {px:5.0f} px -> {ph:.4f} {calib["unit"]}')

    # --- known-value sanity check ---
    # Horizontal line of exactly 100 px representing 10 mm → scale = 0.1 mm/px
    c2 = calculate_scale_from_line_endpoints((0, 0), (100, 0), 10.0, 'mm')
    assert abs(c2['scale'] - 0.1) < 1e-12, 'Sanity check failed'
    print('\nSanity check (100 px = 10 mm -> scale=0.1): ✓')

    # --- zero-distance guard ---
    try:
        calculate_scale_from_line_endpoints((5, 5), (5, 5), 10.0)
        print('ERROR: should have raised ValueError')
    except ValueError as e:
        print(f'Zero-distance guard: {e}  ✓')

    print('\nAll tests passed.')
