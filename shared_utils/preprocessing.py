"""
Preprocessing Utilities
Contrast adjustment and morphological operations — GUI-independent.
Extracted from streamlit_app/tabs/tab2_preprocessing.py
"""

import cv2 as cv
import numpy as np


def adjust_contrast(img, alpha=1.0, beta=0):
    """
    Adjust image contrast and brightness.

    Args:
        img: Grayscale or BGR image
        alpha: Contrast multiplier (>1 increases contrast, <1 decreases)
        beta:  Brightness offset (-100 to +100)

    Returns:
        numpy.ndarray: Adjusted image (same dtype as input)
    """
    return cv.convertScaleAbs(img, alpha=alpha, beta=beta)


def morphological_operation(img, operation, kernel_size=5):
    """
    Apply a morphological operation using an elliptical structuring element.

    Args:
        img:          Grayscale image
        operation:    One of 'erosion', 'dilation', 'opening', 'closing'
        kernel_size:  Side length of the structuring element (odd recommended)

    Returns:
        numpy.ndarray: Processed image

    Raises:
        ValueError: If operation is not one of the four valid options
    """
    valid = ('erosion', 'dilation', 'opening', 'closing')
    if operation not in valid:
        raise ValueError(f"operation must be one of {valid}, got '{operation}'")

    kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )

    ops = {
        'erosion': lambda i, k: cv.erode(i, k),
        'dilation': lambda i, k: cv.dilate(i, k),
        'opening': lambda i, k: cv.morphologyEx(i, cv.MORPH_OPEN, k),
        'closing': lambda i, k: cv.morphologyEx(i, cv.MORPH_CLOSE, k),
    }
    return ops[operation](img, kernel)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test preprocessing utilities')
    parser.add_argument('image', help='Input image path')
    parser.add_argument('--alpha', type=float, default=1.5,
                        help='Contrast multiplier (default 1.5)')
    parser.add_argument('--beta', type=int, default=10,
                        help='Brightness offset (default 10)')
    parser.add_argument('--morph', default='closing',
                        choices=['erosion', 'dilation', 'opening', 'closing'],
                        help='Morphological operation (default closing)')
    parser.add_argument('--kernel', type=int, default=5,
                        help='Morphological kernel size (default 5)')
    parser.add_argument('--output', default='output_preprocessed.png',
                        help='Output image path')
    args = parser.parse_args()

    img = cv.imread(args.image)
    if img is None:
        print(f'ERROR: could not load {args.image}')
        raise SystemExit(1)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if img.ndim == 3 else img
    print(f'Loaded: {args.image}  shape={img.shape}')

    # adjust_contrast
    contrasted = adjust_contrast(gray, alpha=args.alpha, beta=args.beta)
    mean_before = gray.mean()
    mean_after = contrasted.mean()
    print(f'adjust_contrast(alpha={args.alpha}, beta={args.beta}): '
          f'mean {mean_before:.1f} -> {mean_after:.1f}')

    # morphological_operation
    morphed = morphological_operation(contrasted, args.morph, args.kernel)
    print(f'morphological_operation({args.morph!r}, kernel={args.kernel}): '
          f'shape={morphed.shape}')

    # all four operations in sequence as a smoke-test
    for op in ('erosion', 'dilation', 'opening', 'closing'):
        result = morphological_operation(gray, op, kernel_size=3)
        print(f'  {op:10s}: shape={result.shape}  mean={result.mean():.1f}')

    cv.imwrite(args.output, morphed)
    print(f'Saved -> {args.output}')
    print('All tests passed.')
