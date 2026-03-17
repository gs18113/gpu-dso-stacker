#!/usr/bin/env python3
"""Reference Python stacker using OpenCV Lanczos4 warp + kappa-sigma integration."""

import argparse
import csv
import os
import sys

import cv2
import numpy as np
from astropy.io import fits


def parse_csv(filepath):
    """Parse transform_mat.csv, remapping macOS paths to local data/<basename>."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(filepath))), "data")
    frames = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            basename = os.path.basename(row["filepath"].strip())
            local_path = os.path.join(data_dir, basename)
            h = np.array([
                float(row["h00"]), float(row["h01"]), float(row["h02"]),
                float(row["h10"]), float(row["h11"]), float(row["h12"]),
                float(row["h20"]), float(row["h21"]), float(row["h22"]),
            ], dtype=np.float64).reshape(3, 3)
            frames.append({
                "path": local_path,
                "is_reference": int(row["is_reference"].strip()) == 1,
                "H": h,
            })
    return frames


def load_fits(filepath):
    """Load FITS image as float32 numpy array."""
    data = fits.getdata(filepath)
    return data.astype(np.float32)


def kappa_sigma_stack(stack, kappa=3.0, iterations=3):
    """
    Kappa-sigma clipping integration over axis 0.
    stack: (N, H, W) float32 array
    Returns: (H, W) float32 array
    """
    mask = np.zeros(stack.shape, dtype=bool)  # True = clipped

    for _ in range(iterations):
        valid = ~mask
        # Per-pixel mean and std over valid samples
        count = valid.sum(axis=0)  # (H, W)
        sum_vals = np.where(valid, stack, 0.0).sum(axis=0)
        mean = np.where(count > 0, sum_vals / np.maximum(count, 1), 0.0)

        # Bessel-corrected std (ddof=1)
        diff_sq = np.where(valid, (stack - mean[np.newaxis]) ** 2, 0.0)
        sum_sq = diff_sq.sum(axis=0)
        std = np.where(count > 1, np.sqrt(sum_sq / np.maximum(count - 1, 1)), 0.0)

        new_mask = mask | (np.abs(stack - mean[np.newaxis]) > kappa * std[np.newaxis])
        # Don't clip pixels where all values would be masked
        all_clipped = new_mask.all(axis=0)
        new_mask[:, all_clipped] = mask[:, all_clipped]

        if np.array_equal(new_mask, mask):
            break
        mask = new_mask

    # Final mean over surviving pixels
    valid = ~mask
    count = valid.sum(axis=0)
    result = np.where(valid, stack, 0.0).sum(axis=0) / np.maximum(count, 1)
    return result.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Python reference stacker")
    parser.add_argument("-f", "--file", required=True, help="Input CSV file")
    parser.add_argument("-o", "--output", default="output_py.fits", help="Output FITS file")
    parser.add_argument("--kappa", type=float, default=3.0, help="Sigma clipping threshold")
    parser.add_argument("--iterations", type=int, default=3, help="Max clipping iterations")
    args = parser.parse_args()

    frames = parse_csv(args.file)
    ref = next(f for f in frames if f["is_reference"])

    ref_data = load_fits(ref["path"])
    H_ref, W_ref = ref_data.shape
    print(f"Reference frame: {os.path.basename(ref['path'])}  ({W_ref}x{H_ref})")

    warped = []
    for i, frame in enumerate(frames):
        print(f"[{i+1}/{len(frames)}] {os.path.basename(frame['path'])}", flush=True)
        data = load_fits(frame["path"])
        if frame["is_reference"]:
            warped.append(data.copy())
        else:
            w = cv2.warpPerspective(
                data, frame["H"], (W_ref, H_ref),
                flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            warped.append(w)

    print("Integrating (kappa-sigma)…", flush=True)
    stack = np.stack(warped, axis=0)  # (N, H, W)
    result = kappa_sigma_stack(stack, kappa=args.kappa, iterations=args.iterations)

    fits.writeto(args.output, result, overwrite=True)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
