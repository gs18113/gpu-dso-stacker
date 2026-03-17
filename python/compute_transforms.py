#!/usr/bin/env python3
"""
Independently compute homographies from star patterns using astroalign
and compare against the CSV transforms.
"""

import argparse
import csv
import os
import sys

import astroalign
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
                "basename": basename,
                "is_reference": int(row["is_reference"].strip()) == 1,
                "H_csv": h,
            })
    return frames


def load_fits(filepath):
    """Load FITS image as float32 numpy array."""
    data = fits.getdata(filepath)
    return data.astype(np.float32)


def affine_to_homography(transform):
    """
    Convert astroalign AffineTransform to a 3x3 homography in the CSV convention.

    astroalign.find_transform(source, target) returns an AffineTransform whose
    .params matrix maps source → target (frame → ref, the FORWARD direction).

    The CSV stores the BACKWARD map (ref → src), so we invert params to get
    the backward map and compare it against the CSV directly:
        x_src = h00*x_ref + h01*y_ref + h02   (ref → src, backward)
        y_src = h10*x_ref + h11*y_ref + h12
    """
    return np.linalg.inv(transform.params)


def format_matrix(m):
    """Format a 3x3 matrix as a compact string."""
    rows = []
    for row in m:
        rows.append("  [" + "  ".join(f"{v:+.6f}" for v in row) + "]")
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description="Compute transforms via astroalign and compare with CSV")
    parser.add_argument("-f", "--file", required=True, help="Input CSV file")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Translation difference threshold in pixels (default: 0.5)")
    parser.add_argument("--rot-threshold", type=float, default=1e-4,
                        help="Rotation/scale element threshold (default: 1e-4)")
    parser.add_argument("--save-csv", default=None,
                        help="Optional path to save comparison summary CSV")
    args = parser.parse_args()

    frames = parse_csv(args.file)
    ref = next(f for f in frames if f["is_reference"])

    print(f"Loading reference: {ref['basename']}")
    ref_data = load_fits(ref["path"])

    results = []
    print()
    print("=" * 72)
    print(f"{'Frame':<50} {'MaxDiff':>8}  {'Flag'}")
    print("=" * 72)

    for frame in frames:
        if frame["is_reference"]:
            # Identity — astroalign would trivially return identity too, skip
            results.append({
                "basename": frame["basename"],
                "is_reference": True,
                "H_csv": frame["H_csv"],
                "H_computed": np.eye(3),
                "diff": np.zeros((3, 3)),
                "max_diff": 0.0,
                "flagged": False,
                "error": None,
            })
            print(f"{frame['basename']:<50} {'0.000000':>8}  (reference)")
            continue

        print(f"Processing: {frame['basename']}", flush=True)
        src_data = load_fits(frame["path"])

        try:
            transform, _ = astroalign.find_transform(src_data, ref_data)
            H_computed = affine_to_homography(transform)
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            results.append({
                "basename": frame["basename"],
                "is_reference": False,
                "H_csv": frame["H_csv"],
                "H_computed": None,
                "diff": None,
                "max_diff": None,
                "flagged": True,
                "error": str(e),
            })
            print(f"{frame['basename']:<50} {'ERROR':>8}  FLAGGED")
            continue

        diff = np.abs(H_computed - frame["H_csv"])
        max_diff = diff.max()

        # Flag if translation (h02, h12) differs by > threshold, or rot/scale by > rot_threshold
        trans_diff = max(diff[0, 2], diff[1, 2])
        rot_diff = max(diff[0, 0], diff[0, 1], diff[1, 0], diff[1, 1])
        flagged = (trans_diff > args.threshold) or (rot_diff > args.rot_threshold)

        results.append({
            "basename": frame["basename"],
            "is_reference": False,
            "H_csv": frame["H_csv"],
            "H_computed": H_computed,
            "diff": diff,
            "max_diff": max_diff,
            "flagged": flagged,
            "error": None,
        })
        flag_str = "*** FLAGGED ***" if flagged else "ok"
        print(f"{frame['basename']:<50} {max_diff:>8.6f}  {flag_str}")

    # Detailed per-frame report
    print()
    print("=" * 72)
    print("DETAILED COMPARISON")
    print("=" * 72)
    for r in results:
        if r["is_reference"]:
            continue
        print(f"\n{r['basename']}")
        if r["error"]:
            print(f"  ERROR: {r['error']}")
            continue
        print("  CSV homography:")
        print(format_matrix(r["H_csv"]))
        print("  Computed homography (astroalign):")
        print(format_matrix(r["H_computed"]))
        print("  Absolute difference:")
        print(format_matrix(r["diff"]))
        flag_str = " *** FLAGGED ***" if r["flagged"] else ""
        print(f"  Max element diff: {r['max_diff']:.6f}{flag_str}")

    # Summary
    flagged_frames = [r for r in results if r["flagged"] and not r["is_reference"]]
    valid_diffs = [r["max_diff"] for r in results if r["max_diff"] is not None]
    print()
    print("=" * 72)
    print("SUMMARY")
    print(f"  Frames processed : {len(results)}")
    print(f"  Flagged          : {len(flagged_frames)}")
    if valid_diffs:
        print(f"  Max diff overall : {max(valid_diffs):.6f}")
        print(f"  Mean max diff    : {np.mean(valid_diffs):.6f}")

    if args.save_csv:
        with open(args.save_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["basename", "max_diff", "trans_diff_h02", "trans_diff_h12", "flagged", "error"])
            for r in results:
                if r["is_reference"]:
                    writer.writerow([r["basename"], 0.0, 0.0, 0.0, False, ""])
                    continue
                if r["error"]:
                    writer.writerow([r["basename"], "", "", "", True, r["error"]])
                    continue
                writer.writerow([
                    r["basename"],
                    f"{r['max_diff']:.6f}",
                    f"{r['diff'][0,2]:.6f}",
                    f"{r['diff'][1,2]:.6f}",
                    r["flagged"],
                    "",
                ])
        print(f"  Comparison CSV   : {args.save_csv}")


if __name__ == "__main__":
    main()
