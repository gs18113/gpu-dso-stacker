#!/usr/bin/env python3
"""
generate_test_frames.py — Synthetic FITS frame generator for gpu-dso-stacker testing.

Produces a set of FITS images that exercise every stage of the C pipeline:

  Stage 1  Debayer     -- optional Bayer mosaic (--bayer rggb|bggr|grbg|gbrg)
  Stage 2  Star detect -- Moffat PSF stars tuned to the Moffat detector
                          (--moffat-alpha / --moffat-beta match pipeline defaults)
  Stage 3  RANSAC      -- guiding-error transforms; also written to CSV so the
                          11-column path (skip detect+RANSAC) can be tested
  Stage 4  Lanczos     -- non-trivial sub-pixel offsets + rotations
  Stage 5  Integration -- noticeable shot noise + read noise → visible SNR gain

Optionally generates calibration frames (dark, bias, flat, darkflat) that can
be passed directly to --dark / --bias / --flat / --darkflat CLI flags.

Usage examples
--------------
  # Default: 10 frames, 11-column CSV (pre-computed H), luminance FITS
  python generate_test_frames.py -o test_data/

  # 2-column CSV only → exercises star detect + RANSAC at runtime
  python generate_test_frames.py -n 15 --no-homography -o test_noH/

  # Bayer raw frames (RGGB) + calibration sets
  python generate_test_frames.py --bayer rggb --gen-calibration -o raw_test/

  # Larger sensor, heavy noise, strong guiding drift
  python generate_test_frames.py -n 20 --width 2048 --height 1536 \
      --read-noise 40 --max-shift 60 --max-rotation 2.5 -o large_test/

SNR notes
---------
  Single-frame background noise σ ≈ sqrt(sky_bg + read_noise²)
  Single-frame bright-star peak  ≈ star_flux_max / (π · alpha²)   (Moffat core)
  After stacking N frames         : signal scales ×N, noise ×sqrt(N)
  → effective SNR gain            : sqrt(N)
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate synthetic FITS test frames for gpu-dso-stacker.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Frame count
    p.add_argument("-n", "--num-frames", type=int, default=10,
                   help="Number of light frames to generate")
    p.add_argument("-s", "--num-stars",  type=int, default=120,
                   help="Approximate number of stars per frame")
    p.add_argument("-o", "--output-dir", type=str, default="synthetic_frames",
                   help="Output directory for FITS files and CSV")

    # Sensor geometry
    p.add_argument("--width",  type=int, default=1024, help="Image width  in pixels")
    p.add_argument("--height", type=int, default=768,  help="Image height in pixels")

    # Bayer pattern (optional; enables mosaic generation)
    p.add_argument("--bayer", choices=["none", "rggb", "bggr", "grbg", "gbrg"],
                   default="none",
                   help="CFA pattern; 'none' = monochrome luminance output.  "
                        "Sets BAYERPAT in the FITS header and interleaves colour channels.")

    # Noise model
    p.add_argument("--sky-bg",     type=float, default=500.0,
                   help="Uniform sky background level in ADU")
    p.add_argument("--read-noise", type=float, default=20.0,
                   help="Gaussian read-noise sigma in ADU (deliberately visible)")

    # Star brightness
    p.add_argument("--star-flux-min", type=float, default=4000.0,
                   help="Faintest star integrated flux in ADU")
    p.add_argument("--star-flux-max", type=float, default=60000.0,
                   help="Brightest star integrated flux in ADU")

    # PSF — match pipeline defaults so Moffat detection works in 2-col mode
    p.add_argument("--moffat-alpha", type=float, default=2.5,
                   help="Moffat PSF alpha (FWHM ≈ 2·alpha·√(2^(1/beta)−1))")
    p.add_argument("--moffat-beta",  type=float, default=2.0,
                   help="Moffat PSF beta (wing slope)")

    # Guiding-error model
    p.add_argument("--max-shift",    type=float, default=25.0,
                   help="Max random translation offset per frame in pixels")
    p.add_argument("--max-rotation", type=float, default=0.8,
                   help="Max random rotation per frame in degrees")
    p.add_argument("--drift-speed",  type=float, default=0.0,
                   help="Linear drift added per frame in px/frame (simulates "
                        "polar-alignment error; adds to random jitter)")

    # Calibration frame generation
    p.add_argument("--gen-calibration", action="store_true",
                   help="Also generate dark, bias, and flat calibration frames "
                        "and write matching text-list files for --dark / --bias / --flat")
    p.add_argument("--num-calib-frames", type=int, default=5,
                   help="Number of calibration frames per type (dark, bias, flat)")
    p.add_argument("--dark-current",  type=float, default=50.0,
                   help="Mean thermal dark current in ADU (for dark frames)")
    p.add_argument("--hot-pixel-frac",type=float, default=0.001,
                   help="Fraction of pixels that are 'hot' (bright defects)")
    p.add_argument("--vignetting",    type=float, default=0.25,
                   help="Vignetting strength: 0=none, 1=100%% darkening at corner")

    # Output options
    p.add_argument("--csv-name",      type=str, default="frames.csv",
                   help="Filename of the output CSV (placed inside output-dir)")
    p.add_argument("--no-homography", action="store_true",
                   help="Write a 2-column CSV (no pre-computed homographies) to "
                        "exercise the star-detect + RANSAC pipeline stages")
    p.add_argument("--no-star-colors", action="store_true",
                   help="Render all stars as white (no colour variation).  "
                        "In Bayer mode colours are preserved in the mosaic; in "
                        "luminance mode they only affect brightness via ITU weights.")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Moffat PSF helpers
# ---------------------------------------------------------------------------

def moffat_kernel(alpha, beta, radius=None):
    """Return a normalised Moffat kernel array of shape (2r+1, 2r+1)."""
    if radius is None:
        radius = min(int(np.ceil(3.0 * alpha)), 15)
    d = 2 * radius + 1
    ij = np.arange(-radius, radius + 1, dtype=np.float64)
    gx, gy = np.meshgrid(ij, ij)
    k = (1.0 + (gx**2 + gy**2) / alpha**2) ** (-beta)
    return k / k.sum(), radius


def render_moffat_stars(width, height, xs, ys, fluxes, alpha, beta, weights=None):
    """
    Render stars with Moffat PSF on a zero-background float64 image.

    weights : 1-D array of length len(xs), or None (= all 1.0).
        Scales each star's flux independently.  Pass a single colour channel
        column from sample_star_colors() to render one R/G/B plane.
    """
    psf, radius = moffat_kernel(alpha, beta)
    img = np.zeros((height, width), dtype=np.float64)
    if weights is None:
        weights = np.ones(len(xs), dtype=np.float64)
    for x, y, flux, w in zip(xs, ys, fluxes, weights):
        # Sub-pixel offset from nearest integer centre
        xi, yi = int(round(x)), int(round(y))
        dx_sub  = x - xi
        dy_sub  = y - yi
        x0 = max(0, xi - radius);  x1 = min(width,  xi + radius + 1)
        y0 = max(0, yi - radius);  y1 = min(height, yi + radius + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        # Evaluate Moffat centred at sub-pixel position
        gx = np.arange(x0, x1) - (xi + dx_sub)
        gy = np.arange(y0, y1) - (yi + dy_sub)
        gxx, gyy = np.meshgrid(gx, gy)
        p = (1.0 + (gxx**2 + gyy**2) / alpha**2) ** (-beta)
        s = p.sum()
        if s > 0:
            img[y0:y1, x0:x1] += flux * w * p / s
    return img


# ---------------------------------------------------------------------------
# Bayer mosaic helpers
# ---------------------------------------------------------------------------

# Maps pattern name → function (row, col) → channel index {0=R, 1=G, 2=B}
_BAYER_CHANNEL = {
    "rggb": lambda r, c: [0, 1, 1, 2][(r % 2) * 2 + (c % 2)],
    "bggr": lambda r, c: [2, 1, 1, 0][(r % 2) * 2 + (c % 2)],
    "grbg": lambda r, c: [1, 0, 2, 1][(r % 2) * 2 + (c % 2)],
    "gbrg": lambda r, c: [1, 2, 0, 1][(r % 2) * 2 + (c % 2)],
}

def make_bayer_mosaic(r_img, g_img, b_img, pattern):
    """
    Interleave three same-shape float64 channel images into a Bayer mosaic.
    The ITU-R BT.709 weights (R·0.2126 + G·0.7152 + B·0.0722) mean that for
    white stars rendered with equal R=G=B=flux, the debayered luminance ≈ flux.
    """
    h, w  = r_img.shape
    mosaic = np.empty((h, w), dtype=np.float64)

    chan_fn = _BAYER_CHANNEL[pattern]
    channels = [r_img, g_img, b_img]

    # Vectorised assignment for each of the four sub-patterns
    r_even = np.arange(0, h, 2)
    r_odd  = np.arange(1, h, 2)
    c_even = np.arange(0, w, 2)
    c_odd  = np.arange(1, w, 2)

    def assign(rows, cols):
        # Determine which channel this (row_parity, col_parity) maps to
        ch = chan_fn(rows[0] if len(rows) else 0, cols[0] if len(cols) else 0)
        rr, cc = np.ix_(rows, cols)
        mosaic[rr, cc] = channels[ch][rr, cc]

    assign(r_even, c_even)   # (even, even)
    assign(r_even, c_odd)    # (even, odd)
    assign(r_odd,  c_even)   # (odd,  even)
    assign(r_odd,  c_odd)    # (odd,  odd)
    return mosaic


# ---------------------------------------------------------------------------
# Noise + frame assembly
# ---------------------------------------------------------------------------

def add_noise(signal_img, sky_bg, read_noise, rng):
    """
    Add Poisson shot noise (photon statistics) + Gaussian read noise.
    signal_img is float64, no background yet added.
    Returns float32.
    """
    img = signal_img + sky_bg
    img = rng.poisson(np.maximum(img, 0)).astype(np.float64)
    img += rng.normal(0.0, read_noise, img.shape)
    return img.astype(np.float32)


# ---------------------------------------------------------------------------
# Homography helpers
# ---------------------------------------------------------------------------

def make_backward_homography(dx, dy, angle_deg, cx, cy):
    """
    Return the 3×3 backward homography H (ref → src).

    A star at (x_ref, y_ref) in the reference frame appears at
    H @ [x_ref, y_ref, 1]^T in this frame (telescope drifted by (dx,dy) and
    rotated angle_deg about the image centre).  This is exactly the backward
    map the pipeline uses directly for pixel sampling (no inversion needed).
    """
    a     = np.deg2rad(angle_deg)
    cos_a = np.cos(a)
    sin_a = np.sin(a)
    H = np.array([
        [cos_a, -sin_a,  cx * (1.0 - cos_a) + cy * sin_a + dx],
        [sin_a,  cos_a,  cy * (1.0 - cos_a) - cx * sin_a + dy],
        [0.0,    0.0,    1.0],
    ], dtype=np.float64)
    return H


def apply_homography(H, xs, ys):
    """Apply H to coordinate arrays; returns (x_out, y_out)."""
    pts = np.stack([xs, ys, np.ones_like(xs)])   # 3 × N
    out = H @ pts
    out /= out[2:3, :]
    return out[0], out[1]


# ---------------------------------------------------------------------------
# Star field
# ---------------------------------------------------------------------------

def make_star_field(width, height, n_stars, rng, flux_min, flux_max, margin=80):
    """
    Random star positions within the margin and exponential flux distribution
    (many faint stars, few bright ones — mimics stellar luminosity function).
    """
    xs     = rng.uniform(margin, width  - margin, n_stars)
    ys     = rng.uniform(margin, height - margin, n_stars)
    raw    = rng.exponential(1.0, n_stars)
    raw   /= raw.max()
    fluxes = flux_min + raw * (flux_max - flux_min)
    return xs, ys, fluxes


# ---------------------------------------------------------------------------
# Stellar colour model
# ---------------------------------------------------------------------------

# Control points: (B-V colour index, R weight, G weight, B weight)
# Normalised so the brightest channel = 1.0 per spectral class.
#   B-V ≈ -0.3  O/B-type  blue-white hot stars
#   B-V ≈  0.0  A-type    white      (Vega / Sirius)
#   B-V ≈ +0.6  G-type    yellow     (Sun)
#   B-V ≈ +1.0  K-type    orange
#   B-V ≈ +1.6  M-type    deep red
_BV_PTS = np.array([-0.3,  0.0,  0.6,  1.0,  1.6])
_R_PTS  = np.array([ 0.55, 1.00, 1.00, 1.00, 1.00])
_G_PTS  = np.array([ 0.70, 1.00, 0.95, 0.75, 0.55])
_B_PTS  = np.array([ 1.00, 1.00, 0.72, 0.45, 0.25])


def sample_star_colors(n_stars, rng):
    """
    Sample approximate (R, G, B) colour-weight tuples for n_stars using a
    B-V colour-index model.  Distribution is skewed toward G/K-type stars
    (Beta(2.5, 1.5) over [-0.3, 1.6]).  Returns float64 array of shape (N, 3).

    These weights are applied per-channel when rendering in Bayer mode:
        r_signal = flux * colors[:, 0]
        g_signal = flux * colors[:, 1]
        b_signal = flux * colors[:, 2]

    In luminance mode the effective weight is the ITU-R BT.709 dot product:
        lum_w = 0.2126·R + 0.7152·G + 0.0722·B
    """
    bv = rng.beta(2.5, 1.5, n_stars) * 1.9 - 0.3   # ∈ [-0.3, 1.6]
    r  = np.interp(bv, _BV_PTS, _R_PTS)
    g  = np.interp(bv, _BV_PTS, _G_PTS)
    b  = np.interp(bv, _BV_PTS, _B_PTS)
    return np.stack([r, g, b], axis=1)   # (N, 3), values in [0, 1]


# ---------------------------------------------------------------------------
# Calibration frame generation
# ---------------------------------------------------------------------------

def make_bias_frame(width, height, bias_level, read_noise, rng):
    """Zero-exposure: bias pedestal + read noise, no dark current, no signal."""
    img = np.full((height, width), bias_level, dtype=np.float64)
    img += rng.normal(0.0, read_noise, img.shape)
    return img.astype(np.float32)


def make_dark_frame(width, height, bias_level, dark_current, read_noise,
                    hot_pixel_frac, rng):
    """
    Dark frame: bias + thermal current (Poisson) + read noise + hot pixels.
    hot_pixel_frac fraction of pixels are permanently elevated (~5× dark level).
    """
    img = np.full((height, width), bias_level + dark_current, dtype=np.float64)
    # Poisson thermal noise
    img = rng.poisson(np.maximum(img, 0)).astype(np.float64)
    img += rng.normal(0.0, read_noise, img.shape)
    # Hot pixels
    n_hot = max(1, int(hot_pixel_frac * width * height))
    hx = rng.integers(0, width,  n_hot)
    hy = rng.integers(0, height, n_hot)
    img[hy, hx] += rng.uniform(5.0, 20.0, n_hot) * dark_current
    return img.astype(np.float32)


def make_flat_frame(width, height, sky_bg, read_noise, vignetting, rng):
    """
    Flat frame: uniform illumination with radial vignetting + bias pedestal +
    Poisson shot noise + read noise.  Returns raw ADU values — the C calibration
    code (calibration.c) subtracts bias and normalises each flat frame itself
    before stacking, so we must NOT pre-normalise here.

    Illumination level is sky_bg * 10 so flats are well-exposed relative to
    the bias pedestal (sky_bg * 0.5).
    """
    cx, cy = width / 2.0, height / 2.0
    r_max  = np.sqrt(cx**2 + cy**2)
    ys, xs = np.mgrid[0:height, 0:width]
    r      = np.sqrt((xs - cx)**2 + (ys - cy)**2) / r_max
    bias_level = sky_bg * 0.5
    # Cosine^4-like vignetting on top of a bias pedestal
    illum  = bias_level + (1.0 - vignetting * r**2) * sky_bg * 10.0
    img    = rng.poisson(np.maximum(illum, 0)).astype(np.float64)
    img   += rng.normal(0.0, read_noise, img.shape)
    return img.astype(np.float32)


def write_fits(path, data, header_kw=None):
    """Write float32 array as FITS BITPIX=-32."""
    hdr = fits.Header()
    hdr["SIMPLE"] = True
    hdr["BITPIX"] = -32
    hdr["NAXIS"]  = 2
    hdr["NAXIS1"] = data.shape[1]
    hdr["NAXIS2"] = data.shape[0]
    if header_kw:
        for k, v in header_kw.items():
            hdr[k] = v
    fits.writeto(str(path), data, header=hdr, overwrite=True)


def write_text_list(path, fits_paths):
    """Write a newline-separated list of absolute FITS paths (for --dark etc.)."""
    with open(path, "w") as f:
        for p in fits_paths:
            f.write(str(p) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    rng    = np.random.default_rng(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    IW, IH = args.width, args.height   # image dimensions (not to be confused with homography H)
    cx, cy = IW / 2.0, IH / 2.0
    use_bayer = (args.bayer != "none")

    # --- SNR preview ---
    bg_noise  = np.sqrt(args.sky_bg + args.read_noise**2)
    # Moffat core peak for normalised PSF ≈ 1 / (π · alpha²) (approx)
    peak_frac = 1.0 / (np.pi * args.moffat_alpha**2)
    peak_adu  = args.star_flux_max * peak_frac
    snr1      = peak_adu / bg_noise
    snrN      = snr1 * np.sqrt(args.num_frames)

    print(f"gpu-dso-stacker synthetic frame generator")
    print(f"  Sensor       : {IW}×{IH}  Bayer={args.bayer.upper()}")
    print(f"  Frames       : {args.num_frames}  Stars/frame≈{args.num_stars}")
    print(f"  Noise        : sky={args.sky_bg:.0f} ADU  read={args.read_noise:.1f} ADU"
          f"  → bg σ≈{bg_noise:.1f} ADU/px")
    print(f"  PSF          : Moffat α={args.moffat_alpha} β={args.moffat_beta}"
          f"  (matches pipeline detector defaults)")
    print(f"  Guiding err  : ±{args.max_shift:.0f} px  ±{args.max_rotation:.2f}°")
    print(f"  Bright-star SNR : {snr1:.1f}× (single) → {snrN:.1f}× (×{args.num_frames} stack)")
    print()

    # --- Reference star field (defined in reference-frame coordinates) ---
    ref_xs, ref_ys, fluxes = make_star_field(
        IW, IH, args.num_stars, rng,
        args.star_flux_min, args.star_flux_max,
    )

    # Star colours — sampled once, fixed across all frames
    use_colors = not args.no_star_colors
    if use_colors:
        star_colors = sample_star_colors(args.num_stars, rng)   # (N, 3): R, G, B weights ∈ [0,1]
        # ITU-R BT.709 luminance weight per star (used in non-Bayer mode)
        lum_weights = (  0.2126 * star_colors[:, 0]
                       + 0.7152 * star_colors[:, 1]
                       + 0.0722 * star_colors[:, 2])
    else:
        star_colors = None
        lum_weights = None

    fits_paths   = []
    homographies = []

    print("  Generating light frames …")
    for i in range(args.num_frames):
        is_ref = (i == 0)

        if is_ref:
            H       = np.eye(3, dtype=np.float64)
            dx = dy = angle = 0.0
        else:
            dx    = rng.uniform(-args.max_shift,    args.max_shift)
            dy    = rng.uniform(-args.max_shift,    args.max_shift)
            angle = rng.uniform(-args.max_rotation, args.max_rotation)
            dx   += args.drift_speed * i
            dy   += args.drift_speed * i * 0.3
            H     = make_backward_homography(dx, dy, angle, cx, cy)

        # Transform star positions into this frame's coordinate system
        fxs, fys = apply_homography(H, ref_xs, ref_ys)

        if use_bayer:
            # Render each colour channel separately with per-star colour weights.
            # ITU-R BT.709: L = 0.2126·R + 0.7152·G + 0.0722·B
            # White stars (no-color mode): R=G=B=signal → L=signal (perfect round-trip).
            rw = star_colors[:, 0] if use_colors else None
            gw = star_colors[:, 1] if use_colors else None
            bw = star_colors[:, 2] if use_colors else None
            r_sig = render_moffat_stars(IW, IH, fxs, fys, fluxes,
                                        args.moffat_alpha, args.moffat_beta, rw)
            g_sig = render_moffat_stars(IW, IH, fxs, fys, fluxes,
                                        args.moffat_alpha, args.moffat_beta, gw)
            b_sig = render_moffat_stars(IW, IH, fxs, fys, fluxes,
                                        args.moffat_alpha, args.moffat_beta, bw)
            r_ch  = add_noise(r_sig, args.sky_bg, args.read_noise, rng)
            g_ch  = add_noise(g_sig, args.sky_bg, args.read_noise, rng)
            b_ch  = add_noise(b_sig, args.sky_bg, args.read_noise, rng)
            frame = make_bayer_mosaic(r_ch.astype(np.float64),
                                      g_ch.astype(np.float64),
                                      b_ch.astype(np.float64),
                                      args.bayer).astype(np.float32)
        else:
            # Luminance mode: weight each star's flux by its ITU-R luminance weight.
            # White (no-color) stars have lum_weight = 1.0 — no effect.
            signal = render_moffat_stars(IW, IH, fxs, fys, fluxes,
                                         args.moffat_alpha, args.moffat_beta,
                                         lum_weights)
            frame = add_noise(signal, args.sky_bg, args.read_noise, rng)

        fname = f"frame_{i:04d}.fits"
        fpath = out_dir / fname
        kw = {
            "EXPTIME":  (30.0,         "exposure time seconds"),
            "ISREF":    (int(is_ref),  "1=reference frame"),
            "GUIDEDX":  (round(float(dx),    4), "guiding offset X px"),
            "GUIDEDY":  (round(float(dy),    4), "guiding offset Y px"),
            "GUIDEROT": (round(float(angle), 6), "guiding rotation deg"),
            "SKYBG":    (args.sky_bg,   "sky background ADU"),
            "RDNOISE":  (args.read_noise,"read noise sigma ADU"),
        }
        if use_bayer:
            kw["BAYERPAT"] = (args.bayer.upper(), "Bayer CFA pattern")

        write_fits(fpath, frame, kw)
        fits_paths.append(str(fpath.resolve()))
        homographies.append(H)

        tag = "[REF]" if is_ref else f"dx={dx:+6.1f} dy={dy:+6.1f} rot={angle:+5.2f}°"
        print(f"    {fname}  {tag}")

    # --- Calibration frames (optional) ---
    bias_list_path = dark_list_path = flat_list_path = None

    if args.gen_calibration:
        print()
        print("  Generating calibration frames …")
        calib_dir = out_dir / "calib"
        calib_dir.mkdir(exist_ok=True)

        bias_paths = []
        dark_paths = []
        flat_paths = []

        bias_level = args.sky_bg * 0.5   # bias pedestal is half the sky background

        for j in range(args.num_calib_frames):
            # Bias
            bf    = make_bias_frame(IW, IH, bias_level, args.read_noise, rng)
            bpath = calib_dir / f"bias_{j:04d}.fits"
            write_fits(bpath, bf, {"FRAMTYPE": "BIAS", "EXPTIME": 0.0})
            bias_paths.append(str(bpath.resolve()))

            # Dark (same exposure as light frames: 30 s)
            df    = make_dark_frame(IW, IH, bias_level, args.dark_current,
                                    args.read_noise, args.hot_pixel_frac, rng)
            dpath = calib_dir / f"dark_{j:04d}.fits"
            write_fits(dpath, df, {"FRAMTYPE": "DARK", "EXPTIME": 30.0})
            dark_paths.append(str(dpath.resolve()))

            # Flat
            ff    = make_flat_frame(IW, IH, args.sky_bg, args.read_noise,
                                    args.vignetting, rng)
            fpath = calib_dir / f"flat_{j:04d}.fits"
            kw_f  = {"FRAMTYPE": "FLAT", "EXPTIME": 5.0}
            if use_bayer:
                kw_f["BAYERPAT"] = args.bayer.upper()
            write_fits(fpath, ff, kw_f)
            flat_paths.append(str(fpath.resolve()))

            print(f"    bias_{j:04d}.fits  dark_{j:04d}.fits  flat_{j:04d}.fits")

        # Write text-list files for --bias / --dark / --flat
        bias_list_path = out_dir / "bias_list.txt"
        dark_list_path = out_dir / "dark_list.txt"
        flat_list_path = out_dir / "flat_list.txt"
        write_text_list(bias_list_path, bias_paths)
        write_text_list(dark_list_path, dark_paths)
        write_text_list(flat_list_path, flat_paths)
        print(f"    → bias_list.txt  dark_list.txt  flat_list.txt")

    # --- CSV ---
    csv_path = out_dir / args.csv_name
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        if args.no_homography:
            w.writerow(["filepath", "is_reference"])
            for path, i in zip(fits_paths, range(args.num_frames)):
                w.writerow([path, 1 if i == 0 else 0])
        else:
            w.writerow([
                "filepath", "is_reference",
                "h00", "h01", "h02",
                "h10", "h11", "h12",
                "h20", "h21", "h22",
            ])
            for path, H, i in zip(fits_paths, homographies, range(args.num_frames)):
                h = H.flatten()
                w.writerow([
                    path, 1 if i == 0 else 0,
                    f"{h[0]:.10f}", f"{h[1]:.10f}", f"{h[2]:.10f}",
                    f"{h[3]:.10f}", f"{h[4]:.10f}", f"{h[5]:.10f}",
                    f"{h[6]:.10f}", f"{h[7]:.10f}", f"{h[8]:.10f}",
                ])

    # --- Usage summary ---
    mode = "2-column (no H)" if args.no_homography else "11-column (with H)"
    bayer_flag = f" --bayer {args.bayer}" if use_bayer else ""
    print()
    print(f"  CSV ({mode})  →  {csv_path}")
    print()
    print("  dso_stacker commands:")

    base = f"./build/dso_stacker -f {csv_path} -o {out_dir}/stacked.fits{bayer_flag}"
    print(f"    GPU (default)  : {base}")
    print(f"    CPU path       : {base} --cpu")
    if args.no_homography:
        print(f"    (star-detect + RANSAC will run automatically for 2-column CSV)")
    if args.gen_calibration:
        calib_flags = (f" --bias {bias_list_path}"
                       f" --dark {dark_list_path}"
                       f" --flat {flat_list_path}"
                       f" --save-master-frames {out_dir}/masters")
        print(f"    With calibration: {base}{calib_flags}")


if __name__ == "__main__":
    main()
