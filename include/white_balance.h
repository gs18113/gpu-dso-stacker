/*
 * white_balance.h — Per-channel white balance for raw Bayer mosaics.
 *
 * White balance corrects for unequal channel sensitivity in color sensors.
 * Multipliers are applied to the raw Bayer mosaic *after* calibration
 * (dark/flat) and *before* debayering, so each pixel is scaled by exactly
 * one channel multiplier corresponding to its CFA position.
 *
 * Three modes:
 *   WB_CAMERA  — use camera white balance from file metadata (LibRaw cam_mul)
 *   WB_AUTO    — gray-world assumption: scale channels so mean R ≈ mean G ≈ mean B
 *   WB_MANUAL  — user-specified per-channel multipliers
 *
 * BAYER_NONE (monochrome) sensors skip white balance entirely.
 */

#pragma once

#include "dso_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * WbMode — white balance strategy selection.
 * ------------------------------------------------------------------------- */
typedef enum {
    WB_NONE   = 0,   /* no white balance applied                          */
    WB_CAMERA = 1,   /* camera metadata (cam_mul from RAW / FITS keywords)*/
    WB_AUTO   = 2,   /* gray-world: mean_G / mean_{R,B}                  */
    WB_MANUAL = 3    /* user-specified r_mul, g_mul, b_mul                */
} WbMode;

/* -------------------------------------------------------------------------
 * WbParams — white balance configuration for PipelineConfig.
 * ------------------------------------------------------------------------- */
typedef struct {
    WbMode mode;
    float  r_mul;    /* red   multiplier (used when mode == WB_MANUAL)    */
    float  g_mul;    /* green multiplier (used when mode == WB_MANUAL)    */
    float  b_mul;    /* blue  multiplier (used when mode == WB_MANUAL)    */
} WbParams;

/* -------------------------------------------------------------------------
 * bayer_color — return the color channel (0=R, 1=G, 2=B) for a pixel at
 *               position (x, y) given a Bayer pattern.
 *
 * Layout for each pattern (2x2 superpixel):
 *   RGGB: R G   BGGR: B G   GRBG: G R   GBRG: G B
 *         G B         G R         B G         R G
 * ------------------------------------------------------------------------- */
static inline int bayer_color(BayerPattern pat, int x, int y)
{
    int xm = x & 1;
    int ym = y & 1;
    /*
     * Encode the 2x2 CFA as a 4-element lookup: index = ym*2 + xm
     *   [0]=top-left  [1]=top-right  [2]=bot-left  [3]=bot-right
     * Values: 0=R, 1=G, 2=B
     */
    static const int lut[5][4] = {
        {1, 1, 1, 1},  /* BAYER_NONE  — should not be called, but safe  */
        {0, 1, 1, 2},  /* BAYER_RGGB                                    */
        {2, 1, 1, 0},  /* BAYER_BGGR                                    */
        {1, 0, 2, 1},  /* BAYER_GRBG                                    */
        {1, 2, 0, 1},  /* BAYER_GBRG                                    */
    };
    return lut[(int)pat][ym * 2 + xm];
}

/* -------------------------------------------------------------------------
 * wb_apply_bayer — apply white balance multipliers to a raw Bayer mosaic.
 *
 * Each pixel is multiplied in-place by the channel multiplier matching its
 * CFA position.  Returns DSO_OK on success, DSO_ERR_INVALID_ARG on bad
 * input.  BAYER_NONE → immediate return (no-op).
 * ------------------------------------------------------------------------- */
DsoError wb_apply_bayer(float *data, int W, int H,
                        BayerPattern pattern,
                        float r_mul, float g_mul, float b_mul);

/* -------------------------------------------------------------------------
 * wb_auto_compute — compute gray-world white balance multipliers.
 *
 * Averages each Bayer channel separately and returns multipliers that
 * equalise the channel means:
 *   *r_mul = mean_G / mean_R,  *g_mul = 1.0,  *b_mul = mean_G / mean_B
 *
 * Returns DSO_ERR_INVALID_ARG on bad input or BAYER_NONE.
 * Returns DSO_ERR_STAR_DETECT if a channel mean is zero (black frame).
 * ------------------------------------------------------------------------- */
DsoError wb_auto_compute(const float *data, int W, int H,
                         BayerPattern pattern,
                         float *r_mul, float *g_mul, float *b_mul);

#ifdef __cplusplus
}
#endif
