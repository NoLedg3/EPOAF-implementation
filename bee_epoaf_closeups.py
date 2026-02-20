#!/usr/bin/env python3
"""
Improved bee closeup pipeline:
  1. Aggressive local contrast (CLAHE, per-crop adaptive clip)
  2. Gamma midtone boost
  3. NL-Means pre-denoising  (kills grain before amplification)
  4. Adaptive CED  (Weickert, tensor re-estimated mid-diffusion)
  5. Bilateral post-smoothing  (clean flat regions)
  6. Edge-adaptive unsharp masking  (sharpen structure, not noise)
  7. Final stretch + second CLAHE pass

Outputs:
  bee_closeups_epoaf/          – individual after PNGs (320×320)
  bee_closeups_comparisons/    – side-by-side BEFORE | AFTER (640×352)
  bee_montage_comparisons.png  – full grid (8 columns)
"""

import cv2
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# CED / EPOAF primitives
# ─────────────────────────────────────────────────────────────────────────────

def structure_tensor(gray_f32, sigma=1.0, rho=4.0):
    """Compute smoothed structure tensor at scales sigma (inner) and rho (outer)."""
    ks = max(3, 2 * int(3 * sigma) + 1)
    sm = cv2.GaussianBlur(gray_f32, (ks, ks), sigma)
    gx = cv2.Scharr(sm, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(sm, cv2.CV_32F, 0, 1)
    kr = max(3, 2 * int(3 * rho) + 1)
    Jxx = cv2.GaussianBlur(gx * gx, (kr, kr), rho)
    Jxy = cv2.GaussianBlur(gx * gy, (kr, kr), rho)
    Jyy = cv2.GaussianBlur(gy * gy, (kr, kr), rho)
    return Jxx, Jxy, Jyy


def diffusion_tensor(Jxx, Jxy, Jyy, alpha=0.00005, C=1e-6):
    """Build Weickert CED diffusion tensor from structure tensor."""
    trace = Jxx + Jyy
    disc  = np.sqrt(np.maximum((Jxx - Jyy) ** 2 + 4.0 * Jxy ** 2, 0.0))
    lam1  = 0.5 * (trace + disc)
    lam2  = 0.5 * (trace - disc)

    # Major eigenvector (normal to edge)
    v1x = Jxy
    v1y = lam1 - Jxx
    norm = np.sqrt(v1x ** 2 + v1y ** 2 + 1e-12)
    v1x /= norm;  v1y /= norm
    # Tangent direction (along edge)
    v2x = -v1y;   v2y = v1x

    diff = np.maximum(lam1 - lam2, 0.0)
    mu1  = np.full_like(lam1, alpha)
    mu2  = alpha + (1.0 - alpha) * np.exp(-C / (diff ** 2 + 1e-40))

    Dxx = mu1 * v1x * v1x + mu2 * v2x * v2x
    Dxy = mu1 * v1x * v1y + mu2 * v2x * v2y
    Dyy = mu1 * v1y * v1y + mu2 * v2y * v2y
    return Dxx, Dxy, Dyy


def ced_step(u, Dxx, Dxy, Dyy, dt=0.15):
    ux = np.gradient(u, axis=1)
    uy = np.gradient(u, axis=0)
    fx = Dxx * ux + Dxy * uy
    fy = Dxy * ux + Dyy * uy
    return u + dt * (np.gradient(fx, axis=1) + np.gradient(fy, axis=0))


def adaptive_ced(gray_u8, n_iter=30, dt=0.15,
                 sigma=1.2, rho=4.5, alpha=0.00005, C=1e-6,
                 recompute=6):
    """
    CED with periodic structure tensor recomputation so the
    diffusion adapts as the image evolves.
    Returns float32 in [0, 255].
    """
    u = gray_u8.astype(np.float32) / 255.0
    Dxx = Dxy = Dyy = None

    for i in range(n_iter):
        if i % recompute == 0:
            Jxx, Jxy, Jyy = structure_tensor(u, sigma=sigma, rho=rho)
            Dxx, Dxy, Dyy = diffusion_tensor(Jxx, Jxy, Jyy, alpha=alpha, C=C)
        u = np.clip(ced_step(u, Dxx, Dxy, Dyy, dt), 0.0, 1.0)

    return u * 255.0


# ─────────────────────────────────────────────────────────────────────────────
# Pre / post processing helpers
# ─────────────────────────────────────────────────────────────────────────────

def apply_gamma(gray_u8, gamma):
    lut = np.array([min(255, int(((i / 255.0) ** gamma) * 255))
                    for i in range(256)], dtype=np.uint8)
    return cv2.LUT(gray_u8, lut)


def adaptive_clahe(gray_u8, base_clip=3.5, tile=4):
    """
    Adaptive CLAHE: scale clipLimit by local contrast so very flat
    crops are enhanced more aggressively.
    """
    std = float(gray_u8.std())
    # Low-contrast crops get a stronger boost
    clip = base_clip * max(1.0, 12.0 / (std + 1e-3))
    clip = min(clip, 8.0)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return clahe.apply(gray_u8)


def edge_adaptive_usm(gray_u8, amount=1.8, sigma=1.8, edge_boost=2.5):
    """
    Unsharp mask that is stronger at detected edges and softer in
    flat/noisy regions — avoids amplifying grain.
    """
    f   = gray_u8.astype(np.float32)
    blr = cv2.GaussianBlur(f, (0, 0), sigma)
    hf  = f - blr  # high-frequency residual

    # Edge strength mask (normalised to [0, 1])
    gx  = cv2.Scharr(gray_u8, cv2.CV_32F, 1, 0)
    gy  = cv2.Scharr(gray_u8, cv2.CV_32F, 0, 1)
    emag = np.sqrt(gx ** 2 + gy ** 2)
    emax = np.percentile(emag, 95) + 1e-6
    emask = np.clip(emag / emax, 0, 1)
    # Smooth the mask so transitions aren't abrupt
    emask = cv2.GaussianBlur(emask, (0, 0), 3.0)

    # Base USM + extra boost at edges
    result = f + amount * hf + (edge_boost - amount) * emask * hf
    return np.clip(result, 0, 255).astype(np.uint8)


def full_pipeline(crop_gray, target=320,
                  # CED params
                  n_iter=30, dt=0.15, sigma=1.2, rho=4.5,
                  alpha=0.00005, C=1e-6, recompute=6,
                  # pre-denoise
                  nlm_h=9, nlm_template=7, nlm_search=21,
                  # contrast
                  clahe_clip=3.5, gamma=0.70,
                  # post
                  bilateral_d=9, bilateral_sc=35, bilateral_ss=35,
                  usm_amount=1.8, usm_sigma=1.8, usm_edge=2.5,
                  final_clip=2.5):
    """
    Full enhanced pipeline on a raw grayscale crop.
    Returns a target×target uint8 image.
    """
    # ── Phase 0: upscale to working resolution ──────────────────────────────
    h, w = crop_gray.shape
    min_dim = min(h, w)
    scale = (target * 1.5) / min_dim   # work at 1.5× target for headroom
    nw, nh = max(target, int(w * scale)), max(target, int(h * scale))
    work = cv2.resize(crop_gray, (nw, nh), interpolation=cv2.INTER_LANCZOS4)

    # ── Phase 1: local contrast + gamma ─────────────────────────────────────
    work = adaptive_clahe(work, base_clip=clahe_clip, tile=4)
    work = apply_gamma(work, gamma)

    # ── Phase 2: NL-Means denoising (remove grain before amplification) ──────
    work = cv2.fastNlMeansDenoising(
        work, h=nlm_h,
        templateWindowSize=nlm_template,
        searchWindowSize=nlm_search)

    # ── Phase 3: Adaptive CED (EPOAF core) ───────────────────────────────────
    work_f = adaptive_ced(work, n_iter=n_iter, dt=dt,
                          sigma=sigma, rho=rho, alpha=alpha, C=C,
                          recompute=recompute)
    work = np.clip(work_f, 0, 255).astype(np.uint8)

    # ── Phase 4: Bilateral post-smoothing (flattens residual noise in cells) ──
    work = cv2.bilateralFilter(work, d=bilateral_d,
                               sigmaColor=bilateral_sc,
                               sigmaSpace=bilateral_ss)

    # ── Phase 5: Edge-adaptive unsharp masking ───────────────────────────────
    work = edge_adaptive_usm(work, amount=usm_amount,
                             sigma=usm_sigma, edge_boost=usm_edge)

    # ── Phase 6: Second CLAHE pass for final punch ───────────────────────────
    clahe2 = cv2.createCLAHE(clipLimit=final_clip, tileGridSize=(3, 3))
    work = clahe2.apply(work)

    # ── Phase 7: Resize to final target ──────────────────────────────────────
    return cv2.resize(work, (target, target), interpolation=cv2.INTER_LANCZOS4)


# ─────────────────────────────────────────────────────────────────────────────
# Comparison tile and montage builders
# ─────────────────────────────────────────────────────────────────────────────

def make_comparison_tile(before_gray, after_gray, label,
                         tile_w=300, tile_h=300):
    HEADER  = 36
    DIVIDER = 4
    W = tile_w * 2 + DIVIDER
    H = tile_h + HEADER

    tile = np.full((H, W, 3), 22, dtype=np.uint8)

    bef = cv2.resize(before_gray, (tile_w, tile_h), interpolation=cv2.INTER_LANCZOS4)
    aft = cv2.resize(after_gray,  (tile_w, tile_h), interpolation=cv2.INTER_LANCZOS4)

    tile[HEADER:, :tile_w]               = cv2.cvtColor(bef, cv2.COLOR_GRAY2BGR)
    tile[HEADER:, tile_w + DIVIDER:]     = cv2.cvtColor(aft, cv2.COLOR_GRAY2BGR)
    tile[HEADER:, tile_w:tile_w+DIVIDER] = (70, 70, 70)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(tile, "BEFORE", (6, HEADER - 8),
                font, 0.45, (150, 150, 150), 1)
    cv2.putText(tile, "AFTER",  (tile_w + DIVIDER + 6, HEADER - 8),
                font, 0.45, (160, 220, 160), 1)
    cv2.putText(tile, label,
                (tile_w // 2 - 38, HEADER - 8),
                font, 0.42, (255, 210, 80), 1)
    return tile


def build_montage(tiles, cols=8, gap=5):
    if not tiles:
        return None
    h, w = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    canvas = np.full(
        (rows * (h + gap) + gap, cols * (w + gap) + gap, 3),
        12, dtype=np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        y0 = gap + r * (h + gap)
        x0 = gap + c * (w + gap)
        canvas[y0:y0 + h, x0:x0 + w] = t
    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# Before pipeline: CLAHE only (honest baseline — no extra steps)
# ─────────────────────────────────────────────────────────────────────────────

def baseline_pipeline(crop_gray, target=320):
    """Simple CLAHE + upscale baseline for the BEFORE panel."""
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(4, 4))
    enhanced = clahe.apply(crop_gray)
    return cv2.resize(enhanced, (target, target), interpolation=cv2.INTER_LANCZOS4)


# ─────────────────────────────────────────────────────────────────────────────
# Predictions
# ─────────────────────────────────────────────────────────────────────────────

PREDICTIONS = [
    {"x": 1839.219, "y": 1150.174, "width": 65.277,  "height": 51.735, "confidence": 0.865, "detection_id": "08f65668"},
    {"x": 217.184,  "y": 1058.135, "width": 55.164,  "height": 65.585, "confidence": 0.854, "detection_id": "03df690c"},
    {"x": 27.175,   "y": 414.796,  "width": 54.35,   "height": 64.569, "confidence": 0.850, "detection_id": "efad2529"},
    {"x": 1702.12,  "y": 1298.439, "width": 63.061,  "height": 70.336, "confidence": 0.837, "detection_id": "236a143a"},
    {"x": 988.043,  "y": 1126.436, "width": 64.213,  "height": 43.999, "confidence": 0.836, "detection_id": "92d90d0b"},
    {"x": 879.936,  "y": 1136.46,  "width": 72.631,  "height": 54.422, "confidence": 0.835, "detection_id": "d3154521"},
    {"x": 1812.267, "y": 1312.663, "width": 61.007,  "height": 64.53,  "confidence": 0.833, "detection_id": "54362a5e"},
    {"x": 93.294,   "y": 460.846,  "width": 40.21,   "height": 80.371, "confidence": 0.827, "detection_id": "2428d318"},
    {"x": 1426.112, "y": 1009.083, "width": 68.293,  "height": 62.019, "confidence": 0.827, "detection_id": "0f07f7da"},
    {"x": 1166.212, "y": 1072.729, "width": 63.775,  "height": 65.74,  "confidence": 0.824, "detection_id": "ef9cf91d"},
    {"x": 839.258,  "y": 957.177,  "width": 47.793,  "height": 75.764, "confidence": 0.818, "detection_id": "0ab7b573"},
    {"x": 138.539,  "y": 1010.057, "width": 37.371,  "height": 79.05,  "confidence": 0.816, "detection_id": "3a82fa55"},
    {"x": 1388.839, "y": 1084.676, "width": 58.523,  "height": 62.306, "confidence": 0.816, "detection_id": "882638be"},
    {"x": 1141.263, "y": 1319.52,  "width": 60.595,  "height": 82.631, "confidence": 0.815, "detection_id": "bbe3b5a5"},
    {"x": 424.562,  "y": 311.131,  "width": 35.761,  "height": 86.413, "confidence": 0.814, "detection_id": "3e536675"},
    {"x": 352.652,  "y": 1062.947, "width": 53.246,  "height": 60.244, "confidence": 0.811, "detection_id": "4baa3ebb"},
    {"x": 367.173,  "y": 556.172,  "width": 38.193,  "height": 69.17,  "confidence": 0.811, "detection_id": "0f577023"},
    {"x": 965.585,  "y": 1062.219, "width": 68.631,  "height": 51.706, "confidence": 0.810, "detection_id": "31878fb3"},
    {"x": 327.034,  "y": 739.558,  "width": 66.367,  "height": 61.531, "confidence": 0.809, "detection_id": "d00fae47"},
    {"x": 1084.646, "y": 1024.918, "width": 72.762,  "height": 79.908, "confidence": 0.806, "detection_id": "e806d390"},
    {"x": 1465.833, "y": 226.072,  "width": 69.222,  "height": 59.37,  "confidence": 0.803, "detection_id": "d86c3dd4"},
    {"x": 192.349,  "y": 1126.144, "width": 68.613,  "height": 58.763, "confidence": 0.802, "detection_id": "dd58c5a1"},
    {"x": 134.222,  "y": 931.173,  "width": 33.881,  "height": 82.983, "confidence": 0.799, "detection_id": "e4ef736f"},
    {"x": 2013.304, "y": 1006.732, "width": 54.386,  "height": 51.244, "confidence": 0.798, "detection_id": "148969d7"},
    {"x": 516.432,  "y": 900.369,  "width": 59.932,  "height": 52.106, "confidence": 0.793, "detection_id": "e9867337"},
    {"x": 250.559,  "y": 969.872,  "width": 43.071,  "height": 85.89,  "confidence": 0.786, "detection_id": "f4c586ac"},
    {"x": 438.223,  "y": 1024.255, "width": 81.398,  "height": 55.403, "confidence": 0.783, "detection_id": "c06bdef3"},
    {"x": 1349.778, "y": 1138.958, "width": 80.564,  "height": 56.097, "confidence": 0.779, "detection_id": "a7a02ae2"},
    {"x": 697.002,  "y": 691.565,  "width": 70.915,  "height": 51.485, "confidence": 0.778, "detection_id": "7706cb59"},
    {"x": 1114.482, "y": 953.274,  "width": 50.174,  "height": 80.387, "confidence": 0.776, "detection_id": "75e6ea9b"},
    {"x": 724.424,  "y": 1096.199, "width": 62.973,  "height": 70.997, "confidence": 0.768, "detection_id": "3f24f68f"},
    {"x": 1239.435, "y": 1052.378, "width": 54.793,  "height": 71.751, "confidence": 0.766, "detection_id": "4644fea0"},
    {"x": 604.032,  "y": 402.59,   "width": 31.049,  "height": 79.048, "confidence": 0.766, "detection_id": "910a27a8"},
    {"x": 481.548,  "y": 509.315,  "width": 40.705,  "height": 64.056, "confidence": 0.761, "detection_id": "c3830566"},
    {"x": 1484.674, "y": 1039.335, "width": 49.219,  "height": 69.25,  "confidence": 0.760, "detection_id": "bd1289f5"},
    {"x": 587.051,  "y": 302.951,  "width": 34.098,  "height": 80.892, "confidence": 0.759, "detection_id": "e6b196c9"},
    {"x": 72.644,   "y": 560.15,   "width": 34.2,    "height": 78.977, "confidence": 0.757, "detection_id": "76a14863"},
    {"x": 1281.6,   "y": 1327.479, "width": 57.165,  "height": 85.663, "confidence": 0.757, "detection_id": "aedcdf1c"},
    {"x": 463.622,  "y": 865.852,  "width": 55.954,  "height": 53.021, "confidence": 0.756, "detection_id": "a3235c05"},
    {"x": 473.556,  "y": 686.275,  "width": 44.548,  "height": 76.909, "confidence": 0.754, "detection_id": "e7659855"},
    {"x": 504.083,  "y": 705.72,   "width": 44.459,  "height": 77.087, "confidence": 0.753, "detection_id": "77340864"},
    {"x": 88.579,   "y": 1122.673, "width": 61.187,  "height": 58.281, "confidence": 0.751, "detection_id": "55609d9d"},
    {"x": 1211.061, "y": 1296.422, "width": 47.796,  "height": 70.496, "confidence": 0.749, "detection_id": "b6b523ff"},
    {"x": 838.981,  "y": 421.527,  "width": 38.075,  "height": 70.075, "confidence": 0.749, "detection_id": "cbb1d3e3"},
    {"x": 1048.433, "y": 900.969,  "width": 46.466,  "height": 67.261, "confidence": 0.747, "detection_id": "ad72e408"},
    {"x": 1582.244, "y": 674.475,  "width": 55.145,  "height": 73.741, "confidence": 0.746, "detection_id": "08f65b7b"},
    {"x": 510.422,  "y": 1062.844, "width": 62.349,  "height": 55.697, "confidence": 0.745, "detection_id": "f26efbd2"},
    {"x": 234.156,  "y": 547.38,   "width": 84.567,  "height": 63.494, "confidence": 0.744, "detection_id": "8a279142"},
    {"x": 882.171,  "y": 857.573,  "width": 41.959,  "height": 84.571, "confidence": 0.743, "detection_id": "49470286"},
    {"x": 1013.82,  "y": 512.613,  "width": 31.752,  "height": 64.167, "confidence": 0.740, "detection_id": "7597ca95"},
    {"x": 1472.547, "y": 1166.38,  "width": 39.9,    "height": 25.619, "confidence": 0.738, "detection_id": "286db6ed"},
    {"x": 475.065,  "y": 272.731,  "width": 48.717,  "height": 83.956, "confidence": 0.733, "detection_id": "244cb622"},
    {"x": 591.781,  "y": 768.275,  "width": 61.187,  "height": 52.304, "confidence": 0.732, "detection_id": "efe6870d"},
    {"x": 1333.486, "y": 1065.032, "width": 39.251,  "height": 69.101, "confidence": 0.731, "detection_id": "d84ac267"},
    {"x": 1765.557, "y": 382.403,  "width": 32.317,  "height": 74.975, "confidence": 0.730, "detection_id": "6732a406"},
    {"x": 1416.125, "y": 381.988,  "width": 63.25,   "height": 68.556, "confidence": 0.728, "detection_id": "d318450b"},
    {"x": 446.764,  "y": 1109.717, "width": 45.528,  "height": 67.102, "confidence": 0.728, "detection_id": "37dbc7d5"},
    {"x": 629.073,  "y": 977.463,  "width": 58.011,  "height": 74.959, "confidence": 0.727, "detection_id": "afb09a81"},
    {"x": 31.379,   "y": 1047.774, "width": 62.757,  "height": 25.614, "confidence": 0.722, "detection_id": "2293395f"},
    {"x": 496.154,  "y": 403.616,  "width": 34.803,  "height": 70.65,  "confidence": 0.721, "detection_id": "b04bd999"},
    {"x": 70.673,   "y": 972.074,  "width": 50.481,  "height": 77.98,  "confidence": 0.721, "detection_id": "737668b1"},
    {"x": 374.953,  "y": 777.249,  "width": 62.506,  "height": 61.559, "confidence": 0.719, "detection_id": "78e8efe0"},
    {"x": 988.436,  "y": 218.412,  "width": 34.224,  "height": 58.79,  "confidence": 0.718, "detection_id": "9f28fba0"},
    {"x": 1378.299, "y": 813.526,  "width": 30.531,  "height": 61.492, "confidence": 0.718, "detection_id": "b0789fdc"},
    {"x": 1241.124, "y": 694.526,  "width": 52.784,  "height": 42.161, "confidence": 0.708, "detection_id": "bb18abd3"},
    {"x": 573.744,  "y": 536.303,  "width": 61.958,  "height": 65.111, "confidence": 0.707, "detection_id": "564a322b"},
    {"x": 555.05,   "y": 161.452,  "width": 36.177,  "height": 65.644, "confidence": 0.707, "detection_id": "d828cef9"},
    {"x": 1514.165, "y": 553.464,  "width": 63.954,  "height": 64.812, "confidence": 0.707, "detection_id": "3238d81e"},
    {"x": 1228.756, "y": 976.831,  "width": 42.347,  "height": 68.765, "confidence": 0.703, "detection_id": "87650633"},
    {"x": 740.915,  "y": 1114.916, "width": 53.654,  "height": 50.347, "confidence": 0.702, "detection_id": "879b1f7c"},
    {"x": 1636.136, "y": 690.304,  "width": 58.416,  "height": 63.598, "confidence": 0.701, "detection_id": "08c2e596"},
    {"x": 1696.256, "y": 1116.641, "width": 56.348,  "height": 50.024, "confidence": 0.697, "detection_id": "6a941572"},
    {"x": 1456.777, "y": 499.737,  "width": 85.024,  "height": 49.263, "confidence": 0.697, "detection_id": "ea1c99ec"},
    {"x": 1935.939, "y": 1152.118, "width": 45.244,  "height": 30.709, "confidence": 0.696, "detection_id": "95090c8c"},
    {"x": 1229.269, "y": 619.429,  "width": 56.776,  "height": 38.775, "confidence": 0.693, "detection_id": "70522bc0"},
    {"x": 800.005,  "y": 396.757,  "width": 51.745,  "height": 76.538, "confidence": 0.692, "detection_id": "f6b989fc"},
    {"x": 578.112,  "y": 621.676,  "width": 60.012,  "height": 80.323, "confidence": 0.691, "detection_id": "31dd8808"},
    {"x": 968.001,  "y": 933.907,  "width": 51.681,  "height": 64.763, "confidence": 0.688, "detection_id": "1b1e0424"},
    {"x": 654.152,  "y": 310.745,  "width": 34.947,  "height": 88.107, "confidence": 0.685, "detection_id": "c14f18b5"},
    {"x": 1350.994, "y": 1012.864, "width": 51.596,  "height": 75.727, "confidence": 0.685, "detection_id": "991b4cc2"},
    {"x": 1239.068, "y": 426.448,  "width": 62.833,  "height": 72.787, "confidence": 0.684, "detection_id": "c3a3463d"},
    {"x": 1414.786, "y": 586.153,  "width": 40.726,  "height": 70.38,  "confidence": 0.684, "detection_id": "9e99f72b"},
    {"x": 679.929,  "y": 226.091,  "width": 45.123,  "height": 81.113, "confidence": 0.680, "detection_id": "a533ae76"},
    {"x": 610.699,  "y": 878.95,   "width": 61.321,  "height": 50.767, "confidence": 0.678, "detection_id": "f928f762"},
    {"x": 865.047,  "y": 168.144,  "width": 74.01,   "height": 35.361, "confidence": 0.677, "detection_id": "c0a6a1a6"},
    {"x": 983.535,  "y": 455.919,  "width": 60.389,  "height": 39.673, "confidence": 0.675, "detection_id": "f330c76b"},
    {"x": 585.022,  "y": 195.582,  "width": 46.966,  "height": 63.486, "confidence": 0.673, "detection_id": "28b99ad8"},
    {"x": 982.889,  "y": 378.905,  "width": 61.552,  "height": 64.974, "confidence": 0.669, "detection_id": "0ad0fca7"},
    {"x": 1151.291, "y": 1103.154, "width": 56.43,   "height": 55.168, "confidence": 0.667, "detection_id": "72c1a435"},
    {"x": 340.961,  "y": 358.508,  "width": 72.952,  "height": 45.65,  "confidence": 0.662, "detection_id": "557fb7f3"},
    {"x": 523.832,  "y": 1236.693, "width": 84.415,  "height": 58.94,  "confidence": 0.655, "detection_id": "9c8a2469"},
    {"x": 624.78,   "y": 1107.402, "width": 75.593,  "height": 54.795, "confidence": 0.655, "detection_id": "b00cd0a3"},
    {"x": 455.083,  "y": 355.113,  "width": 29.038,  "height": 82.422, "confidence": 0.654, "detection_id": "0d29457b"},
    {"x": 779.626,  "y": 802.893,  "width": 45.897,  "height": 70.237, "confidence": 0.653, "detection_id": "e6533e0f"},
    {"x": 1612.234, "y": 1065.289, "width": 75.505,  "height": 54.46,  "confidence": 0.652, "detection_id": "aae9578e"},
    {"x": 1654.739, "y": 350.714,  "width": 48.587,  "height": 81.042, "confidence": 0.650, "detection_id": "cd051d60"},
    {"x": 731.789,  "y": 251.796,  "width": 42.041,  "height": 76.786, "confidence": 0.649, "detection_id": "d87f61d2"},
    {"x": 304.142,  "y": 961.649,  "width": 82.928,  "height": 47.769, "confidence": 0.648, "detection_id": "0fa3c784"},
    {"x": 1954.438, "y": 1056.899, "width": 33.248,  "height": 79.157, "confidence": 0.647, "detection_id": "7f78d55e"},
    {"x": 1951.575, "y": 247.448,  "width": 52.146,  "height": 53.326, "confidence": 0.646, "detection_id": "dcf8508d"},
    {"x": 1620.458, "y": 209.798,  "width": 63.101,  "height": 72.372, "confidence": 0.640, "detection_id": "53ef7f72"},
    {"x": 1352.068, "y": 930.259,  "width": 51.757,  "height": 66.583, "confidence": 0.637, "detection_id": "dc8ebcd5"},
    {"x": 1506.701, "y": 822.339,  "width": 36.425,  "height": 62.805, "confidence": 0.634, "detection_id": "435ebc2c"},
    {"x": 2004.738, "y": 294.021,  "width": 33.508,  "height": 55.385, "confidence": 0.634, "detection_id": "1ebea03f"},
    {"x": 800.427,  "y": 680.029,  "width": 60.517,  "height": 57.56,  "confidence": 0.634, "detection_id": "46ef507b"},
    {"x": 1056.804, "y": 497.381,  "width": 44.997,  "height": 60.476, "confidence": 0.629, "detection_id": "fbd7a098"},
    {"x": 1551.691, "y": 886.705,  "width": 40.157,  "height": 59.34,  "confidence": 0.628, "detection_id": "d3922925"},
    {"x": 792.766,  "y": 257.149,  "width": 46.673,  "height": 67.761, "confidence": 0.627, "detection_id": "792617d0"},
    {"x": 540.4,    "y": 286.024,  "width": 52.795,  "height": 65.811, "confidence": 0.620, "detection_id": "b44766c9"},
    {"x": 1381.377, "y": 427.71,   "width": 39.617,  "height": 77.102, "confidence": 0.619, "detection_id": "7ce79c33"},
    {"x": 1616.807, "y": 1163.175, "width": 52.284,  "height": 62.964, "confidence": 0.619, "detection_id": "8608d851"},
    {"x": 1260.83,  "y": 824.257,  "width": 49.415,  "height": 62.082, "confidence": 0.618, "detection_id": "5466903c"},
    {"x": 642.307,  "y": 1004.058, "width": 59.64,   "height": 59.033, "confidence": 0.614, "detection_id": "bdd3dc79"},
    {"x": 1565.744, "y": 541.181,  "width": 58.915,  "height": 69.696, "confidence": 0.612, "detection_id": "f0071660"},
    {"x": 1134.97,  "y": 469.821,  "width": 49.387,  "height": 53.037, "confidence": 0.608, "detection_id": "fad542f5"},
    {"x": 1710.383, "y": 836.326,  "width": 68.247,  "height": 55.484, "confidence": 0.608, "detection_id": "5d34146d"},
    {"x": 1818.419, "y": 1071.473, "width": 55.711,  "height": 49.797, "confidence": 0.606, "detection_id": "4a839019"},
    {"x": 1885.379, "y": 617.842,  "width": 39.582,  "height": 63.328, "confidence": 0.604, "detection_id": "17814d92"},
    {"x": 954.962,  "y": 598.697,  "width": 58.807,  "height": 65.755, "confidence": 0.600, "detection_id": "95f77e2f"},
    {"x": 1212.812, "y": 906.234,  "width": 43.747,  "height": 82.766, "confidence": 0.599, "detection_id": "f62b1122"},
    {"x": 1280.488, "y": 523.838,  "width": 42.984,  "height": 80.24,  "confidence": 0.599, "detection_id": "38bcd4c1"},
    {"x": 776.281,  "y": 833.945,  "width": 53.681,  "height": 66.239, "confidence": 0.599, "detection_id": "2548663b"},
    {"x": 413.718,  "y": 385.648,  "width": 41.63,   "height": 70.551, "confidence": 0.599, "detection_id": "2b91ed43"},
    {"x": 548.208,  "y": 819.183,  "width": 46.537,  "height": 66.285, "confidence": 0.599, "detection_id": "f4b0199b"},
    {"x": 1294.783, "y": 799.933,  "width": 49.326,  "height": 50.776, "confidence": 0.596, "detection_id": "012771fb"},
    {"x": 1970.838, "y": 497.274,  "width": 38.83,   "height": 63.398, "confidence": 0.594, "detection_id": "d88fc65e"},
    {"x": 1559.665, "y": 948.617,  "width": 40.124,  "height": 55.139, "confidence": 0.588, "detection_id": "eb4a663a"},
    {"x": 1500.488, "y": 917.791,  "width": 43.161,  "height": 68.564, "confidence": 0.588, "detection_id": "dac79099"},
    {"x": 1933.26,  "y": 590.816,  "width": 36.166,  "height": 59.065, "confidence": 0.585, "detection_id": "33ed37a5"},
    {"x": 1001.071, "y": 309.776,  "width": 51.219,  "height": 57.514, "confidence": 0.583, "detection_id": "c520c140"},
    {"x": 588.328,  "y": 811.92,   "width": 73.576,  "height": 55.419, "confidence": 0.582, "detection_id": "3c32d85c"},
    {"x": 1833.074, "y": 994.44,   "width": 62.947,  "height": 52.35,  "confidence": 0.579, "detection_id": "fee7ac81"},
    {"x": 1535.328, "y": 464.183,  "width": 75.171,  "height": 54.141, "confidence": 0.579, "detection_id": "bfcbc3de"},
    {"x": 572.742,  "y": 1301.084, "width": 78.816,  "height": 52.072, "confidence": 0.577, "detection_id": "28d45b9f"},
    {"x": 844.486,  "y": 580.356,  "width": 50.994,  "height": 71.39,  "confidence": 0.577, "detection_id": "2ea941e2"},
    {"x": 1637.09,  "y": 993.253,  "width": 48.853,  "height": 43.873, "confidence": 0.576, "detection_id": "b5dcc021"},
    {"x": 916.506,  "y": 527.829,  "width": 44.916,  "height": 67.443, "confidence": 0.576, "detection_id": "62268d6e"},
    {"x": 748.853,  "y": 544.428,  "width": 54.963,  "height": 57.778, "confidence": 0.574, "detection_id": "a2d7283f"},
    {"x": 1973.129, "y": 422.276,  "width": 59.406,  "height": 50.542, "confidence": 0.569, "detection_id": "810063e2"},
    {"x": 1429.753, "y": 1058.558, "width": 55.727,  "height": 57.169, "confidence": 0.569, "detection_id": "f0d84dbc"},
    {"x": 1247.238, "y": 758.665,  "width": 51.928,  "height": 41.131, "confidence": 0.568, "detection_id": "bcd5026f"},
    {"x": 872.327,  "y": 578.383,  "width": 46.857,  "height": 70.776, "confidence": 0.564, "detection_id": "ef178686"},
    {"x": 1067.001, "y": 122.54,   "width": 33.39,   "height": 67.1,   "confidence": 0.561, "detection_id": "013ff2df"},
    {"x": 420.445,  "y": 502.579,  "width": 82.606,  "height": 39.127, "confidence": 0.560, "detection_id": "6aed353d"},
    {"x": 1756.243, "y": 1151.833, "width": 56.83,   "height": 55.39,  "confidence": 0.559, "detection_id": "47f165c9"},
    {"x": 966.439,  "y": 762.024,  "width": 43.271,  "height": 73.238, "confidence": 0.556, "detection_id": "d66c9dc6"},
    {"x": 1215.325, "y": 113.495,  "width": 39.532,  "height": 63.451, "confidence": 0.553, "detection_id": "b5c35838"},
    {"x": 386.545,  "y": 678.038,  "width": 63.94,   "height": 62.463, "confidence": 0.553, "detection_id": "5c961c47"},
    {"x": 1851.804, "y": 675.72,   "width": 49.361,  "height": 53.83,  "confidence": 0.552, "detection_id": "4dd89f7c"},
    {"x": 802.462,  "y": 200.206,  "width": 62.263,  "height": 58.929, "confidence": 0.552, "detection_id": "3547b8dd"},
    {"x": 281.22,   "y": 909.164,  "width": 73.017,  "height": 45.933, "confidence": 0.550, "detection_id": "879b418a"},
    {"x": 1732.123, "y": 1004.621, "width": 33.73,   "height": 59.992, "confidence": 0.550, "detection_id": "26d1ad43"},
    {"x": 1173.105, "y": 422.231,  "width": 38.729,  "height": 66.502, "confidence": 0.544, "detection_id": "8eb7e91f"},
    {"x": 1047.858, "y": 372.107,  "width": 40.496,  "height": 56.252, "confidence": 0.536, "detection_id": "119a500e"},
    {"x": 1066.199, "y": 762.51,   "width": 49.314,  "height": 47.905, "confidence": 0.535, "detection_id": "341b19b1"},
    {"x": 1691.032, "y": 1151.974, "width": 100.229, "height": 29.027, "confidence": 0.534, "detection_id": "3bfd932c"},
    {"x": 1564.763, "y": 801.645,  "width": 70.089,  "height": 51.23,  "confidence": 0.532, "detection_id": "cc499194"},
    {"x": 1192.745, "y": 660.464,  "width": 58.746,  "height": 54.092, "confidence": 0.521, "detection_id": "28b6ae2b"},
    {"x": 1067.148, "y": 211.316,  "width": 51.38,   "height": 55.579, "confidence": 0.516, "detection_id": "cae69f5f"},
    {"x": 1249.823, "y": 112.412,  "width": 35.867,  "height": 65.126, "confidence": 0.513, "detection_id": "4f24120a"},
    {"x": 1220.048, "y": 554.371,  "width": 52.552,  "height": 53.626, "confidence": 0.508, "detection_id": "b8570964"},
    {"x": 568.059,  "y": 386.8,    "width": 43.967,  "height": 56.469, "confidence": 0.507, "detection_id": "4c13e888"},
]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def bbox_to_pixels(det, img_h, img_w, pad=0.55):
    cx, cy = det["x"], det["y"]
    pw = det["width"]  * (1 + pad)
    ph = det["height"] * (1 + pad)
    x1 = max(0,     int(cx - pw / 2))
    y1 = max(0,     int(cy - ph / 2))
    x2 = min(img_w, int(cx + pw / 2))
    y2 = min(img_h, int(cy + ph / 2))
    return x1, y1, x2, y2


def main():
    img_path  = Path("/Users/noledge/Downloads/B E E/data/0007.png")
    base_dir  = Path("/Users/noledge/Downloads/B E E/data")
    out_after = base_dir / "bee_closeups_epoaf"
    out_cmp   = base_dir / "bee_closeups_comparisons"
    out_after.mkdir(exist_ok=True)
    out_cmp.mkdir(exist_ok=True)

    print(f"Loading {img_path.name} …")
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(img_path)
    img_h, img_w = img.shape
    print(f"  {img_w}×{img_h}  mean={img.mean():.1f}")

    TARGET       = 320
    TILE_W       = 240   # per-panel width in comparison tile
    TILE_H       = 240
    MONTAGE_COLS = 7

    PIPELINE = dict(
        target       = TARGET,
        # CED
        n_iter       = 30,
        dt           = 0.15,
        sigma        = 1.2,
        rho          = 4.5,
        alpha        = 0.00005,
        C            = 1e-6,
        recompute    = 6,
        # pre-denoise
        nlm_h        = 9,
        nlm_template = 7,
        nlm_search   = 21,
        # contrast
        clahe_clip   = 3.5,
        gamma        = 0.70,
        # post
        bilateral_d  = 9,
        bilateral_sc = 35,
        bilateral_ss = 35,
        usm_amount   = 1.8,
        usm_sigma    = 1.8,
        usm_edge     = 2.8,
        final_clip   = 2.5,
    )

    comparison_tiles = []

    for idx, det in enumerate(PREDICTIONS):
        x1, y1, x2, y2 = bbox_to_pixels(det, img_h, img_w, pad=0.55)
        crop_raw = img[y1:y2, x1:x2]
        if crop_raw.size == 0:
            print(f"  [{idx+1:03d}] SKIP – empty crop")
            continue

        before = baseline_pipeline(crop_raw, target=TARGET)
        after  = full_pipeline(crop_raw, **PIPELINE)

        conf_str = f"{det['confidence']:.3f}".replace(".", "p")
        did      = det["detection_id"][:8]
        stem     = f"bee_{idx+1:03d}_conf{conf_str}_{did}"

        cv2.imwrite(str(out_after / f"{stem}.png"), after)

        label = f"#{idx+1:03d} {det['confidence']:.2f}"
        tile  = make_comparison_tile(before, after, label,
                                     tile_w=TILE_W, tile_h=TILE_H)
        cv2.imwrite(str(out_cmp / f"{stem}_cmp.png"), tile)
        comparison_tiles.append(tile)

        h_raw, w_raw = crop_raw.shape
        print(f"  [{idx+1:03d}/{len(PREDICTIONS)}] {stem}  raw {w_raw}×{h_raw}")

    # ── Montage ───────────────────────────────────────────────────────────────
    print(f"\nBuilding montage ({len(comparison_tiles)} tiles, "
          f"{MONTAGE_COLS} cols) …")
    montage = build_montage(comparison_tiles, cols=MONTAGE_COLS)
    if montage is not None:
        mpath = base_dir / "bee_montage_comparisons.png"
        cv2.imwrite(str(mpath), montage)
        mh, mw = montage.shape[:2]
        print(f"Montage → {mpath.name}  ({mw}×{mh} px)")

    print(f"\nDone.")
    print(f"  Individual after  : {out_after}")
    print(f"  Comparisons       : {out_cmp}")
    print(f"  Montage           : {base_dir / 'bee_montage_comparisons.png'}")


if __name__ == "__main__":
    main()
