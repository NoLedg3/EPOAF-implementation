#!/usr/bin/env python3
"""
  1. Aggressive local contrast (CLAHE, per-crop adaptive clip)
  2. Gamma midtone boost
  3. NL-Means pre-denoising  (kills grain before amplification)
  4. Adaptive CED  (Weickert, tensor re-estimated mid-diffusion)
  5. Bilateral post-smoothing  (clean flat regions)
  6. Edge-adaptive unsharp masking  (sharpen structure, not noise)
  7. Final stretch + second CLAHE pass
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
    img_path  = 
    base_dir  = 
    out_after = 
    out_cmp   =
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
        mpath = base_dir / "montage_comparisons.png"
        cv2.imwrite(str(mpath), montage)
        mh, mw = montage.shape[:2]
        print(f"Montage → {mpath.name}  ({mw}×{mh} px)")

    print(f"\nDone.")
    print(f"  Individual after  : {out_after}")
    print(f"  Comparisons       : {out_cmp}")
    print(f"  Montage           : {base_dir / 'montage_comparisons.png'}")


if __name__ == "__main__":
    main()
