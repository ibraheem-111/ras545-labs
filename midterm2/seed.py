import numpy as np
import cv2
from skimage.morphology import reconstruction, binary_erosion
from skimage.morphology.convex_hull import convex_hull_image
from skimage.feature import corner_harris, corner_peaks, corner_subpix
from skimage.util import invert

def isolate_maze_with_reconstruction(gray, seed_half=40, second_seed_half=240,
                                     bin_thresh=0.70, selem1=(11,11), selem2=(30,30)):
    # gray: float32/float64 in [0,1] recommended
    if gray.dtype != np.float32 and gray.dtype != np.float64:
        gray = gray.astype(np.float32) / 255.0
    h, w = gray.shape

    # Seed 1 (center window) → reconstruction by dilation
    seed = np.zeros_like(gray)
    seed[h//2-seed_half:h//2+seed_half, w//2-seed_half:w//2+seed_half] = \
        gray[h//2-seed_half:h//2+seed_half, w//2-seed_half:w//2+seed_half]
    rec1 = reconstruction(seed, gray)                                      # :contentReference[oaicite:14]{index=14}
    rec1 = (rec1 >= bin_thresh).astype(np.float32)

    # Seed 2 (larger) → reconstruction by erosion with bigger SE
    seed2 = np.ones_like(rec1)
    seed2[h//2-second_seed_half:h//2+second_seed_half, w//2-second_seed_half:w//2+second_seed_half] = \
        rec1[h//2-second_seed_half:h//2+second_seed_half, w//2-second_seed_half:w//2+second_seed_half]
    se1 = np.ones(selem1, np.uint8)
    rec2 = reconstruction(seed2, rec1, method='erosion', footprint=se1)        # :contentReference[oaicite:15]{index=15}

    # Outer box corners via convex hull → erosion → corner detection
    rec2_inv = invert(rec2)                                                # :contentReference[oaicite:16]{index=16}
    hull = convex_hull_image(rec2_inv)                                     # :contentReference[oaicite:17]{index=17}
    se2 = np.ones(selem2, np.uint8)
    hull_eroded = binary_erosion(hull, footprint=se2)                          # :contentReference[oaicite:18]{index=18}

    harris = corner_harris(hull_eroded.astype(float))
    coords = corner_peaks(harris, min_distance=5, threshold_rel=0.02)      # :contentReference[oaicite:19]{index=19}
    coords_subpix = corner_subpix(rec2_inv.astype(float), coords, window_size=13)

    # coords_subpix can contain >4 points; keep 4 extreme ones by convex hull in OpenCV
    if coords_subpix is None or len(coords_subpix) < 4:
        raise RuntimeError("Could not find 4 corners; tune sizes/thresholds.")
    pts = coords_subpix[:, ::-1]  # (x,y), convert from (row,col) to (x,y)
    pts = pts[~np.isnan(pts).any(axis=1)]
    if len(pts) > 4:
        # keep 4 corners using OpenCV convex hull + polygon approx
        hull_pts = cv2.convexHull(pts.astype(np.float32))
        peri = cv2.arcLength(hull_pts, True)
        approx = cv2.approxPolyDP(hull_pts, 0.02*peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4,2)
        else:
            # fallback: take four farthest points by pairwise distance
            from itertools import combinations
            best = max(combinations(pts, 4), key=lambda comb:
                       cv2.contourArea(np.array(comb, dtype=np.float32)))
            pts = np.array(best, dtype=np.float32)

    # order: TL, TR, BR, BL
    def order_corners(p):
        s = p.sum(1); d = np.diff(p, axis=1).ravel()
        ordered = np.zeros((4,2), np.float32)
        ordered[0] = p[np.argmin(s)]
        ordered[2] = p[np.argmax(s)]
        ordered[1] = p[np.argmin(d)]
        ordered[3] = p[np.argmax(d)]
        return ordered

    return order_corners(pts.astype(np.float32)), (rec2 > 0.5).astype(np.uint8)*255