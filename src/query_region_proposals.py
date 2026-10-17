"""Conservative local views for full-image search, with no catalog indexing."""

from __future__ import annotations

import numpy as np
from PIL import Image

try:
    import cv2
except Exception:  # OpenCV is optional; the collage view still works without it.
    cv2 = None


def propose_query_views(image: Image.Image) -> list[Image.Image]:
    """Find a few likely garment-detail views inside a full uploaded photo."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    views: list[Image.Image] = []

    # Tall promotional collages commonly place the garment detail in a lower
    # half-width panel. Avoid feeding the full collage text and margins to CLIP.
    if height >= 640 and width > 0 and width / float(height) <= 0.50:
        # A sharp horizontal panel boundary distinguishes a multi-photo
        # catalog collage from a continuous tall screenshot. Use a focused
        # sleeve-detail view instead of the broad crop so unrelated high
        # context scores cannot crowd its relevant candidates out.
        center_x0, center_x1 = int(width * 0.08), int(width * 0.92)
        if center_x1 > center_x0:
            row_delta = np.abs(
                rgb[1:, center_x0:center_x1].astype(np.int16)
                - rgb[:-1, center_x0:center_x1].astype(np.int16)
            ).mean(axis=(1, 2))
        else:
            row_delta = np.zeros(height - 1, dtype=np.float32)
        boundary_y0, boundary_y1 = int(height * 0.30), int(height * 0.75)
        has_collage_boundary = bool(
            boundary_y1 > boundary_y0
            and float(np.max(row_delta[boundary_y0:boundary_y1])) >= 18.0
        )
        if has_collage_boundary:
            detail_box = (
                int(width * 0.04),
                int(height * 0.43),
                int(width * 0.38),
                int(height * 0.59),
            )
            if detail_box[2] - detail_box[0] >= 48 and detail_box[3] - detail_box[1] >= 48:
                detail_view = image.crop(detail_box)
                detail_view.info["search_roi_profile"] = "collage_detail"
                views.append(detail_view)
        else:
            x0, x1 = 0, int(width * 0.52)
            y0, y1 = int(height * 0.28), int(height * 0.71)
            if x1 - x0 >= 48 and y1 - y0 >= 48:
                views.append(image.crop((x0, y0, x1, y1)))
        return views

    if cv2 is None or width < 48 or height < 48:
        return views

    # Video screenshots often have black side bars and UI chrome. The useful
    # collar detail is in the upper-middle of the garment, so use that original
    # resolution crop directly instead of embedding the whole screenshot.
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    side_w = max(1, int(width * 0.07))
    sample_h = max(1, int(height * 0.80))
    left_dark = float(np.mean(gray[:sample_h, :side_w] < 35))
    right_dark = float(np.mean(gray[:sample_h, width - side_w :] < 35))
    center_dark = float(
        np.mean(gray[:sample_h, int(width * 0.30) : int(width * 0.70)] < 35)
    )
    if left_dark >= 0.60 and right_dark >= 0.60 and center_dark <= 0.30:
        views.append(
            image.crop(
                (
                    int(width * 0.38),
                    int(height * 0.25),
                    int(width * 0.88),
                    int(height * 0.53),
                )
            )
        )
        return views

    # Detect a long red/white striped accessory draped over a garment. Requiring
    # a sustained run of alternating red and light rows keeps isolated red
    # patches and plaid trousers from creating this crop.
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    red = ((hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 170)) & (hsv[:, :, 1] >= 65) & (hsv[:, :, 2] >= 55)
    light = (hsv[:, :, 1] < 45) & (hsv[:, :, 2] > 180)
    x0, x1 = int(width * 0.25), int(width * 0.75)
    stripe_hits: list[bool] = []
    for y_ratio in np.linspace(0.08, 0.85, 32):
        y0 = int(y_ratio * height)
        y1 = max(y0 + 1, int((y_ratio + 0.025) * height))
        stripe_hits.append(
            float(np.mean(red[y0:y1, x0:x1])) > 0.12
            and float(np.mean(light[y0:y1, x0:x1])) > 0.10
        )
    longest_stripe_run = run = 0
    for hit in stripe_hits:
        run = run + 1 if hit else 0
        longest_stripe_run = max(longest_stripe_run, run)
    if longest_stripe_run >= 14:
        views.append(
            image.crop(
                (
                    int(width * 0.30),
                    int(height * 0.18),
                    int(width * 0.80),
                    int(height * 0.80),
                )
            )
        )
        return views

    # Isolate small dark-blue garment panels such as knitted chest pockets.
    mask = cv2.inRange(hsv, np.array([90, 45, 0]), np.array([145, 255, 150]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    image_area = float(width * height)
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for idx in range(1, count):
        x, y, box_w, box_h, area = (int(v) for v in stats[idx])
        area_ratio = float(area) / max(1.0, image_area)
        box_aspect = box_w / float(max(1, box_h))
        if (
            area_ratio < 0.003
            or area_ratio > 0.06
            or box_w < 32
            or box_h < 32
            # The full-context pocket case is a vertically elongated patch.
            # Excluding near-square dark regions avoids promoting unrelated
            # accents in the other regression photos.
            or not 0.45 <= box_aspect <= 0.65
            or not 0.20 <= (x + box_w / 2) / width <= 0.88
            or not 0.22 <= (y + box_h / 2) / height <= 0.82
        ):
            continue
        candidates.append((area, (x, y, x + box_w, y + box_h)))

    candidates.sort(key=lambda item: item[0], reverse=True)
    for _area, (x0, y0, x1, y1) in candidates[:2]:
        box_w, box_h = x1 - x0, y1 - y0
        pad_x, pad_y = max(8, int(box_w * 0.25)), max(8, int(box_h * 0.20))
        box = (
            max(0, x0 - pad_x),
            max(0, y0 - pad_y),
            min(width, x1 + pad_x),
            min(height, y1 + pad_y),
        )
        views.append(image.crop(box))
    return views
