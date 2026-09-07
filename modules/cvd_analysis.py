"""
Person 2 - CVD & Color Analysis

This module:
1. Gets colors from Person 1's chart data
2. Detects whether labels exist
3. Simulates Protanopia, Deuteranopia and Tritanopia
4. Detects similar colors
5. Detects low contrast
6. Warns when information may rely only on color
"""

import numpy as np

try:
    # Works when imported as part of the `modules` package (e.g. app.py
    # doing `from modules.cvd_analysis import get_colors_and_labels`).
    from .cvd import simulate_cvd
except ImportError:
    # Works when this file is run/imported directly as a standalone
    # script (no package context), e.g. `python cvd_analysis.py`.
    from cvd import simulate_cvd


DEFAULT_PALETTE = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
]

BG_AREA_FRACTION = 0.80
BG_FILL_DENSITY = 0.50
NOISE_PIXEL_FRACTION = 0.02

COLOR_DISTANCE_THRESHOLD = 60
BRIGHTNESS_THRESHOLD = 40


def _is_background_or_noise(region, total_pixels):
    bx, by, bw, bh = region["bbox"]

    bbox_area = max(bw * bh, 1)

    area_fraction = bbox_area / total_pixels
    fill_density = region["pixel_count"] / bbox_area
    pixel_fraction = region["pixel_count"] / total_pixels

    is_full_canvas_fill = (
        area_fraction > BG_AREA_FRACTION
        and fill_density > BG_FILL_DENSITY
    )

    is_negligible = pixel_fraction < NOISE_PIXEL_FRACTION

    return is_full_canvas_fill or is_negligible


def simulate_all_cvd(colors):
    """
    Simulate all three major types of color vision deficiency.
    """

    results = {
        "protanopia": [],
        "deuteranopia": [],
        "tritanopia": []
    }

    for color in colors:
        results["protanopia"].append(
            simulate_cvd(color, "protanopia")
        )

        results["deuteranopia"].append(
            simulate_cvd(color, "deuteranopia")
        )

        results["tritanopia"].append(
            simulate_cvd(color, "tritanopia")
        )

    return results


def color_distance(color1, color2):
    """
    Calculate how different two RGB colors are.
    """

    c1 = np.array(color1, dtype=float)
    c2 = np.array(color2, dtype=float)

    return float(np.linalg.norm(c1 - c2))


def calculate_brightness(color):
    """
    Calculate the perceived brightness of an RGB color.
    """

    r, g, b = color

    return 0.299 * r + 0.587 * g + 0.114 * b


def find_problematic_pairs(colors, cvd_results):
    """
    Find colors that become too similar or have low contrast
    under CVD simulation.
    """

    issues = []

    for deficiency in [
        "protanopia",
        "deuteranopia",
        "tritanopia"
    ]:

        simulated_colors = cvd_results[deficiency]

        for i in range(len(simulated_colors)):

            for j in range(i + 1, len(simulated_colors)):

                color1 = simulated_colors[i]
                color2 = simulated_colors[j]

                distance = color_distance(color1, color2)

                brightness_difference = abs(
                    calculate_brightness(color1)
                    - calculate_brightness(color2)
                )

                if distance < COLOR_DISTANCE_THRESHOLD:

                    issues.append({
                        "type": "similar_colors",
                        "deficiency": deficiency,
                        "color_1": colors[i],
                        "color_2": colors[j],
                        "distance": round(distance, 2)
                    })

                elif brightness_difference < BRIGHTNESS_THRESHOLD:

                    issues.append({
                        "type": "low_contrast",
                        "deficiency": deficiency,
                        "color_1": colors[i],
                        "color_2": colors[j],
                        "brightness_difference": round(
                            brightness_difference, 2
                        )
                    })

    return issues


def get_colors_and_labels(chart_data):
    """
    Process Person 1's ChartData and return all
    color accessibility information needed by Person 3.
    """

    source_type = chart_data["source_type"]

    if source_type == "image":

        regions = chart_data["chart_elements"]["color_regions"]

        total_pixels = (
            chart_data["raw_array"].shape[0]
            * chart_data["raw_array"].shape[1]
        )

        colors = [
            r["color"]
            for r in regions
            if not _is_background_or_noise(r, total_pixels)
        ]

        labels_present = (
            len(chart_data["chart_elements"]["labels"]) > 0
        )

    elif source_type == "csv":

        structure = chart_data["csv_structure"]
        cat_col = structure["likely_category_column"]

        if cat_col is None or chart_data["dataframe"] is None:

            return {
                "colors": [],
                "labels_present": False,
                "cvd_results": {
                    "protanopia": [],
                    "deuteranopia": [],
                    "tritanopia": []
                },
                "problematic_pairs": [],
                "color_only_warning": True
            }

        n_categories = (
            chart_data["dataframe"][cat_col].nunique()
        )

        colors = [
            DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)]
            for i in range(n_categories)
        ]

        labels_present = False

    else:

        raise ValueError(
            f"Unknown source_type: {source_type!r}"
        )

    cvd_results = simulate_all_cvd(colors)

    problematic_pairs = find_problematic_pairs(
        colors,
        cvd_results
    )

    return {
        "colors": colors,
        "labels_present": labels_present,
        "cvd_results": cvd_results,
        "problematic_pairs": problematic_pairs,
        "color_only_warning": not labels_present
    }


if __name__ == "__main__":
    print("cvd_analysis.py loaded with no errors!")