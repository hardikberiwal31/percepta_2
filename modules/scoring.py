"""
modules/scoring.py
Person 4 - Accessibility Scoring

Combines Person 2's CVD/contrast math with Person 3's reconstructed
visualization metadata into a single 0-100 accessibility score, plus
plain-language recommendations.

Score breakdown (100 pts total):
    40  CVD distinguishability  - re-simulates Percepta's OWN final colors
                                   (not the original chart's) under the
                                   selected CVD type and counts remaining
                                   collisions. This closes the loop instead
                                   of just trusting a preset name.
    20  Contrast                - average WCAG contrast ratio of the final
                                   colors against a white background.
    15  Shape differentiation   - fraction of legend entries with a
                                   distinct shape.
    10  Pattern differentiation - fraction of legend entries with a
                                   distinct (shape, hatch) combination.
    10  Labeling                - whether every element has a real text
                                   label.
     5  Color independence      - whether the chart still needs color at
                                   all to be read (patterns or >1 shape).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

try:
    from .cvd import simulate_cvd
    from .cvd_analysis import (
        color_distance,
        calculate_brightness,
        COLOR_DISTANCE_THRESHOLD,
        BRIGHTNESS_THRESHOLD,
    )
    from .visualization import contrast_ratio
except ImportError:
    from cvd import simulate_cvd
    from cvd_analysis import (
        color_distance,
        calculate_brightness,
        COLOR_DISTANCE_THRESHOLD,
        BRIGHTNESS_THRESHOLD,
    )
    from visualization import contrast_ratio


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def _cvd_distinguishability_score(legend: List[Dict[str, Any]], cvd_type: str) -> Dict[str, Any]:
    """
    Re-simulates Percepta's final palette under the given CVD type using
    Person 2's own simulate_cvd(), then reuses Person 2's own thresholds
    to count remaining collisions. Fewer remaining issues -> higher score.
    """
    if len(legend) < 2:
        return {"score": 40.0, "remaining_issues": 0, "total_pairs_checked": 0}

    colors_rgb = [hex_to_rgb(entry["color"]) for entry in legend]
    simulated = [simulate_cvd(c, cvd_type) for c in colors_rgb]

    issues = 0
    total_pairs = 0
    for i in range(len(simulated)):
        for j in range(i + 1, len(simulated)):
            total_pairs += 1
            distance = color_distance(simulated[i], simulated[j])
            brightness_diff = abs(
                calculate_brightness(simulated[i]) - calculate_brightness(simulated[j])
            )
            if distance < COLOR_DISTANCE_THRESHOLD or brightness_diff < BRIGHTNESS_THRESHOLD:
                issues += 1

    ratio_ok = 1 - (issues / max(total_pairs, 1))
    return {
        "score": round(40 * max(ratio_ok, 0), 1),
        "remaining_issues": issues,
        "total_pairs_checked": total_pairs,
    }


def _contrast_score(legend: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not legend:
        return {"score": 0, "average_contrast_ratio": 0.0}

    ratios = [contrast_ratio(entry["color"], "#FFFFFF") for entry in legend]
    avg_ratio = sum(ratios) / len(ratios)

    if avg_ratio >= 7.0:
        pts = 20
    elif avg_ratio >= 4.5:
        pts = 15
    elif avg_ratio >= 3.0:
        pts = 10
    else:
        pts = 5

    return {"score": pts, "average_contrast_ratio": round(avg_ratio, 2)}


def _shape_score(legend: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not legend:
        return {"score": 0, "unique_shapes": 0}
    unique_shapes = len({entry["shape"] for entry in legend})
    ratio = min(unique_shapes / len(legend), 1.0)
    return {"score": round(15 * ratio, 1), "unique_shapes": unique_shapes}


def _pattern_score(legend: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not legend:
        return {"score": 0, "unique_combinations": 0}
    combos = {(entry["shape"], entry["hatch"]) for entry in legend}
    ratio = min(len(combos) / len(legend), 1.0)
    return {"score": round(10 * ratio, 1), "unique_combinations": len(combos)}


def _label_score(legend: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not legend:
        return {"score": 0, "all_labeled": False}
    all_labeled = all(bool(entry.get("label")) for entry in legend)
    return {"score": 10 if all_labeled else 4, "all_labeled": all_labeled}


def _color_independence_score(viz_metadata: Dict[str, Any]) -> Dict[str, Any]:
    legend = viz_metadata.get("legend", [])
    uses_patterns = any(entry["hatch"] for entry in legend)
    uses_shapes = len({entry["shape"] for entry in legend}) > 1
    independent = uses_patterns or uses_shapes
    return {"score": 5 if independent else 0, "color_independent": independent}


def calculate_accessibility_score(viz_metadata: Dict[str, Any], cvd_type: str) -> Dict[str, Any]:
    """
    Main Person 4 entry point. Scores Person 3's OUTPUT (viz_metadata),
    not the original chart - the whole point is measuring how accessible
    the *fixed* visualization actually is.

    Usage:
        from modules.scoring import calculate_accessibility_score
        score = calculate_accessibility_score(viz_result.metadata, cvd_type="deuteranopia")

    Returns:
        {
            "total_score": int 0-100,
            "breakdown": {component_name: {...}, ...},
            "recommendations": [str, ...],
        }
    """
    legend = viz_metadata.get("legend", [])

    breakdown = {
        "cvd_distinguishability": _cvd_distinguishability_score(legend, cvd_type),
        "contrast": _contrast_score(legend),
        "shape_differentiation": _shape_score(legend),
        "pattern_differentiation": _pattern_score(legend),
        "labeling": _label_score(legend),
        "color_independence": _color_independence_score(viz_metadata),
    }

    total = sum(component["score"] for component in breakdown.values())

    return {
        "total_score": round(min(max(total, 0), 100)),
        "breakdown": breakdown,
        "recommendations": _build_recommendations(breakdown),
    }


def _build_recommendations(breakdown: Dict[str, Any]) -> List[str]:
    recs = []

    if breakdown["cvd_distinguishability"]["remaining_issues"] > 0:
        recs.append(
            f"{breakdown['cvd_distinguishability']['remaining_issues']} color pair(s) are still hard "
            "to tell apart under this CVD type - try the high_contrast preset."
        )
    if breakdown["contrast"]["average_contrast_ratio"] < 4.5:
        recs.append("Average contrast is below the WCAG AA minimum (4.5:1) - try the high_contrast preset.")
    if breakdown["shape_differentiation"]["unique_shapes"] < 2:
        recs.append("Only one shape is in use - add shape variety so categories don't rely on color alone.")
    if not breakdown["labeling"]["all_labeled"]:
        recs.append("Some elements are missing readable labels - add text labels or a fuller legend.")
    if not breakdown["color_independence"]["color_independent"]:
        recs.append("This chart still depends on color alone - enable patterns or shapes.")
    if not recs:
        recs.append("This visualization meets Percepta's accessibility bar for the selected CVD type.")

    return recs


if __name__ == "__main__":
    demo_metadata = {
        "cvd_type": "deuteranopia",
        "preset": "high_contrast",
        "legend": [
            {"label": "critical", "color": "#8A3D00", "shape": "triangle", "hatch": "xx", "blink": True},
            {"label": "monitor", "color": "#5F5B1A", "shape": "square", "hatch": "//", "blink": False},
            {"label": "stable", "color": "#00664A", "shape": "circle", "hatch": "", "blink": False},
        ],
    }
    result = calculate_accessibility_score(demo_metadata, cvd_type="deuteranopia")
    print("Total score:", result["total_score"])
    for name, component in result["breakdown"].items():
        print(f"  {name}: {component}")
    print("Recommendations:", result["recommendations"])
