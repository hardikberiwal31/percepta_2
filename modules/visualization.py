"""
modules/visualization.py
Person 3 - Accessible Visualization & Presets

Consumes:
    chart_data  -> Person 1's ChartData dict (from build_chart_data()):
        {
            "source_type": "image" | "csv",
            "raw_array": np.ndarray | None,
            "dataframe": pd.DataFrame | None,
            "chart_elements": {"color_regions": [...], "labels": [...]} | None,
            "csv_structure": {"categorical_columns": [...], "numerical_columns": [...],
                               "likely_category_column": str|None,
                               "likely_value_column": str|None} | None,
        }

    analysis    -> Person 2's analysis dict (from get_colors_and_labels()):
        {
            "colors": [(R, G, B), ...],            # 0-255 int tuples
            "labels_present": bool,
            "cvd_results": {"protanopia": [...], "deuteranopia": [...], "tritanopia": [...]},
            "problematic_pairs": [
                {"type": "similar_colors"|"low_contrast", "deficiency": str,
                 "color_1": (R,G,B), "color_2": (R,G,B), ...}, ...
            ],
            "color_only_warning": bool,
        }

Produces:
    - A reconstructed matplotlib chart that never depends on color alone:
      CVD-safe palette, shapes/markers, patterns/hatching, labels, WCAG
      contrast, semantic color->shape mapping, legend.
    - 9 presets total: 3 per CVD type (protanopia, deuteranopia, tritanopia).
    - `blink` metadata (semantic danger cue) — the frontend owns the actual
      blink animation / ON-OFF switch.
    - A preset auto-picked from Person 2's real findings when the caller
      doesn't force one (more problems detected for a CVD type -> stronger
      preset; no labels at all -> stronger preset too).

Person 4 should import `generate_accessible_visualization` in pipeline.py.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless-safe backend for Streamlit / servers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import pandas as pd
except ImportError:  # pandas is required only for the CSV path
    pd = None


# ---------------------------------------------------------------------------
# 1. CVD-safe palettes (Okabe-Ito / Wong — validated for all 3 CVD types)
# ---------------------------------------------------------------------------

CVD_PALETTES = {
    "protanopia": ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#000000", "#D55E00"],
    "deuteranopia": ["#0072B2", "#D55E00", "#F0E442", "#CC79A7", "#009E73", "#56B4E9", "#000000", "#E69F00"],
    "tritanopia": ["#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00", "#000000", "#56B4E9", "#F0E442"],
}

# ---------------------------------------------------------------------------
# 2. Semantic mapping: meaning -> color / shape / pattern / marker / blink
# ---------------------------------------------------------------------------

SEMANTIC_MAP = {
    "safe": {"color": "#009E73", "shape": "circle", "marker": "o", "hatch": "", "blink": False},
    "stable": {"color": "#009E73", "shape": "circle", "marker": "o", "hatch": "", "blink": False},
    "normal": {"color": "#009E73", "shape": "circle", "marker": "o", "hatch": "", "blink": False},
    "warning": {"color": "#F0E442", "shape": "square", "marker": "s", "hatch": "//", "blink": False},
    "monitor": {"color": "#F0E442", "shape": "square", "marker": "s", "hatch": "//", "blink": False},
    "caution": {"color": "#F0E442", "shape": "square", "marker": "s", "hatch": "//", "blink": False},
    "danger": {"color": "#D55E00", "shape": "triangle", "marker": "^", "hatch": "xx", "blink": True},
    "critical": {"color": "#D55E00", "shape": "triangle", "marker": "^", "hatch": "xx", "blink": True},
}

SHAPE_UNICODE = {"circle": "●", "square": "■", "triangle": "▲"}


def get_semantic_style(label: str) -> Optional[Dict[str, Any]]:
    """Looks up a semantic style by label/category text. Returns None if no match."""
    key = (label or "").strip().lower()
    return SEMANTIC_MAP.get(key)


# ---------------------------------------------------------------------------
# 3. Presets — 3 per CVD type = 9 total
# ---------------------------------------------------------------------------

PRESETS = {
    cvd: {
        "standard": {"palette": palette, "use_patterns": True, "use_shapes": True, "min_contrast": 4.5},
        # NOTE: this is capped at 5.5, not the WCAG AAA target of 7.0.
        # Testing showed pushing every color to 7:1 forces heavy, near-
        # uniform darkening, which compresses hue differences between
        # colors that started out perfectly distinct -- actively hurting
        # CVD distinguishability while improving contrast-vs-background.
        # 5.5 is a meaningfully stronger boost than "standard" without
        # collapsing the palette.
        "high_contrast": {"palette": palette, "use_patterns": True, "use_shapes": True, "min_contrast": 5.5},
        "pattern_focus": {"palette": palette, "use_patterns": True, "use_shapes": False, "min_contrast": 4.5},
    }
    for cvd, palette in CVD_PALETTES.items()
}


def get_preset(cvd_type: str, preset_name: str = "standard") -> Dict[str, Any]:
    cvd_type = cvd_type.lower()
    if cvd_type not in PRESETS:
        raise ValueError(f"Unknown CVD type '{cvd_type}'. Expected one of {list(PRESETS)}")
    if preset_name not in PRESETS[cvd_type]:
        raise ValueError(f"Unknown preset '{preset_name}'. Expected one of {list(PRESETS[cvd_type])}")
    return PRESETS[cvd_type][preset_name]


def list_all_presets() -> List[Tuple[str, str]]:
    """Returns all (cvd_type, preset_name) combinations — 9 total."""
    return [(cvd, preset) for cvd in PRESETS for preset in PRESETS[cvd]]


def auto_select_preset(analysis: Dict[str, Any], cvd_type: str) -> str:
    """
    Picks a preset using Person 2's real findings instead of guessing blind:
      - No labels anywhere on the chart (color_only_warning) -> go strongest.
      - Multiple problem pairs detected for this specific CVD type -> go strongest.
      - One problem pair -> standard is enough (patterns/shapes already added).
      - No problems detected for this CVD type -> still "standard" (Percepta
        always adds non-color cues; we just don't over-push contrast).
    """
    cvd_type = cvd_type.lower()
    severity = sum(
        1 for issue in analysis.get("problematic_pairs", [])
        if issue.get("deficiency") == cvd_type
    )
    if analysis.get("color_only_warning") or severity >= 2:
        return "high_contrast"
    return "standard"


# ---------------------------------------------------------------------------
# 4. Color helpers (Person 2 hands us RGB int tuples, not hex)
# ---------------------------------------------------------------------------

def rgb_tuple_to_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"#{int(r):02X}{int(g):02X}{int(b):02X}"


def _hex_to_rgb01(hex_color: str) -> Tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def relative_luminance(hex_color: str) -> float:
    r, g, b = _hex_to_rgb01(hex_color)

    def channel(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = channel(r), channel(g), channel(b)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(color_a: str, color_b: str) -> float:
    """WCAG contrast ratio: 1:1 (no contrast) to 21:1 (max contrast)."""
    l1, l2 = relative_luminance(color_a), relative_luminance(color_b)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def ensure_min_contrast(color_hex: str, background_hex: str = "#FFFFFF", min_ratio: float = 4.5) -> str:
    """Darkens a color step-by-step until it meets min_ratio against the background."""
    r, g, b = _hex_to_rgb01(color_hex)
    factor = 1.0
    while factor > 0.1:
        candidate = f"#{int(r * factor * 255):02X}{int(g * factor * 255):02X}{int(b * factor * 255):02X}"
        if contrast_ratio(candidate, background_hex) >= min_ratio:
            return candidate
        factor -= 0.05
    return f"#{int(r * factor * 255):02X}{int(g * factor * 255):02X}{int(b * factor * 255):02X}"


# ---------------------------------------------------------------------------
# 5. Turning Person 1's ChartData into plottable (label, value) elements
# ---------------------------------------------------------------------------

def build_elements_from_chart_data(chart_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Normalizes Person 1's image-based or CSV-based ChartData into a flat
    list of {"label": str, "value": float} elements Person 3 can plot.
    """
    source_type = chart_data.get("source_type")

    if source_type == "csv":
        return _elements_from_csv(chart_data)
    elif source_type == "image":
        return _elements_from_image(chart_data)
    else:
        raise ValueError(f"Unknown source_type: {source_type!r}")


def _elements_from_csv(chart_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    df = chart_data.get("dataframe")
    structure = chart_data.get("csv_structure") or {}
    cat_col = structure.get("likely_category_column")
    val_col = structure.get("likely_value_column")

    if df is None or cat_col is None or val_col is None:
        return []

    grouped = df.groupby(cat_col)[val_col].mean()
    return [{"label": str(label), "value": float(value)} for label, value in grouped.items()]


def _elements_from_image(chart_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    chart_elements = chart_data.get("chart_elements") or {}
    regions = chart_elements.get("color_regions") or []
    labels = chart_elements.get("labels") or []

    elements = []
    for i, region in enumerate(regions):
        label = labels[i] if i < len(labels) else f"Region {i + 1}"
        elements.append({
            "label": label,
            "value": float(region["pixel_count"]),
            "original_color": region.get("color"),
        })
    return elements


def get_axis_labels(chart_data: Dict[str, Any]) -> Tuple[str, str]:
    """Best-effort x/y axis labels from Person 1's structure."""
    if chart_data.get("source_type") == "csv":
        structure = chart_data.get("csv_structure") or {}
        return (structure.get("likely_category_column") or "Category",
                structure.get("likely_value_column") or "Value")
    return "Element", "Pixel Count"


# ---------------------------------------------------------------------------
# 6. Core reconstruction engine
# ---------------------------------------------------------------------------

class VisualizationResult:
    """Simple result container: the figure + everything Person 4 needs."""

    def __init__(self, figure: plt.Figure, metadata: Dict[str, Any]):
        self.figure = figure
        self.metadata = metadata


class AccessibleVisualizer:
    """
    Rebuilds a chart so it never relies on color alone.
    Input:  chart_data (Person 1's dict) + analysis (Person 2's dict)
    Output: matplotlib Figure + metadata dict (consumed by Person 4 / frontend)
    """

    def __init__(self, chart_data: Dict[str, Any], analysis: Dict[str, Any], chart_type: str = "bar"):
        self.chart_data = chart_data
        self.analysis = analysis
        self.chart_type = chart_type.lower()
        self.elements = build_elements_from_chart_data(chart_data)
        self.x_label, self.y_label = get_axis_labels(chart_data)

    def build(self, cvd_type: str = "deuteranopia", preset_name: Optional[str] = None) -> VisualizationResult:
        if not self.elements:
            raise ValueError(
                "No plottable elements found in chart_data — check that "
                "csv_structure has a category/value column, or that the "
                "image extraction found color regions."
            )

        cvd_type = cvd_type.lower()
        if preset_name is None:
            preset_name = auto_select_preset(self.analysis, cvd_type)

        preset = get_preset(cvd_type, preset_name)
        palette = preset["palette"]
        use_patterns = preset["use_patterns"]
        use_shapes = preset["use_shapes"]
        min_contrast = preset["min_contrast"]

        # Force on non-color cues if Person 2 flagged that the source chart
        # relies on color with zero labels — this is the worst case.
        if self.analysis.get("color_only_warning"):
            use_patterns = True

        builders = {
            "bar": self._build_bar,
            "line": self._build_line,
            "pie": self._build_pie,
            "scatter": self._build_scatter,
        }
        if self.chart_type not in builders:
            raise ValueError(f"Unsupported chart_type '{self.chart_type}'")

        fig, styled_elements = builders[self.chart_type](palette, use_patterns, use_shapes, min_contrast)

        metadata = {
            "cvd_type": cvd_type,
            "preset": preset_name,
            "chart_type": self.chart_type,
            "elements": styled_elements,
            "legend": self._build_legend_metadata(styled_elements),
            "accessibility_context": {
                "labels_present_in_source": self.analysis.get("labels_present", False),
                "color_only_warning": self.analysis.get("color_only_warning", False),
                "problems_for_this_cvd_type": sum(
                    1 for issue in self.analysis.get("problematic_pairs", [])
                    if issue.get("deficiency") == cvd_type
                ),
            },
        }
        return VisualizationResult(figure=fig, metadata=metadata)

    # -- internal helpers -------------------------------------------------

    def _style_for_element(self, el: Dict[str, Any], idx: int, palette: List[str]) -> Dict[str, Any]:
        semantic = get_semantic_style(el["label"])
        if semantic is not None:
            color = semantic["color"]
            hatch, marker, shape, blink = semantic["hatch"], semantic["marker"], semantic["shape"], semantic["blink"]
        else:
            color = palette[idx % len(palette)]
            hatch = ["", "//", "xx", "\\\\", "++", "..", "oo", "**"][idx % 8]
            marker = ["o", "s", "^", "D", "v", "P", "X", "*"][idx % 8]
            shape = ["circle", "square", "triangle", "diamond"][idx % 4]
            blink = False
        return {
            "label": el["label"],
            "value": el["value"],
            "color": color,
            "hatch": hatch,
            "marker": marker,
            "shape": shape,
            "blink": blink,
        }

    def _build_bar(self, palette, use_patterns, use_shapes, min_contrast):
        fig, ax = plt.subplots(figsize=(8, 5))
        styled = []
        for i, el in enumerate(self.elements):
            s = self._style_for_element(el, i, palette)
            color = ensure_min_contrast(s["color"], "#FFFFFF", min_contrast)
            hatch = s["hatch"] if use_patterns else ""
            ax.bar(s["label"], s["value"], color=color, hatch=hatch, edgecolor="black", linewidth=1.2)
            ax.text(i, s["value"], f'{s["value"]:.1f}', ha="center", va="bottom", fontsize=9)
            s["color"] = color
            styled.append(s)
        ax.set_title(self.chart_data.get("title", "Accessible Visualization"))
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        self._add_legend(ax, styled, use_patterns)
        fig.tight_layout()
        return fig, styled

    def _build_line(self, palette, use_patterns, use_shapes, min_contrast):
        fig, ax = plt.subplots(figsize=(8, 5))
        xs = [e["label"] for e in self.elements]
        ys = [e["value"] for e in self.elements]
        color = ensure_min_contrast(palette[0], "#FFFFFF", min_contrast)
        ax.plot(xs, ys, color=color, marker="o" if use_shapes else None, markersize=8, linewidth=2)

        styled = []
        for i, el in enumerate(self.elements):
            st = self._style_for_element(el, i, palette)
            st["color"] = color
            styled.append(st)

        ax.set_title(self.chart_data.get("title", "Accessible Visualization"))
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        fig.tight_layout()
        return fig, styled

    def _build_pie(self, palette, use_patterns, use_shapes, min_contrast):
        fig, ax = plt.subplots(figsize=(6, 6))
        styled, colors, hatches, labels, values = [], [], [], [], []
        for i, el in enumerate(self.elements):
            s = self._style_for_element(el, i, palette)
            color = ensure_min_contrast(s["color"], "#FFFFFF", min_contrast)
            s["color"] = color
            colors.append(color)
            hatches.append(s["hatch"] if use_patterns else "")
            labels.append(f'{s["label"]} ({s["value"]:.1f})')
            values.append(s["value"])
            styled.append(s)

        wedges, _ = ax.pie(values, colors=colors, startangle=90, wedgeprops={"edgecolor": "black"})
        for wedge, hatch in zip(wedges, hatches):
            wedge.set_hatch(hatch)
        ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set_title(self.chart_data.get("title", "Accessible Visualization"))
        fig.tight_layout()
        return fig, styled

    def _build_scatter(self, palette, use_patterns, use_shapes, min_contrast):
        fig, ax = plt.subplots(figsize=(8, 5))
        styled = []
        for i, el in enumerate(self.elements):
            s = self._style_for_element(el, i, palette)
            color = ensure_min_contrast(s["color"], "#FFFFFF", min_contrast)
            marker = s["marker"] if use_shapes else "o"
            ax.scatter(i, el["value"], color=color, marker=marker, s=120, edgecolor="black")
            s["color"] = color
            styled.append(s)
        ax.set_title(self.chart_data.get("title", "Accessible Visualization"))
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        fig.tight_layout()
        return fig, styled

    def _add_legend(self, ax, styled_elements, use_patterns):
        handles, seen = [], set()
        for s in styled_elements:
            key = (s["color"], s["hatch"])
            if key in seen:
                continue
            seen.add(key)
            handles.append(mpatches.Patch(
                facecolor=s["color"],
                hatch=s["hatch"] if use_patterns else "",
                edgecolor="black",
                label=s["label"],
            ))
        if handles:
            ax.legend(handles=handles, loc="best")

    def _build_legend_metadata(self, styled_elements) -> List[Dict[str, Any]]:
        legend, seen = [], set()
        for s in styled_elements:
            if s["label"] in seen:
                continue
            seen.add(s["label"])
            legend.append({
                "label": s["label"],
                "color": s["color"],
                "shape": s["shape"],
                "shape_symbol": SHAPE_UNICODE.get(s["shape"], "●"),
                "hatch": s["hatch"],
                "blink": s["blink"],
            })
        return legend


# ---------------------------------------------------------------------------
# 7. Convenience entry point — this is what Person 4 imports into pipeline.py
# ---------------------------------------------------------------------------

def generate_accessible_visualization(
    chart_data: Dict[str, Any],
    analysis: Dict[str, Any],
    cvd_type: str = "deuteranopia",
    preset_name: Optional[str] = None,
    chart_type: str = "bar",
) -> VisualizationResult:
    """
    Usage in Person 4's pipeline.py:

        from modules.visualization import generate_accessible_visualization
        result = generate_accessible_visualization(chart_data, analysis, cvd_type="protanopia")
        result.figure.savefig("output.png")
        metadata = result.metadata   # feed into scoring.py and ai.py

    `preset_name` is optional — leave it out and Percepta auto-picks
    "standard" vs "high_contrast" based on how bad Person 2's analysis
    says this CVD type is for this chart.
    `chart_type` defaults to "bar" since neither Person 1's image nor CSV
    extraction currently classifies chart type; pass "line"/"pie"/"scatter"
    explicitly if you know it.
    """
    visualizer = AccessibleVisualizer(chart_data, analysis, chart_type=chart_type)
    return visualizer.build(cvd_type=cvd_type, preset_name=preset_name)


# ---------------------------------------------------------------------------
# 8. Self-test / demo — only runs when this file is executed directly.
#    Uses synthetic dicts shaped exactly like Person 1 / Person 2's real
#    output, so it proves the integration contract without requiring
#    their files to be importable from this exact location.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if pd is None:
        raise SystemExit("pandas is required to run this demo: pip install pandas")

    # ---- Stand-in for Person 1's build_chart_data() output (CSV path) ----
    demo_df = pd.DataFrame({
        "status": ["stable", "monitor", "critical"],
        "risk_score": [2.0, 5.5, 9.0],
    })
    demo_chart_data = {
        "source_type": "csv",
        "raw_array": None,
        "dataframe": demo_df,
        "chart_elements": None,
        "csv_structure": {
            "categorical_columns": ["status"],
            "numerical_columns": ["risk_score"],
            "likely_category_column": "status",
            "likely_value_column": "risk_score",
        },
        "title": "Patient Status Overview",
    }

    # ---- Stand-in for Person 2's get_colors_and_labels() output ----
    demo_analysis = {
        "colors": [(0, 255, 0), (255, 255, 0), (255, 0, 0)],
        "labels_present": False,
        "cvd_results": {"protanopia": [], "deuteranopia": [], "tritanopia": []},
        "problematic_pairs": [
            {"type": "similar_colors", "deficiency": "deuteranopia",
             "color_1": (0, 255, 0), "color_2": (255, 255, 0), "distance": 42.1},
            {"type": "low_contrast", "deficiency": "deuteranopia",
             "color_1": (255, 255, 0), "color_2": (255, 0, 0), "brightness_difference": 12.0},
        ],
        "color_only_warning": True,
    }

    print("Available presets (9 total):")
    for cvd, preset in list_all_presets():
        print(f"  - {cvd} / {preset}")

    chosen_preset = auto_select_preset(demo_analysis, "deuteranopia")
    print(f"\nAuto-selected preset for deuteranopia given this analysis: {chosen_preset}")

    result = generate_accessible_visualization(demo_chart_data, demo_analysis, cvd_type="deuteranopia")
    result.figure.savefig("demo_output.png", dpi=150)
    print("Saved demo_output.png")
    print("Legend metadata:", result.metadata["legend"])
    print("Accessibility context:", result.metadata["accessibility_context"])
