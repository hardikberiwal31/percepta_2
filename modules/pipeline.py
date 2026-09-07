"""
modules/pipeline.py
Person 4 - Final Pipeline

Connects: Input -> CVD Analysis -> Visualization -> Scoring -> AI Explanation

This is the single function app.py (Streamlit) should call.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

try:
    from .input_extraction import build_chart_data
    from .cvd_analysis import get_colors_and_labels
    from .visualization import generate_accessible_visualization
    from .scoring import calculate_accessibility_score
    from .ai import generate_ai_explanation
except ImportError:
    from input_extraction import build_chart_data
    from cvd_analysis import get_colors_and_labels
    from visualization import generate_accessible_visualization
    from scoring import calculate_accessibility_score
    from ai import generate_ai_explanation


def run_pipeline(
    file_path: str,
    cvd_type: str = "deuteranopia",
    preset_name: Optional[str] = None,
    chart_type: str = "bar",
) -> Dict[str, Any]:
    """
    Runs the full Percepta pipeline end to end on an uploaded file.

    Args:
        file_path: path to a .jpg/.jpeg/.png/.csv file on disk.
        cvd_type: "protanopia" | "deuteranopia" | "tritanopia".
        preset_name: "standard" | "high_contrast" | "pattern_focus", or
                     None to let Person 3 auto-pick based on severity.
        chart_type: "bar" | "line" | "pie" | "scatter". Neither Person 1's
                    image nor CSV extraction currently classifies chart
                    type, so this defaults to "bar" unless you know better.

    Returns:
        {
            "chart_data": dict,     # Person 1's ChartData
            "analysis": dict,       # Person 2's CVD/color analysis
            "figure": matplotlib.figure.Figure,
            "viz_metadata": dict,   # Person 3's legend/preset/element metadata
            "score": dict,          # Person 4's 0-100 accessibility score
            "explanation": dict,    # Person 4's AI explanation
        }

    Raises:
        ValueError: unsupported file type, or no plottable data found
                    (e.g. a CSV with no usable category/value columns).
    """
    chart_data = build_chart_data(file_path)
    analysis = get_colors_and_labels(chart_data)

    viz_result = generate_accessible_visualization(
        chart_data,
        analysis,
        cvd_type=cvd_type,
        preset_name=preset_name,
        chart_type=chart_type,
    )

    score = calculate_accessibility_score(viz_result.metadata, cvd_type=cvd_type)
    explanation = generate_ai_explanation(chart_data, analysis, viz_result.metadata, score)

    return {
        "chart_data": chart_data,
        "analysis": analysis,
        "figure": viz_result.figure,
        "viz_metadata": viz_result.metadata,
        "score": score,
        "explanation": explanation,
    }


if __name__ == "__main__":
    import os
    import pandas as pd

    demo_df = pd.DataFrame({
        "status": ["stable", "monitor", "critical"],
        "risk_score": [2.0, 5.5, 9.0],
    })
    demo_df.to_csv("demo_pipeline.csv", index=False)

    try:
        result = run_pipeline("demo_pipeline.csv", cvd_type="protanopia")
        result["figure"].savefig("demo_pipeline_output.png", dpi=150)

        print("=== FULL PIPELINE OK ===")
        print("Score:", result["score"]["total_score"])
        print("Recommendations:", result["score"]["recommendations"])
        print()
        print(result["explanation"]["full_text"])
    finally:
        os.remove("demo_pipeline.csv")
