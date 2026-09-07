"""
modules/ai.py
Person 4 - AI Explanation

Explains, in plain language:
    1. What accessibility problems were detected (from Person 2 + score)
    2. What Percepta changed (from Person 3's metadata)
    3. Why the new visualization is more accessible (from the score)

Design choice: the explanation is built deterministically from structured
data first - this always works, has zero network dependency, and is safe
for a live hackathon demo. It is then OPTIONALLY rewritten in a friendlier
voice by an LLM if an API key is available in the environment. If the LLM
call fails or no key is set, the deterministic text is used unchanged -
the pipeline never breaks because of a missing/expired key or a flaky
connection.

Supported keys (checked in this order): ANTHROPIC_API_KEY, OPENAI_API_KEY.
Neither `anthropic` nor `openai` packages are required unless you set one
of these - both imports are lazy and wrapped in try/except.
"""

from __future__ import annotations

import os
from typing import Any, Dict

SHAPE_UNICODE = {"circle": "●", "square": "■", "triangle": "▲", "diamond": "◆"}


def _describe_problems(analysis: Dict[str, Any], cvd_type: str) -> str:
    issues = [p for p in analysis.get("problematic_pairs", []) if p.get("deficiency") == cvd_type]

    if not issues and not analysis.get("color_only_warning"):
        return f"No significant color-accessibility problems were detected for {cvd_type}."

    parts = []
    if analysis.get("color_only_warning"):
        parts.append("the chart communicated its information through color alone, with no labels")
    if issues:
        similar = [i for i in issues if i["type"] == "similar_colors"]
        low_contrast = [i for i in issues if i["type"] == "low_contrast"]
        if similar:
            parts.append(f"{len(similar)} pair(s) of colors become nearly indistinguishable under {cvd_type}")
        if low_contrast:
            parts.append(f"{len(low_contrast)} pair(s) of colors have too little brightness contrast under {cvd_type}")

    return f"Under {cvd_type}, " + "; and ".join(parts) + "."


def _describe_changes(viz_metadata: Dict[str, Any]) -> str:
    legend = viz_metadata.get("legend", [])
    preset = viz_metadata.get("preset", "standard")
    parts = [f"Percepta rebuilt the chart using the '{preset}' preset for {viz_metadata.get('cvd_type')}."]

    shapes_used = {entry["shape"] for entry in legend}
    hatches_used = {entry["hatch"] for entry in legend if entry["hatch"]}

    if len(shapes_used) > 1:
        shape_list = ", ".join(f"{SHAPE_UNICODE.get(s, '')} {s}" for s in shapes_used)
        parts.append(f"Each category now has its own shape marker ({shape_list}).")
    if hatches_used:
        parts.append("Distinct fill patterns (hatching) were added so categories stay visible even in grayscale.")
    parts.append("A colorblind-safe palette replaced the original colors, and every element now has a text label.")

    blinking = [entry["label"] for entry in legend if entry.get("blink")]
    if blinking:
        parts.append(f"High-priority categories ({', '.join(blinking)}) are flagged for an on-screen blink cue.")

    return " ".join(parts)


def _describe_why_it_helps(score: Dict[str, Any]) -> str:
    total = score.get("total_score", 0)
    breakdown = score.get("breakdown", {})
    parts = [f"The reconstructed visualization scores {total}/100 on Percepta's accessibility scale."]

    cvd = breakdown.get("cvd_distinguishability", {})
    if cvd.get("remaining_issues", 1) == 0:
        parts.append("Every color pair is now distinguishable under the simulated color vision deficiency.")

    contrast = breakdown.get("contrast", {})
    if contrast.get("average_contrast_ratio", 0) >= 4.5:
        parts.append(f"Average contrast is {contrast['average_contrast_ratio']}:1, meeting the WCAG AA guideline.")

    if breakdown.get("color_independence", {}).get("color_independent"):
        parts.append("Because shape and pattern now carry the meaning, the chart stays readable even without color.")

    return " ".join(parts)


def _build_rule_based_explanation(
    analysis: Dict[str, Any],
    viz_metadata: Dict[str, Any],
    score: Dict[str, Any],
) -> Dict[str, str]:
    cvd_type = viz_metadata.get("cvd_type", "deuteranopia")
    return {
        "problems_detected": _describe_problems(analysis, cvd_type),
        "what_changed": _describe_changes(viz_metadata),
        "why_it_helps": _describe_why_it_helps(score),
    }


def _try_llm_polish(explanation: Dict[str, str]) -> Dict[str, str]:
    """
    Optional: rewrite the rule-based explanation in a friendlier voice.
    Silently returns the original explanation on ANY failure (no key,
    package not installed, network error, bad response, ...).
    """
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not anthropic_key and not openai_key:
        return explanation

    prompt = (
        "Rewrite the following accessibility explanation in a warm, clear, "
        "non-technical tone for a hospital dashboard user. Keep it to three "
        "short paragraphs matching these three points, in this order, "
        "without inventing new facts:\n\n"
        f"1. Problems detected: {explanation['problems_detected']}\n"
        f"2. What changed: {explanation['what_changed']}\n"
        f"3. Why it helps: {explanation['why_it_helps']}\n"
    )

    try:
        if anthropic_key:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            response = client.messages.create(
                model="claude-sonnet-5",
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(block.text for block in response.content if hasattr(block, "text"))
        else:
            import openai
            client = openai.OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
            )
            text = response.choices[0].message.content

        if text and text.strip():
            explanation = dict(explanation)
            explanation["polished_text"] = text.strip()
        return explanation

    except Exception:
        return explanation


def generate_ai_explanation(
    chart_data: Dict[str, Any],
    analysis: Dict[str, Any],
    viz_metadata: Dict[str, Any],
    score: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Main Person 4 entry point.

    Usage:
        from modules.ai import generate_ai_explanation
        explanation = generate_ai_explanation(chart_data, analysis, viz_result.metadata, score)
        print(explanation["full_text"])

    Returns:
        {
            "problems_detected": str,
            "what_changed": str,
            "why_it_helps": str,
            "polished_text": str,   # only present if an LLM call succeeded
            "full_text": str,       # always present - what the UI should render
        }
    """
    explanation = _build_rule_based_explanation(analysis, viz_metadata, score)
    explanation = _try_llm_polish(explanation)

    if "polished_text" in explanation:
        explanation["full_text"] = explanation["polished_text"]
    else:
        explanation["full_text"] = (
            f"**What we found:** {explanation['problems_detected']}\n\n"
            f"**What we changed:** {explanation['what_changed']}\n\n"
            f"**Why it helps:** {explanation['why_it_helps']}"
        )

    return explanation


if __name__ == "__main__":
    demo_analysis = {
        "problematic_pairs": [
            {"type": "similar_colors", "deficiency": "deuteranopia", "color_1": (0, 255, 0), "color_2": (255, 255, 0)},
        ],
        "color_only_warning": True,
    }
    demo_viz_metadata = {
        "cvd_type": "deuteranopia",
        "preset": "high_contrast",
        "legend": [
            {"label": "critical", "color": "#8A3D00", "shape": "triangle", "hatch": "xx", "blink": True},
            {"label": "monitor", "color": "#5F5B1A", "shape": "square", "hatch": "//", "blink": False},
            {"label": "stable", "color": "#00664A", "shape": "circle", "hatch": "", "blink": False},
        ],
    }
    demo_score = {
        "total_score": 87,
        "breakdown": {
            "cvd_distinguishability": {"remaining_issues": 1},
            "contrast": {"average_contrast_ratio": 7.22},
            "color_independence": {"color_independent": True},
        },
    }

    result = generate_ai_explanation({}, demo_analysis, demo_viz_metadata, demo_score)
    print(result["full_text"])
