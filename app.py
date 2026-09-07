"""
app.py
Percepta - Streamlit frontend

Wires together:
    modules.input_extraction  (Person 1)
    modules.cvd_analysis      (Person 2)
    modules.visualization     (Person 3)
    modules.scoring, modules.ai, modules.pipeline (Person 4)

Run with:
    streamlit run app.py
"""

import os
import tempfile

import streamlit as st

from modules.pipeline import run_pipeline
from modules.visualization import list_all_presets

st.set_page_config(page_title="Percepta", page_icon=":art:", layout="wide")

st.title("Percepta")
st.caption(
    "AI-powered accessibility for data visualizations, built for people "
    "with Color Vision Deficiency (CVD)."
)

with st.sidebar:
    st.header("1. Upload")
    uploaded_file = st.file_uploader(
        "Upload a chart image or dataset",
        type=["jpg", "jpeg", "png", "csv"],
    )

    st.header("2. Settings")
    cvd_type = st.selectbox(
        "Simulate for",
        options=["protanopia", "deuteranopia", "tritanopia"],
        index=1,
        help="Which type of color vision deficiency to optimize for.",
    )

    preset_options = ["auto"] + sorted({preset for _, preset in list_all_presets()})
    preset_choice = st.selectbox(
        "Preset",
        options=preset_options,
        index=0,
        help=(
            "'auto' lets Percepta pick standard vs. high_contrast based on "
            "how many real problems were detected for this CVD type."
        ),
    )
    preset_name = None if preset_choice == "auto" else preset_choice

    chart_type = st.selectbox(
        "Chart type",
        options=["bar", "line", "pie", "scatter"],
        index=0,
        help="Neither the image nor CSV extraction currently detects this automatically.",
    )

    run_clicked = st.button("Analyze & Fix", type="primary", use_container_width=True)


if not uploaded_file:
    st.info("Upload a chart image (JPG/PNG) or a dataset (CSV) from the sidebar to get started.")
    st.stop()

if not run_clicked:
    st.info("Adjust settings in the sidebar, then click **Analyze & Fix**.")
    st.stop()

# Person 1's build_chart_data() expects a real file path, so persist the
# upload to a temp file first.
suffix = os.path.splitext(uploaded_file.name)[1]
tmp_path = None
try:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    with st.spinner("Analyzing colors, simulating CVD, and rebuilding the visualization..."):
        result = run_pipeline(
            tmp_path,
            cvd_type=cvd_type,
            preset_name=preset_name,
            chart_type=chart_type,
        )
except Exception as exc:
    st.error(f"Something went wrong while processing this file: {exc}")
    st.stop()
finally:
    if tmp_path and os.path.exists(tmp_path):
        os.remove(tmp_path)

chart_data = result["chart_data"]
analysis = result["analysis"]
score = result["score"]
explanation = result["explanation"]
viz_metadata = result["viz_metadata"]

col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("Accessible Visualization")
    st.pyplot(result["figure"], use_container_width=True)

    if chart_data.get("source_type") == "image" and chart_data.get("raw_array") is not None:
        with st.expander("Original uploaded chart"):
            st.image(chart_data["raw_array"], use_container_width=True)

with col_right:
    st.subheader("Accessibility Score")
    st.metric("Overall score", f"{score['total_score']} / 100")
    st.progress(score["total_score"] / 100)

    with st.expander("Score breakdown"):
        for name, component in score["breakdown"].items():
            st.write(f"**{name.replace('_', ' ').title()}**: {component.get('score', 0)} pts")

    st.subheader("Recommendations")
    for rec in score["recommendations"]:
        st.write(f"- {rec}")

st.divider()
st.subheader("AI Explanation")
st.markdown(explanation["full_text"])

st.divider()
st.subheader("Legend")
legend = viz_metadata.get("legend", [])
if legend:
    legend_cols = st.columns(len(legend))
    for col, entry in zip(legend_cols, legend):
        with col:
            st.markdown(f"### {entry['shape_symbol']}")
            st.write(entry["label"])
            st.color_picker(
                "Color",
                value=entry["color"],
                key=f"legend_{entry['label']}",
                disabled=True,
                label_visibility="collapsed",
            )
            if entry["blink"]:
                st.caption("Flagged for blinking cue")

st.divider()
if analysis.get("color_only_warning"):
    st.warning(
        "The original visualization relied on color alone with no labels. "
        "Percepta added labels, shapes, and patterns to fix this."
    )
