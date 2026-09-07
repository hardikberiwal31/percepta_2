"""
modules/input_extraction.py
Person 1 - Input & Data Extraction

Responsibilities:
    1. Accept .jpg, .png, .csv uploads.
    2. Detect whether the input is a "chart" (image) or a "dataset" (csv).
    3. For images: extract chart elements - color regions (bars/lines/
       points), and attempt text extraction for labels/axes/legend via OCR
       (best-effort; falls back gracefully if OCR isn't installed).
    4. For CSVs: identify columns, categorical vs numerical data, and
       likely relationships between columns (e.g. which column is a
       category axis vs a value axis).
    5. Package everything into one common `ChartData` structure that
       Person 2 (CVD & Color Analysis) can consume regardless of
       whether the original input was an image or a CSV.

Output contract - the ChartData structure (see build_chart_data()):
    {
        "source_type": "image" | "csv",
        "raw_array": np.ndarray | None,       # RGB image array, if image
        "dataframe": pd.DataFrame | None,      # parsed CSV, if csv
        "chart_elements": {
            "color_regions": [ {"color": (R,G,B), "pixel_count": int,
                                 "bbox": (x, y, w, h)}, ... ],
            "labels": [ str, ... ],             # OCR text found, best-effort
        } | None,
        "csv_structure": {
            "categorical_columns": [str, ...],
            "numerical_columns": [str, ...],
            "likely_category_column": str | None,
            "likely_value_column": str | None,
        } | None,
    }
"""

import os
import numpy as np
import pandas as pd
import cv2

# OCR is optional - the extractor works without it, just returns an
# empty labels list if pytesseract/tesseract isn't installed on this
# machine. This keeps the module usable even before OCR is set up.
try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


# ---------------------------------------------------------------------
# File loading + input type detection
# ---------------------------------------------------------------------

def load_file(file_path: str) -> dict:
    """
    Detect file type from extension and load it appropriately.

    Args:
        file_path: path to a .jpg, .jpeg, .png, or .csv file.

    Returns:
        dict with "source_type" ("image"/"csv") and the loaded content
        under "raw_array" or "dataframe".
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext in (".jpg", ".jpeg", ".png"):
        img_bgr = cv2.imread(file_path)
        if img_bgr is None:
            raise ValueError(f"Could not read image file: {file_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return {"source_type": "image", "raw_array": img_rgb, "dataframe": None}

    elif ext == ".csv":
        df = pd.read_csv(file_path)
        return {"source_type": "csv", "raw_array": None, "dataframe": df}

    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. Expected .jpg, .jpeg, .png, or .csv"
        )


def detect_input_type(file_path: str) -> str:
    """
    Quick classification of the upload as "chart" (image) or "dataset" (csv).

    Args:
        file_path: path to the uploaded file.

    Returns:
        "chart" or "dataset".
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".jpg", ".jpeg", ".png"):
        return "chart"
    elif ext == ".csv":
        return "dataset"
    else:
        raise ValueError(f"Unsupported file type '{ext}'")


# ---------------------------------------------------------------------
# Image-based chart element extraction
# ---------------------------------------------------------------------

def extract_color_regions(image_array: np.ndarray, num_colors: int = 6,
                           min_region_fraction: float = 0.01) -> list:
    """
    Detect distinct color regions in a chart image - a scoped-down
    stand-in for full bar/line/point detection. Uses K-Means to find
    dominant colors, then finds the bounding box of each color's
    pixels via a color mask. Small/noise regions (background
    anti-aliasing etc.) below min_region_fraction are dropped.

    Args:
        image_array: np.ndarray (H, W, 3), RGB, uint8.
        num_colors: how many dominant colors to look for.
        min_region_fraction: minimum fraction of total pixels a color
                              must cover to be considered a real
                              chart element (filters out noise).

    Returns:
        List of dicts: {"color": (R,G,B), "pixel_count": int,
                         "bbox": (x, y, w, h)}
        sorted by pixel_count descending.
    """
    h, w = image_array.shape[:2]
    total_pixels = h * w

    pixels = image_array.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    _compactness, labels, centers = cv2.kmeans(
        pixels, num_colors, None, criteria, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS
    )
    labels_2d = labels.reshape(h, w)

    regions = []
    for i, center in enumerate(centers):
        mask = (labels_2d == i).astype(np.uint8)
        pixel_count = int(mask.sum())

        if pixel_count / total_pixels < min_region_fraction:
            continue  # too small to be a meaningful chart element

        ys, xs = np.where(mask == 1)
        if len(xs) == 0:
            continue
        bbox = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min()))

        color = tuple(int(round(v)) for v in center)
        regions.append({"color": color, "pixel_count": pixel_count, "bbox": bbox})

    regions.sort(key=lambda r: -r["pixel_count"])
    return regions


def extract_labels(image_array: np.ndarray) -> list:
    """
    Best-effort OCR text extraction for axis labels, legend text, etc.
    Returns an empty list if OCR isn't available on this machine rather
    than crashing - so the rest of the pipeline still works without it.

    Args:
        image_array: np.ndarray (H, W, 3), RGB, uint8.

    Returns:
        List of non-empty text strings detected in the image.
    """
    if not OCR_AVAILABLE:
        return []

    try:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        raw_text = pytesseract.image_to_string(gray)
        lines = [line.strip() for line in raw_text.split("\n")]
        return [line for line in lines if line]
    except Exception:
        # Tesseract binary might be missing even if pytesseract (the
        # Python wrapper) is installed - fail gracefully either way.
        return []


def extract_chart_elements(image_array: np.ndarray) -> dict:
    """
    Full image-extraction entry point: color regions + labels.

    Args:
        image_array: np.ndarray (H, W, 3), RGB, uint8.

    Returns:
        {"color_regions": [...], "labels": [...]}
    """
    return {
        "color_regions": extract_color_regions(image_array),
        "labels": extract_labels(image_array),
    }


# ---------------------------------------------------------------------
# CSV-based structure extraction
# ---------------------------------------------------------------------

def classify_columns(df: pd.DataFrame) -> dict:
    """
    Split a DataFrame's columns into categorical vs numerical.

    Args:
        df: the uploaded CSV.

    Returns:
        {"categorical_columns": [...], "numerical_columns": [...]}
    """
    categorical, numerical = [], []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numerical.append(col)
        else:
            categorical.append(col)
    return {"categorical_columns": categorical, "numerical_columns": numerical}


def detect_relationships(df: pd.DataFrame, column_types: dict) -> dict:
    """
    Best-effort guess at which categorical column is the "category axis"
    and which numerical column is the "value axis" - e.g. for a bar
    chart rebuilt from this CSV, what goes on X vs Y.

    Args:
        df: the uploaded CSV.
        column_types: output of classify_columns().

    Returns:
        {"likely_category_column": str | None, "likely_value_column": str | None}
    """
    categorical = column_types["categorical_columns"]
    numerical = column_types["numerical_columns"]

    likely_category_column = None
    likely_value_column = None

    # Prefer a low-cardinality categorical column (few unique values =
    # good candidate for grouping/x-axis categories)
    if categorical:
        cardinalities = {col: df[col].nunique() for col in categorical}
        likely_category_column = min(cardinalities, key=cardinalities.get)

    # Prefer the first numerical column as the value/y-axis
    if numerical:
        likely_value_column = numerical[0]

    return {
        "likely_category_column": likely_category_column,
        "likely_value_column": likely_value_column,
    }


def extract_csv_structure(df: pd.DataFrame) -> dict:
    """
    Full CSV-extraction entry point: column types + likely relationships.

    Args:
        df: the uploaded CSV.

    Returns:
        {"categorical_columns": [...], "numerical_columns": [...],
         "likely_category_column": str | None, "likely_value_column": str | None}
    """
    column_types = classify_columns(df)
    relationships = detect_relationships(df, column_types)
    return {**column_types, **relationships}


# ---------------------------------------------------------------------
# Main entry point - builds the shared ChartData structure
# ---------------------------------------------------------------------

def build_chart_data(file_path: str) -> dict:
    """
    Main Person 1 entry point. Loads the file, runs the appropriate
    extraction pipeline, and returns the standardized ChartData
    structure for Person 2 to consume.

    Args:
        file_path: path to the uploaded .jpg, .jpeg, .png, or .csv file.

    Returns:
        dict matching the ChartData contract (see module docstring).
    """
    loaded = load_file(file_path)

    chart_data = {
        "source_type": loaded["source_type"],
        "raw_array": loaded["raw_array"],
        "dataframe": loaded["dataframe"],
        "chart_elements": None,
        "csv_structure": None,
    }

    if loaded["source_type"] == "image":
        chart_data["chart_elements"] = extract_chart_elements(loaded["raw_array"])

    elif loaded["source_type"] == "csv":
        chart_data["csv_structure"] = extract_csv_structure(loaded["dataframe"])

    return chart_data


if __name__ == "__main__":
    # ---- Smoke test 1: image path ----
    test_img = np.zeros((100, 150, 3), dtype=np.uint8)
    test_img[:, :50] = [220, 50, 50]     # red bar
    test_img[:, 50:100] = [60, 180, 75]  # green bar
    test_img[:, 100:] = [255, 225, 25]   # yellow bar
    cv2.imwrite("test_chart.png", cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR))

    chart_data_image = build_chart_data("test_chart.png")
    print("=== Image test ===")
    print("source_type:", chart_data_image["source_type"])
    print("color_regions found:", len(chart_data_image["chart_elements"]["color_regions"]))
    for region in chart_data_image["chart_elements"]["color_regions"]:
        print("  -", region)
    print("labels found (OCR available:", OCR_AVAILABLE, "):",
          chart_data_image["chart_elements"]["labels"])

    # ---- Smoke test 2: CSV path ----
    test_df = pd.DataFrame({
        "quarter": ["Q1", "Q2", "Q3", "Q4"],
        "revenue": [100, 120, 115, 140],
        "region": ["North", "South", "North", "South"],
    })
    test_df.to_csv("test_data.csv", index=False)

    chart_data_csv = build_chart_data("test_data.csv")
    print("\n=== CSV test ===")
    print("source_type:", chart_data_csv["source_type"])
    print("csv_structure:", chart_data_csv["csv_structure"])

    # Cleanup test artifacts
    os.remove("test_chart.png")
    os.remove("test_data.csv")
