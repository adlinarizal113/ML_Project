from pathlib import Path
import json

import pandas as pd
import streamlit as st

# =====================================
# 1. FILE PATHS AND CONFIGURATION
# =====================================

ROOT = Path(__file__).resolve().parent

DATA = ROOT / "data" / "miros_cleaned.csv"
DATA_DIR = ROOT / "data"
EXCEL = ROOT / "MIROS_original.xlsx"

MODEL = ROOT / "model" / "pipeline.joblib"
SCHEMA = ROOT / "model" / "input_schema.json"

TARGET = "Type of Crash"

st.set_page_config(
    page_title="MIROS Accident Severity",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Road Accident Severity Prediction")
st.caption(
    "Pahang, Malaysia | Machine Learning Project | Prototype"
)

# =====================================
# 2. LOAD DATASET
# =====================================

@st.cache_data(show_spinner=False)
def load_csv(path):
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_excel(path):
    book = pd.ExcelFile(path)
    sheet = "undergrad" if "undergrad" in book.sheet_names else book.sheet_names[0]
    return pd.read_excel(book, sheet_name=sheet)


def get_dataset():
    if DATA.exists():
        return load_csv(DATA), "Cleaned Dataset"

    if EXCEL.exists():
        return load_excel(EXCEL), "Original MIROS Dataset (Testing)"

    # Support the original MIROS filename in either repository root or data/.
    candidates = []
    for folder in (ROOT, DATA_DIR):
        if folder.exists():
            candidates.extend(
                p for p in folder.iterdir()
                if p.is_file()
                and p.suffix.lower() in {".xlsx", ".xls"}
                and "miros" in p.name.lower()
                and not p.name.startswith("~$")
            )
    if candidates:
        chosen = sorted(candidates, key=lambda p: str(p).lower())[0]
        return load_excel(chosen), "Original MIROS Dataset (Testing)"

    return None, None


# =====================================
# 3. SIDEBAR NAVIGATION
# =====================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    ["Dashboard", "Prediction", "About"]
)

st.sidebar.divider()
st.sidebar.caption("MIROS Machine Learning Project")


# =====================================
# 4. DASHBOARD
# =====================================

if page == "Dashboard":

    st.header("📊 Road Accident Dashboard")
