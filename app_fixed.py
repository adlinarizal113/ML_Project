from pathlib import Path
import json

import pandas as pd
import streamlit as st

# =====================================
# 1. FILE PATHS AND CONFIGURATION
# =====================================

ROOT = Path(__file__).parent

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

    # Find other MIROS Excel filenames without renaming the original file.
    if DATA_DIR.exists():
        candidates = sorted(
            (p for p in DATA_DIR.iterdir()
             if p.is_file() and p.suffix.lower() in {".xlsx", ".xls"}
             and "miros" in p.name.lower() and not p.name.startswith("~$")),
            key=lambda p: p.name.lower()
        )
        if candidates:
            return load_excel(candidates[0]), "Original MIROS Dataset (Testing)"

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

    st.write(
        "Explore road accident data in Pahang, Malaysia."
    )

    try:
        df, source = get_dataset()

    except Exception as error:
        st.error(f"Unable to load dataset: {error}")
        st.stop()

    if df is None:

        st.warning("Dataset is not available yet.")

        st.info(
            "Add your MIROS Excel file into the data folder "
            "for testing, or wait for Gabriellyn to provide "
            "miros_cleaned.csv."
        )

    else:

        if source == "Original MIROS Dataset (Testing)":
            st.info(
                "Currently using the original MIROS dataset "
                "for interface testing. This is not the final "
                "cleaned dataset."
            )
        else:
            st.success("Cleaned dataset loaded successfully.")

        # DATASET OVERVIEW
        st.subheader("Dataset Overview")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Records",
            f"{len(df):,}"
        )

        col2.metric(
            "Total Variables",
            len(df.columns)
        )

        col3.metric(
            "Missing Values",
            f"{int(df.isna().sum().sum()):,}"
        )

        st.divider()

        # FILTER DATA
        st.subheader("Filter Accident Data")

        filtered_df = df.copy()

        filter_columns = [
            "Vehicle Type",
            "Weather",
            "Road Condition"
        ]

        filter_layout = st.columns(3)

        for index, column in enumerate(filter_columns):

            if column in df.columns:

                options = sorted(
                    df[column]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                with filter_layout[index]:

                    selected = st.multiselect(
                        column,
                        options,
                        key=f"filter_{column}"
                    )

                if selected:

                    filtered_df = filtered_df[
                        filtered_df[column]
                        .astype(str)
                        .isin(selected)
                    ]

            else:

                with filter_layout[index]:
                    st.caption(
                        f"{column} is not available."
                    )

        st.write(
            f"Showing **{len(filtered_df):,}** "
            f"of **{len(df):,}** records"
        )

        if filtered_df.empty:
            st.warning(
                "No records match the selected filters."
            )

        st.divider()

        # ACCIDENT SEVERITY
        st.subheader("Accident Severity Distribution")

        if TARGET in filtered_df.columns:

            severity_counts = (
                filtered_df[TARGET]
                .fillna("Missing")
                .astype(str)
                .value_counts()
            )

            st.bar_chart(severity_counts)

        else:
            st.warning(
                "Type of Crash column was not found."
            )

        st.divider()

        # VEHICLE AND WEATHER CHARTS
        chart1, chart2 = st.columns(2)

        with chart1:

            st.subheader("Vehicle Type Distribution")

            if "Vehicle Type" in filtered_df.columns:

                vehicle_counts = (
                    filtered_df["Vehicle Type"]
                    .fillna("Missing")
                    .astype(str)
                    .value_counts()
                    .head(10)
                )

                st.bar_chart(vehicle_counts)

            else:
                st.info("Vehicle Type column not available.")

        with chart2:

            st.subheader("Weather Conditions")

            if "Weather" in filtered_df.columns:

                weather_counts = (
                    filtered_df["Weather"]
                    .fillna("Missing")
                    .astype(str)
                    .value_counts()
                )

                st.bar_chart(weather_counts)

            else:
                st.info("Weather column not available.")

        st.divider()

        # ROAD CONDITION
        st.subheader("Road Condition Distribution")

        if "Road Condition" in filtered_df.columns:

            road_counts = (
                filtered_df["Road Condition"]
                .fillna("Missing")
                .astype(str)
                .value_counts()
            )

            st.bar_chart(road_counts)

        else:
            st.info("Road Condition column not available.")

        st.divider()

        # DATASET PREVIEW
        st.subheader("Dataset Preview")

        st.dataframe(
            filtered_df.head(100),
            use_container_width=True
        )

        st.caption(
            "Dashboard statistics are descriptive only. "
            "They do not establish the causes of accidents."
        )


# =====================================
# 5. PREDICTION
# =====================================

elif page == "Prediction":

    st.header("🔮 Predict Accident Severity")

    st.write(
        "Enter accident information to predict "
        "the severity category."
    )

    if not MODEL.exists() or not SCHEMA.exists():

        st.warning(
            "Prediction is not available yet."
        )

        st.info(
            "Waiting for Nurin (ML Engineer) to provide "
            "the trained model and input schema."
        )

    else:

        import joblib

        try:

            schema = json.loads(
                SCHEMA.read_text(encoding="utf-8")
            )

            features = schema["features"]

            if not features or not all(
                "name" in item and "type" in item
                for item in features
            ):
                st.error("Invalid input schema.")
                st.stop()

            values = {}

            with st.form("prediction_form"):

                for item in features:

                    name = item["name"]
                    field_type = item["type"]

                    if field_type == "category":

                        options = item.get("options", [])

                        if not options:
                            st.error(
                                f"No options defined for {name}"
                            )
                            st.stop()

                        values[name] = st.selectbox(
                            name,
                            options
                        )

                    elif field_type == "number":

                        values[name] = st.number_input(
                            name,
                            value=float(
                                item.get("default", 0)
                            ),
                            min_value=item.get("min"),
                            max_value=item.get("max")
                        )

                    else:
                        st.error(
                            f"Unsupported input type: {name}"
                        )
                        st.stop()

                submitted = st.form_submit_button(
                    "Predict Severity"
                )

            if submitted:

                # Load only a trusted model from the team.
                pipeline = joblib.load(MODEL)

                sample = pd.DataFrame(
                    [values],
                    columns=[
                        item["name"]
                        for item in features
                    ]
                )

                prediction = pipeline.predict(sample)[0]

                st.success(
                    f"Predicted Accident Severity: {prediction}"
                )

                if hasattr(pipeline, "predict_proba"):

                    probabilities = pipeline.predict_proba(
                        sample
                    )[0]

                    labels = getattr(
                        pipeline,
                        "classes_",
                        None
                    )

                    if labels is not None:

                        st.subheader(
                            "Prediction Probabilities"
                        )

                        probability_data = pd.Series(
                            probabilities,
                            index=[
                                str(label)
                                for label in labels
                            ]
                        )

                        st.bar_chart(probability_data)

                st.caption(
                    "Educational prediction only. "
                    "Not an official road safety assessment."
                )

        except Exception as error:

            st.error(
                f"Model integration error: {error}"
            )

            st.info(
                "Please ask Nurin to verify the model, "
                "input features and package versions."
            )


# =====================================
# 6. ABOUT PROJECT
# =====================================

else:

    st.header("ℹ️ About This Project")

    st.subheader("Project Title")

    st.write(
        "Road Accident Severity Prediction "
        "Using Machine Learning in Pahang, Malaysia"
    )

    st.subheader("Project Objective")

    st.write(
        "This project aims to develop a machine learning "
        "system to classify road accident severity using "
        "Malaysian MIROS accident data."
    )

    st.subheader("Main Dataset")

    st.write(
        "MIROS-UMP DAX Challenge 2026 Dataset"
    )

    st.subheader("Target Variable")

    st.write("Type of Crash (Proposed)")

    st.subheader("Project Team")

    st.write("Project Leader: Gabriella")
    st.write("Data Engineer: Gabriellyn")
    st.write("Data Analyst: Zul")
    st.write("ML Engineer: Nurin")
    st.write("System Developer: Adlina")

    st.divider()

    st.info(
        "This system classifies accident severity "
        "within the scope of the training data. "
        "It does not predict whether an accident "
        "will occur."
    )
