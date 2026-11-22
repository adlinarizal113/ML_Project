# MIROS Road Accident Severity Prediction — System Developer Starter

## Open in VS Code
1. Open this project folder in VS Code.
2. Open Terminal > New Terminal.
3. Create a virtual environment:
   - Windows: `py -m venv .venv`
   - Activate (PowerShell): `.\\.venv\\Scripts\\Activate.ps1`
4. Install: `python -m pip install -r requirements.txt`
5. Run: `python -m streamlit run app.py`

## Team handoffs
- **Gabriellyn (Data Engineer)**: place validated CSV at `data/miros_cleaned.csv`. Keep exact MIROS column names, especially `Type of Crash`. Do not publish sensitive raw data without permission.
- **Zul (Data Analyst)**: propose useful EDA plots and interpretations based on the cleaned data.
- **Nurin (ML Engineer)**: provide `model/pipeline.joblib` containing the FULL fitted preprocessing + classifier pipeline and `model/input_schema.json` with EXACT training feature names, data types and actual category labels. The example schema is NOT based on verified category values and must not be used as the final schema.
- **Adlina (System Developer)**: maintain app, input validation, GitHub, tests, deployment and documentation.

**Important:** The example JSON is a formatting reference only. Never copy it to `input_schema.json` unless Nurin verifies that all feature names, categories and types match the trained model. Load joblib files only from trusted team members. Check leakage and group splitting by accident identifier with Nurin.

## Project layout
```
app.py
requirements.txt
README.md
data/miros_cleaned.csv          # supplied later; not included
model/pipeline.joblib           # supplied later; not included
model/input_schema.json        # supplied later; not included
model/input_schema.example.json
```

## Deployment
Push the code to GitHub after checking dataset-sharing permissions. Deploy with Streamlit Community Cloud or an assignment-approved platform. Confirm the Python and package versions with Nurin and pin them before deployment.
