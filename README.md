# CompositeAI

AI-Based Strength Prediction and Stacking Sequence Optimization of Aerospace Composite Laminates.

## Project Overview

CompositeAI is a Streamlit platform for aerospace composite laminate analysis. Current stabilized app scope includes:

- ANN/MLP tensile-strength prediction
- dataset exploration, profiling, EDA, feature engineering, and preprocessing views
- CLT analysis and AI-vs-CLT comparison
- stacking-sequence optimization
- dataset-driven material benchmark comparison
- PDF / HTML / CSV report outputs
- Gemini assistant with non-fatal fallback behavior

Locked ML dataset:

```text
data/kaggle/2/composite_material_strength.csv
```

Benchmark material dataset:

```text
data/kaggle/3/aerospace_structural_design_dataset.csv
```

Confirmed ML target column:

```text
tensile_strength_mpa
```

## Navigation

- `PROJECT`
  - Dashboard
  - About Project
  - Workflow
- `DATA`
  - Dataset Explorer
  - Dataset Profile
  - EDA
  - Feature Engineering
  - Preprocessing
- `AI`
  - Model Performance
  - Strength Prediction
- `ENGINEERING ANALYSIS`
  - CLT Analysis
  - Stacking Optimizer
  - AI vs CLT Comparison
  - Composite vs Aerospace Metals
- `REPORTS`
  - Report Generator
- `TOOLS`
  - Gemini Assistant

## Features

- Stabilized single-source navigation via `src/navigation.py`.
- Central session-state bootstrap via `src/state_manager.py`.
- Dashboard, About Project, and Workflow overview pages.
- Strength prediction page using locked ANN/MLP artifacts.
- CLT analysis page exposing validated material card and reference check.
- AI-vs-CLT comparison page with explicit comparability limits.
- Composite-vs-metals benchmark page using a separate engineering reference database.
- Report Generator with Executive Summary, Prediction Results, AI Metrics, CLT Analysis, Optimization Results, Composite vs Aerospace Metals, and Conclusions.
- Streamlit dashboard with aerospace-themed responsive UI.
- Modular `src/` package for preprocessing, training, prediction, optimization, utilities, config, and Gemini integration.
- Gemini assistant support for low-confidence explanations, stacking sequence suggestions, laminate comparison, material recommendations, report generation, and aerospace composite Q&A.
- Dataset import for CSV, XLSX, JSON, and multiple files at once.
- Dataset organization for raw, Kaggle, NASA, papers, uploaded, processed, merged, and metadata assets.
- Automatic schema standardization with synonym mapping and logged column mappings.
- Dataset validation for missing values, duplicates, numeric ranges, ply orientations, fibre volume fractions, densities, negative mechanical properties, and type mismatches.
- Weighted dataset quality score from 0 to 100.
- Processed dataset versioning as `dataset_v1.csv`, `dataset_v2.csv`, and metadata history in `data/metadata/metadata.json`.
- Dataset explorer and profile pages with preview, statistics, missing-value charts, correlation matrix, feature distributions, and processed CSV download.
- EDA page for dataset summary, feature types, missing matrix, duplicates, correlations, distributions, boxplots, scatterplots, category counts, skewness, kurtosis, target distribution, and class balance.
- Feature engineering page for stacking symmetry, balanced laminate flags, ply counts, orientation percentages, angle statistics, entropy, thickness features, material/resin families, strength-to-weight ratio, layup complexity, and interaction terms.
- Data preprocessing page for outlier detection, missing value imputation, categorical encoding, scaling, feature selection, clean dataset export, metadata export, and Joblib pipeline export.
- Gemini dataset analysis assistant for read-only quality summaries and preprocessing recommendations.
- Gemini EDA/preprocessing advisor for dataset insights, outlier explanations, feature suggestions, correlation explanations, leakage risks, and preprocessing recommendations.
- AI-vs-CLT comparison page with explicit material compatibility checks, common-unit conversion only through backend, calculation trace, and no fabricated ANN-vs-lambda comparison.
- `.env` support for `GEMINI_API_KEY`.

## Report Outputs

Report Generator now supports:

- PDF report
- HTML report
- CSV report summary

Generated artifacts are saved in:

```text
reports/
```

## Runtime Note

Full PDF generation requires `reportlab` inside:

```text
CompositeAI/.venv312
```

Install command used during validation:

```bash
.venv312/bin/python -m pip install reportlab
```

## Step 1 Dataset Validation

Validated source dataset:

```text
data/kaggle/2/composite_material_strength.csv
```

Confirmed columns:

- `fiber_type`
- `resin_type`
- `density_g_cm3`
- `layer_count`
- `curing_temperature_c`
- `fiber_volume_fraction`
- `void_content_pct`
- `tensile_strength_mpa`

Confirmed target:

```text
tensile_strength_mpa
```

Validation command:

```bash
python src/validate_training_data.py --lock
```

The dataset has no missing values, no duplicate rows, no negative numeric values, valid fibre volume fractions, valid void percentages, and positive integer layer counts. Target values are positive, with IQR outliers requiring domain review. The dataset does not contain stacking sequence, ply-by-ply orientation, fibre angle, individual ply, or laminate layup columns.

The current dataset can be used for tensile-strength prediction but is insufficient by itself for true stacking-sequence optimization.

## Step 2 Preprocessing Decisions

Locked training dataset:

```text
data/training/composite_strength_training.csv
```

Feature/target split:

- Target: `tensile_strength_mpa`
- Categorical features: `fiber_type`, `resin_type`
- Numerical features: `density_g_cm3`, `layer_count`, `curing_temperature_c`, `fiber_volume_fraction`, `void_content_pct`

Missing values:

- Current dataset has 0 missing values in every column.
- Pipeline includes `SimpleImputer(strategy="median")` for numerical features and `SimpleImputer(strategy="most_frequent")` for categorical features as a reproducible safety layer.

Duplicates:

- Current dataset has 0 duplicate rows.

Outliers:

- 168 target IQR outliers are detected in `tensile_strength_mpa`.
- Outlier range: 3222.1 MPa to 4667.3 MPa.
- Outliers are retained because no objectively invalid values have been identified.
- Domain review is required before any filtering experiment.

Encoding:

- Categorical columns use `OneHotEncoder(handle_unknown="ignore")`.
- This prevents Streamlit inference from crashing on unseen `fiber_type` or `resin_type` values.

Scaling:

- Numerical columns use `StandardScaler`.
- Scaling is included because Step 4 may evaluate scale-sensitive baseline models.
- Target is not scaled in Step 2.

Leakage prevention:

- `src/preprocessing.py` builds the reusable sklearn `ColumnTransformer`.
- The final preprocessing object must be fitted only on the training split during Step 4.
- Step 2 smoke tests may fit a small sample only to validate structure; this is not model training.

Preprocessing config:

```text
data/training/preprocessing_config.json
```

## Step 3 Feature Specification

ML-ready baseline dataset:

```text
data/training/ml_ready_features.csv
```

Feature specification:

```text
data/training/feature_specification.json
```

Baseline X columns:

- `fiber_type`
- `resin_type`
- `density_g_cm3`
- `layer_count`
- `curing_temperature_c`
- `fiber_volume_fraction`
- `void_content_pct`

Target y column:

```text
tensile_strength_mpa
```

Feature analysis findings:

- Target distribution is right-skewed with skewness `0.8870`.
- Target quartiles: Q1 `1134.975`, median `1485.8`, Q3 `1968.175`, IQR `833.2`.
- Strong multicollinearity was not found among numerical input features.
- Numerical feature correlations with target are weak to moderate:
  - `layer_count`: Pearson `0.384335`, Spearman `0.388974`
  - `void_content_pct`: Pearson `-0.284542`, Spearman `-0.279585`
  - `fiber_volume_fraction`: Pearson `0.202657`, Spearman `0.192932`
  - `curing_temperature_c`: Pearson `0.174698`, Spearman `0.172954`
  - `density_g_cm3`: Pearson `-0.010531`, Spearman `-0.010476`
- The 168 target IQR outlier rows are retained. Their input features remain inside the full dataset feature ranges.
- Leakage review passed: `tensile_strength_mpa` is excluded from X, and no feature is calculated from the target.
- No additional engineered features are required for the baseline model because the dataset does not include stacking sequence, ply thickness, or ply orientation columns.

Step 3 EDA plots:

```text
data/training/step3_plots/
```

Step 4 must use:

- X columns from `data/training/feature_specification.json`
- y column `tensile_strength_mpa`
- preprocessing pipeline from `src/preprocessing.py`
- fit preprocessing only on the training split

## Step 4 Model Training and Comparison

Notebook:

```text
notebooks/step4_model_training.ipynb
```

Local validation run:

```bash
python src/train.py
```

Training setup:

- Dataset: `data/training/ml_ready_features.csv`
- Train/test split: 80/20
- Train rows: 8,000
- Test rows: 2,000
- Random state: `42`
- Preprocessing fitted only inside each sklearn `Pipeline` on training rows

Model comparison:

| Model | Test MAE | Test RMSE | Test R2 |
| --- | ---: | ---: | ---: |
| ANN/MLP | 32.5122 | 43.3776 | 0.9952 |
| Gradient Boosting | 47.9699 | 64.9924 | 0.9893 |
| Random Forest | 60.4610 | 79.2733 | 0.9840 |
| Linear Regression | 97.2777 | 137.0174 | 0.9523 |

XGBoost was not trained in the local validation run because `xgboost` was not installed in the active Python environment. The Colab notebook includes an optional install/import path for XGBoost.

Selected model:

- Model: ANN/MLP
- Test MAE: 32.5122
- Test RMSE: 43.3776
- Test R2: 0.9952

Overfitting check:

- ANN/MLP train R2: 0.9951
- ANN/MLP test R2: 0.9952
- No major overfitting signal in this validation run.

Outlier experiment:

- Primary all-data experiment retained all target outliers.
- Secondary outlier-filtered training-only experiment removed 131 training outliers.
- Filtered training-only result was worse: RMSE 52.1361 vs 43.3776.
- Primary all-data model remains selected.

Step 4 artifacts:

```text
saved_models/best_strength_model.joblib
saved_models/model_metadata.json
data/training/model_comparison.csv
data/training/outlier_experiment.csv
reports/step4_model_plots/
```

## Step 5 Model Validation

Validation command:

```bash
python src/model_validation.py
```

Final selected model:

- Model: ANN/MLP
- Test MAE: 32.5122 MPa
- Test RMSE: 43.3776 MPa
- Test R2: 0.9952
- Final decision: READY

Validation findings:

- Leakage check: PASS
- Saved model validation: PASS
- Test set remains untouched for final artifact verification.
- Predictions are numeric and contain no NaN/inf values.
- No negative predictions were found.
- Predictions remain inside observed dataset target range.
- No meaningful overfitting was detected.

5-fold CV on training data only:

| Model | CV MAE | CV RMSE | CV R2 |
| --- | ---: | ---: | ---: |
| ANN/MLP | 32.9956 +/- 1.1678 | 43.5481 +/- 1.8829 | 0.9951 +/- 0.0004 |
| Gradient Boosting | 47.7454 +/- 0.8345 | 65.2267 +/- 1.3181 | 0.9889 +/- 0.0005 |

Seed robustness for ANN/MLP:

- Seeds tested: 42, 7, 21, 100
- MAE mean/std: 32.4093 +/- 1.9896
- RMSE mean/std: 42.7879 +/- 3.0939
- R2 mean/std: 0.9952 +/- 0.0006

Subgroup validation:

- Fiber groups all have at least 493 test samples.
- Resin groups all have at least 488 test samples.
- Carbon and Epoxy groups show higher errors than other categories, but still high R2.

Step 5 artifacts:

```text
data/training/model_validation_report.json
reports/step5_model_validation.md
reports/step5_model_validation/
```

## Step 6 Strength Prediction

Prediction command/module:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Prediction implementation:

- Loads `saved_models/best_strength_model.joblib`.
- Uses saved sklearn Pipeline containing preprocessing and ANN/MLP model.
- Accepts exactly seven model inputs:
  - `fiber_type`
  - `resin_type`
  - `density_g_cm3`
  - `layer_count`
  - `curing_temperature_c`
  - `fiber_volume_fraction`
  - `void_content_pct`
- Returns `predicted_tensile_strength_mpa`.
- Displays validation metrics from metadata: MAE, RMSE, and R2.
- Does not display fabricated confidence values.
- Warns when numeric input is outside observed training-data range.

Streamlit integration:

- Sidebar page: `Strength Prediction`
- User enters material/process values.
- App calls `src.predict.predict_strength`.
- App displays predicted tensile strength and validated model metrics.

## Step 7 Stacking Sequence Representation

Representation:

- Stacking sequence is an ordered list of ply orientations in degrees.
- Default allowed angles: `[-45, 0, 45, 90]`.
- Allowed angles are configurable through `SequenceConfig.allowed_angles`.
- Sequence order is preserved; sequences are never sorted.

Implemented utilities:

- `LaminateSequence` stores ordered sequence, allowed angles, and ply count.
- `validate_sequence` checks ply count, allowed angles, numeric validity, symmetry, and balance.
- `is_symmetric` checks mirror symmetry about the mid-plane.
- `is_balanced` checks equal `+theta` and `-theta` counts for allowed angle pairs.
- `estimate_search_space_size` estimates candidate count before generation.
- `generate_candidate_sequences` performs bounded candidate generation only.

Critical limitation:

- The current ANN model does not use stacking sequence as an input.
- The current repository cannot evaluate sequence-dependent strength yet.
- Sequence-dependent strength requires one of these future routes:
  - sequence-specific measured/simulated dataset,
  - Classical Laminate Theory or mechanics-based failure model,
  - hybrid physics-informed ML route.

Step 7 artifact:

```text
data/training/stacking_sequence_specification.json
```

## Step 8 CLT Sequence-Dependent Evaluation

Important separation:

> The existing ANN predicts tensile strength from material/process features. Sequence-dependent evaluation is performed separately using mechanics-based laminate analysis when verified material properties are available.

Material-property audit:

- Current repository has no complete verified orthotropic lamina material card.
- Missing verified CLT inputs: `E1`, `E2`, `G12`, `nu12`, `ply_thickness`.
- Missing verified strength allowables for failure optimization: `Xt`, `Xc`, `Yt`, `Yc`, `S`.
- Existing datasets contain partial density/modulus/strength/thickness fields, but not complete sequence-aware lamina data.

Implemented mechanics layer:

- `LaminaMaterial` with explicit SI units.
- `LaminateLoadCase` with `Nx`, `Ny`, `Nxy`, `Mx`, `My`, `Mxy`.
- Reduced stiffness matrix `Q`.
- Transformed stiffness matrix `Qbar`.
- Laminate `A`, `B`, `D`, and `ABD` matrices.
- Mid-plane strain, curvature, global/local ply strain, and global/local ply stress.
- Maximum Stress failure index only when all strength allowables are supplied.

Optimization status:

- Optimization is not enabled in Step 8.
- No load capacity, best sequence, or failure-index objective is reported without verified material properties and load case.
- `src/optimizer.py` exposes readiness status only.

Step 8 artifact:

```text
data/training/clt_specification.json
```

## Step 9 Sequence Data Integration

Investigated source:

- Dataset: `Dataset for Beyond Double-Double Theory: n-Directional Stacking Sequence Optimisation in Composite Laminates`
- Concept DOI: `10.5281/zenodo.15864524`
- Resolved record DOI: `10.5281/zenodo.15864525`
- Publisher/source: Zenodo and TU Delft Research Portal
- License: MIT
- Type: optimization software/results code, not experimental measurements
- Raw files preserved under `data/sequence/tu_delft_zenodo_15864524/raw/`

Verified available fields from source code:

- Stacking sequences for 48- and 64-ply symmetric/balanced cases.
- Material stiffness inputs: `E11 = 127.55e9 Pa`, `E22 = 13.03e9 Pa`, `G12 = 6.41e9 Pa`, `nu12 = 0.3`.
- Ply thickness: `0.005 in`, represented in source as `0.005 * 25.4 / 1000 m = 0.000127 m`.
- Load cases: `Nxx = 1 lb/in`, `Nyy = 0 or 1 lb/in`, converted in source to N/m.
- Reference outputs in code comments: `lambda_cs` failure load and `lambda_cb` buckling load.

Compatibility decision:

- Usable for current CLT stiffness/ABD/stress calculations: yes, with parsed material stiffness card.
- Usable for current Maximum Stress failure optimization: no.
- Missing current failure allowables: `Xt`, `Xc`, `Yt`, `Yc`, `S`.
- Source uses strain allowables from Haftka 1993 inside its own failure-load routine; this is not the same input form as current Maximum Stress stress-allowable implementation.
- Sequence-aware ML training is not started; source is optimization code/results, not a row-wise experimental dataset.

Step 9 artifacts:

```text
data/sequence/tu_delft_zenodo_15864524/metadata.json
data/sequence/tu_delft_zenodo_15864524/processed/source_records.csv
data/sequence/dataset_inventory.json
src/sequence_data.py
```

## Step 10 Final Demonstrator

Final project architecture:

- **System A — ANN strength prediction:** material/process features -> validated ANN/MLP -> `tensile_strength_mpa`.
- **System B — CLT stacking optimizer:** material card + stacking sequence + load case -> CLT + source-compatible strain failure -> `lambda_cs`.
- These systems remain separate. Stacking sequence is not fed into the ANN.

Failure methodology:

- Method: source-compatible strain-allowable route from preserved TU Delft/Zenodo scripts.
- Source allowables:
  - `epsilon_1_allowable = 0.008`
  - `epsilon_2_allowable = 0.029`
  - `gamma_12_allowable = 0.015`
- Objective: maximize allowable failure load factor `lambda_cs`.
- Buckling `lambda_cb` is not optimized because buckling evaluator is not implemented in this project.

Reference validation:

- Source case: `D_Case1_pymoo_load.py`
- Reference `lambda_cs`: `10394.81`
- Reproduced `lambda_cs`: `10319.4275`
- Difference: `-0.7252%`
- Status: pass within documented 1% tolerance.

Optimization demonstrator:

- Search method: bounded random search for large constrained search spaces.
- Default constraints: 48 plies, symmetric, balanced, allowed angles `[-45, 0, 45, 90]`.
- Demonstrator result is best sequence found within configured search budget, not a proven global optimum.
- Baseline comparison uses same material, load case, and failure criterion.

Streamlit integration:

- `Strength Prediction` page: ANN/MLP tensile-strength predictor.
- `Stacking Optimizer` page: CLT/strain-failure optimization demonstrator.
- UI includes material card, laminate constraints, load case, baseline comparison, laminate visualization, ply failure table, and engineering disclaimer.

Step 10 artifact:

```text
data/sequence/optimization_validation_report.json
```

## Step 11 AI vs Physics Comparison

Step 11 adds backend comparison only. No Streamlit page is implemented yet.

Comparison systems:

- **ANN:** predicts `tensile_strength_mpa` from seven material/process features.
- **CLT:** calculates source-compatible `lambda_cs`, a dimensionless load factor that scales supplied CLT loads.
- Direct `tensile_strength_mpa` vs `lambda_cs` comparison is prohibited because the units and physical quantities differ.

Material compatibility:

- Direct numerical comparison requires explicit evidence that the ANN sample material/process data and CLT material card represent the same physical material system.
- ANN category names such as `Carbon` are not treated as equivalent to the TU Delft material card.
- If material equivalence is not verified, result status is `NOT DIRECTLY COMPARABLE`.

Allowed common quantities:

- `equivalent_laminate_tensile_stress_mpa`: allowed only for verified same-material cases with uniaxial tensile `Nx > 0`, zero `Ny/Nxy/Mx/My/Mxy`, base load in `N/m`, known ply thickness, known ply count, and finite `lambda_cs`.
- `tensile_load_capacity_n_per_m`: allowed under the same conditions, using laminate thickness to convert ANN stress to load per unit width.

Comparison equations:

```text
failure_Nx_N_per_m = lambda_cs * base_Nx_N_per_m
total_thickness_m = ply_count * ply_thickness_m
CLT_equivalent_stress_MPa = failure_Nx_N_per_m / total_thickness_m / 1e6
ANN_load_capacity_N_per_m = predicted_tensile_strength_mpa * 1e6 * total_thickness_m
relative_difference_percent = abs(ANN_common_value - CLT_common_value) / abs(CLT_common_value) * 100
```

Important distinction:

- TU Delft reference validation remains `TU Delft lambda_cs` vs local CLT `lambda_cs`.
- AI-vs-CLT comparison is separate and only runs when `ComparisonCase` passes compatibility and conversion checks.

Step 11 artifacts:

```text
src/ai_clt_comparison.py
tests/test_ai_clt_comparison.py
data/training/ai_clt_comparison_specification.json
```

## Step 12 Streamlit AI vs CLT Integration

Navigation:

```text
COMPOSITE ENGINEERING
- Stacking Optimizer
- AI vs CLT Comparison
```

The page implements three user-facing states:

- Valid comparison: backend confirms same material case, supported tensile load, known geometry, same common quantity, and same units.
- Not comparable: ANN result and CLT result are displayed separately with no difference calculation.
- Invalid input: unsupported stacking sequence, missing load, invalid units, or prediction input errors are shown without stack traces.

Current default state:

- ANN output: tensile strength in `MPa`.
- CLT output: `lambda_cs`, dimensionless load factor.
- Direct numerical comparison: not available.
- Reason: current ANN dataset cannot be directly mapped to the TU Delft CLT material card, and raw ANN MPa is not equivalent to raw CLT `lambda_cs`.

Step 12 artifact:

```text
tests/test_app_ai_clt_page.py
```

## Folder Structure

```text
CompositeAI/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
├── data/
│   ├── raw/
│   ├── processed/
│   ├── external/
│   ├── kaggle/
│   ├── nasa/
│   ├── papers/
│   ├── uploaded/
│   ├── merged/
│   └── metadata/
├── models/
├── saved_models/
├── notebooks/
├── reports/
├── assets/
└── src/
    ├── __init__.py
    ├── preprocessing.py
    ├── train.py
    ├── predict.py
    ├── clt.py
    ├── ai_clt_comparison.py
    ├── sequence_data.py
    ├── stacking_sequence.py
    ├── optimizer.py
    ├── utils.py
    ├── config.py
    ├── gemini_client.py
    ├── gemini_service.py
    ├── dataset_loader.py
    ├── dataset_validator.py
    ├── dataset_merger.py
    ├── dataset_standardizer.py
    ├── dataset_profiler.py
    ├── dataset_versioning.py
    ├── eda.py
    ├── feature_engineering.py
    ├── preprocessing_pipeline.py
    ├── visualization.py
    ├── outlier_detection.py
    ├── encoding.py
    └── scaling.py
```

## Installation

```bash
cd CompositeAI
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Gemini assistant support, create `.env` from `.env.example`:

```bash
cp .env.example .env
```

Then set:

```text
GEMINI_API_KEY=your_google_ai_studio_api_key_here
GEMINI_PRIMARY_MODEL=gemini-2.5-flash
GEMINI_FALLBACK_MODELS=gemini-2.5-flash-lite,gemini-3.1-flash-lite
```

## Running Instructions

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, then use the sidebar to navigate the project pages.

## Dataset Management Workflow

1. Open **Dataset Import**.
2. Upload one or more CSV, XLSX, or JSON files, or select existing files from `data/kaggle/`, `data/nasa/`, `data/papers/`, `data/raw/`, or `data/uploaded/`.
3. Click **Process Dataset**.
4. CompositeAI merges files, standardizes schema, validates quality, profiles data, saves `data/merged/merged_latest.csv`, and versions processed data in `data/processed/`.
5. Open **Dataset Explorer** or **Dataset Profile** to inspect data, view charts, and download the processed dataset.
6. Use Gemini dataset analysis only for summaries and risk recommendations; it never edits or fabricates dataset values.

## EDA and Preprocessing Workflow

1. Open **Exploratory Data Analysis** and select a processed dataset.
2. Select the target variable to inspect target distribution and target correlations.
3. Review missing values, duplicates, numerical statistics, skewness, kurtosis, correlations, distributions, boxplots, scatterplots, pair plots, and class balance.
4. Open **Feature Engineering** and enable or disable laminate-specific engineered features.
5. Open **Data Preprocessing** to detect outliers, optionally remove them, impute missing values, encode categorical columns, scale numerical columns, and select features.
6. Export the clean dataset as CSV or Excel, export preprocessing metadata as JSON, or export the sklearn-compatible preprocessing pipeline as Joblib.

## Future Roadmap

- Implement Step 4 machine learning training.
- Add regression model comparison and experiment tracking.
- Train baseline regressors using scikit-learn, XGBoost, LightGBM, CatBoost, and TensorFlow.
- Add uncertainty estimation and confidence calibration.
- Connect prediction outputs to Gemini low-confidence explanation workflows.
- Implement stacking sequence optimization with Optuna.
- Add SHAP-based model explainability.
- Generate PDF or HTML engineering reports.
- Add saved model registry and reproducible experiment tracking.
