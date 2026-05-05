# ForeCas9: gRNA Editing Efficiency Prediction

ForeCas9 is an end-to-end machine learning pipeline for predicting CRISPR-Cas9 gRNA cutting efficiency from guide sequence data. The project processes raw Azimuth datasets, extracts interpretable sequence features, trains regression models, and exposes an API-backed frontend for real-time prediction.

## Overview

The goal of ForeCas9 is to demonstrate a reproducible bioinformatics workflow that:

- preprocesses Azimuth V1 and V2 gRNA datasets
- extracts biologically relevant sequence features
- trains and evaluates regression models for cutting efficiency
- integrates a FastAPI backend with a React frontend
- surfaces prediction confidence and feature diagnostics

This work is designed to support a final-year dissertation by documenting the full implementation pipeline and experimental results.

## Key Features

- Dataset preprocessing for Azimuth V1 and V2
- Custom feature extraction including GC content, homopolymers, nucleotide composition, positional encoding, and dinucleotides
- Trainable model pipeline supporting Linear Regression, Random Forest, and Gradient Boosting
- Model tuning through grid search for tree-based regressors
- Interactive frontend for sequence input and feature breakdown
- API-driven architecture with configurable model selection

## Repository Structure

```
ForeCas9/
├── api/                      # FastAPI backend service
├── data/                     # Raw and processed data files
├── frontend/                 # React + Vite user interface
├── models/                   # Saved trained model artifacts
├── src/                      # Core Python pipeline code
│   ├── preprocessing/        # Data cleaning scripts
│   ├── feature_extraction/   # Sequence feature engineering
│   ├── training/             # Model training and evaluation
│   ├── service/              # Prediction wrapper for inference
│   └── evaluation/           # Evaluation utilities
├── tests/                    # Unit and integration tests
└── README.md
```

## Data Sources

This project uses the following public datasets:

- **Azimuth V1** (Human single-guide RNA data)
- **Azimuth V2** (expanded guide scoring data)

The raw Azimuth datasets are included under `data/raw/` and are processed into cleaned CSV files under `data/processed/`.

> Note: Azimuth datasets are redistributed under their original license, which is included in `data/LICENSES/`.

## Development Environment

1. Clone the repository:

```bash
git clone <repository-url>
cd ForeCas9
```

2. Create and activate a Python virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. Install Python dependencies:

```powershell
pip install -r requirements.txt
```

4. Install frontend dependencies:

```powershell
cd frontend
npm install
```

## Running the Backend API

From the repository root:

```powershell
python -m uvicorn api.app:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at `http://127.0.0.1:8000`.

## Running the Frontend

From the `frontend` folder:

```bash
npm run dev
```

Open the displayed local URL in your browser to use the prediction interface.

## Data Processing Workflow

1. Generate cleaned V1 data:

```bash
python -m src.preprocessing.preprocess_v1
```

2. Generate cleaned V2 data:

```bash
python -m src.preprocessing.preprocess_v2
```

3. Combine cleaned V1 and V2 datasets:

```bash
python -m src.preprocessing.combine_datasets
```

## Feature Extraction

Generate feature matrices for model training:

```bash
python -m src.feature_extraction.extract_features_v1 \
  --input-path data/processed/v1_cleaned.csv \
  --output-path data/processed/v1_features.csv
```

```bash
python -m src.feature_extraction.extract_features_v1 \
  --input-path data/processed/v2_cleaned.csv \
  --output-path data/processed/v2_features.csv
```

```bash
python -m src.feature_extraction.extract_features_v1 \
  --input-path data/processed/combined_cleaned.csv \
  --output-path data/processed/combined_features.csv
```

## Training Models

Train a model using the generic training script:

```bash
python -m src.training.train_model --model-type gb --data-path data/processed/combined_features.csv
```

Supported model types:

- `linear` — Linear Regression
- `rf` — Random Forest Regressor
- `gb` — Gradient Boosting Regressor

## Model Selection

The API loads the model specified by the `FORECAS9_MODEL` environment variable.

Example:

```powershell
$env:FORECAS9_MODEL = 'gradient_boosting_v2'
python -m uvicorn api.app:app --reload
```

Available model files are stored in `models/`.

## Evaluation and Results

Metrics are computed on held-out test sets and saved under `data/processed/`.

Current key results:

- `gradient_boosting_v2` (V2-only):
  - MAE = `0.1330`
  - RMSE = `0.1734`
  - R² = `0.2143`
- `gradient_boosting_combined` (V1 + V2):
  - MAE = `0.2395`
  - RMSE = `0.4245`
  - R² = `0.0810`

These metrics indicate that the Azimuth V2-trained model currently provides the strongest prediction performance in this pipeline.

## Testing

Run the Python test suite with:

```bash
pytest
```

The repository includes unit tests for feature extraction, prediction service logic, integration coverage for data artifacts, and model artifact sanity checks.

## Notes for Dissertation

This project demonstrates an end-to-end predictive pipeline with:

- documented dataset cleaning and validation
- feature engineering based on sequence biochemistry
- reusable model training and hyperparameter tuning
- an API-backed web interface for user-driven prediction
- a clear separation between data, model, and presentation layers

## Author

Final-year project implementation for CRISPR gRNA efficiency prediction and interactive demonstration.