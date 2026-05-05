# ForeCas9: gRNA Editing Accuracy Prediction

ForeCas9 is a machine learning project designed to predict CRISPR-Cas9 gRNA cutting efficiency, exploring how sequence features and biological context influence Cas9 activity.

## Project Objectives

- Clean and preprocess CRISPR gRNA datasets and their efficiencies
- Train machine learning models to predict cutting efficiency
- Compare performance between model types
- Build simple GUI for bioinformaticians to input their gRNA and receive an efficiency score alongide justification

## Datasets

This project uses the publicly available **Azimuth** datasets released by Microsoft Research:

- **Azimuth V1** (v1_data.xlsx)  
- **Azimuth V2** (v2_data.xlsx)  

Source: https://github.com/MicrosoftResearch/Azimuth/tree/master/azimuth/data

The Azimuth datasets are redistributed under the original Azimuth license (included in this repository's data folder as `AZIMUTH_LICENSE`).

This repository now supports both Azimuth V1 and Azimuth V2 preprocessing and model training, including a new V2 model pipeline that can be selected via `FORECAS9_MODEL`.

This project also uses the publicly available **Wang et al** datasets:

Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC3972032/#SD2


## Repository Structure

ForeCas9/
├── data/ # Raw and processed datasets (Excel/CSV)
├── processed/ # Cleaned datasets after preprocessing
├── src/ # Source code (preprocessing, models, training scripts)
├── notebooks/ # Exploratory analysis & prototyping
├── models/ # Saved models / checkpoints
└── README.md

## Training

- Generate cleaned V1 data: `python -m src.preprocessing.preprocess_v1`
- Generate cleaned V2 data: `python -m src.preprocessing.preprocess_v2`
- Extract V1 features: `python -m src.feature_extraction.extract_features_v1 --input-path data/processed/v1_cleaned.csv --output-path data/processed/v1_features.csv`
- Extract V2 features: `python -m src.feature_extraction.extract_features_v1 --input-path data/processed/v2_cleaned.csv --output-path data/processed/v2_features.csv`
- Combine V1 and V2 cleaned datasets: `python -m src.preprocessing.combine_datasets`
- Extract combined features: `python -m src.feature_extraction.extract_features_v1 --input-path data/processed/combined_cleaned.csv --output-path data/processed/combined_features.csv`
- Train a combined gradient boosting model: `python -m src.training.train_model --model-type gb --data-path data/processed/combined_features.csv`
- Use `FORECAS9_MODEL` to select a different model for the API.
