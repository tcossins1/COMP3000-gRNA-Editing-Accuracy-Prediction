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


## Repository Structure

ForeCas9/
├── data/ # Raw and processed datasets (Excel/CSV)
├── processed/ # Cleaned datasets after preprocessing
├── src/ # Source code (preprocessing, models, training scripts)
├── notebooks/ # Exploratory analysis & prototyping
├── models/ # Saved models / checkpoints
└── README.md