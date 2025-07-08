# Adaptive-CP-TC-Intensity
A physically-informed adaptive conformal prediction framework for tropical cyclone intensity forecasting, featuring stage-aware dynamic calibration and uncertainty quantification.
[![DOI](https://doi.org/10.5281/zenodo.15796798)]
This repository contains the code and data used in the study:  
**"Meteorologically-Informed Adaptive Conformal Prediction for Tropical Cyclone Intensity Forecasting"**.

We propose an adaptive Conformal Prediction framework informed by typhoon evolution stages, which dynamically adjusts prediction intervals for tropical cyclone intensity forecasting. The methods combine deep learning models with conformal prediction and are validated using both observational data and case studies. By applying the method to several real-world cases, we demonstrate that it enhances the ability of the forecasting system to handle RI conditions. This adaptive framework provides a new direction for probabilistic forecasting of extreme weather. More details in paper.

## Repository Structure
├── pth/ # This directory contains the saved weight files (`.pth`) for the Transformer-based models used in this study. These checkpoints store the trained parameters of the models at different forecasting lead times or training stages.
├── visualization_results/ # Figures generated from the analysis
├── dataprocessing.py # Data cleaning, transformation, and preparation functions
├── dynamic_alpha_mapping.py # Module for mapping typhoon stages to alpha values dynamically
├── imputed_data.xls # Preprocessed and gap-filled input dataset
├── ML_models.py # comparison machine learning models (e.g., XGBoost, LightGBM, CatBoost)
├── typhoon_transformer.py # Transformer-based model structure for temporal forecasting
├── UQ_methods.py # Uncertainty Quantification methods including McDropout, Monte Carlo, and Conformal Prediction
├── weighted_score_mapping.py # Custom score weighting methods used for evaluation

## Environment Configuration
This project requires:

- Python ≥ 3.8
- `numpy`, `pandas`, `scikit-learn`
- `xgboost`, `matplotlib`, `seaborn`
- `mapie`, `torch` (if using Transformer model)

You can install dependencies using either pip or conda:
Create conda environment from `environment.yml`:

```conda env create -f environment.yml```

## Dataset
`imputed_data.xls`: Includes all cleaned and imputed variables used for model training and evaluation. Features include wind speed, pressure, time, typhoon stage indicators, etc.

## Model training and testing
Load or inspect `imputed_data.xls`, which contains the input features and labels.
Select and run models from `ML_models.py` or `typhoon_transformer.py`.

## Contact
For details of the model development methodology, data source, and potential applications, please refer to our paper. Additional questions can be directed to Fan Meng (`meng@nuist.edu.cn`).
