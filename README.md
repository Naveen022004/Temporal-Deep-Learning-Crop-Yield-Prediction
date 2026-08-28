# Temporal Deep Learning for Crop Yield Prediction

## B.Tech CSE Project

A reproducible machine-learning project that predicts crop yield from historical agricultural records and environmental data. The project compares a temporal LSTM model with Ridge Regression because the available annual dataset is relatively small.

## Project Architecture

```text
FAOSTAT Crop Yield + NASA POWER Environmental Data
                    |
                    v
              data/raw/
                    |
                    v
          Preprocessing and Cleaning
                    |
                    v
            Feature Engineering
                    |
                    v
           5-Year Temporal Sequences
                    |
          +---------+---------+
          |                   |
          v                   v
         LSTM              Ridge Baseline
          |                   |
          +---------+---------+
                    v
              Evaluation
                    |
          +---------+---------+
          v                   v
       models/            results/
```

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── Temporal_Crop_Yield_Prediction_Real_Data.ipynb
├── nasa_power_environment.py
├── src/
│   ├── __init__.py
│   ├── nasa_power_environment.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── evaluate_model.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── results/
│   ├── predictions/
│   └── figures/
└── docs/
    └── project_report.md
```

## Data Sources

### FAOSTAT Crop Production and Yield
The project uses publicly available agricultural statistics and filters historical records for India, including Wheat, Rice, and Maize.

### NASA POWER Environmental Data
Environmental variables include:
- Temperature (T2M)
- Corrected precipitation (PRECTOTCORR)
- Relative humidity (RH2M)
- Solar radiation (ALLSKY_SFC_SW_DWN)

### Optional Satellite Extension
MODIS NDVI can be integrated through Google Earth Engine as an advanced extension.

## Machine-Learning Workflow

1. Download agricultural and environmental data.
2. Clean missing and invalid values.
3. Encode crop categories and scale numerical features.
4. Build consecutive 5-year temporal sequences.
5. Split training and test data chronologically.
6. Train an LSTM model with regularization and early stopping.
7. Train a Ridge Regression baseline.
8. Inverse-transform target predictions before evaluation.
9. Compare MAE, RMSE, and R².
10. Save predictions, figures, and trained models in their dedicated directories.

## Installation

```bash
pip install -r requirements.txt
```

## Run in Google Colab

Open `Temporal_Crop_Yield_Prediction_Real_Data.ipynb` and select **Runtime → Run all**. The notebook downloads the required real data automatically.

## Evaluation Metrics

- **MAE**: Average absolute prediction error.
- **RMSE**: Penalizes larger prediction errors.
- **R²**: Measures performance relative to predicting the target mean.

A negative R² means the evaluated model performs worse than a mean-prediction baseline.

## Limitations

Annual national-level agricultural data provides a limited number of temporal samples. Therefore, results should be treated as experimental academic results rather than production agricultural forecasts.

## Future Improvements

- Add district-level crop data.
- Add soil and irrigation variables.
- Integrate MODIS NDVI.
- Use monthly or seasonal environmental features.
- Evaluate additional models such as Random Forest, XGBoost, and GRU.
