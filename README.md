# Temporal Deep Learning for Crop Yield Prediction Using Real Agricultural and Environmental Data

## B.Tech CSE Project

This project predicts crop yield using a temporal deep-learning model (LSTM) and **real publicly available data**.

## Data Sources

### 1. FAOSTAT – Crop Production and Yield
Official Food and Agriculture Organization statistics.

Notebook download URL:
`https://bulks-faostat.fao.org/production/Production_Crops_Livestock_E_All_Data_(Normalized).zip`

The notebook filters real records for **India** and uses:
- Wheat
- Rice
- Maize
- Historical annual yield

### 2. NASA POWER – Environmental Data
Real environmental observations are downloaded automatically using the NASA POWER API:
- Temperature (T2M)
- Corrected precipitation (PRECTOTCORR)
- Relative humidity (RH2M)
- Solar radiation (ALLSKY_SFC_SW_DWN)

### 3. Optional Satellite Data – MODIS NDVI
For an advanced extension, the project can use:
`MODIS/061/MOD13Q1`

MODIS NDVI extraction requires Google Earth Engine authentication. The core project intentionally runs without credentials using real FAOSTAT and NASA POWER data.

## Project Workflow

FAOSTAT Crop Yield + NASA POWER Environment
→ Data Cleaning
→ Feature Encoding
→ 5-Year Temporal Sequences
→ LSTM
→ Crop Yield Prediction
→ MAE, RMSE and R² Evaluation

## Run in Google Colab

Open `Temporal_Crop_Yield_Prediction_Real_Data.ipynb` in Google Colab and select **Runtime → Run all**.

The notebook automatically downloads the required real datasets; no local dataset upload is required.

## Academic Note

This implementation is suitable for a B.Tech student project because it uses real sources, reproducible data acquisition, temporal feature engineering and an LSTM model. Results should be reported as experimental results and not as production agricultural forecasts.
