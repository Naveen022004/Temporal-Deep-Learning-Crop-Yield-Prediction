# Project Report Guide

## Objective
Predict crop yield using historical agricultural yield and environmental variables.

## Data Sources
- FAOSTAT crop yield records for India.
- NASA POWER environmental observations.

## Features
Temperature, precipitation, relative humidity, solar radiation, and crop identity.

## Models
- LSTM for temporal sequence learning.
- Ridge Regression as a small-data baseline.

## Evaluation
Report MAE, RMSE, and R² on chronologically later test data.

## Limitation
Annual national-level data provides a limited number of samples, so experimental results should not be presented as production agricultural forecasts.
