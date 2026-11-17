# Rossmann Store Sales Forecasting

An end-to-end machine learning project for forecasting daily sales across Rossmann stores. The repository covers data preparation, exploratory data analysis, feature engineering, traditional ML, time-series modeling, LSTM experiments, and deployment through a Streamlit application.

## Overview

The goal of this project is to predict daily store sales using store characteristics, promotion status, holidays, customer counts, competition distance, and other retail variables.

Multiple modeling approaches were evaluated, including LightGBM, time-series forecasting techniques, and LSTM deep learning. A preprocessing and LightGBM pipeline was selected for deployment because it combines strong tabular performance with low-latency, real-time inference.

## Project Workflow

| Notebook | Purpose |
|---|---|
| `01_data_preparation.ipynb` | Load and inspect the source data, check data quality, and prepare datasets for analysis. |
| `02_merge_and_eda.ipynb` | Merge store and sales data and explore patterns, distributions, correlations, and seasonality. |
| `03_feature_engineering_and_ml.ipynb` | Build features and train machine learning models. |
| `04_time_series_forecasting.ipynb` | Explore time-series forecasting methods and temporal patterns. |
| `05_LSTM_Deep_Learning.ipynb` | Experiment with an LSTM neural network for sequential sales forecasting. |
| `06_Model_Deployment_and_Inference.ipynb` | Load the serialized pipeline and validate deployment-ready inference. |

## Repository Contents

- `01_data_preparation.ipynb` — data preparation
- `02_merge_and_eda.ipynb` — merging and exploratory analysis
- `03_feature_engineering_and_ml.ipynb` — feature engineering and ML experiments
- `04_time_series_forecasting.ipynb` — forecasting analysis
- `05_LSTM_Deep_Learning.ipynb` — deep learning experiments
- `06_Model_Deployment_and_Inference.ipynb` — model deployment validation
- `app (1) (1).py` — Streamlit sales prediction application
- `final_test.csv` — test dataset
- `Sales_Forecasting_PPT.pptx` — project presentation

## Streamlit Application

The Streamlit app accepts store-level inputs and predicts daily sales using the saved machine learning pipeline.

### Input Features

- Store type
- Assortment type
- State holiday status
- Number of customers
- Competition distance
- Active promotion status

### Run the Application

```bash
pip install streamlit pandas joblib
streamlit run "app (1) (1).py"
```

The app loads the serialized pipeline from:

```text
../models/sales_pipeline_latest.pkl
```

Place the pipeline file in a `models` directory at the expected location before running the application.

## Technologies

- Python
- Jupyter Notebook
- Pandas
- Scikit-learn
- LightGBM
- TensorFlow or Keras for LSTM experiments
- Joblib
- Streamlit

## Model Deployment Approach

The final deployment artifact is a Scikit-learn pipeline containing preprocessing and the selected LightGBM model. Combining preprocessing and prediction in one serialized pipeline helps ensure that categorical encoding, numerical scaling, and feature alignment remain consistent between training and inference.

The Streamlit interface demonstrates real-time prediction using unseen user input.

## Getting Started

1. Clone the repository:

```bash
git clone https://github.com/prashantsingh08/Project-6-Rossmann_store_sales.git
cd Project-6-Rossmann_store_sales
```

2. Install the required Python packages.

```bash
pip install pandas scikit-learn lightgbm joblib streamlit
```

3. Open the notebooks in Jupyter or launch the Streamlit app.

```bash
jupyter notebook
streamlit run "app (1) (1).py"
```

## Notes

- The repository does not include a package configuration or requirements file, so dependencies should be installed manually.
- Relative model paths may need to be adjusted to match the local project structure.
- The supplied app filename contains spaces and parentheses and can be run using the command shown above.

## Author

**Prashant Singh**

GitHub: [@prashantsingh08](https://github.com/prashantsingh08)
