# Zindi Liquidity Stress Early Warning Challenge

This repository contains my team's solution for the Zindi Liquidity Stress Challenge. The goal is to predict if a mobile-money user will experience liquidity stress in the next 30 days, using their transaction history from the past 6 months.

## Objective
The competition metric is a mix of two things:
- **60% Log Loss**: This heavily penalizes being overly confident and wrong.
- **40% ROC-AUC**: This rewards the model for ranking stressed users correctly.

## Project Structure
We set this up following the standard Cookiecutter Data Science layout to keep things organized.

```text
├── README.md
├── data/                 <- Raw, cleaned, and feature-engineered datasets
├── notebooks/            <- Chronological Jupyter notebooks (01 to 11) tracking the development phases
├── src/                  <- Reusable python scripts (like custom metrics)
├── submissions/          <- Final prediction files for upload
└── ml_env/               <- Virtual environment
```

## The Pipeline
We broke the project down into 11 phases, tracked sequentially in the `notebooks/` folder. Here is how the approach evolved:

- **Phases 1-3 (EDA & Baselines):** Started with standard exploratory data analysis, cleaned up the raw data, and did some basic feature engineering. Got a simple LightGBM baseline running.
- **Phase 4-5 (Tuning & Deep Learning):** Brought in Optuna for hyperparameter tuning and MLflow to track everything. We also built a PyTorch Neural Network baseline (with GELU, AdamW, and early stopping), but tree-ensembles ultimately proved much stronger on this tabular dataset.
- **Phases 6-7 (Ensembling & Advanced Tuning):** Scaled up to an ensemble of LightGBM, XGBoost, and CatBoost, and heavily tuned them using Stratified K-Fold cross-validation.
- **Phases 8-9 (SHAP Insights):** Used SHAP values and KDE plots to look under the hood. It turned out the models cared way more about sudden balance drops and zero-balance events than raw historical averages, so we pruned a lot of the noisy static features.
- **Phases 10-11 (Temporal Features & Calibration):** This was the real breakthrough. We engineered deep behavioral features like drawdowns (peak-to-trough collapse), trajectory slopes, deterioration streaks, and customer-normalized z-scores. Finally, we built a Logistic Regression meta-learner on the unconstrained logits of the base models and ran a calibration audit, proving the stacker was naturally calibrated for the punishing LogLoss metric.

## Results
- **Local CV Log Loss:** `0.2606`
- **Local CV ROC-AUC:** `0.8957`

These scores are the best ones we had. They are completely leak-free and validated on 5-Fold Stratified CV Out-Of-Fold predictions without using any overfitting calibration hacks.

## Running the Code
1. Activate the environment: `source ml_env/bin/activate`
2. Start MLflow (optional, if you want to see the tuning logs): `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 8080`
3. Run through the notebooks in numerical order (`01` to `11`).
4. Grab the final output from the `submissions/` folder.
