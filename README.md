# Predictive Analytics System (CPU-friendly)

End-to-end machine learning project designed for real-world workflow:
data validation → training pipeline → evaluation → inference API → Docker → CI.

## Goals
- Build a reproducible ML pipeline in Python
- Track experiments and results clearly
- Provide a FastAPI inference service (CPU)
- Use tests + linting + GitHub Actions CI

## Repo Structure
- src/ - Python package source code
- notebooks/ - EDA and experiments
- configs/ - configuration files (YAML)
- scripts/ - helper scripts / CLI entrypoints
- tests/ - automated tests (pytest)
- docs/ - documentation and presentation notes
- data/ - local-only data (not committed)
- reports/ - experiment reports and summaries

## Problem Statement

Customer churn is a critical business problem for subscription-based companies.
Acquiring a new customer is significantly more expensive than retaining an existing one.

The goal of this project is to predict whether a customer will churn (cancel the service)
in the near future based on their demographics, account information, and service usage.

The model’s predictions can be used by retention teams to proactively target
high-risk customers with personalized offers or interventions.

## Machine Learning Task

- Task type: Binary classification
- Target variable: `Churn`
- Positive class: Customer churns (`Yes`)
- Negative class: Customer stays (`No`)

## Evaluation Strategy

The primary evaluation metric for this project is ROC-AUC.

Reasoning:
- The dataset is moderately imbalanced.
- The business goal is to rank customers by churn risk.
- ROC-AUC evaluates the model’s ability to distinguish between churners and non-churners
  across different decision thresholds.

Secondary metrics such as precision, recall, and F1-score will be used to analyze
performance at specific operating points.

## Data Splitting

The dataset will be split into training, validation, and test sets
using a stratified random split based on the target variable.

Stratification ensures that the proportion of churned customers
is consistent across all splits.
