## Feature Engineering v1

### Added Features
- Tenure bucket
- Contract stability level
- Charges per tenure ratio
- Binary service indicators

### Results
- Baseline ROC-AUC: ~0.84
- With engineered features: ~0.85

### Observations
- Tenure-based features improved separation of early churners.
- Contract stability reduced confusion for long-term customers.
- Feature gains are modest but consistent.

## Model Comparison

| Model                     | ROC-AUC |
|---------------------------|---------|
| Logistic Regression       | ~0.840 |
| Gradient Boosting (HGB)   | ~0.863  |

Gradient boosting outperforms the linear baseline by capturing
non-linear interactions in the data.
