# Baseline Model Evaluation

## Overall Performance
- ROC-AUC: ~0.84
- The model shows good ranking ability for churn risk.

## Threshold Behavior
- At threshold 0.5:
  - Precision is moderate
  - Recall is relatively low
- Lowering the threshold increases recall at the cost of precision.

## Error Characteristics
- The model tends to miss some short-tenure churners.
- Longer contracts are predicted as low-risk more consistently.

## Slice Analysis Insights
- Month-to-month contracts have the highest predicted churn rates.
- Customers with short tenure show significantly higher churn probability.

These observations guide future feature engineering and threshold selection.
