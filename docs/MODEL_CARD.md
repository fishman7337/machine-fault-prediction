# Model Card

## Model Name

Predictive Maintenance Fault Classifier

## Model Type

Scikit-learn soft-voting binary classifier.

Base estimators:

- Decision tree
- Random forest
- Gradient boosting
- AdaBoost
- K-nearest neighbors
- Logistic regression

## Intended Use

The model estimates whether a factory machine row is normal or abnormal based on sensor readings and product quality. It is intended for predictive maintenance experimentation, assessment work, and prototype operational workflows.

## Not Intended For

- Direct safety-critical automated shutdown decisions without domain validation.
- Use on unseen factories, sensors, or operating regimes without recalibration.
- Root-cause diagnosis. The model predicts status, not the underlying failure mode.

## Inputs

Raw input columns must match `docs/DATA_CARD.md`.

## Output

The prediction workflow returns:

- `predicted_probability_fault`
- `predicted_machine_status`

The classification threshold is selected on a dedicated validation split, then reported metrics
are calculated once on a separate test split. The selected threshold is saved with the model
artifact.

## Evaluation

The training CLI reports:

- Accuracy
- Balanced accuracy
- Precision
- Recall
- F1
- ROC AUC
- Average precision
- Selected threshold

Because faults are rare, balanced accuracy, recall, F1, and average precision should be considered alongside accuracy.

## Limitations

- The dataset is imbalanced.
- Missing values are imputed rather than recovered from source systems.
- The model does not include temporal sequence behavior.
- The included threshold is optimized on a dedicated validation split and should be reviewed against business costs of false negatives and false positives.

## Monitoring Recommendations

- Track input schema and null rates.
- Track target distribution when labels become available.
- Track prediction score distribution.
- Track recall and precision by time period.
- Re-run statistical tests and model evaluation after material data drift.
