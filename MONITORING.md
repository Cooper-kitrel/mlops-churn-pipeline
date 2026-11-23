# Drift Monitoring Analysis

**Script:** `python -m src.monitor_drift`
**Reference data:** the training split (5,634 rows), identical to the split used by `src/train.py`
**Production data:** the 1,409-row held-out split with simulated post-deployment changes (`simulate_drift()` in `src/monitor_drift.py`)
**Threshold:** alert (exit code 1) when more than 15% of features drift (`drift.drift_share_threshold` in `configs/config.yaml`)

## Results

| Feature | Method | Clean score | Drifted score | Drifted? |
|---|---|---|---|---|
| tenure | Wasserstein distance (normed) | 0.025 | 0.484 | Yes |
| TotalCharges | Wasserstein distance (normed) | 0.044 | 0.323 | Yes |
| MonthlyCharges | Wasserstein distance (normed) | 0.029 | 0.291 | Yes |
| Contract | Jensen-Shannon distance | 0.004 | 0.173 | Yes |
| Other 15 features | Jensen-Shannon distance | ≤ 0.024 | ≤ 0.024 (unchanged) | No |

A feature counts as drifted when its distance score reaches 0.1.

**Drift share: 4 / 19 = 21.1%, which is above the 15% threshold, so the script exits with code 1.**

**Control run:** `python -m src.monitor_drift --no-simulate` compares the reference against the *unmodified* holdout. It finds 0 drifted features and exits with code 0, so the alert comes from the simulated changes and not from sampling noise.

## 1. Which features showed drift and why?

Each drifted feature corresponds to a business or engineering event that I simulated:

- **tenure** (largest drift): an acquisition campaign. 40% of production customers are brand new (0–6 months), which pulls the distribution sharply toward zero.
- **MonthlyCharges**: a price increase of about 15% plus noise, which shifts the whole distribution upward.
- **TotalCharges**: drifts mostly as a *consequence* of the two changes above, because TotalCharges ≈ tenure × MonthlyCharges. I also injected 5% missing values to mimic an upstream pipeline bug.
- **Contract**: a promotion moves about half of the one-year and two-year customers onto month-to-month contracts.

Demographic and service-add-on features (gender, Partner, StreamingTV, and so on) were not changed, and Evidently correctly reports no drift for them.

## 2. Would this drift likely affect model performance?

Yes. The drifted features are the model's strongest churn signals: short tenure, month-to-month contracts and high monthly charges all point toward churn. All three moved in the "higher churn risk" direction at the same time.

I scored the same 1,409 holdout customers with the baseline logistic regression model, first unchanged and then with the simulated drift applied:

| | Clean | Drifted | Change |
|---|---|---|---|
| Predicted churn rate | 41.2% | 47.8% | +6.6 pts |
| ROC-AUC | 0.841 | 0.808 | −0.033 |
| Precision | 0.504 | 0.450 | −0.054 |
| Recall | 0.783 | 0.810 | +0.027 |
| Accuracy | 0.738 | 0.686 | −0.052 |

The model flags more customers as churners, and precision falls. About 55% of flagged customers are now false alarms, up from about 50%, so more retention offers would go to customers who were not going to leave. ROC-AUC is still above the 0.80 CI threshold (0.808), but only barely: further drift in the same direction would push the model below it.

*Caveat:* the simulation changes the features but keeps the original churn labels, so these numbers measure how the model *reacts* to shifted inputs, not true production accuracy. In production, churn labels arrive weeks later, which is why input-drift monitoring serves as the early warning.

## 3. Recommended action

**Investigate first, then retrain.**

1. **Investigate immediately.** The 5% missing `TotalCharges` values indicate a data-quality problem (a pipeline bug), not a change in customer behavior. Retraining on broken data would bake the bug into the model, so fix the upstream source first.
2. **Confirm the business causes.** Check with the pricing and marketing teams whether a price change, an acquisition campaign or a contract promotion actually happened. If so, the drift is real and will persist.
3. **Retrain** once a few weeks of labeled post-change data exist, because the reference distribution no longer represents current customers. After retraining, update the drift reference to the new training data.
4. **Continue monitoring** in the meantime: track the predicted churn rate as a daily proxy metric, and compare against real churn labels as they arrive.

"Continue monitoring" alone would not be enough. The drift hits the model's most predictive features, performance is already measurably worse, and ROC-AUC is one step away from failing the 0.80 quality gate.