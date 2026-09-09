# Model Comparison Report — Customer Churn

Generated from the MLflow `customer-churn` experiment (experiment_id `1`), tracking database: legacy SQLite backend used by the training runs before the tracking-URI fix.

Data: 5,000 customers, 21.7% churn rate. Held-out test split: 1,000 rows, 21.7% churn. Holdout churn rate matches the stratified training split.

## Results

| Model                | Precision | Recall | F1     | ROC-AUC | PR-AUC  | Accuracy | Train Acc |
|----------------------|-----------|--------|--------|---------|---------|----------|-----------|
| logistic_regression  | 0.3963    | 0.6959 | 0.5050 | 0.7647  | 0.4718  | 0.7040   | 0.6855    |
| random_forest        | 0.5241    | 0.3502 | 0.4199 | 0.7428  | 0.4578  | 0.7900   | 1.0000    |
| gradient_boosting    | 0.5543    | 0.2350 | 0.3301 | 0.7327  | 0.4451  | 0.7930   | 0.9713    |

### Per-metric ranking

| Metric       | 1st (best)        | 2nd                | 3rd                |
|--------------|-------------------|--------------------|--------------------|
| Precision    | gradient_boosting | random_forest      | logistic_regression |
| Recall       | logistic_regression | random_forest    | gradient_boosting  |
| F1           | logistic_regression | random_forest    | gradient_boosting  |
| ROC-AUC      | logistic_regression | random_forest    | gradient_boosting  |
| PR-AUC       | logistic_regression | random_forest    | gradient_boosting  |
| Accuracy     | gradient_boosting | random_forest      | logistic_regression |

## Which metric should be prioritized for customer churn

For customer churn, the business action is a **retention campaign** (discount, outreach, offer). The cost of the two error types is asymmetric:

- **False negative** (model says "stays", customer actually churns): the customer leaves with no intervention. Cost is a **lost customer / lost revenue** — precisely the outcome the business is trying to prevent.
- **False positive** (model says "churns", customer actually stays): the retention offer is wasted. Cost is limited to the offer/discount spend.

Because losing a repeat customer almost always costs more than funding a discount for a false alarm, the model must **maximize recall** — find as many true churners as possible — while keeping precision from collapsing to avoid an unbounded retention budget.

For ranking models on imbalanced churn data, **PR-AUC is the appropriate primary threshold-independent metric**: unlike ROC-AUC, it directly reflects performance on the rare (churn) class and is not inflated by the large majority (stay) class. ROC-AUC should be treated as secondary. Precision/recall/F1 are threshold-dependent; F1 is a reasonable secondary summary when the false-negative/false-positive costs are roughly balanced.

## Recommended model

**`logistic_regression` is selected.**

- Highest recall (0.696), PR-AUC (0.472), ROC-AUC (0.765), and F1 (0.505) of the three models.
- Catches ~70% of true churners versus ~35% (random_forest) and ~24% (gradient_boosting). Under the retention business objective, that is the behavior that matters.
- Generalizes cleanly: train accuracy 0.686 vs test 0.704 (no overfitting). In contrast, random_forest (train 1.000) and gradient_boosting (train 0.971) show clear overfitting coupled with weak holdout recall.
- Additionally the most interpretable model for a business-facing retention program.

gradient_boosting/random_forest win only on precision and raw accuracy; that higher precision comes at roughly a 2–3x loss in churner capture, which conflicts with the retention objective. If the business ever values precision more (e.g., a very expensive offer), random_forest is the fallback, with the caveat that its perfect training accuracy signals overfitting.

## Provenance and caveats

- 19 runs exist in the experiment: 17 `FINISHED` with logged metrics, 1 `RUNNING` (no metrics logged), 1 `FAILED` (metrics logged, same values as its FINISHED duplicates). All FINISHED runs for a model share identical metrics because training uses `random_state=42` on identical data; each model's numbers are from its latest FINISHED run.
- The runs were recorded in the legacy SQLite tracking DB (target of the URL-encoding bug); the `customer-churn` experiment is not present in the corrected `mlflow.db` until the next training run.
- Representative run IDs — logistic_regression: `d2bac073a7c2479683e52140647f0d2e`, random_forest: `d705d6500d824496a39fd3d81c53200e`, gradient_boosting: `9b05040c58c94b02bfb4daf9f03e4be9`.