# Freight Rate Prediction — solution

LightGBM regressor on log-rate. Backtest MAPE **5.46%** on a strict
time split (train Jan–Aug 2025 → test Sep–Oct 2025).

## Reproduce

```bash
python -m pip install -r requirements.txt
python src/model.py   # prints backtest MAPE, writes validation_predictions.csv,
                      # fills data/december_chart_inputs.csv, writes metrics.json
python score.py --predictions validation_predictions.csv \
  --december-predictions data/december_chart_inputs.csv
```

Outputs: `validation_predictions.csv` (12,000 rows), filled
`data/december_chart_inputs.csv` (31 rows), chart at
`scorer_results/candidate_december.png`, run metrics in `metrics.json`.

## Method (one paragraph each)

**Validation.** Time-forward only: dev = 2025-01-01…08-31, holdout =
2025-09-01…10-31, scored with MAPE. Random K-fold was rejected — the
deployment window (Nov–Dec) lies strictly after training (Jan–Oct), so
shuffled folds leak the future and overstate skill.

**Data quality.** `weight`: 300 missing, 292 negative, 2.5% capped at
47,500. Fix: `abs()` (negative magnitudes match the positive
distribution: mean 31,724 vs 31,415) then per-equipment median fill;
flags kept as features, dropped after ablation showed zero gain.
`market_index`: 0.8% missing, filled from the batch's daily median —
a deliberately transductive step, since any scoring batch carries
~150–200 rows per date. `quote_signal`: complete. `posted_rate` has
heavy-tail outliers (rpm up to 14); the log target absorbs them.
8 validation cities never appear in training; 1,461 of 4,214
validation lanes (35%) are unseen — hence shrinkage-encoded
city/lane priors with global fallbacks instead of one-hots.

**Model.** 8 features: `distance`, `inv_distance`, `equipment`,
shrinkage-encoded lane/city log-rpm priors, cleaned weight,
`quote_signal`. Log target, MAPE objective, 110 rounds (backtest
`best_iteration`). The ablation (`src/ablate_market.py`) killed the
obvious second-order story: adding per-row `market_index`/calendar
features *hurt* the backtest by +0.76pp (5.60% → 6.36%), so December
is driven by the fixed lane's economics, not the calendar — the flat
$835.58 chart is the honest output of that finding, matching the
lane's training anchor ($823 ± $46 over 21 loads).

## Files

| path | what |
|---|---|
| `src/features.py` | cleaning + shrinkage encoders (fit on train only) |
| `src/model.py` | backtest, final fit, both output files |
| `src/ablate_market.py` | market/day-feature ablation |
| `report.md` | full report (validation, split, chart) |
