# Freight Rate Prediction — Validation, Split & Findings Report

**Backtest MAPE: 5.4575%** (train 2025-01-01…08-31 → test 2025-09-01…10-31).
Final model refit on all 48,000 training rows with identical hyperparameters.
December lane (Lexington → Fort Wayne, 360 mi, Dry Van, 32,000 lb): **$835.58/day**
— see `scorer_results/candidate_december.png`.

## 1. Data split & validation approach

The deployment window (validation.csv: 2025-11-01…12-31) lies strictly after
the labeled window (train_test.csv: 2025-01-01…10-31). I validated with a
single time-forward split — dev Jan–Aug, holdout Sep–Oct — scored with MAPE,
the scale-free metric appropriate for rates spanning $57–$25,533. Random or
stratified K-fold was rejected: shuffling mixes October rows into training
folds and leaks the future, overstating skill on exactly the regime that
matters (forward extrapolation). The Sep–Oct holdout is also seasonally the
closest proxy to Nov–Dec. All preprocessing statistics (weight medians,
shrinkage priors, market medians) are fit on the dev side only; the holdout
is transformed with frozen stats, except the per-date market median, which is
computed within each batch — matching how a production batch of ~150–200
loads/day would be scored.

## 2. Exploration: key findings

- **Rate ≈ distance × rate-per-mile(distance, equipment).**
  Correlation(distance, posted_rate) = 0.91. Rate-per-mile decays smoothly
  with distance (2.92 $/mi under 150 mi → 1.90 $/mi over 2,500 mi) and shifts
  by equipment (median rpm: Dry Van 2.05, Flatbed 2.22, Reefer 2.31).
- **Weight is nearly inert.** Correlation(weight, rate) = 0.03; within a
  fixed equipment × distance cell, rpm moves only ~8% from the lightest to
  the heaviest bin. It still earns its place (ablation: −0.5pp), but as a
  small correction, not a driver.
- **`quote_signal` is a weak per-row signal, not leakage.**
  Correlation with rate is −0.04 globally; quote×distance reaches 0.90 only
  because distance dominates both sides. Within distance bins its
  correlation with rpm is 0.005–0.08 — kept, small positive effect.
- **Weekly market wave.** Daily-mean market_index oscillates ~0.85 → ~1.18
  across the week (trough Sunday, peak Wednesday); daily-mean rpm tracks it
  (daily correlation 0.58). This motivated a market/day feature set.

## 3. Data-quality issues & fixes

| issue | scope | treatment |
|---|---|---|
| `weight` missing | 300 train / 165 val | per-equipment median fill on cleaned weights |
| `weight` negative | 292 train / 145 val | `abs()` — negative magnitudes match the positive distribution (mean 31,724 vs 31,415): sign-entry errors, not real values |
| `weight` capped at 47,500 | 2.5% of rows | left as-is (legal max); flag tested, dropped on zero gain |
| `market_index` missing | 374 train / 249 val | batch daily-median fill (transductive, documented above) |
| `posted_rate` heavy tail (rpm to 14, rates to $25k) | ~0.6% rows | log target absorbs it; no row deletion |
| 8 unseen validation cities; 1,461/4,214 unseen lanes | val-wide | shrinkage-encoded priors with global fallbacks; no one-hots |
| coordinates | exact per city (std 0) | redundant with city identity; unused |

## 4. Model & why it won

LightGBM, log-rate target, MAPE objective, 8 features
(`distance`, `inv_distance`, `equipment`, lane/city shrinkage priors,
cleaned weight, `quote_signal`), 110 boosting rounds taken from the
backtest's `best_iteration`, refit on all data for submission.
Objective choice mattered: MAPE-objective 5.49% vs RMSE/huber 5.54% on the
backtest; raw-target training was far worse (6.26%). Small trees won
(`num_leaves=31`, `min_child_samples=100`); the data regime is smooth and
larger trees only fit noise. Deterministic seeds are exact (seed-invariant
to 4dp), so no ensemble was needed (seed-averaged MAPE identical).

The decisive experiment was the ablation in `src/ablate_market.py`:
strong-features-only 5.60% vs +market/day features 6.36% (**+0.76pp worse**).
The weekly wave in market_index does not transfer to per-row accuracy — the
row-level values are noisy around the regime and the trees chase them. Per
the same evidence, calendar features were dropped entirely.

## 5. December chart

Only the date varies; the lane economics are fixed. Since date features hurt
the backtest, December predictions come from the lane model with the daily
market regime imputed from validation-batch weekday donors — $835.58 every
day, inside the lane's training anchor ($823 ± $46, n=21). A model that drew
a lively December curve here would be expressing an effect the backtest
proved it cannot estimate. The flat line is the finding, not a placeholder.

## 6. Reproduce

```bash
python -m pip install -r requirements.txt
python src/model.py
python score.py --predictions validation_predictions.csv \
  --december-predictions data/december_chart_inputs.csv
```
