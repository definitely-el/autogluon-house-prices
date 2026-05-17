# Can a Model Trained in 5 Minutes Beat a Weekend of Feature Engineering?
## A Practitioner's Case Study of AutoGluon on Ames Housing

*House prices are a solved problem — or so I thought. Then I let AutoGluon run for two minutes and quietly closed my hyperparameter tuning notebook.*

---

## What Is AutoGluon, and Why Should You Care?

AutoGluon is an open-source AutoML framework built by Amazon that takes a radically different philosophy from most ML tooling. Where scikit-learn hands you a Swiss Army knife and expects you to assemble the blade, AutoGluon asks one question: *what column is the target?* Everything else — imputation, encoding, scaling, model selection, hyperparameter tuning, and ensemble construction — happens automatically.

Under the hood, AutoGluon trains a portfolio of models in parallel: LightGBM, XGBoost, CatBoost, Random Forests, Extra Trees, and neural networks. Depending on the preset and time budget, it then attempts multi-layer stacked generalization on top of cross-validated out-of-fold predictions from those base models. The final output is a weighted ensemble designed to outperform any single model in the portfolio.

This is meaningfully different from tools like MLJAR (which generates reproducible Python code and broad explainability reports per model) or DALEX (which focuses almost entirely on post-hoc explanation). AutoGluon's philosophy is performance first, transparency second — a trade-off worth examining carefully.

---

## The Case Study: Predicting House Prices in Ames, Iowa

The **Ames Housing dataset** contains 2,930 residential property sales with 79 features — everything from basement square footage and garage type to neighborhood zoning and kitchen quality. The target is `SalePrice`, a continuous regression problem with a right-skewed distribution that a log-transform would normally improve. AutoGluon, as we'll see, handles this internally.

Why this dataset? Because it's *messy in realistic ways*: 23 features have missing values, many are ordinal categories with meaningful ordering (e.g. quality ratings: Poor → Excellent), and the price relationship is deeply non-linear. It's exactly the kind of problem where manual feature engineering traditionally pays off.

### Setup

The data was loaded locally, split 80/20 (2,344 train / 586 test rows), and fed directly to AutoGluon. No imputation, no encoding, no scaling — just the raw DataFrame and the target column name.

### What AutoGluon Produced

**`medium_quality` preset** (2-minute time budget): AutoGluon trained 4 base models — LightGBMXT, LightGBM, RandomForest, and CatBoost — then combined the two best into a `WeightedEnsemble_L2` (96% LightGBMXT, 4% CatBoost by weight). Test RMSE: **$23,202**.

**`best_quality` preset** (10-minute budget): AutoGluon 1.5 introduced **Dynamic Stacking (DyStack)** — a self-diagnostic that first trains on a data subset to detect whether multi-layer stacking would cause overfitting. On this dataset, it did: DyStack spent 2.5 minutes running a sub-fit, concluded that stacking overfitting was occurring, and automatically disabled multi-layer stacking for the main run. The result was a `WeightedEnsemble_L2` combining LightGBMXT (38%), LightGBM (38%), and CatBoost (24%). Test RMSE: **$23,774**.

For comparison, a manually-built scikit-learn pipeline (GradientBoostingRegressor, 300 trees, median imputation, one-hot encoding, standard scaling) landed at **$25,352** — taking about 12 seconds to train.

| Method | RMSE (Test) |
|--------|-------------|
| Scikit-learn GBM (manual) | $25,352 |
| AutoGluon `medium_quality` | $23,202 |
| AutoGluon `best_quality` | $23,774 |

The result that catches attention: `medium_quality` beat `best_quality`. This happened because DyStack disabled stacking in the best_quality run, so both presets ended up fitting L1-only ensembles — but `medium_quality` had a tighter time budget that favored the faster, more optimized LightGBMXT configuration. The "more is better" assumption about presets doesn't always hold.

Both AutoGluon presets beat the manual baseline by $2,000–$2,100 (~8–9% RMSE reduction) with zero feature engineering.

---

## Explainability: The One Thing AutoGluon Doesn't Hand You

AutoGluon's ensemble isn't directly interpretable. The `predictor.feature_importance()` method runs permutation importance across the full ensemble and correctly surfaces the expected drivers: **Overall Qual**, **Gr Liv Area**, and **Neighborhood** as the top three — all sensible from a domain perspective.

But permutation importance answers *which features matter globally*, not *why the model made a specific prediction*. For that, I wrapped the full predictor with **SHAP's KernelExplainer**, using 100 test samples with 30 background reference points.

The SHAP summary plot confirmed the story: high overall quality and large living area push predictions up sharply; below-average quality pulls them down. The waterfall plot for an individual house — actual $161,000, predicted $175,527 — showed the model being dragged down by a low Overall Qual score (5/10), small above-grade living area (1,224 sq ft), an "Abnormal" sale condition, and an old garage (built 1960), while nearly every other feature contributed near zero. The $14,527 overprediction is explained by the model not fully accounting for the interaction of those negative signals.

The practical caveat: KernelExplainer is slow — about 11 minutes for 100 rows on a CPU-only machine. For production explainability at scale, this approach needs optimization (more background samples as tradeoff for accuracy, or restricting to a TreeExplainer-compatible base model).

This is one area where **MLJAR has a practical edge**: it generates SHAP plots automatically for each model in its report. If explainability is a first-class requirement — regulatory compliance, loan underwriting, clinical support — MLJAR's built-in reporting pipeline is easier to audit than AutoGluon with SHAP bolted on post-hoc.

---

## Critical Assessment: Should You Use This in Production?

**The case for it:**

AutoGluon is impressive as a rapid prototyping and benchmarking tool. For internal decision-support systems, exploratory forecasting, or any situation where getting to a strong baseline quickly matters, the performance-to-effort ratio is hard to match. DyStack's self-correction behavior — detecting stacking overfitting and adapting automatically — is a genuine sign of framework maturity that manual pipelines can't easily replicate. The `optimize_for_deployment` preset, which distills the ensemble into a leaner model, is a feature most AutoML tools don't offer out of the box.

**The case against it:**

The model artifact is large and opaque. Even without stacking, a weighted LightGBM ensemble is difficult to explain to a non-technical stakeholder, hard to audit for bias, and requires re-running the full pipeline for retraining. XGBoost, FastAI, and NeuralNet models were skipped entirely in this run due to the time budget — meaning the 10-minute ceiling meaningfully constrained what `best_quality` could explore. On a machine without CUDA, neural network candidates are slow enough to crowd out everything else.

For **production ML systems** with strict latency SLAs, regulatory explainability requirements, or teams that need to reason about model behavior under distribution shift — AutoGluon alone is insufficient. You'd need to pair it with monitoring, SHAP-based explanations, and documentation of what the ensemble contains and why.

**My recommendation:** Use AutoGluon to establish a performance ceiling quickly. If a day of feature engineering can't beat it, that tells you something important about the dataset. If you ship to production, use `optimize_for_deployment`, add SHAP for explanation, and increase the time budget significantly (30–60 minutes) so slower models like XGBoost and neural networks actually get to run.

The two-minute model won. But understanding *why* it won — and what it silently skipped — is where the real practitioner judgment lives.

---

*Full code, notebook, and figures available in the repository.*
