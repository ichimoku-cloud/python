# Machine learning deep dives in Python

Runnable, heavily explained scripts for the models you reach for most:
linear and logistic regression, neural networks (from scratch and in Keras),
gradient boosting, and time-series forecasting.

Each script does more than fit a model. It asks the questions that decide
whether a model is any good: does it beat a naive guess, which features
drive it, is it overfitting, does it hold up across different splits, and
can its numbers be trusted. The answers are printed in plain English next to
the numbers, and the key plots are saved to `outputs/`.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python linear_regression.py
```

Every script runs with no arguments on a dataset bundled with scikit-learn
(or, for the forecasting scripts, a clearly labelled synthetic sales series),
so nothing needs downloading. Add `--show` to open plot windows as well.

## What's here

Work through them top to bottom if you're learning; jump straight to one if you
need it.

| Script | Model | What you'll learn |
| --- | --- | --- |
| `linear_regression.py` | Linear, Ridge, Lasso | Reading coefficients, p-values and confidence intervals; checking the assumptions (residuals, constant variance, multicollinearity); cross-validation; when regularisation helps |
| `logistic_regression.py` | Logistic regression | Why accuracy can lie; precision vs recall; choosing a threshold; odds ratios; whether predicted probabilities can be trusted (calibration) |
| `neural_net_from_scratch.py` | Two-layer network in numpy | Forward pass, loss and backpropagation written out by hand; gradient checking; how width causes under- and overfitting; what a bad learning rate looks like |
| `neural_net_keras.py` | Deep network (Keras) | Scaling, dropout, L2, early stopping, learning-rate schedules, reading learning curves, and an honest comparison with linear regression |
| `model_comparison.py` | 10+ regressors | Cross-validated leaderboard with error bars; spotting statistical ties; blending different model families; permutation importance |
| `time_series_xgboost.py` | XGBoost for forecasting | Lag and calendar features without leakage; beating the "same as last season" baseline; walk-forward validation; why trees can't extrapolate a trend and how to fix it |
| `forecast_prophet.py` | Prophet | Trend and seasonality decomposition; true multi-step holdout; checking the uncertainty interval; error by forecast horizon |

Shared code lives in `toolkit/`:

- `data.py` loads a CSV or demo dataset and prints a first look (missing values, constant columns, strongest links to the target, a data-leakage warning).
- `evaluate.py` scores a model, compares it with a naive baseline, and gives a bias/variance verdict.
- `plots.py` draws the standard diagnostic plots and saves them.

`extras/text_to_speech.py` is a small offline text-to-speech utility, unrelated to ML, kept from the original repository.

## Using your own data

```bash
python linear_regression.py   --csv data/houses.csv --target price
python logistic_regression.py --csv data/churn.csv  --target churned
python model_comparison.py    --csv data/houses.csv --target price --log-target
python time_series_xgboost.py --csv data/sales.csv  --date-column Month --target "Monthly Sales"
```

Text columns are one-hot encoded and constant columns dropped automatically.
Missing values are filled with the median inside each model's pipeline, so
the test set never influences how training data is filled. CSVs in `data/`
are git-ignored so private data doesn't get committed by accident.

## Cheat sheets

### Is my model overfitting?

Compare the score on data the model trained on with data it has never seen.

| Train | Test | Diagnosis | What to try |
| --- | --- | --- | --- |
| High | High, close to train | Healthy | Ship it, or try for more |
| Low | Low | Underfitting (high bias) | More flexible model, better features, less regularisation |
| High | Clearly lower | Overfitting (high variance) | More data, more regularisation, dropout, early stopping, simpler model |
| Lower | Higher | Lucky split, or dropout active only in training | Confirm with cross-validation |

Use three sets when you tune: train to fit, validation to choose settings,
and test touched once at the very end. If you tune against the test set, it
stops being a fair test.

### Which metric?

| Metric | Use it when |
| --- | --- |
| MAE | You want the typical miss in real units, and outliers shouldn't dominate |
| RMSE | Big misses are much worse than small ones |
| R² | You want "share of variation explained" (1 is perfect, 0 is no better than the average) |
| MAPE | You want a percentage; avoid it when true values can be zero or near zero |
| Accuracy | Classes are roughly balanced |
| Precision / recall | False alarms and misses have different costs |
| ROC AUC | You care about ranking cases, independent of the threshold |

### Which loss and output layer?

| Problem | Output layer | Loss |
| --- | --- | --- |
| Regression | 1 unit, no activation | `mse` (or `mae` / `huber` with outliers) |
| Yes / no | 1 unit, sigmoid | `binary_crossentropy` |
| One of many classes | N units, softmax | `sparse_categorical_crossentropy` |

### Five habits that prevent most mistakes

1. Always compare with a dumb baseline. If you can't beat "guess the average" or "same as last season", stop.
2. Fit scalers, imputers and encoders on training data only.
3. Never shuffle time series. Train on the past, test on the future.
4. Trust cross-validation over a single split, especially on small data.
5. When two models tie, pick the simpler one.

## Requirements

Python 3.10 or newer. `requirements.txt` lists everything; the core
scripts (linear and logistic regression, the numpy neural net and model
comparison) need only numpy, pandas, scikit-learn, scipy,
matplotlib and statsmodels. TensorFlow, XGBoost and Prophet are only needed
for the scripts that use them. LightGBM and CatBoost are optional extras that
`model_comparison.py` picks up automatically if installed.
