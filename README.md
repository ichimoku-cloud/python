# Machine learning deep dives in Python

This repo collects the models I use most: linear and logistic regression, neural networks, gradient boosting and time-series forecasting. Each one is a script that runs on its own, with comments that explain what's happening and why.

## Getting started

```bash
pip install -r requirements.txt
python linear_regression.py
```

There's nothing to download. The scripts run on sample data that comes with scikit-learn. The two forecasting scripts use made-up sales data instead, and they say so when they run.

## The scripts

- `linear_regression.py` predicts a number and shows which inputs actually matter.
- `logistic_regression.py` predicts yes or no and covers the trade-off between false alarms and misses.
- `neural_net_from_scratch.py` builds a small neural network by hand, so you can see how it learns.
- `neural_net_keras.py` trains a deep network and checks whether it really beats a simpler model.
- `model_comparison.py` runs a dozen models on the same data to see which one comes out ahead.
- `time_series_xgboost.py` forecasts monthly sales and compares the forecast with simply repeating last year's numbers.
- `forecast_prophet.py` forecasts with Prophet by splitting a series into its trend and seasonal patterns.

The `toolkit/` folder holds the shared code for loading data, scoring models and drawing charts.

## Using your own data

Point any script at a CSV and name the column you want to predict:

```bash
python linear_regression.py --csv data/houses.csv --target price
```

For the time-series script, also name the date column with `--date-column`.

## A few things worth remembering

Always compare your model with a dumb baseline, like guessing the average. If you can't beat it, the model isn't adding anything.

If a model scores much better on its training data than on new data, it has memorised rather than learned. That's overfitting. If it scores badly on both, it's too simple for the problem.

With time series, never shuffle the data. Train on the past and test on the future.

When two models perform about the same, pick the simpler one.
