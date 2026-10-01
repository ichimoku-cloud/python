"""Forecasting a time series with XGBoost.

Gradient-boosted trees don't understand time. The trick is to turn time into
ordinary columns ("which month is it?", "what were sales last month and the
same month last year?") and let the trees learn from those.

Time series break three habits that are fine everywhere else:
  - Never shuffle. Train on the past, test on the future, or you are
    predicting yesterday using tomorrow.
  - Never let the test set pick anything, including when to stop training.
    The original version of this script used the test set for early stopping,
    which quietly leaks the answers. Here a validation slice is carved off the
    end of the training period instead.
  - Always compare with "same as last season". It is shockingly hard to beat,
    and a model that can't is not worth deploying.

With no --csv the script builds a SYNTHETIC monthly sales series (trend,
yearly seasonality and noise) so it runs out of the box. Point it at your own
data for real insight.

Run:  python time_series_xgboost.py
      python time_series_xgboost.py --csv data/sales.csv --date-column Month --target "Monthly Sales"
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

from toolkit import plots
from toolkit.evaluate import regression_report

SEASON_LENGTH = {"D": 7, "B": 5, "W": 52, "M": 12, "MS": 12, "ME": 12, "Q": 4, "QS": 4, "QE": 4, "h": 24, "H": 24}


def synthetic_monthly_sales(years: int = 15) -> pd.Series:
    """Clearly fake data: upward trend + yearly cycle + a December spike + noise."""
    rng = np.random.default_rng(7)
    dates = pd.date_range("2010-01-01", periods=12 * years, freq="MS")
    t = np.arange(len(dates))
    trend = 200_000 + 1_500 * t
    seasonal = 40_000 * np.sin(2 * np.pi * (dates.month - 3) / 12)
    december = np.where(dates.month == 12, 60_000, 0)
    noise = rng.normal(0, 12_000, len(dates))
    return pd.Series(trend + seasonal + december + noise, index=dates, name="sales")


def load_series(args) -> pd.Series:
    if not args.csv:
        print("Using SYNTHETIC monthly sales (no --csv given).")
        return synthetic_monthly_sales()
    if not (args.date_column and args.target):
        raise SystemExit("With --csv, also pass --date-column and --target.")
    frame = pd.read_csv(args.csv, parse_dates=[args.date_column])
    series = frame.set_index(args.date_column)[args.target].sort_index()
    return series.dropna().astype(float)


def make_features(series: pd.Series, season: int) -> pd.DataFrame:
    """Every feature for a row uses only values from *before* that row."""
    frame = pd.DataFrame({"y": series})
    idx = series.index

    # Calendar features that make sense for the data's frequency. The old
    # script added "hour" to monthly data, a column that is always zero, and
    # day-of-week of the 1st of each month, which is pure noise.
    by_season = {
        24: {"hour": idx.hour, "dayofweek": idx.dayofweek},
        7: {"dayofweek": idx.dayofweek, "dayofyear": idx.dayofyear, "month": idx.month},
        5: {"dayofweek": idx.dayofweek, "month": idx.month},
        52: {"weekofyear": idx.isocalendar().week.to_numpy(), "month": idx.month},  # dt.weekofyear is gone from pandas
        12: {"month": idx.month, "quarter": idx.quarter},
        4: {"quarter": idx.quarter},
    }
    for name, values in {**by_season.get(season, {"month": idx.month}), "year": idx.year}.items():
        if pd.Series(values).nunique() > 1:
            frame[name] = values

    # Lags: the value 1, 2, 3 steps ago and one full season ago.
    for lag in sorted({1, 2, 3, season}):
        frame[f"lag_{lag}"] = series.shift(lag)

    # Rolling averages, shifted by one so today's value never leaks into its own feature.
    frame["rolling_mean_3"] = series.shift(1).rolling(3).mean()
    frame[f"rolling_mean_{season}"] = series.shift(1).rolling(season).mean()
    # Year-on-year change captures momentum.
    frame["yoy_change"] = series.shift(1) - series.shift(1 + season)
    return frame.dropna()


def build_model(n_estimators=2000):
    # In XGBoost 2+, early stopping is set on the model, not passed to fit().
    return xgb.XGBRegressor(n_estimators=n_estimators, learning_rate=0.03, max_depth=4, subsample=0.8,
                            colsample_bytree=0.8, early_stopping_rounds=100, random_state=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv")
    parser.add_argument("--date-column")
    parser.add_argument("--target")
    parser.add_argument("--test-periods", type=int, default=24, help="How many final periods to hold out.")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    plots.show_mode(args.show)

    series = load_series(args)
    freq = pd.infer_freq(series.index) or "M"
    season = SEASON_LENGTH.get(freq, SEASON_LENGTH.get(freq[:1], 12))
    print(f"{len(series)} observations from {series.index[0]:%Y-%m-%d} to {series.index[-1]:%Y-%m-%d}; "
          f"frequency '{freq}', season length {season}")

    # ---------------------------------------------------------- understand it
    yearly = series.groupby(series.index.year).mean()
    growth = yearly.pct_change().dropna()
    print(f"Average year-on-year growth: {growth.mean():+.1%}")
    by_period = series.groupby(series.index.month if season == 12 else series.index.dayofweek).mean()
    print(f"Strongest period: {by_period.idxmax()} ({by_period.max() / by_period.mean() - 1:+.0%} vs average); "
          f"weakest: {by_period.idxmin()} ({by_period.min() / by_period.mean() - 1:+.0%})")

    # ---------------------------------------------------------- features/split
    data = make_features(series, season)
    X, y = data.drop(columns="y"), data["y"]
    split = len(data) - args.test_periods
    X_train, X_test, y_train, y_test = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]
    print(f"\nTraining on {len(X_train)} periods, testing on the last {len(X_test)}. "
          f"Features: {', '.join(X.columns)}")

    # Two ways to frame the target:
    #   level           predict sales directly
    #   seasonal change predict how much sales differ from the same period
    #                   last season, then add that back on
    # Trees can only output values they have seen, so on a growing series the
    # "level" model hits a ceiling. Predicting the change sidesteps it.
    lag_col = f"lag_{season}"
    modes = {"level": (lambda X_, y_: y_, lambda X_, p_: p_),
             "seasonal change": (lambda X_, y_: y_ - X_[lag_col], lambda X_, p_: p_ + X_[lag_col])}

    def fit(X_fit, y_fit, mode):
        to_target, _ = modes[mode]
        target = to_target(X_fit, y_fit)
        # Early stopping watches the last season of the TRAINING data, never the test set.
        val = season if len(X_fit) > 3 * season else max(1, len(X_fit) // 5)
        probe = build_model().fit(X_fit.iloc[:-val], target.iloc[:-val],
                                  eval_set=[(X_fit.iloc[-val:], target.iloc[-val:])], verbose=False)
        # Refit on everything with the chosen tree count, so the most recent
        # periods (the validation slice) are learned from too.
        trees = probe.best_iteration + 1
        return xgb.XGBRegressor(**{**probe.get_params(), "n_estimators": trees,
                                   "early_stopping_rounds": None}).fit(X_fit, target)

    def predict(model, X_new, mode):
        _, from_target = modes[mode]
        return pd.Series(from_target(X_new, model.predict(X_new)), index=X_new.index)

    # ------------------------------------------------------------- walk-forward
    # Choose between the two framings using only the training period.
    # Walk-forward validation trains on an expanding history and tests on the
    # next season, several times over, so one lucky window can't decide it.
    print("\nWalk-forward validation on the training period (MAPE per fold):")
    cv_mape = {}
    for mode in modes:
        fold_errors = []
        for tr, te in TimeSeriesSplit(n_splits=4, test_size=season).split(X_train):
            m = fit(X_train.iloc[tr], y_train.iloc[tr], mode)
            p = predict(m, X_train.iloc[te], mode)
            fold_errors.append(np.mean(np.abs((y_train.iloc[te] - p) / y_train.iloc[te])))
        cv_mape[mode] = np.mean(fold_errors)
        print(f"  {mode:<16} {'  '.join(f'{e:.1%}' for e in fold_errors)}   average {cv_mape[mode]:.1%}")
    chosen = min(cv_mape, key=cv_mape.get)
    print(f"Chosen framing: {chosen}")

    final = fit(X_train, y_train, chosen)
    pred = predict(final, X_test, chosen)
    other = next(m for m in modes if m != chosen)
    pred_other = predict(fit(X_train, y_train, other), X_test, other)

    # --------------------------------------------------------------- baselines
    # These are one-step-ahead forecasts: each test period uses the real values
    # of the periods before it. That's the right test for "forecast next month",
    # but it flatters the model if you actually need a year ahead (see below).
    naive = X_test["lag_1"]
    seasonal_naive = X_test[lag_col]
    regression_report(y_test, naive, "Baseline: same as last period", y_train=y_train)
    base = regression_report(y_test, seasonal_naive, "Baseline: same as last season", y_train=y_train)
    regression_report(y_test, pred_other, f"XGBoost, {other} (for comparison)", y_train=y_train)
    ours = regression_report(y_test, pred, f"XGBoost, {chosen} (chosen)", y_train=y_train)
    improvement = 1 - ours["mae"] / base["mae"]
    change = f"{improvement:.0%} lower MAE" if improvement >= 0 else f"{-improvement:.0%} higher MAE"
    print(f"\nChosen XGBoost vs the seasonal baseline: {change}. ", end="")
    print("Worth using." if improvement > 0.1 else
          "Not clearly better than the baseline; a simpler method may do." if improvement > -0.05 else
          "Worse than copying last season. Don't ship this yet.")

    # -------------------------------------------------------------- importance
    gain = pd.Series(final.get_booster().get_score(importance_type="gain")).reindex(X.columns).fillna(0)
    gain = gain / gain.sum()
    print("\nWhat the model leans on (share of total gain):")
    for name, value in gain.sort_values(ascending=False).head(8).items():
        print(f"  {name:<20} {value:.1%}")
    plots.bar_ranking(gain, "ts_feature_importance", "XGBoost feature importance (gain)", "Share of gain")

    # ---------------------------------------------------- forecast the future
    # To go several steps ahead we feed each prediction back in as the next
    # lag. Errors compound, which is why long-range forecasts get worse.
    final_all = fit(X, y, chosen)
    history = series.copy()
    step = pd.tseries.frequencies.to_offset(freq)
    for _ in range(season):
        next_date = history.index[-1] + step
        extended = pd.concat([history, pd.Series([np.nan], index=[next_date])])
        row = make_features(extended.fillna(0), season).iloc[[-1]][X.columns]
        history.loc[next_date] = float(predict(final_all, row, chosen).iloc[0])
    future = history.iloc[-season:]
    print(f"\nNext {season} periods (recursive forecast): total {future.sum():,.0f}, "
          f"{future.sum() / series.iloc[-season:].sum() - 1:+.1%} vs the last {season}")
    print("Trees can't extrapolate: a 'level' model never predicts above the highest value it has seen,")
    print("which is why the 'seasonal change' framing usually wins on a growing series.")

    fig, ax = plt.subplots(figsize=(13, 5))
    series.plot(ax=ax, label="actual", color="#555")
    pred.plot(ax=ax, label=f"XGBoost, {chosen} (one step ahead)", color="#2e86ab")
    seasonal_naive.plot(ax=ax, label="same as last season", color="#aaa", linestyle=":")
    future.plot(ax=ax, label="future forecast", color="#d1495b", linestyle="--")
    ax.axvline(X_test.index[0], color="grey", linestyle="--", linewidth=0.8)
    ax.set(title="Actuals, test-period predictions and forecast", xlabel="", ylabel=series.name)
    ax.legend()
    plots.finish(fig, "ts_forecast")


if __name__ == "__main__":
    main()
