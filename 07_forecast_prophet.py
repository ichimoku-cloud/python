"""Forecasting with Prophet, and checking whether you should trust it.

Prophet splits a series into trend + seasonality + holidays and fits each
part separately. That makes it quick to use and easy to explain ("sales are
growing 6% a year, and every December adds a spike"), and it handles gaps
and outliers gracefully.

What this script adds beyond fit-and-plot:
  - an honest holdout test against the "same as last season" baseline
  - Prophet's own rolling cross-validation (forecast horizon vs. error)
  - the uncertainty interval, and whether reality actually lands inside it
  - the decomposition in plain numbers

With no --csv it uses the same SYNTHETIC monthly sales as 06_time_series_xgboost.py,
so you can compare the two approaches directly.

Run:  python 07_forecast_prophet.py
      python 07_forecast_prophet.py --csv data/revenue.csv --date-column Date --target IAP
Needs: pip install prophet
"""

import argparse
import logging
from importlib import import_module

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from toolkit import plots
from toolkit.evaluate import regression_report

logging.getLogger("prophet.plot").disabled = True  # optional plotly warning we don't need

try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
except ImportError:
    raise SystemExit("Prophet isn't installed. Run: pip install prophet") from None

logging.getLogger("cmdstanpy").disabled = True  # Prophet's optimiser logs every fit

# Reuse the series loader so both forecasting scripts read data the same way.
ts = import_module("06_time_series_xgboost")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv")
    parser.add_argument("--date-column")
    parser.add_argument("--target")
    parser.add_argument("--test-periods", type=int, default=24)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    plots.show_mode(args.show)

    series = ts.load_series(args)
    freq = pd.infer_freq(series.index) or "MS"
    season = ts.SEASON_LENGTH.get(freq, 12)

    # Prophet insists on two columns named exactly ds (date) and y (value).
    frame = pd.DataFrame({"ds": series.index, "y": series.values})
    train, test = frame.iloc[:-args.test_periods], frame.iloc[-args.test_periods:]

    # Multiplicative seasonality means "December is 20% above trend", which
    # fits most growing businesses better than "December is +40k". Switch to
    # "additive" if the seasonal swings stay the same size as the series grows.
    def new_model():
        return Prophet(seasonality_mode="multiplicative", yearly_seasonality=True,
                       weekly_seasonality=season in (5, 7), daily_seasonality=season == 24,
                       interval_width=0.8)

    model = new_model().fit(train)
    forecast = model.predict(test[["ds"]])
    pred = forecast["yhat"].to_numpy()

    # --------------------------------------------------------------- holdout
    # Unlike the XGBoost script, this is a genuine multi-step forecast: Prophet
    # never sees any test values. Judge it against copying last season.
    seasonal_naive = series.shift(season).iloc[-args.test_periods:].to_numpy()
    base = regression_report(test.y, seasonal_naive, "Baseline: same as last season", y_train=train.y)
    ours = regression_report(test.y, pred, f"Prophet, {args.test_periods} periods ahead", y_train=train.y)
    improvement = 1 - ours["mae"] / base["mae"]
    print(f"\nProphet vs the seasonal baseline: "
          f"{f'{improvement:.0%} lower' if improvement >= 0 else f'{-improvement:.0%} higher'} MAE")

    # The 80% interval should contain roughly 80% of real values. Much lower
    # means Prophet is overconfident; much higher means the band is too wide
    # to be useful.
    inside = np.mean((test.y.to_numpy() >= forecast["yhat_lower"]) & (test.y.to_numpy() <= forecast["yhat_upper"]))
    print(f"Share of actual values inside the 80% interval: {inside:.0%}", end=" ")
    print("(about right)" if 0.65 <= inside <= 0.95 else "(interval is miscalibrated; treat it with care)")

    # --------------------------------------------------------- decomposition
    full = new_model().fit(frame)
    parts = full.predict(frame)
    years = (parts.ds.iloc[-1] - parts.ds.iloc[0]).days / 365.25
    growth = (parts.trend.iloc[-1] / parts.trend.iloc[0]) ** (1 / years) - 1
    print("\nWhat Prophet sees in the full history:")
    print(f"  Trend: {growth:+.1%} per year on average")
    yearly = parts.groupby(parts.ds.dt.month)["yearly"].mean()
    print(f"  Seasonality: best month {yearly.idxmax()} ({yearly.max():+.0%} vs trend), "
          f"worst month {yearly.idxmin()} ({yearly.min():+.0%})")
    changepoints = full.changepoints[np.abs(np.nanmean(full.params["delta"], axis=0)) > 0.01]
    if len(changepoints):
        print(f"  Trend shifts detected around: {', '.join(d.strftime('%Y-%m') for d in changepoints[:5])}")

    # ------------------------------------------------ rolling cross-validation
    # Prophet's own backtest: repeatedly cut the history, forecast forward, and
    # record how error changes with the horizon. It usually rises the further
    # ahead you look; a flat line means the series is very regular.
    days_per_period = {12: 30.5, 4: 91.3, 52: 7, 7: 1, 5: 1, 24: 1 / 24}[season]
    horizon = f"{int(days_per_period * season)} days"
    initial = f"{int(days_per_period * season * 4)} days"
    cv = cross_validation(full, initial=initial, period=f"{int(days_per_period * season / 2)} days",
                          horizon=horizon, disable_tqdm=True)
    perf = performance_metrics(cv, rolling_window=0)
    perf["periods_ahead"] = np.ceil(perf.horizon.dt.days / days_per_period).clip(lower=1).astype(int)
    by_horizon = perf.groupby("periods_ahead")["mape"].mean()
    print("\nBacktest error by distance ahead (periods: MAPE):")
    print("  " + "   ".join(f"{h}: {m:.1%}" for h, m in by_horizon.iloc[:: max(1, len(by_horizon) // 6)].items()))

    # ------------------------------------------------------------------ plots
    future = full.make_future_dataframe(periods=season, freq=freq)
    outlook = full.predict(future)
    fig = full.plot(outlook, figsize=(13, 5))
    fig.axes[0].set_title(f"Prophet forecast, next {season} periods (shaded = 80% interval)")
    plots.finish(fig, "prophet_forecast")
    plots.finish(full.plot_components(outlook, figsize=(11, 6)), "prophet_components")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(by_horizon.index, by_horizon.values * 100, marker="o")
    ax.set(xlabel="Periods ahead", ylabel="MAPE (%)", title="Forecast error grows with the horizon")
    plots.finish(fig, "prophet_error_by_horizon")


if __name__ == "__main__":
    main()
