"""Which model should I use? Line them up and let cross-validation decide.

This replaces the ensemble section of the old models.py. It trains ten-plus
regressors on identical folds, ranks them with error bars, blends the best
few, and explains what the winner relies on.

Two things this teaches that a single train/test split hides:
  - Rankings are noisy. If two models' error bars overlap, they are a tie,
    and the simpler or faster one should win.
  - Averaging several good, *different* models often beats the best single
    one, because their mistakes partly cancel out.

XGBoost, LightGBM and CatBoost are used if installed and skipped if not.

Run:  python model_comparison.py
      python model_comparison.py --csv data/houses.csv --target price --log-target
"""

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import BayesianRidge, Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from toolkit import plots
from toolkit.data import load_dataset, make_parser, profile
from toolkit.evaluate import regression_report

warnings.filterwarnings("ignore", category=UserWarning)


# Rough grouping used to build a diverse blend.
FAMILY = {
    "Linear regression": "linear", "Ridge": "linear", "Lasso": "linear", "Bayesian ridge": "linear",
    "Random forest": "trees", "Gradient boosting": "trees", "Hist gradient boosting": "trees",
    "XGBoost": "trees", "LightGBM": "trees", "CatBoost": "trees",
    "K-nearest neighbours": "other", "Support vector (RBF)": "other",
}


def candidate_models() -> dict:
    """Each entry is a full pipeline, so every model gets identical preprocessing.

    Scaling only matters for models that measure distances or penalise
    coefficient size (linear, SVR, KNN). Trees ignore it, but it does no harm.
    """
    def pipe(estimator):
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), estimator)

    models = {
        "Linear regression": pipe(LinearRegression()),
        "Ridge": pipe(Ridge(alpha=1.0)),
        "Lasso": pipe(Lasso(alpha=0.1, max_iter=20000)),
        "Bayesian ridge": pipe(BayesianRidge()),
        "K-nearest neighbours": pipe(KNeighborsRegressor(n_neighbors=10)),
        "Support vector (RBF)": pipe(SVR(C=10)),
        "Random forest": pipe(RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=0)),
        "Gradient boosting": pipe(GradientBoostingRegressor(random_state=0)),
        "Hist gradient boosting": pipe(HistGradientBoostingRegressor(random_state=0)),
    }
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = pipe(XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=4,
                                              subsample=0.8, colsample_bytree=0.8, random_state=0))
    except ImportError:
        pass
    try:
        from lightgbm import LGBMRegressor
        models["LightGBM"] = pipe(LGBMRegressor(n_estimators=400, learning_rate=0.05, random_state=0, verbose=-1))
    except ImportError:
        pass
    try:
        from catboost import CatBoostRegressor
        models["CatBoost"] = pipe(CatBoostRegressor(verbose=0, random_state=0))
    except ImportError:
        pass
    return models


class AverageBlend:
    """The simplest ensemble: average the predictions of already-chosen models."""

    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)


def main() -> None:
    parser = make_parser(__doc__.splitlines()[0])
    parser.add_argument("--log-target", action="store_true",
                        help="Model log(target). Helps when the target is skewed, like prices or incomes.")
    args = parser.parse_args()
    plots.show_mode(args.show)

    X, y, about = load_dataset(args.csv, args.target, task="regression")
    print(about)
    profile(X, y)

    # The test set is locked away until the very end. All model selection
    # happens with cross-validation on the training part, so the final score
    # is not flattered by our own choices.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = candidate_models()
    if args.log_target:
        if (y <= 0).any():
            raise SystemExit("--log-target needs every target value to be above zero.")
        # Fit on log(y), predict, then convert back. Skewed targets often
        # become well-behaved on a log scale.
        models = {name: TransformedTargetRegressor(m, func=np.log, inverse_func=np.exp) for name, m in models.items()}

    print(f"\nComparing {len(models)} models with 5-fold cross-validation on {len(X_train):,} training rows...")
    folds = KFold(n_splits=5, shuffle=True, random_state=0)
    rows = []
    for name, model in models.items():
        cv = cross_validate(model, X_train, y_train, cv=folds,
                            scoring=("neg_root_mean_squared_error", "r2"), n_jobs=-1)
        rmse = -cv["test_neg_root_mean_squared_error"]
        rows.append({"model": name, "rmse": rmse.mean(), "rmse_std": rmse.std(),
                     "r2": cv["test_r2"].mean(), "fit_seconds": cv["fit_time"].mean()})
    board = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    board.index += 1

    print("\nLeaderboard (lower RMSE is better; ± is the spread across folds):")
    for rank, r in board.iterrows():
        print(f"  {rank:>2}. {r.model:<24} RMSE {r.rmse:9,.3f} ± {r.rmse_std:<8,.3f} "
              f"R² {r.r2:6.3f}   {r.fit_seconds:6.2f}s per fit")

    best = board.iloc[0]
    # "Within one standard deviation of the best" is a common, forgiving
    # definition of a statistical tie.
    tied = board[board.rmse <= best.rmse + best.rmse_std]
    print(f"\nWithin one std of the winner ({len(tied)}): {', '.join(tied.model)}")
    if len(tied) > 1:
        print("These are effectively tied. Pick the simplest or fastest of them, not just the nominal #1.")

    if FAMILY.get(best.model) == "linear":
        print(f"\nA linear model wins. Common with {len(X_train):,} rows: trees and boosting need more data")
        print("to find interactions, and until then their extra flexibility mostly fits noise.")

    fig, ax = plots.plt.subplots(figsize=(8, 0.4 * len(board) + 1.5))
    ordered = board.iloc[::-1]
    ax.barh(ordered.model, ordered.rmse, xerr=ordered.rmse_std, color="#2e86ab", capsize=3)
    ax.set(xlabel="Cross-validated RMSE (lower is better)", title="Model comparison")
    plots.finish(fig, "comparison_leaderboard")

    # ------------------------------------------------------------------ blend
    # Take the best model from each family. Three near-identical linear models
    # make the same mistakes, so averaging them changes nothing; a linear
    # model and a tree model err in different places.
    top_names = board.assign(family=board.model.map(FAMILY)).drop_duplicates("family").model.head(3).tolist()
    print(f"\nBlending the best of each model family: {', '.join(top_names)}")
    blend = AverageBlend([models[n] for n in top_names]).fit(X_train, y_train)
    champion = models[best.model].fit(X_train, y_train)

    single = regression_report(y_test, champion.predict(X_test), f"Best single model ({best.model}) on the test set",
                               y_train=y_train)
    blended = regression_report(y_test, blend.predict(X_test), "Family blend on the test set", y_train=y_train)
    if blended["rmse"] < single["rmse"]:
        print("\nThe blend beats the single best model: its members make different mistakes that partly cancel.")
    else:
        print("\nThe blend doesn't beat the best single model here. Averaging only helps when the weaker")
        print("members are close in accuracy; a clearly worse member drags the average down.")

    plots.actual_vs_predicted(y_test, blend.predict(X_test), "comparison_blend_actual_vs_predicted",
                              "Family blend: test set")

    # --------------------------------------------------------- what it relies on
    # Permutation importance: shuffle one column and measure how much the
    # error rises. Unlike a tree's built-in importance, it works for any model
    # and is measured on unseen data, so it isn't fooled by memorised noise.
    print(f"\nWhat {best.model} relies on (RMSE increase when that feature is shuffled):")
    started = time.time()
    result = permutation_importance(champion, X_test, y_test, scoring="neg_root_mean_squared_error",
                                    n_repeats=10, random_state=0, n_jobs=-1)
    importance = pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)
    for name, value in importance.head(10).items():
        print(f"  {name:<28} {value:+,.3f}")
    useless = importance[importance <= 0].index.tolist()
    if useless:
        print(f"Shuffling these changed nothing (candidates to drop): {useless}")
    print(f"(computed in {time.time() - started:.1f}s)")
    plots.bar_ranking(importance, "comparison_permutation_importance",
                      f"Permutation importance: {best.model}", "RMSE increase when shuffled")


if __name__ == "__main__":
    main()
