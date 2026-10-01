"""Logistic regression: the linear model for yes/no questions.

Despite the name, this is a classifier. It squashes a straight-line score
through a sigmoid to get a probability between 0 and 1, then calls anything
above a threshold (0.5 by default) a "yes".

The deep-dive parts:
  - accuracy alone can lie, so we look at precision, recall and ROC AUC
  - the 0.5 threshold is a choice, not a law; we show what moving it costs
  - coefficients become odds ratios, which are easy to explain to anyone
  - we check whether the predicted probabilities can be trusted (calibration)

Run:  python 02_logistic_regression.py
      python 02_logistic_regression.py --csv data/churn.csv --target churned
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from toolkit import plots
from toolkit.data import load_dataset, make_parser, profile
from toolkit.evaluate import classification_report, diagnose_fit


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    args = make_parser(__doc__.splitlines()[0], task="classification").parse_args()
    plots.show_mode(args.show)

    section("1. The data")
    X, y, about = load_dataset(args.csv, args.target, task="classification")
    print(about)
    if y.nunique() != 2:
        raise SystemExit(f"This script handles two classes; '{y.name}' has {y.nunique()}.")
    # Map whatever labels the CSV uses (yes/no, True/False, 3/7) to 0 and 1.
    classes = sorted(y.unique())
    y = (y == classes[1]).astype(int)
    print(f"Positive class (1) = {classes[1]!r}, negative class (0) = {classes[0]!r}")
    profile(X, y)

    share = y.mean()
    print(f"\nClass balance: {share:.1%} positive")
    if min(share, 1 - share) < 0.2:
        print("Imbalanced classes: accuracy will look good even for a useless model. Watch recall and AUC.")

    # stratify keeps the class balance identical in train and test.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    section("2. Fit and score")
    # Scaling matters here: the default L2 penalty treats every coefficient
    # equally, which is only fair if every feature is on the same scale.
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=5000))
    model.fit(X_train, y_train)

    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    scores = classification_report(y_test, pred, prob, "Test set at the default 0.5 threshold")
    diagnose_fit(model.score(X_train, y_train), scores["accuracy"], metric="accuracy")

    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_auc = cross_val_score(model, X, y, cv=folds, scoring="roc_auc")
    print(f"\n10-fold ROC AUC: {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_test, prob, ax=ax)
    ax.plot([0, 1], [0, 1], "--", color="grey", label="coin flip")
    ax.set_title("ROC curve: further top-left is better")
    ax.legend()
    plots.finish(fig, "logreg_roc")

    section("3. Choosing the threshold")
    # Lowering the threshold catches more positives (recall up) at the cost
    # of more false alarms (precision down). Which side to favour is a
    # business decision: missing a cancer is worse than a false alarm,
    # wrongly blocking a customer may be worse than missing some fraud.
    print(f"{'threshold':>9}  {'precision':>9}  {'recall':>6}  {'flagged':>7}")
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        p = (prob >= t).astype(int)
        print(f"{t:>9.1f}  {precision_score(y_test, p, zero_division=0):>9.1%}  "
              f"{recall_score(y_test, p):>6.1%}  {p.mean():>7.1%}")

    section("4. What drives the prediction (odds ratios)")
    # A coefficient is the change in log-odds per one-standard-deviation
    # increase. Exponentiating turns it into an odds ratio: 2.0 means the
    # odds of a positive double, 0.5 means they halve.
    coefs = pd.Series(model[-1].coef_[0], index=X.columns)
    odds = np.exp(coefs).reindex(coefs.abs().sort_values(ascending=False).index)
    for name, ratio in odds.head(10).items():
        effect = f"x{ratio:.2f} odds" if ratio >= 1 else f"x{ratio:.2f} odds (÷{1 / ratio:.1f})"
        print(f"  {name:<28} {effect}")
    plots.bar_ranking(coefs, "logreg_coefficients", "Logistic regression coefficients",
                      "Change in log-odds per 1 std of feature")
    print("\nSame caution as linear regression: overlapping features share credit unpredictably.")

    section("5. Can we trust the probabilities?")
    # If the model says "70%" on a group of cases, about 70% of them should
    # really be positive. A calibration curve checks exactly that.
    true_rate, predicted_rate = calibration_curve(y_test, prob, n_bins=8, strategy="quantile")
    gap = np.abs(true_rate - predicted_rate).mean()
    print(f"Average gap between predicted and observed rates: {gap:.1%}")
    print("Under ~5% is well calibrated. A larger gap means rank the cases by score, but don't")
    print("quote the raw probabilities to anyone without recalibrating (CalibratedClassifierCV).")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(predicted_rate, true_rate, "o-", label="model")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfectly calibrated")
    ax.set(xlabel="Predicted probability", ylabel="Observed positive rate", title="Calibration")
    ax.legend()
    plots.finish(fig, "logreg_calibration")


if __name__ == "__main__":
    main()
