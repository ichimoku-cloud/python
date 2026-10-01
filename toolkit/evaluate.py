"""Scoring a model and saying, in plain words, what the score means.

A number on its own rarely tells you much. "RMSE 54" is good or terrible
depending on the scale of the target, so every report here also compares the
model with a naive baseline that just guesses the average.
"""

import numpy as np
from sklearn import metrics


def regression_report(y_true, y_pred, label="Model", y_train=None) -> dict:
    """Print and return the core regression scores.

    y_train is optional. When given, the naive baseline guesses the training
    mean, which is the fair comparison (the model never saw the test set
    either).
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    mae = metrics.mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    r2 = metrics.r2_score(y_true, y_pred)

    reference = np.mean(y_train) if y_train is not None else np.mean(y_true)
    baseline_mae = metrics.mean_absolute_error(y_true, np.full_like(y_true, reference))
    lift = 1 - mae / baseline_mae if baseline_mae else float("nan")

    print(f"\n{label}")
    print(f"  MAE   {mae:,.3f}   average miss, in the target's own units")
    print(f"  RMSE  {rmse:,.3f}   like MAE but punishes big misses harder")
    print(f"  R²    {r2:.3f}     share of the target's variation the model explains")

    # MAPE is only meaningful when no true value is zero or near it.
    if np.all(np.abs(y_true) > 1e-9):
        mape = metrics.mean_absolute_percentage_error(y_true, y_pred)
        print(f"  MAPE  {mape:.1%}    average miss as a percentage")
    else:
        mape = None

    verdict = f"{lift:.0%} lower MAE" if lift >= 0 else f"{-lift:.0%} HIGHER MAE (worse than guessing)"
    print(f"  vs. always guessing the average: {verdict}")
    if rmse > 1.5 * mae:
        print("  Note: RMSE is well above MAE, so a few predictions miss by a lot. Check the residual plot.")

    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape, "lift_vs_mean": lift}


def classification_report(y_true, y_pred, y_prob=None, label="Model") -> dict:
    """Print and return the core classification scores."""
    acc = metrics.accuracy_score(y_true, y_pred)
    majority = max(np.mean(y_true), 1 - np.mean(y_true))

    print(f"\n{label}")
    print(f"  Accuracy  {acc:.1%}   (always guessing the most common class scores {majority:.1%})")

    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = metrics.f1_score(y_true, y_pred)
    print(f"  Precision {precision:.1%}   when it says 'positive', how often it is right")
    print(f"  Recall    {recall:.1%}   of the real positives, how many it caught")
    print(f"  F1        {f1:.3f}   balance of the two")
    print(f"  Confusion: {tp} true pos, {tn} true neg, {fp} false alarms, {fn} misses")

    result = {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}
    if y_prob is not None:
        auc = metrics.roc_auc_score(y_true, y_prob)
        print(f"  ROC AUC   {auc:.3f}   chance a random positive is ranked above a random negative")
        result["auc"] = auc
    return result


def diagnose_fit(train_score: float, test_score: float, metric: str = "R²", higher_is_better: bool = True) -> str:
    """Turn a train/test pair into a bias-variance verdict.

    The thresholds are rules of thumb, not laws. They are written for scores on
    a 0-1 scale where higher is better (R², accuracy, AUC).
    """
    if not higher_is_better:
        train_score, test_score = -train_score, -test_score

    gap = train_score - test_score
    print(f"\nFit check ({metric}): train {abs(train_score):.3f}, test {abs(test_score):.3f}, gap {gap:+.3f}")

    if higher_is_better and train_score < 0.5:
        verdict = ("Underfitting (high bias). The model can't even fit the data it learned from. "
                   "Try a more flexible model, better features, or less regularisation.")
    elif gap > 0.10:
        verdict = ("Overfitting (high variance). It fits the training data noticeably better than new data. "
                   "Try more data, more regularisation, dropout, early stopping, or a simpler model.")
    elif gap < -0.05:
        verdict = ("Test beats train by a margin. Usually a small or lucky test split, or regularisation "
                   "such as dropout that is only active during training. Confirm with cross-validation.")
    else:
        verdict = "Healthy. Train and test scores are close, so the model generalises about as well as it fits."

    print(f"  {verdict}")
    return verdict
