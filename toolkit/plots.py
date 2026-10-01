"""The handful of plots that answer "is this model any good, and why?"

Every plot is saved to outputs/<name>.png so the scripts work on a server or
in CI. Pass --show to a script to also open the windows.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = Path("outputs")
_SHOW = False


def show_mode(enabled: bool) -> None:
    """Call once at the start of a script with args.show."""
    global _SHOW
    _SHOW = enabled
    if not enabled:
        plt.switch_backend("Agg")  # draw straight to files, never open a window


def finish(fig, name: str) -> Path:
    """Save the figure, optionally show it, then free its memory."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    if _SHOW:
        plt.show()
    plt.close(fig)
    print(f"  saved {path}")
    return path


def actual_vs_predicted(y_true, y_pred, name: str, title: str = "Actual vs predicted"):
    """Points on the dashed line are perfect predictions.

    Look for a fan shape (errors grow with size) or a curve (the model is
    missing a non-linear pattern).
    """
    y_true, y_pred = np.ravel(y_true), np.ravel(y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolor="none")
    low, high = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([low, high], [low, high], "--", color="grey", label="perfect prediction")
    ax.set(xlabel="Actual", ylabel="Predicted", title=title)
    ax.legend()
    return finish(fig, name)


def residuals(y_true, y_pred, name: str):
    """Residual = actual minus predicted.

    A good model leaves residuals that look like random noise around zero:
    no curve, no funnel, roughly bell-shaped.
    """
    y_true, y_pred = np.ravel(y_true), np.ravel(y_pred)
    resid = y_true - y_pred
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    left.scatter(y_pred, resid, alpha=0.6, edgecolor="none")
    left.axhline(0, color="grey", linestyle="--")
    left.set(xlabel="Predicted", ylabel="Residual", title="Residuals vs predicted (want: no pattern)")
    right.hist(resid, bins=30, edgecolor="white")
    right.set(xlabel="Residual", ylabel="Count", title="Residual distribution (want: centred on 0)")
    return finish(fig, name)


def bar_ranking(values: pd.Series, name: str, title: str, xlabel: str, top: int = 15):
    """Horizontal bars, biggest at the top. Used for coefficients and importances."""
    values = values.reindex(values.abs().sort_values(ascending=False).index).head(top)[::-1]
    colours = ["#d1495b" if v < 0 else "#2e86ab" for v in values]
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(values) + 1.5))
    ax.barh(values.index.astype(str), values.values, color=colours)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set(title=title, xlabel=xlabel)
    return finish(fig, name)


def learning_curves(history: dict, name: str, metrics=("loss",)):
    """Training vs validation curves from a Keras-style history dict.

    Both falling together: still learning, could train longer.
    Validation flattens or rises while training keeps falling: overfitting
    has started, which is exactly what early stopping watches for.
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4), squeeze=False)
    for ax, metric in zip(axes[0], metrics, strict=True):
        ax.plot(history[metric], label=f"train {metric}")
        if f"val_{metric}" in history:
            ax.plot(history[f"val_{metric}"], label=f"validation {metric}")
            best = int(np.argmin(history[f"val_{metric}"]))
            ax.axvline(best, color="grey", linestyle=":", label=f"best epoch ({best + 1})")
        ax.set(xlabel="Epoch", ylabel=metric, title=f"Learning curve: {metric}")
        ax.legend()
    return finish(fig, name)


def correlation_heatmap(frame: pd.DataFrame, name: str, max_cols: int = 20):
    """Feature-to-feature correlations. Bright off-diagonal blocks mean features
    that carry the same information (multicollinearity)."""
    numeric = frame.select_dtypes("number")
    if numeric.shape[1] > max_cols:
        numeric = numeric.iloc[:, :max_cols]
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(0.5 * len(corr) + 3, 0.45 * len(corr) + 2))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90)
    ax.set_yticks(range(len(corr)), corr.columns)
    fig.colorbar(im, ax=ax, shrink=0.8, label="correlation")
    ax.set_title("Feature correlations")
    return finish(fig, name)
