"""Loading data and taking a first look at it.

Every script can run on your own CSV (--csv path --target column) or, with no
arguments, on a small dataset that ships inside scikit-learn. Nothing needs to
be downloaded, so a fresh clone runs straight away.
"""

import argparse

import numpy as np
import pandas as pd
from sklearn import datasets


def add_data_args(parser: argparse.ArgumentParser, task: str) -> None:
    """Add the --csv / --target / --show options every script shares."""
    demo = "diabetes progression" if task == "regression" else "breast cancer diagnosis"
    parser.add_argument("--csv", help=f"Path to your own CSV. Without it we use the built-in {demo} data.")
    parser.add_argument("--target", help="Name of the column you want to predict (required with --csv).")
    parser.add_argument("--show", action="store_true", help="Open plot windows as well as saving them to outputs/.")


def load_dataset(csv=None, target=None, task="regression"):
    """Return features X (DataFrame), target y (Series) and a one-line description."""
    if csv:
        if not target:
            raise SystemExit("When you pass --csv you also need --target <column name>.")
        frame = pd.read_csv(csv)
        if target not in frame.columns:
            raise SystemExit(f"Column '{target}' is not in {csv}. Columns are: {list(frame.columns)}")

        # A row with no answer can't teach the model anything, so drop it.
        frame = frame.dropna(subset=[target])
        y = frame.pop(target)

        # Models need numbers. Turn text columns into 0/1 indicator columns.
        # Missing numbers are left alone; each model's pipeline fills them in.
        text_cols = frame.select_dtypes(exclude="number").columns.tolist()
        if text_cols:
            print(f"One-hot encoding text columns: {text_cols}")
            frame = pd.get_dummies(frame, columns=text_cols, drop_first=True, dtype=float)

        # A column with one value everywhere carries no information and breaks
        # some statistics (its scaled version is all zeros).
        constant = [c for c in frame.columns if frame[c].nunique(dropna=True) <= 1]
        if constant:
            print(f"Dropping constant columns: {constant}")
            frame = frame.drop(columns=constant)
        return frame, y, f"{csv} -> predicting '{target}'"

    if task == "regression":
        raw = datasets.load_diabetes(as_frame=True)
        about = "scikit-learn diabetes data: predict disease progression one year out (442 patients)"
    else:
        raw = datasets.load_breast_cancer(as_frame=True)
        about = "scikit-learn breast cancer data: predict malignant (0) vs benign (1) from 30 cell measurements"
    return raw.data, raw.target, about


def profile(X: pd.DataFrame, y: pd.Series, top: int = 8) -> None:
    """Print the handful of facts worth knowing before you fit anything."""
    print(f"\nRows: {len(X):,}   Features: {X.shape[1]}")

    missing = X.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        print("Missing values: none")
    else:
        print("Missing values (share of rows):")
        print(missing.head(top).map("{:.1%}".format).to_string())

    constant = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    if constant:
        print(f"Constant columns (carry no information, consider dropping): {constant}")

    print(f"\nTarget '{y.name}': mean {y.mean():.3g}, std {y.std():.3g}, min {y.min():.3g}, max {y.max():.3g}")

    # Correlation only sees straight-line relationships, but it is a fast
    # first hint at which features will matter.
    numeric = X.select_dtypes("number")
    if not numeric.empty and y.nunique() > 1:
        corr = numeric.corrwith(y.astype(float)).dropna()
        ranked = corr.reindex(corr.abs().sort_values(ascending=False).index).head(top)
        print("\nStrongest straight-line links to the target:")
        for name, value in ranked.items():
            direction = "rises with" if value > 0 else "falls as"
            print(f"  {name:<28} r = {value:+.2f}  (target {direction} it)")

        # Real-world features almost never predict the target this well. When
        # one does, it is usually computed from the target or only known after
        # the fact ("leakage"), and the model will collapse in real use.
        suspicious = corr[corr.abs() > 0.95].index.tolist()
        if suspicious:
            print(f"\nWARNING: {suspicious} track the target almost perfectly (|r| > 0.95).")
            print("Check they aren't derived from the target or only known afterwards (data leakage).")


def make_parser(description: str, task: str = "regression") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    add_data_args(parser, task)
    return parser


def set_seed(seed: int = 42) -> None:
    """Make results repeatable from run to run."""
    np.random.seed(seed)
