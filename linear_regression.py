"""Linear regression, start to finish.

Linear regression draws the best straight line (or flat plane, with many
features) through the data. It is rarely the most accurate model, but it is
the most explainable one, and it is the baseline every fancier model has to
beat. This script fits it and then asks the questions people usually skip:

  1. Does it beat a naive guess?
  2. Which features matter, and are those effects real or noise?
  3. Are the assumptions behind the numbers actually met?
  4. Does it hold up across different splits of the data?
  5. Does regularisation (Ridge / Lasso) help?

Run:  python linear_regression.py
      python linear_regression.py --csv data/houses.csv --target price
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from toolkit import plots
from toolkit.data import load_dataset, make_parser, profile
from toolkit.evaluate import diagnose_fit, regression_report

try:
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.outliers_influence import variance_inflation_factor
except ImportError:  # statsmodels adds p-values and assumption tests; everything else works without it
    sm = None


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    args = make_parser(__doc__.splitlines()[0]).parse_args()
    plots.show_mode(args.show)

    # ------------------------------------------------------------------ data
    section("1. The data")
    X, y, about = load_dataset(args.csv, args.target, task="regression")
    print(about)
    profile(X, y)
    plots.correlation_heatmap(X.assign(**{str(y.name): y}), "linreg_correlations")

    # Hold back 20% the model never sees until the very end. That held-back
    # set is the only honest estimate of how it will do on new data.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ----------------------------------------------------------------- model
    section("2. Fit and score")
    # Scaling every feature to mean 0 / std 1 does not change the predictions
    # of plain linear regression, but it makes the coefficients comparable:
    # each one becomes "change in target per one-standard-deviation change in
    # this feature". The imputer fills gaps with the median so missing values
    # don't crash the fit.
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearRegression())
    model.fit(X_train, y_train)

    train_pred, test_pred = model.predict(X_train), model.predict(X_test)
    regression_report(y_train, train_pred, "Training set", y_train=y_train)
    test_scores = regression_report(y_test, test_pred, "Test set (the number that counts)", y_train=y_train)
    diagnose_fit(model.score(X_train, y_train), test_scores["r2"])

    plots.actual_vs_predicted(y_test, test_pred, "linreg_actual_vs_predicted", "Linear regression: test set")
    plots.residuals(y_test, test_pred, "linreg_residuals")

    # ------------------------------------------------------------ coefficients
    section("3. What drives the prediction")
    coefs = pd.Series(model[-1].coef_, index=X.columns)
    print(f"Intercept (prediction for a perfectly average row): {model[-1].intercept_:,.3f}\n")
    print("Effect of a one-standard-deviation increase in each feature, biggest first:")
    for name, value in coefs.reindex(coefs.abs().sort_values(ascending=False).index).items():
        print(f"  {name:<28} {value:+,.3f}")
    plots.bar_ranking(coefs, "linreg_coefficients", "Standardised coefficients",
                      "Change in target per 1 std of feature")

    print("\nCaution: a coefficient is the effect of one feature *holding the others fixed*.")
    print("When features overlap heavily (see the VIF check below), individual coefficients")
    print("become unstable and can even flip sign, even though predictions stay fine.")

    # ------------------------------------------------------- statistical view
    if sm is not None:
        section("4. Are the effects real? (statsmodels OLS)")
        # Reuse the pipeline's own imputer and scaler so the numbers line up with section 3.
        scaled = pd.DataFrame(model[:-1].transform(X_train), columns=X.columns, index=X_train.index)
        ols = sm.OLS(y_train.astype(float), sm.add_constant(scaled)).fit()

        table = pd.DataFrame({"coef": ols.params, "p_value": ols.pvalues,
                              "ci_low": ols.conf_int()[0], "ci_high": ols.conf_int()[1]}).drop("const")
        table = table.sort_values("p_value")
        print(table.round(4).to_string())
        weak = table.index[table.p_value > 0.05].tolist()
        print("\nA p-value under 0.05 means the effect is unlikely to be pure chance.")
        print("If the 95% confidence interval crosses zero, we can't even be sure of the direction.")
        if weak:
            print(f"Not clearly distinguishable from zero here: {weak}")
        print(f"Adjusted R² (penalises useless features): {ols.rsquared_adj:.3f}")

        section("5. Checking the assumptions")
        resid = ols.resid

        # Normal residuals: matters for p-values and intervals, not for predictions.
        skew = stats.skew(resid)
        print(f"Residual skew: {skew:+.2f} ", end="")
        print("(close to 0, fine)" if abs(skew) < 0.5 else "(lopsided: try a log transform of the target)")

        # Constant spread: if errors grow with the prediction, standard errors are wrong.
        _, bp_p, _, _ = het_breuschpagan(resid, ols.model.exog)
        print(f"Breusch-Pagan p-value: {bp_p:.3f} ", end="")
        print("(error spread looks constant)" if bp_p > 0.05
              else "(error spread changes with the prediction: consider log target or robust errors)")

        # Multicollinearity: VIF above ~5 means a feature is largely explained by the others.
        vif = pd.Series([variance_inflation_factor(scaled.values, i) for i in range(scaled.shape[1])],
                        index=X.columns).sort_values(ascending=False)
        high = vif[vif > 5]
        print("\nVariance inflation factor (above 5 = overlaps heavily with other features):")
        print(vif.head(10).round(2).to_string())
        if not high.empty:
            print(f"Overlapping features: {list(high.index)}. Their individual coefficients are unreliable.")
    else:
        print("\n(Install statsmodels for p-values, confidence intervals and assumption tests.)")

    # ------------------------------------------------------ cross-validation
    section("6. Does it hold up? (10-fold cross-validation)")
    # One train/test split can be lucky or unlucky. Cross-validation repeats
    # the experiment on 10 different splits and reports the spread.
    folds = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_r2 = cross_val_score(model, X, y, cv=folds, scoring="r2")
    print(f"R² across folds: mean {cv_r2.mean():.3f}, std {cv_r2.std():.3f}, "
          f"worst {cv_r2.min():.3f}, best {cv_r2.max():.3f}")
    if cv_r2.std() > 0.1:
        print("The score swings a lot between splits, so treat any single test score with suspicion.")

    # --------------------------------------------------------- regularisation
    section("7. Ridge and Lasso")
    # Both add a penalty for large coefficients, trading a little bias for
    # less variance. Ridge shrinks everything; Lasso can shrink features all
    # the way to zero, which doubles as automatic feature selection. The CV
    # versions pick the penalty strength (alpha) for us.
    alphas = np.logspace(-3, 3, 50)
    ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=alphas))
    lasso = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LassoCV(alphas=alphas, cv=5, random_state=42, max_iter=20000))

    for name, candidate in [("Plain", model), ("Ridge", ridge), ("Lasso", lasso)]:
        scores = cross_val_score(candidate, X, y, cv=folds, scoring="r2")
        print(f"  {name:<6} CV R² {scores.mean():.3f} ± {scores.std():.3f}")

    lasso.fit(X_train, y_train)
    dropped = X.columns[np.isclose(lasso[-1].coef_, 0)].tolist()
    print(f"\nLasso chose alpha = {lasso[-1].alpha_:.4g}")
    print(f"Features Lasso set to zero: {dropped or 'none'}")
    print("If regularised scores match the plain model, the plain model wasn't overfitting.")

    # ------------------------------------------------------- under the hood
    section("8. Under the hood")
    # Linear regression has a closed-form answer: the coefficients that
    # minimise squared error solve the "normal equation". numpy's least
    # squares solver gives the same numbers sklearn did.
    Z = model[:-1].transform(X_train)
    Z = np.column_stack([np.ones(len(Z)), Z])
    beta, *_ = np.linalg.lstsq(Z, y_train.to_numpy(dtype=float), rcond=None)
    match = np.allclose(beta[1:], model[-1].coef_, atol=1e-6)
    print(f"numpy least-squares coefficients match sklearn: {match}")
    print("See neural_net_from_scratch.py for the gradient-descent route to the same idea.")


if __name__ == "__main__":
    main()
