"""A deep neural network for tabular regression, done properly with Keras.

This replaces the original neural_nets.py. Same idea (stacked Dense layers
with dropout), with the fixes that usually decide whether a network trains
well or not:

  - inputs are scaled (unscaled inputs are the number one cause of a network
    that "just doesn't learn")
  - batches of 32 instead of 1 (faster, and smoother gradients)
  - early stopping keeps the best epoch instead of the last one
  - the learning rate drops automatically when progress stalls
  - the result is compared with plain linear regression, because on small
    tabular datasets a neural net often doesn't win, and you should know

Run:  python 04_neural_net_keras.py
      python 04_neural_net_keras.py --csv data/crime.csv --target total_crime_reported_per_1_million_res
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow's start-up chatter

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from toolkit import plots
from toolkit.data import load_dataset, make_parser, profile
from toolkit.evaluate import diagnose_fit, regression_report


def build_model(n_features: int, learning_rate: float, dropout_rate: float, l2: float):
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers

    # A funnel: wide first layer to find many simple patterns, narrower layers
    # to combine them. Dropout randomly switches off a share of units on each
    # training step so the network can't lean on any single one; L2 keeps
    # weights small. Both fight overfitting.
    reg = regularizers.l2(l2)
    model = tf.keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(128, activation="relu", kernel_regularizer=reg),
        layers.Dropout(dropout_rate),
        layers.Dense(64, activation="relu", kernel_regularizer=reg),
        layers.Dropout(dropout_rate),
        layers.Dense(32, activation="relu", kernel_regularizer=reg),
        layers.Dense(1),  # no activation: regression outputs any real number
    ])

    # Loss choices worth knowing:
    #   regression      "mse" (penalises big misses), "mae" or "huber" (robust to outliers)
    #   yes/no          sigmoid output + "binary_crossentropy"
    #   many classes    softmax output + "sparse_categorical_crossentropy"
    # Adam adapts the step size per weight and is a strong default.
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
    return model


def main() -> None:
    parser = make_parser(__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=500, help="Upper limit; early stopping usually ends sooner.")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--l2", type=float, default=1e-4)
    args = parser.parse_args()
    plots.show_mode(args.show)

    import tensorflow as tf

    tf.keras.utils.set_random_seed(42)  # seeds Python, numpy and TensorFlow together

    X, y, about = load_dataset(args.csv, args.target, task="regression")
    print(about)
    profile(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the scaler on training data only. Fitting on everything would leak
    # information about the test set into training.
    prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    X_train_s = prep.fit_transform(X_train).astype("float32")
    X_test_s = prep.transform(X_test).astype("float32")

    # Scaling the target too keeps the loss in a comfortable range for the
    # optimiser. We undo it before reporting, so the errors are in real units.
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train_s = ((y_train - y_mean) / y_std).to_numpy("float32")

    model = build_model(X_train_s.shape[1], args.learning_rate, args.dropout, args.l2)
    model.summary(print_fn=lambda line: print("  " + line))

    callbacks = [
        # Stop once validation loss hasn't improved for 30 epochs, and roll
        # back to the best weights rather than keeping the last ones.
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True),
        # Halve the learning rate after 10 flat epochs: big steps early, fine-tuning later.
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6),
    ]
    history = model.fit(X_train_s, y_train_s, validation_split=0.2, epochs=args.epochs, batch_size=32,
                        callbacks=callbacks, verbose=0).history

    best = int(np.argmin(history["val_loss"]))
    print(f"\nTrained {len(history['loss'])} epochs; best validation loss at epoch {best + 1}.")
    if len(history["loss"]) == args.epochs:
        print("Hit the epoch limit before early stopping fired; it may still be improving. Try more --epochs.")

    def predict(features):
        return model.predict(features, verbose=0).ravel() * y_std + y_mean

    train_pred, test_pred = predict(X_train_s), predict(X_test_s)
    regression_report(y_train, train_pred, "Neural net: training set", y_train=y_train)
    nn = regression_report(y_test, test_pred, "Neural net: test set", y_train=y_train)

    from sklearn.metrics import r2_score
    diagnose_fit(r2_score(y_train, train_pred), nn["r2"])
    print("  (Dropout is switched off when predicting, which is why train can look better than it did")
    print("   during training. The learning curve below shows the during-training view.)")

    plots.learning_curves(history, "keras_learning_curves", metrics=("loss", "mae"))
    plots.actual_vs_predicted(y_test, test_pred, "keras_actual_vs_predicted", "Neural net: test set")
    plots.residuals(y_test, test_pred, "keras_residuals")

    # The honest comparison. A deep net has thousands of weights; linear
    # regression has one per feature. If they score the same, prefer the
    # simple one: it's faster, explainable and harder to break.
    linear = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearRegression()).fit(X_train, y_train)
    lin = regression_report(y_test, linear.predict(X_test), "Linear regression on the same split", y_train=y_train)

    print(f"\nNeural net weights: {model.count_params():,}   Linear regression weights: {X.shape[1] + 1}")
    diff = nn["rmse"] - lin["rmse"]
    if diff < -0.02 * lin["rmse"]:
        print(f"The neural net wins by {-diff:,.2f} RMSE on this split, a hint of non-linear patterns.")
        print(f"With only {len(y_test)} test rows, confirm with cross-validation before paying for the complexity.")
    elif diff > 0.02 * lin["rmse"]:
        print(f"Linear regression wins by {diff:,.2f} RMSE. Typical for small tabular data: not enough rows to")
        print("feed a network. Gradient-boosted trees (05_model_comparison.py) are usually the stronger choice here.")
    else:
        print("They tie (within 2%). Prefer linear regression: same accuracy, far simpler.")


if __name__ == "__main__":
    main()
