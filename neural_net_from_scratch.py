"""A neural network in plain numpy, so nothing is magic.

Libraries like Keras and PyTorch hide the three ideas that make every neural
network work. This file writes them out by hand:

  1. Forward pass   multiply by weights, add a bias, apply a non-linearity, repeat
  2. Loss           one number saying how wrong the predictions are
  3. Backprop       the chain rule, run backwards, gives each weight's share of the blame
                    so gradient descent can nudge every weight downhill

The data is "two moons": two interlocking crescents no straight line can
separate. Logistic regression (a network with no hidden layer) gets stuck;
one small hidden layer bends the boundary and solves it.

Run:  python neural_net_from_scratch.py
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from toolkit import plots

rng = np.random.default_rng(0)


# --------------------------------------------------------------- building blocks
def relu(z):
    # Keeps positives, zeroes negatives. Without a non-linearity like this,
    # stacking layers is pointless: many linear layers collapse into one.
    return np.maximum(0, z)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def binary_cross_entropy(y, p):
    # Punishes confident wrong answers very hard: predicting 0.99 for a true 0
    # costs far more than predicting 0.6.
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


class TinyNetwork:
    """Input -> hidden layer (ReLU) -> one output (sigmoid probability)."""

    def __init__(self, n_inputs: int, n_hidden: int):
        # He initialisation: random weights scaled to the layer size so the
        # signal neither explodes nor fades as it passes through ReLU layers.
        # Starting every weight at zero would make all hidden units identical
        # for ever, so randomness is essential.
        self.W1 = rng.normal(0, np.sqrt(2 / n_inputs), (n_inputs, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, np.sqrt(2 / n_hidden), (n_hidden, 1))
        self.b2 = np.zeros(1)

    def forward(self, X):
        # Keep the intermediate values; backprop needs them.
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.p = sigmoid(self.z2).ravel()
        return self.p

    def backward(self, y):
        """Gradients of the loss with respect to every weight, via the chain rule."""
        n = len(y)
        # Sigmoid + cross-entropy has a famously tidy derivative: prediction minus truth.
        dz2 = (self.p - y).reshape(-1, 1) / n
        grads = {"W2": self.a1.T @ dz2, "b2": dz2.sum(axis=0)}
        # Pass the blame back through W2, then through ReLU (whose slope is 1
        # where the unit was active and 0 where it was switched off).
        dz1 = (dz2 @ self.W2.T) * (self.z1 > 0)
        grads["W1"] = self.X.T @ dz1
        grads["b1"] = dz1.sum(axis=0)
        return grads

    def step(self, grads, learning_rate):
        # Gradient descent: move each weight a small step against its gradient.
        for name, g in grads.items():
            setattr(self, name, getattr(self, name) - learning_rate * g)


# ----------------------------------------------------------------- sanity check
def gradient_check(net, X, y, eps=1e-5):
    """Compare backprop's gradients with a brute-force numerical estimate.

    If they disagree, the backward pass has a bug. Worth doing once whenever
    you hand-write gradients.
    """
    net.forward(X)
    analytic = net.backward(y)["W1"]
    numeric = np.zeros_like(net.W1)
    for i in range(net.W1.shape[0]):
        for j in range(net.W1.shape[1]):
            original = net.W1[i, j]
            net.W1[i, j] = original + eps
            up = binary_cross_entropy(y, net.forward(X))
            net.W1[i, j] = original - eps
            down = binary_cross_entropy(y, net.forward(X))
            net.W1[i, j] = original
            numeric[i, j] = (up - down) / (2 * eps)
    return np.linalg.norm(analytic - numeric) / (np.linalg.norm(analytic) + np.linalg.norm(numeric))


def train(n_hidden, X_train, y_train, X_val, y_val, epochs=3000, learning_rate=0.5):
    net = TinyNetwork(X_train.shape[1], n_hidden)
    history = {"loss": [], "val_loss": []}
    for _ in range(epochs):
        p = net.forward(X_train)
        history["loss"].append(binary_cross_entropy(y_train, p))
        net.step(net.backward(y_train), learning_rate)
        history["val_loss"].append(binary_cross_entropy(y_val, net.forward(X_val)))
    return net, history


def accuracy(net, X, y):
    return np.mean((net.forward(X) >= 0.5) == y)


def plot_boundary(ax, net, X, y, title):
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200),
                         np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 200))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    ax.contourf(xx, yy, net.forward(grid).reshape(xx.shape), levels=20, cmap="RdBu", alpha=0.6)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolor="k", s=15)
    ax.set_title(title)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--show", action="store_true", help="Open plot windows as well as saving them.")
    args = parser.parse_args()
    plots.show_mode(args.show)

    X, y = make_moons(n_samples=600, noise=0.25, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=0)

    print("Gradient check (relative difference, want < 1e-6):", end=" ")
    print(f"{gradient_check(TinyNetwork(2, 8), X_train[:50], y_train[:50]):.2e}")

    # One hidden unit is nearly a linear model; 128 is far more than this problem needs.
    results = {}
    for hidden in [1, 8, 128]:
        net, history = train(hidden, X_train, y_train, X_val, y_val)
        results[hidden] = (net, history)
        val = np.array(history["val_loss"])
        best_epoch = int(val.argmin())
        print(f"\nHidden units: {hidden:>3}   train acc {accuracy(net, X_train, y_train):.1%}   "
              f"validation acc {accuracy(net, X_val, y_val):.1%}")
        print(f"  validation loss: best {val.min():.3f} at epoch {best_epoch + 1}, final {val[-1]:.3f}", end="")
        # The tell-tale sign of overfitting: training loss keeps falling while
        # validation loss turns back up. Accuracy can hide this for a while.
        if val[-1] > val.min() * 1.1:
            print("  <- validation loss rose again: overfitting, early stopping would have helped")
        else:
            print()

    print("\nWhat to notice:")
    print("  - 1 hidden unit can only bend the boundary once, so it plateaus (underfitting).")
    print("  - A handful of units is enough to wrap around the moons.")
    print("  - Many more units barely improve accuracy, and the validation loss creeps back up")
    print("    as the network starts memorising noise. Capacity beyond what the data needs costs you.")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (hidden, (net, _)) in zip(axes, results.items(), strict=True):
        plot_boundary(ax, net, X_val, y_val, f"{hidden} hidden unit(s): {accuracy(net, X_val, y_val):.0%}")
    plots.finish(fig, "scratch_nn_decision_boundaries")
    plots.learning_curves(results[128][1], "scratch_nn_learning_curve")

    # Learning rate is the knob that most often breaks training.
    print("\nLearning-rate sweep with 16 hidden units, 1,500 epochs:")
    for lr in [0.001, 0.5, 20.0, 50.0]:
        _, history = train(16, X_train, y_train, X_val, y_val, epochs=1500, learning_rate=lr)
        loss = np.array(history["loss"])
        biggest_jump = np.max(np.diff(loss))
        if not np.isfinite(loss[-1]) or biggest_jump > 0.05:
            note = f"unstable: loss jumped up by {biggest_jump:.2f} in one step (steps too big, overshooting)"
        elif loss[-1] > 0.3:
            note = "too small: still crawling downhill"
        else:
            note = "converged nicely"
        print(f"  lr = {lr:<6} final loss {loss[-1]:.3f}  {note}")


if __name__ == "__main__":
    main()
