"""The classical classifier.

The classical classifier is a fully connected feed-forward neural network with 3 input
neurons and l layers. For the NN we use 3 neurons for each layer (to be as equivalent
as possible to the PQC). The loss function used is binary crossentropy and adam as
optimizer, and the model runs for a maximum of 80 epochs.

The original notebook used Keras. Here the same network is built with scikit-learn so
that tensorflow is not needed; build_keras_mlp() still gives the Keras version.
The inputs are standardised first: without it the x1 * x2 input (up to ~39) makes the
narrow 5-layer network predict a single label.
"""

import warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def mlp_param_count(n_inputs=3, width=3, n_layers=1):
    """Number of weights and biases: 12 l + 4 for the 3-3-...-3-1 network."""
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")
    sizes = [n_inputs] + [width] * n_layers + [1]
    return sum((a + 1) * b for a, b in zip(sizes[:-1], sizes[1:]))


@dataclass
class MLPBaseline:
    n_layers: int = 1
    width: int = 3
    epochs: int = 80
    activation: str = "relu"
    learning_rate: float = 1e-3
    batch_size: int = 32
    seed: int = 0
    standardize: bool = True
    model: Pipeline = field(init=False, repr=False)

    def __post_init__(self):
        if self.n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        mlp = MLPClassifier(
            hidden_layer_sizes=(self.width,) * self.n_layers,
            activation=self.activation,
            solver="adam",
            learning_rate_init=self.learning_rate,
            batch_size=self.batch_size,
            max_iter=self.epochs,
            random_state=self.seed,
            n_iter_no_change=self.epochs + 1,  # no early stopping: run all epochs
            tol=0.0,
        )
        steps = [("scale", StandardScaler())] if self.standardize else []
        self.model = Pipeline(steps + [("mlp", mlp)])

    @property
    def network(self):
        return self.model.named_steps["mlp"]

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.network.set_params(batch_size=min(self.batch_size, len(X)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.model.fit(X, np.ravel(y).astype(int))
        return self

    def predict(self, X):
        return self.model.predict(np.asarray(X, dtype=float)).astype(int)

    def predict_proba(self, X):
        return self.model.predict_proba(np.asarray(X, dtype=float))[:, 1]

    @property
    def loss_history(self):
        return list(getattr(self.network, "loss_curve_", []))

    @property
    def n_params(self):
        net = self.network
        if hasattr(net, "coefs_"):
            return int(sum(w.size for w in net.coefs_) + sum(b.size for b in net.intercepts_))
        return mlp_param_count(3, self.width, self.n_layers)


def build_keras_mlp(n_inputs=3, width=3, n_layers=1, activation="relu"):
    """The original Keras network (needs tensorflow)."""
    import keras

    layers = [keras.Input(shape=(n_inputs,))]
    layers += [keras.layers.Dense(width, activation=activation) for _ in range(n_layers)]
    layers.append(keras.layers.Dense(1, activation="sigmoid"))
    model = keras.Sequential(layers)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
