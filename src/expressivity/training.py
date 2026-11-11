"""Training of the PQC and the NN, and the full set of experiments.

Model names follow the table of the report: 'S' and 'D' denote the type of
architecture (shallow = 1 layer, deep = 5 layers), 'NN' the neural network and
'E1,2,3' the encoding scheme used for the PQC. E.g. SNN, DNN, SE1, ..., DE3.
Other depths can be written as 'E3L15' (PQC) or 'NNL2' (NN).

Run all experiments from the terminal with:

    expressivity-benchmark --seeds 0 1 2 --out results/benchmarks.csv --figures results/figures
"""

import argparse
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from .classical_models import MLPBaseline
from .datasets import make_dataset
from .evaluation import accuracy, save_benchmark_figures, summary_table
from .quantum_circuits import PQCClassifier

SHALLOW_LAYERS = 1
DEEP_LAYERS = 5
DEFAULT_MODELS = ("SNN", "DNN", "SE1", "SE2", "SE3", "DE1", "DE2", "DE3")


@dataclass
class TrainingResult:
    model: str
    family: str  # "pqc" or "mlp"
    n_layers: int
    n_params: int
    train_accuracy: float
    test_accuracy: float
    initial_train_accuracy: float  # with the random starting weights (PQC only)
    initial_test_accuracy: float
    loss_history: list
    seconds: float
    encoding: str = None
    params: np.ndarray = field(default=None, repr=False)
    initial_params: np.ndarray = field(default=None, repr=False)
    estimator: object = field(default=None, repr=False)

    def summary(self):
        """One row of the results table."""
        return {
            "model": self.model,
            "family": self.family,
            "encoding": self.encoding or "",
            "n_layers": self.n_layers,
            "n_params": self.n_params,
            "train_accuracy": round(self.train_accuracy, 4),
            "test_accuracy": round(self.test_accuracy, 4),
            "initial_train_accuracy": round(self.initial_train_accuracy, 4),
            "initial_test_accuracy": round(self.initial_test_accuracy, 4),
            "final_loss": round(self.loss_history[-1], 6) if self.loss_history else np.nan,
            "iterations": len(self.loss_history),
            "seconds": round(self.seconds, 3),
        }


def parse_model_code(code):
    """'DE3' -> ('pqc', 5, 'E3'), 'SNN' -> ('mlp', 1, None), 'E3L15' -> ('pqc', 15, 'E3')."""
    code = code.upper().strip()
    m = re.fullmatch(r"([SD])(NN|E[123])", code)
    if m:
        depth = SHALLOW_LAYERS if m.group(1) == "S" else DEEP_LAYERS
        kind = m.group(2)
    else:
        m = re.fullmatch(r"(NN|E[123])L(\d+)", code)
        if not m:
            raise ValueError(f"unknown model code {code!r}")
        kind, depth = m.group(1), int(m.group(2))
    return ("mlp", depth, None) if kind == "NN" else ("pqc", depth, kind)


def train_pqc(X_train, y_train, X_test, y_test, encoding="E3", n_layers=5, maxiter=80,
              method="COBYLA", seed=42, name=None, entangler="cz"):
    """Train the PQC. The loss is optimised with COBYLA as the derivatives of the
    loss function cannot be accessed by the classical optimizer."""
    clf = PQCClassifier(n_qubits=np.shape(X_train)[1], n_layers=n_layers,
                        encoding=encoding, entangler=entangler)
    theta0 = clf.init_params(seed)

    # Save the current cost to plot later
    tracking_cost = []

    # Loss function to pass to classical optimizer
    def cost_to_optimize(params):
        cost = clf.loss(params, X_train, y_train)
        tracking_cost.append(cost)
        return cost

    start = time.perf_counter()
    if clf.n_params == 0:
        final_params = theta0
        cost_to_optimize(theta0)
    else:
        final_params = minimize(cost_to_optimize, theta0, method=method,
                                options={"maxiter": maxiter}).x
    seconds = time.perf_counter() - start

    enc = f"E{encoding}" if isinstance(encoding, int) else str(encoding).upper()
    return TrainingResult(
        model=name or f"{enc}L{n_layers}",
        family="pqc",
        n_layers=n_layers,
        n_params=clf.n_params,
        train_accuracy=accuracy(y_train, clf.predict(final_params, X_train)),
        test_accuracy=accuracy(y_test, clf.predict(final_params, X_test)),
        initial_train_accuracy=accuracy(y_train, clf.predict(theta0, X_train)),
        initial_test_accuracy=accuracy(y_test, clf.predict(theta0, X_test)),
        loss_history=tracking_cost,
        seconds=seconds,
        encoding=enc,
        params=final_params,
        initial_params=theta0,
        estimator=clf,
    )


def train_mlp(X_train, y_train, X_test, y_test, n_layers=1, epochs=80, width=3, seed=42,
              name=None):
    """Train the NN (binary crossentropy, adam). There are no "initial" accuracies for
    the NN, so they are reported as NaN."""
    model = MLPBaseline(n_layers=n_layers, width=width, epochs=epochs, seed=seed)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    seconds = time.perf_counter() - start
    return TrainingResult(
        model=name or f"NNL{n_layers}",
        family="mlp",
        n_layers=n_layers,
        n_params=model.n_params,
        train_accuracy=accuracy(y_train, model.predict(X_train)),
        test_accuracy=accuracy(y_test, model.predict(X_test)),
        initial_train_accuracy=float("nan"),
        initial_test_accuracy=float("nan"),
        loss_history=model.loss_history,
        seconds=seconds,
        estimator=model,
    )


def train_model(code, X_train, y_train, X_test, y_test, maxiter=80, epochs=80, seed=42):
    """Train the model given by its name in the report's table (e.g. 'SNN', 'DE2')."""
    family, depth, encoding = parse_model_code(code)
    if family == "mlp":
        return train_mlp(X_train, y_train, X_test, y_test, n_layers=depth, epochs=epochs,
                         seed=seed, name=code.upper())
    return train_pqc(X_train, y_train, X_test, y_test, encoding=encoding, n_layers=depth,
                     maxiter=maxiter, seed=seed, name=code.upper())


def run_experiment(dataset, models=DEFAULT_MODELS, test_size=0.5, maxiter=80, epochs=80,
                   seed=42):
    """Train every model on the same train/test split of one dataset."""
    X_train, X_test, y_train, y_test = dataset.split(test_size=test_size, seed=seed)
    return [train_model(code, X_train, y_train, X_test, y_test, maxiter, epochs, seed)
            for code in models]


def run_benchmarks(datasets=("linear", "difference", "product"), models=DEFAULT_MODELS,
                   n_points=1000, margin=0.3, seeds=(42,), maxiter=80, epochs=80,
                   verbose=False):
    """All datasets x all models, repeated for every seed. Returns a pandas DataFrame.

    Running the models multiple times and averaging the accuracies makes the results
    more robust against the networks' starting weights.
    """
    import pandas as pd

    rows = []
    for seed in seeds:
        for name in datasets:
            data = make_dataset(name, n_points=n_points, margin=margin, seed=seed)
            for res in run_experiment(data, models, maxiter=maxiter, epochs=epochs, seed=seed):
                rows.append({"dataset": data.name, "seed": seed, **res.summary()})
                if verbose:
                    print(f"seed={seed:<4} {data.name:<10} {res.model:<5} "
                          f"train={res.train_accuracy:.3f} test={res.test_accuracy:.3f} "
                          f"params={res.n_params}")
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare the PQC and the NN on the 3 datasets.")
    parser.add_argument("--datasets", nargs="+", default=["linear", "difference", "product"])
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--n-points", type=int, default=1000)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--maxiter", type=int, default=80, help="COBYLA iterations (PQC)")
    parser.add_argument("--epochs", type=int, default=80, help="training epochs (NN)")
    parser.add_argument("--out", type=Path, default=Path("results/benchmarks.csv"))
    parser.add_argument("--figures", type=Path, default=None, help="folder for the figures")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    df = run_benchmarks(args.datasets, args.models, args.n_points, args.margin, args.seeds,
                        args.maxiter, args.epochs, verbose=not args.quiet)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {len(df)} rows to {args.out}\n")
    print(summary_table(df).to_string())
    if args.figures is not None:
        for path in save_benchmark_figures(df, args.figures, n_points=args.n_points,
                                           margin=args.margin, seed=args.seeds[0]):
            print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
