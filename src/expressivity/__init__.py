"""Expressivity of the generating set: PQCs vs classical neural networks.

Code of the Applied Quantum Algorithms mini project (T.2.E). Example:

    from expressivity import make_dataset, train_pqc

    data = make_dataset("difference", n_points=1000, seed=0)
    X_train, X_test, y_train, y_test = data.split()
    result = train_pqc(X_train, y_train, X_test, y_test, encoding="E3", n_layers=1)
"""

from .classical_models import MLPBaseline, build_keras_mlp, mlp_param_count
from .datasets import Dataset, make_dataset
from .evaluation import accuracy, aggregate_benchmarks, summary_table
from .quantum_circuits import ENCODINGS, PQCClassifier

__version__ = "0.1.0"

# The functions of training.py are only imported when they are asked for, so that
# `python -m expressivity.training` does not load training.py twice.
_TRAINING_NAMES = ["DEFAULT_MODELS", "TrainingResult", "run_benchmarks", "run_experiment",
                   "train_mlp", "train_model", "train_pqc"]


def __getattr__(name):
    if name in _TRAINING_NAMES:
        from . import training
        return getattr(training, name)
    raise AttributeError(f"module 'expressivity' has no attribute {name!r}")


__all__ = ["ENCODINGS", "Dataset", "MLPBaseline", "PQCClassifier", "accuracy",
           "aggregate_benchmarks", "build_keras_mlp", "make_dataset", "mlp_param_count",
           "summary_table", *_TRAINING_NAMES]
