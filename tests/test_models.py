import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from expressivity.classical_models import MLPBaseline, build_keras_mlp, mlp_param_count
from expressivity.datasets import make_dataset
from expressivity.evaluation import (
    accuracy,
    aggregate_benchmarks,
    plot_accuracy_comparison,
    plot_data,
    plot_loss_history,
    plot_prediction_comparison,
    save_benchmark_figures,
    summary_table,
)
from expressivity.training import main, parse_model_code, run_benchmarks, train_mlp, train_pqc


@pytest.fixture(scope="module")
def dataset_1():
    return make_dataset("linear", n_points=400, seed=0).split(seed=0)


@pytest.fixture(scope="module")
def dataset_2():
    return make_dataset("difference", n_points=400, seed=0).split(seed=0)


def test_nn_parameters(dataset_1):
    X_train, _, y_train, _ = dataset_1
    for layers, expected in [(1, 16), (5, 64)]:
        assert mlp_param_count(3, 3, layers) == expected
        assert MLPBaseline(n_layers=layers, epochs=2).fit(X_train, y_train).n_params == expected


def test_nn_learns_dataset_1(dataset_1):
    X_train, X_test, y_train, y_test = dataset_1
    res = train_mlp(X_train, y_train, X_test, y_test, n_layers=1, seed=0)
    # ~19% of the points get a random label, so ~0.9 is the best possible accuracy
    assert res.test_accuracy > 0.8
    assert len(res.loss_history) == 80


def test_keras_network():
    pytest.importorskip("keras")
    assert build_keras_mlp(n_layers=1).count_params() == 16


def test_pqc_training_lowers_the_loss(dataset_2):
    X_train, X_test, y_train, y_test = dataset_2
    res = train_pqc(X_train, y_train, X_test, y_test, encoding="E3", n_layers=1, maxiter=40)
    assert res.n_params == 3
    assert min(res.loss_history) < res.loss_history[0]


def test_encoding_3_better_than_encoding_1(dataset_2):
    X_train, X_test, y_train, y_test = dataset_2
    e1 = train_pqc(X_train, y_train, X_test, y_test, encoding="E1", n_layers=1, seed=0)
    e3 = train_pqc(X_train, y_train, X_test, y_test, encoding="E3", n_layers=1, seed=0)
    assert e1.test_accuracy == pytest.approx(0.5)
    assert e3.test_accuracy > 0.6


@pytest.mark.parametrize("code,expected", [("SNN", ("mlp", 1, None)), ("DE3", ("pqc", 5, "E3")),
                                           ("E3L15", ("pqc", 15, "E3"))])
def test_model_names(code, expected):
    assert parse_model_code(code) == expected
    with pytest.raises(ValueError):
        parse_model_code("XE4")


@pytest.fixture(scope="module")
def small_benchmark():
    return run_benchmarks(datasets=("linear", "difference"), models=("SNN", "SE1", "SE3"),
                          n_points=120, seeds=(0, 1), maxiter=10, epochs=5)


def test_benchmark_and_table(small_benchmark):
    assert len(small_benchmark) == 2 * 3 * 2
    assert set(aggregate_benchmarks(small_benchmark)["runs"]) == {2}
    table = summary_table(small_benchmark)
    assert list(table.index) == ["linear", "difference"]
    assert list(table.columns) == ["SNN", "SE1", "SE3"]


def test_accuracy():
    assert accuracy([0, 1, 1, 0], [0, 1, 0, 0]) == 0.75
    with pytest.raises(ValueError):
        accuracy([0, 1], [0])


def test_figures_and_command_line(tmp_path, small_benchmark, dataset_1):
    X_train, _, y_train, _ = dataset_1
    figs = [plot_data(X_train, y_train),
            plot_prediction_comparison(X_train, y_train, y_train, y_train),
            plot_accuracy_comparison(small_benchmark),
            plot_loss_history([train_mlp(X_train, y_train, X_train, y_train, epochs=3)])]
    assert all(isinstance(f, plt.Figure) for f in figs)
    plt.close("all")

    assert all(p.exists() for p in save_benchmark_figures(small_benchmark, tmp_path, 60))
    out = tmp_path / "bench.csv"
    main(["--datasets", "linear", "--models", "SNN", "SE3", "--n-points", "60",
          "--maxiter", "5", "--epochs", "3", "--out", str(out), "--quiet"])
    assert len(pd.read_csv(out)) == 2
    assert np.isnan(pd.read_csv(out).loc[0, "initial_test_accuracy"])  # SNN has no initial
