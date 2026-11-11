import numpy as np
import pytest

from expressivity.datasets import (
    ROTATION_FUNCTIONS,
    assign_labels,
    generator_expectation,
    make_dataset,
    sample_points,
)
from expressivity.quantum_circuits import random_unitary


def test_points_are_in_the_square():
    X = sample_points(500, rng=1)
    assert X.shape == (500, 2)
    assert X.min() >= 0 and X.max() <= 2 * np.pi


@pytest.mark.parametrize("name", list(ROTATION_FUNCTIONS))
def test_expected_value_is_cos_f_over_2(name):
    # C2 Rz(f) C2 = exp(-i f ZZ / 2), so the real part is cos(f/2) for any U
    f = ROTATION_FUNCTIONS[name]
    X = sample_points(200, rng=3)
    for seed in range(3):
        expval = generator_expectation(X, f, random_unitary(4, seed))
        np.testing.assert_allclose(expval, np.cos(f(X[:, 0], X[:, 1]) / 2), atol=1e-12)


def test_labels_and_random_band():
    expval = np.linspace(-1, 1, 2001)
    labels = assign_labels(expval, margin=0.3, rng=0)
    sure = np.abs(expval) >= 0.3
    np.testing.assert_array_equal(labels[sure], expval[sure] >= 0)
    assert 0.3 < labels[~sure].mean() < 0.7  # both labels in the band
    with pytest.raises(ValueError):
        assign_labels(expval, margin=-1)


@pytest.mark.parametrize("name", [1, 2, 3, "linear", "dataset2", "product"])
def test_datasets_are_balanced(name):
    data = make_dataset(name, n_points=400, seed=5)
    assert data.X.shape == (400, 2)
    assert data.class_balance == 0.5


def test_dataset_1_boundary_is_x2_equal_pi():
    data = make_dataset("linear", n_points=400, margin=0.0, seed=0)
    np.testing.assert_array_equal(data.y, data.X[:, 1] <= np.pi)


def test_same_seed_same_data():
    a, b = make_dataset("product", 200, seed=11), make_dataset("product", 200, seed=11)
    np.testing.assert_array_equal(a.X, b.X)
    np.testing.assert_array_equal(a.y, b.y)
    assert not np.array_equal(a.X, make_dataset("product", 200, seed=12).X)


def test_features_and_split():
    data = make_dataset("difference", n_points=200, seed=0)
    np.testing.assert_allclose(data.features[:, 2], data.X[:, 0] * data.X[:, 1])
    X_train, X_test, y_train, y_test = data.split(seed=0)
    assert X_train.shape == X_test.shape == (100, 3)
    assert y_train.mean() == y_test.mean() == 0.5


@pytest.mark.parametrize("kwargs", [{"name": "circle"}, {"n_points": 7}, {"n_points": 1}])
def test_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        make_dataset(**{"name": "linear", "n_points": 10, **kwargs})
