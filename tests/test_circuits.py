from functools import reduce

import numpy as np
import pytest
from scipy.linalg import expm

from expressivity.quantum_circuits import (
    CNOT,
    CZ,
    PQCClassifier,
    apply_gate,
    random_unitary,
    resolve_encoding,
    rx,
    ry,
    rz,
    zero_state,
)

PAULIS = {"x": np.array([[0, 1], [1, 0]]), "y": np.array([[0, -1j], [1j, 0]]),
          "z": np.diag([1, -1])}


def full_matrix(gate, qubits, n):
    """Slow reference: the 2**n x 2**n matrix of a gate acting on `qubits`."""
    k = len(qubits)
    order = list(qubits) + [q for q in range(n) if q not in qubits]
    perm = []
    for idx in range(2**n):
        bits = [(idx >> (n - 1 - p)) & 1 for p in range(n)]
        real = [0] * n
        for pos, q in enumerate(order):
            real[q] = bits[pos]
        perm.append(int("".join(map(str, real)), 2))
    P = np.eye(2**n)[:, perm]
    return P @ np.kron(gate, np.eye(2 ** (n - k))) @ P.T


@pytest.mark.parametrize("gate,pauli", [(rx, "x"), (ry, "y"), (rz, "z")])
def test_rotations(gate, pauli):
    for theta in (-2.1, 0.0, 0.7, np.pi):
        np.testing.assert_allclose(gate(theta), expm(-0.5j * theta * PAULIS[pauli]), atol=1e-12)
    np.testing.assert_allclose(gate(np.array([0.1, 0.9]))[1], gate(0.9))


@pytest.mark.parametrize("gate,qubits", [(ry(1.1), [2]), (CNOT, [2, 0]), (CZ, [1, 2]),
                                         (random_unitary(8, 2), [1, 2, 0])])
def test_apply_gate_matches_full_matrix(gate, qubits):
    rng = np.random.default_rng(0)
    state = rng.standard_normal((4, 8)) + 1j * rng.standard_normal((4, 8))
    expected = state @ full_matrix(gate, qubits, 3).T
    np.testing.assert_allclose(apply_gate(state, gate, qubits), expected, atol=1e-12)


def test_cnot_first_qubit_is_control():
    state = apply_gate(zero_state(2), rx(np.pi), [0])  # |10>
    np.testing.assert_allclose(np.abs(apply_gate(state, CNOT, [0, 1])[0]), [0, 0, 0, 1], atol=1e-12)


def test_apply_gate_errors():
    with pytest.raises(ValueError):
        apply_gate(zero_state(2), CNOT, [0])
    with pytest.raises(ValueError):
        apply_gate(zero_state(2), rx(0.1), [2])


def test_encoding_names():
    assert resolve_encoding("E3") == resolve_encoding(3) == resolve_encoding("encoding_3")
    with pytest.raises(ValueError):
        resolve_encoding("E7")


@pytest.mark.parametrize("encoding", ["E1", "E2", "E3"])
@pytest.mark.parametrize("entangler", ["cz", "cnot"])
def test_pqc_matches_full_matrix_version(encoding, entangler):
    clf = PQCClassifier(n_layers=2, encoding=encoding, entangler=entangler)
    rng = np.random.default_rng(4)
    params = clf.init_params(4)
    X = rng.uniform(0, 2 * np.pi, (5, 3))
    gates = {"rx": rx, "ry": ry, "rz": rz}
    ent = CZ if entangler == "cz" else CNOT
    for x, psi in zip(X, clf.state(params, X)):
        ref = np.eye(8)[0].astype(complex)
        for g in clf.encoding_gates:
            ref = reduce(np.kron, [gates[g](x[q]) for q in range(3)]) @ ref
        for layer in params.reshape(2, 3):
            ref = reduce(np.kron, [rx(t) for t in layer]) @ ref
            ref = full_matrix(ent, [1, 2], 3) @ full_matrix(ent, [0, 1], 3) @ ref
        np.testing.assert_allclose(psi, ref, atol=1e-10)


def test_encoding_1_does_not_see_the_data():
    # Rz on |0> is only a phase, so the E1 classifier assigns the same label everywhere
    clf = PQCClassifier(n_layers=4, encoding="E1")
    X = np.random.default_rng(0).uniform(0, 2 * np.pi, (40, 3))
    scores = clf.scores(clf.init_params(3), X)
    np.testing.assert_allclose(scores, scores[0], atol=1e-12)


def test_pqc_outputs_and_errors():
    clf = PQCClassifier(n_layers=3)
    params = clf.init_params(0)
    X = np.random.default_rng(1).uniform(0, 2 * np.pi, (50, 3))
    scores = clf.scores(params, X)
    assert np.all((scores >= 0) & (scores <= 1)) and np.ptp(scores) > 0.1
    np.testing.assert_array_equal(clf.predict(params, X), scores > 0.5)
    assert clf.n_params == 9
    with pytest.raises(ValueError):
        clf.scores(np.zeros(4), X)
    with pytest.raises(ValueError):
        PQCClassifier(entangler="swap")
    assert isinstance(clf.draw(params, X[0]), str)


def test_same_as_cirq():
    cirq = pytest.importorskip("cirq")
    clf = PQCClassifier(n_layers=3)
    params, x = clf.init_params(7), np.array([0.4, 2.2, 5.1])
    expected = cirq.Simulator().simulate(clf.to_cirq(params, x)).final_state_vector
    assert abs(np.vdot(expected, clf.state(params, x)[0])) == pytest.approx(1.0, abs=1e-5)
