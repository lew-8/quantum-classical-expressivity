"""The PQC used for classification and a small simulator to run it.

A PQC can be defined by the mapping rho(0) -> rho(t; Theta), where the initial state
is mapped onto a final state characterised by a set of parameters Theta. These
parameters can be separated into two categories, namely fixed and tunable
parameters x and theta. The fixed parameters are responsible for the input
encoding while the free parameters are optimised during training.

The circuits only have 2 or 3 qubits, so instead of cirq the state vector is
simulated directly with numpy (for all data points at once). Qubit 0 is the
leftmost bit, as in cirq, and R_P(a) = exp(-i a P / 2) as in cirq and pennylane.
"""

from dataclasses import dataclass

import numpy as np

CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
CZ = np.diag([1, 1, 1, -1]).astype(complex)

# Encoding layers S(x) of the report (gates applied in this order on every qubit)
ENCODINGS = {
    "E1": ("rz",),
    "E2": ("rz", "ry"),
    "E3": ("rz", "ry", "rz"),
}


# ---------------------------------------------------------------- gates
# Each rotation takes one angle (gives a 2x2 matrix) or an array of angles,
# one per data point (gives an array of 2x2 matrices).

def rx(theta):
    c, s = np.cos(np.asarray(theta) / 2), np.sin(np.asarray(theta) / 2)
    return np.stack([np.stack([c, -1j * s], -1), np.stack([-1j * s, c], -1)], -2)


def ry(theta):
    c, s = np.cos(np.asarray(theta) / 2), np.sin(np.asarray(theta) / 2)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2).astype(complex)


def rz(theta):
    phase = np.exp(-0.5j * np.asarray(theta, dtype=float))
    zero = np.zeros_like(phase)
    return np.stack([np.stack([phase, zero], -1), np.stack([zero, phase.conj()], -1)], -2)


ROTATIONS = {"rx": rx, "ry": ry, "rz": rz}


# ---------------------------------------------------------------- simulator

def zero_state(n_qubits, batch=1):
    """|0...0> for every data point, shape (batch, 2**n_qubits)."""
    state = np.zeros((batch, 2**n_qubits), dtype=complex)
    state[:, 0] = 1
    return state


def apply_gate(state, gate, qubits):
    """Apply a gate on the given qubits of every state in the batch.

    gate is either one matrix for all data points or one matrix per data point.
    """
    qubits = list(qubits)
    batch, dim = state.shape
    n, k = int(np.log2(dim)), len(qubits)
    if 2**n != dim:
        raise ValueError("state dimension is not a power of two")
    if len(set(qubits)) != k or min(qubits) < 0 or max(qubits) >= n:
        raise ValueError(f"invalid qubits {qubits} for {n} qubits")
    if gate.shape[-2:] != (2**k, 2**k):
        raise ValueError(f"gate of shape {gate.shape} does not act on {k} qubit(s)")

    # move the target qubits to the end, multiply, and move them back
    psi = state.reshape((batch,) + (2,) * n)
    targets, ends = [q + 1 for q in qubits], list(range(n + 1 - k, n + 1))
    psi = np.moveaxis(psi, targets, ends)
    shape = psi.shape
    psi = psi.reshape(batch, -1, 2**k)
    if gate.ndim == 2:
        psi = psi @ gate.T
    else:
        psi = np.einsum("bij,brj->bri", gate, psi)
    return np.moveaxis(psi.reshape(shape), ends, targets).reshape(batch, dim)


def random_unitary(dim, rng=None):
    """Random unitary, like scipy.stats.unitary_group.rvs(dim)."""
    rng = np.random.default_rng(rng)
    z = (rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def resolve_encoding(encoding):
    """'E3', 3 or 'encoding3' -> the gates of that encoding."""
    key = f"E{encoding}" if isinstance(encoding, (int, np.integer)) else str(encoding)
    key = key.upper().replace("ENCODING", "E").replace("_", "").replace(" ", "")
    if key not in ENCODINGS:
        raise ValueError(f"unknown encoding {encoding!r}, choose from {list(ENCODINGS)}")
    return ENCODINGS[key]


# ---------------------------------------------------------------- the PQC

@dataclass(frozen=True)
class PQCClassifier:
    """The PQC is comprised of an encoding and a variational part, W(theta) S(x) |0>.

    - Encoding layer that uses x_i as angles to encode data (one feature per qubit).
    - Variational layers of Rx and linearly-connected CZ (or CNOT). Each sequence of
      rotations + entangling gates is considered as 1 variational layer.
    - The output is the amplitude of the 11...1 string; label 1 if it is above 0.5.

    With 'E1' the encoding is only Rz on |0>, which just adds a global phase, so the
    output does not depend on the data. This is why the E1 classifier assigns the
    same label to the whole dataset.
    """

    n_qubits: int = 3
    n_layers: int = 5
    encoding: str = "E3"
    entangler: str = "cz"
    threshold: float = 0.5

    def __post_init__(self):
        resolve_encoding(self.encoding)
        if self.n_qubits < 1 or self.n_layers < 0:
            raise ValueError("need n_qubits >= 1 and n_layers >= 0")
        if self.entangler not in ("cz", "cnot"):
            raise ValueError("entangler must be 'cz' or 'cnot'")

    @property
    def n_params(self):
        # one Rx angle per qubit per layer
        return self.n_qubits * self.n_layers

    @property
    def encoding_gates(self):
        return resolve_encoding(self.encoding)

    def init_params(self, seed=None):
        # initial parameters, uniform in [-2 pi, 2 pi] as in the original notebook
        return np.random.default_rng(seed).uniform(-2 * np.pi, 2 * np.pi, size=self.n_params)

    def _check(self, params, X):
        params = np.asarray(params, dtype=float).ravel()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if params.size != self.n_params:
            raise ValueError(f"expected {self.n_params} parameters, got {params.size}")
        if X.shape[1] != self.n_qubits:
            raise ValueError(f"expected {self.n_qubits} features (one per qubit), got {X.shape[1]}")
        return params, X

    def state(self, params, X):
        """Final state of the circuit for every data point."""
        params, X = self._check(params, X)
        state = zero_state(self.n_qubits, len(X))
        # Encoding layer that uses x_i as angles to encode data.
        for gate in self.encoding_gates:
            for q in range(self.n_qubits):
                state = apply_gate(state, ROTATIONS[gate](X[:, q]), [q])
        # Variational layers of Rx and linearly-connected CZ
        ent = CZ if self.entangler == "cz" else CNOT
        for layer in params.reshape(self.n_layers, self.n_qubits):
            for q, theta in enumerate(layer):
                state = apply_gate(state, rx(theta), [q])
            for q in range(self.n_qubits - 1):
                state = apply_gate(state, ent, [q, q + 1])
        return state

    def scores(self, params, X):
        # Determine the relevant output of the circuit:
        # this is the amplitude of the 11...1 string
        return np.abs(self.state(params, X)[:, -1])

    def predict(self, params, X):
        return (self.scores(params, X) > self.threshold).astype(int)

    def loss(self, params, X, y):
        # Based on output of the circuit, compute the loss (square loss).
        return float(np.mean((self.scores(params, X) - np.ravel(y)) ** 2))

    def to_cirq(self, params, x):
        """The same circuit for one data point as a cirq.Circuit (needs cirq)."""
        import cirq

        params, X = self._check(params, x)
        qubits = cirq.LineQubit.range(self.n_qubits)
        gates = {"rx": cirq.rx, "ry": cirq.ry, "rz": cirq.rz}
        ent = cirq.CZ if self.entangler == "cz" else cirq.CNOT
        circuit = cirq.Circuit()
        for gate in self.encoding_gates:
            circuit.append(gates[gate](X[0, q]).on(qubits[q]) for q in range(self.n_qubits))
        for layer in params.reshape(self.n_layers, self.n_qubits):
            circuit.append(cirq.rx(t).on(qubits[q]) for q, t in enumerate(layer))
            circuit.append(ent(qubits[q], qubits[q + 1]) for q in range(self.n_qubits - 1))
        return circuit

    def draw(self, params, x):
        """Text diagram of the circuit (cirq if installed, otherwise a short summary)."""
        try:
            return self.to_cirq(params, x).to_text_diagram(transpose=False)
        except ImportError:
            enc = " ".join(g.upper() for g in self.encoding_gates)
            return (f"{self.n_qubits} qubits | encoding: {enc} | "
                    f"{self.n_layers} layers of RX + {self.entangler.upper()} | "
                    f"output: amplitude of |{'1' * self.n_qubits}>")
