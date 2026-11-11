# Theoretical background

## Parameterized quantum circuits

A PQC can be defined by the mapping

ρ(0) ↦ ρ(t; Θ)

where the initial state ρ is mapped onto a final state ρ(Θ) which is characterised by a set of parameters Θ.
These parameters can be separated into two categories, namely fixed and tunable parameters x and θ. Of course in
real-world quantum computers one should take into account the specific computer's properties (for example its
physical geometry, as it impacts the way local operations are permitted to act on the system). Since this
approach is hardware independent, all parameters not regarding input encoding or training are omitted. The
fixed parameters are responsible for the input encoding while the free parameters are optimised during training.

A static state preparation circuit S(x) must be applied to the initial ground state in order to prepare such a
quantum state. The data is non-linearly mapped to the feature space where a hyperplane is built to divide the
labeled samples.

## Quantum variational classifier and quantum kernel estimator

In the quantum variational classifier the data is encoded into a state, x ↦ |ψ(x)⟩ = S(x)|0⟩^⊗n, and then a
variational circuit W(θ) is applied to it. The final state is measured and labels are assigned according to the
probability of measuring the value y. An empirical risk is then used as a loss function and classical
optimization methods are employed to find the parameters θ that minimize the loss. This is the method used in
this project.

In the quantum kernel estimator the quantum computer estimates the kernel K(xᵢ, xⱼ) = |⟨ψ(xᵢ)|ψ(xⱼ)⟩|² for the
points of the training set, and then a classical optimizer is used to maximize

Σᵢ αᵢ − ½ Σᵢⱼ yᵢ yⱼ αᵢ αⱼ K(xᵢ, xⱼ)   subject to   Σᵢ αᵢ yᵢ = 0.

Afterwards the quantum computer is used again to estimate the kernel K(xᵢ, s) for a new point s, and with the
optimal solution α* the point is classified as m(s) = sign(Σᵢ yᵢ αᵢ* K(xᵢ, s)).

## Feature maps and kernels

Feature maps map the input data from its original feature space to a higher-dimensional feature space,
x ↦ Φ(x), e.g. (x₁, x₂) ↦ (x₁, x₂, x₁², x₂², …). The purpose is to transform the data into a space where it is
more easily separable by a linear classifier. Adding x₁x₂ as a third input in this project is a simple example.

Kernels are functions that measure the similarity between pairs of data points in the feature space,
K(x, y) = ⟨Φ(x), Φ(y)⟩. They compute the inner product without explicitly calculating the transformation (the
kernel trick), which avoids working directly in high-dimensional spaces.

A kernel has a corresponding feature map when it satisfies Mercer's condition: for any set of points
x₁, …, xₙ the matrix Kᵢⱼ = K(xᵢ, xⱼ) must be positive semi-definite, meaning zᵀKz ≥ 0 for any vector z.

## Two results used in the analysis

**The generator only depends on f.** C₂ (I ⊗ Rz(f)) C₂ = exp(−i f Z⊗Z / 2) = cos(f/2) I − i sin(f/2) Z⊗Z. Since
⟨Z⊗Z⟩ is real, the real part of the expected value is cos(f/2) for any state, whatever the encoding and the
random unitary U. So the decision boundaries are where f(x₁, x₂) = π, 3π, …: the line x₂ = π for dataset 1, the
lines x₁ − x₂ = ±π for dataset 2 and hyperbolae x₁x₂ = (2k+1)π for dataset 3. Points with |cos(f/2)| < 0.3 get a
random label; for dataset 1 these are about 19% of the points, so about 0.9 is the best possible accuracy.

**Encoding 1 cannot learn.** |0⟩ is an eigenstate of Rz, so Rz(x)|0⟩ = e^(−ix/2)|0⟩. With Encoding 1 the
encoded state is |000⟩ times a global phase, which no measurement can see. The output of the PQC is the same for
every point, so it assigns the same label to the whole dataset and gets 0.5 accuracy on balanced data, however
many layers are used. Encoding 2 adds Ry(x), which changes the probabilities of |0⟩ and |1⟩, and Encoding 3 adds
a final Rz(x), which gives a relative phase that the Rx layers can turn into probabilities.

## Parameters

| Model | Parameters for l layers | l = 1 | l = 5 |
|---|---|---|---|
| PQC (3 qubits, one Rx per qubit per layer) | 3l | 3 | 15 |
| NN (3 inputs, l layers of 3 neurons, 1 output) | 12l + 4 | 16 | 64 |

## References

- V. Havlíček et al., "Supervised learning with quantum-enhanced feature spaces", Nature 567, 209 (2019),
  arXiv:1804.11326.
- M. Schuld, R. Sweke, J. J. Meyer, "Effect of data encoding on the expressive power of variational quantum
  machine-learning models", Phys. Rev. A 103, 032430 (2021), arXiv:2008.08605.
