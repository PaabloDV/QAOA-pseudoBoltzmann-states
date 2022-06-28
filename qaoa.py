import numpy as np
from numpy import pi as π
import math

# **********************************
# EXTENDED-QAOA SIMULATOR
# **********************************

# Definition of required gate
σz = np.array([[1, 0], [0, -1]])
σx = np.array([[0, 1], [1, 0]])
σy = -1j * σz @ σx
i2 = np.eye(2)


def local_operator(operator, N, i):
    return np.kron(np.eye(2 ** (N - i - 1)), np.kron(operator, np.eye(2 ** i)))


def product(operators):
    output = 1
    for op in operators:
        output = np.kron(op, output)
    return output


def apply_op(O, ψ):
    N = round(np.log2(ψ.size))
    for i in range(N):
        ψ = (ψ.reshape(-1, 2) @ O.T).transpose()
    return ψ.flatten()


def Ry(θ):
    return np.cos(θ / 2) * i2 - 1j * np.sin(θ / 2) * σy


def expH(γ, E):
    return np.exp((-1j * γ) * E)


def U1(λ):
    return np.cos(λ / 2) * i2 - 1j * np.sin(λ / 2) * σz


# Algorithm
def QAOA(θ, γ, N, E, λ=-π / 2):
    d = 2 ** N
    ψ2 = np.abs(apply_op(Ry(θ) @ U1(λ), expH(γ, E) * np.ones(d) / np.sqrt(d))) ** 2
    return ψ2


# **********************************
# OBSERVABLES TO MEASURE
# **********************************

# Optimal angles
from scipy.optimize import minimize


def optimization(N, E, initial_point=None, method="L-BFGS-B", cost_function='GS-Probability',**kwargs):
    """Solve a QAOA problem of 'N' qubits with energies 'E' sorted by
    index in the states of the qubit register, minimizing the two angles
    θ and γ in the QAOA ansatz."""

    def objective_function_gsprobability(x, N, E, Emin_ndx):
        θ, γ = x
        ψ2 = QAOA(θ, γ, N, E, λ=-π / 2)
        return -np.sum(ψ2[Emin_ndx])

    def objective_function_energy(x, N, E, Emin_ndx):
        θ, γ = x
        ψ2 = QAOA(θ, γ, N, E, λ=-π / 2)
        return np.sum(E*ψ2)

    if initial_point is None:
        θ0 = [(math.pi / 3) + (np.random.rand() - 0.5) * 10 ** (-2)]
        #θ0 = [(0.9) + (np.random.rand() - 0.5) * 10 ** (-2)]
        γ0 = [0.1 + (np.random.rand() - 0.5) * 10 ** (-2)]
        initial_point = θ0 + γ0

    if cost_function=='GS-Probability':
        #Maximize the probability of the ground state
        objective_function = objective_function_gsprobability
    elif cost_function=='Energy':
        #Minimize the expectation value of the Hamiltonian
        objective_function = objective_function_energy

    result = minimize(
        objective_function,
        initial_point,
        args=(N, E, np.where(E == np.min(E))),
        method=method,
        bounds=[(0, π / 2), (0, 1.5)],
        **kwargs,
    )

    θ, γ = result.x
    return θ, γ

# Performance in optimization

def GS_probability(ψ2, E):
    ndx = np.where(E == np.min(E))
    return np.sum(ψ2[ndx])

def GS_degeneracy(E):
    return len(np.where(E == np.min(E))[0])

def Enhancement(N, GSprob, E):
    return (GSprob * 2 ** N)/GS_degeneracy(E)


def GS_probability_mc(βmc, E):
    """Given a Boltzmann distribution with inverse temperature βmc and energy vector E, compute the 
    probability amplitude of the ground state"""
    p = np.exp(-βmc * E)
    p /= np.sum(p)
    return GS_probability(p, E)


# Effective temperature
from sklearn.linear_model import LinearRegression


def BoltzmannFit(ψ2, E, fraction=1):
    nstates = round(fraction * len(ψ2))
    x = E[:nstates].reshape(-1, 1)
    y = np.log(ψ2[:nstates])
    reg = LinearRegression().fit(x, y)
    β = -reg.coef_[0]
    org = reg.intercept_
    score = reg.score(x, y)
    return β, org, score