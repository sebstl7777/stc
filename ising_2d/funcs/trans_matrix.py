import numpy as np
from itertools import product

def generate_rows(L):
    """All 2^L row configurations (spins ±1)."""
    return np.array(list(product([1, -1], repeat=L)))


def row_energy(row, J):
    """Horizontal energy for one row, periodic."""
    L = len(row)
    return -J * sum(row[i] * row[(i+1) % L] for i in range(L))


def inter_row_energy(row1, row2, J):
    """Vertical coupling between two rows."""
    return -J * sum(row1[i] * row2[i] for i in range(len(row1)))


def build_transfer_matrix(L, beta, J):
    rows = generate_rows(L)
    n = len(rows)
    
    T = np.zeros((n, n))
    horiz_energies = np.array([row_energy(row, J) for row in rows])
    
    for i in range(n):
        for j in range(n):
            vert_E = inter_row_energy(rows[i], rows[j], J)
            total_E = horiz_energies[i] + vert_E
            T[i, j] = np.exp(-beta * total_E)
    
    return T


def partition_function_2d(L, beta, J):
    """Partition function via transfer matrix, periodic L×L."""
    T = build_transfer_matrix(L, beta, J)
    eigenvalues = np.linalg.eigvals(T)
    Z = np.sum(eigenvalues ** L)
    return np.real(Z)


def free_energy_per_spin(L, beta, J):
    Z = partition_function_2d(L, beta, J)
    N = L * L
    return - (1 / (beta * N)) * np.log(Z)
