import numpy as np

def run_stc_ising(M, L, beta, J=1.0):
    """STC for 2D Ising. Proposal: 1D open columns."""
    N = L * L
    samples = np.zeros((M, L, L), dtype=np.int8)
    samples[:, 0, :] = np.random.choice([1, -1], size=(M, L))
    p_align = np.exp(beta * J) / (np.exp(beta * J) + np.exp(-beta * J))
    for i in range(1, L):
        rand_vals = np.random.rand(M, L)
        multipliers = np.where(rand_vals < p_align, 1, -1)
        samples[:, i, :] = samples[:, i-1, :] * multipliers

    kept_vert_E = -J * np.sum(samples[:, :-1, :] * samples[:, 1:, :], axis=(1, 2))
    broken_vert_E = -J * np.sum(samples[:, -1, :] * samples[:, 0, :], axis=1)
    broken_horiz_E = -J * np.sum(samples * np.roll(samples, shift=-1, axis=2), axis=(1, 2))
    
    E_broken = broken_vert_E + broken_horiz_E
    E_total = kept_vert_E + E_broken
    log_w = -beta * E_broken
    max_log_w = np.max(log_w)
    weights = np.exp(log_w - max_log_w)
    norm_w = weights / np.sum(weights)

    log_Z0 = L * np.log(2) + L * (L - 1) * np.log(2 * np.cosh(beta * J))
    avg_shifted_weight = np.mean(weights)
    log_Z = log_Z0 + np.log(avg_shifted_weight) + max_log_w
    
    free_energy_per_spin = - (1 / (beta * N)) * log_Z

    mag = np.sum(samples, axis=(1, 2))
    abs_mag = np.abs(mag)
    E_mean = np.sum(norm_w * E_total)
    E2_mean = np.sum(norm_w * (E_total ** 2))
    
    abs_M_mean = np.sum(norm_w * abs_mag)
    M2_mean = np.sum(norm_w * (mag ** 2))
    Cv = (beta ** 2 / N) * (E2_mean - E_mean ** 2)
    chi = (beta / N) * (M2_mean - abs_M_mean ** 2)
    
    return {
        "Log_Z": log_Z,
        "Free_Energy_per_spin": free_energy_per_spin,
        "Energy_per_spin": E_mean / N,
        "Magnetization_per_spin": abs_M_mean / N,
        "Specific_Heat": Cv,
        "Susceptibility": chi
    }

import numpy as np

def run_stc_comb_ising(M, L, beta, J=1.0):
    """
    STC for 2D Ising Model using a Comb Spanning Tree proposal.
    """
    N = L * L
    samples = np.zeros((M, L, L), dtype=np.int8)
    
    # Probability to align with a connected neighbor
    p_align = np.exp(beta * J) / (np.exp(beta * J) + np.exp(-beta * J))
    
    # ==========================================
    # 1. GENERATE EXACT SAMPLES FROM THE COMB
    # ==========================================
    # Step A: Generate the Backbone (Row 0)
    samples[:, 0, 0] = np.random.choice([1, -1], size=M) # Root of the tree
    for j in range(1, L):
        rand_vals = np.random.rand(M)
        multipliers = np.where(rand_vals < p_align, 1, -1)
        samples[:, 0, j] = samples[:, 0, j-1] * multipliers
        
    # Step B: Generate the Teeth (Rows 1 to L-1)
    for i in range(1, L):
        rand_vals = np.random.rand(M, L)
        multipliers = np.where(rand_vals < p_align, 1, -1)
        samples[:, i, :] = samples[:, i-1, :] * multipliers

    # ==========================================
    # 2. CALCULATE ENERGIES & WEIGHTS
    # ==========================================
    # We must explicitly calculate the total energy, and the tree energy
    
    # 1. Total Energy of the 2D configuration
    horiz_E_total = -J * np.sum(samples * np.roll(samples, shift=-1, axis=2), axis=(1, 2))
    vert_E_total = -J * np.sum(samples * np.roll(samples, shift=-1, axis=1), axis=(1, 2))
    E_total = horiz_E_total + vert_E_total
    
    # 2. Energy of the intact Comb Tree bonds
    # Backbone bonds: row 0, cols 0 to L-2
    E_backbone = -J * np.sum(samples[:, 0, :-1] * samples[:, 0, 1:], axis=1)
    # Teeth bonds: all cols, rows 0 to L-2
    E_teeth = -J * np.sum(samples[:, :-1, :] * samples[:, 1:, :], axis=(1, 2))
    E_tree = E_backbone + E_teeth
    
    # 3. Energy of the Broken Bonds
    E_broken = E_total - E_tree
    
    # Importance Weight: w = exp(-beta * E_broken)
    log_w = -beta * E_broken
    max_log_w = np.max(log_w)
    weights = np.exp(log_w - max_log_w)
    norm_w = weights / np.sum(weights)
    
    # Effective Sample Size diagnostic
    ESS = 1.0 / np.sum(norm_w ** 2)
    
    # ==========================================
    # 3. OBSERVABLES & PARTITION FUNCTION
    # ==========================================
    # Z_tree for ANY spanning tree of size N is 2 * (2 cosh(beta J))^(N-1)
    log_Z_tree = np.log(2) + (N - 1) * np.log(2 * np.cosh(beta * J))
    log_Z = log_Z_tree + np.log(np.mean(weights)) + max_log_w
    free_energy = - (1 / (beta * N)) * log_Z
    
    mag = np.sum(samples, axis=(1, 2))
    abs_mag = np.abs(mag)
    
    E_mean = np.sum(norm_w * E_total)
    E2_mean = np.sum(norm_w * (E_total ** 2))
    abs_M_mean = np.sum(norm_w * abs_mag)
    M2_mean = np.sum(norm_w * (mag ** 2))
    
    Cv = (beta ** 2 / N) * (E2_mean - E_mean ** 2)
    chi = (beta / N) * (M2_mean - abs_M_mean ** 2)
    
    return {
        "Log_Z": log_Z,
        "Free_Energy_per_spin": free_energy,
        "Specific_Heat": Cv,
        "Susceptibility": chi,
        "ESS": ESS
    }
