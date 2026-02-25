import numpy as np

def mcmc_ising_baseline(L, beta, J, num_steps, burn_in):
    """Metropolis MCMC for 2D Ising. Returns (energies, magnetizations)."""
    N = L * L
    lattice = np.random.choice([1, -1], size=(L, L))
    dE_values = [-8*J, -4*J, 0, 4*J, 8*J]
    probabilities = {dE: np.exp(-beta * dE) for dE in dE_values}
    
    def total_energy(lat):
        horiz = -J * np.sum(lat * np.roll(lat, shift=-1, axis=1))
        vert = -J * np.sum(lat * np.roll(lat, shift=-1, axis=0))
        return horiz + vert

    energies = np.zeros(num_steps)
    magnetizations = np.zeros(num_steps)
    E = total_energy(lattice)
    M = np.sum(lattice)
    
    total_sweeps = burn_in + num_steps

    for step in range(total_sweeps):
        for _ in range(N):
            i, j = np.random.randint(0, L, 2)
            spin = lattice[i, j]
            neighbors = (lattice[(i+1)%L, j] + lattice[(i-1)%L, j] +
                         lattice[i, (j+1)%L] + lattice[i, (j-1)%L])
            dE = 2 * J * spin * neighbors
            if dE <= 0 or np.random.rand() < probabilities[dE]:
                lattice[i, j] *= -1
                E += dE
                M += 2 * lattice[i, j]
        if step >= burn_in:
            idx = step - burn_in
            energies[idx] = E
            magnetizations[idx] = M

    return energies, magnetizations

def calculate_mcmc_observables(energies, magnetizations, L, beta):
    """Observables from MCMC trajectories."""
    N = L * L
    E_mean = np.mean(energies) / N
    abs_M_mean = np.mean(np.abs(magnetizations)) / N
    
    E2_mean = np.mean(energies**2)
    M2_mean = np.mean(magnetizations**2)
    specific_heat = (beta**2 / N) * (E2_mean - np.mean(energies)**2)
    susceptibility = (beta / N) * (M2_mean - np.mean(np.abs(magnetizations))**2)
    
    return {
        "Magnetization_per_spin": abs_M_mean,
        "Energy_per_spin": E_mean,
        "Specific_Heat": specific_heat,
        "Susceptibility": susceptibility
    }
