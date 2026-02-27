import numpy as np

try:
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

def init_stc_state(size, given_puzzle=None):
    """Initialize the probability tensor and the mask for fixed clues."""
    logits = np.zeros((size, size, size))
    fixed_mask = np.zeros((size, size), dtype=bool)
    if given_puzzle is not None:
        filled = given_puzzle != 0
        fixed_mask = filled
        r_idx, c_idx = np.where(filled)
        val = given_puzzle[filled] - 1
        logits[r_idx, c_idx, :] = -1e4
        logits[r_idx, c_idx, val] = 1e4
    return logits, fixed_mask

def get_probabilities(logits):
    """Convert logits to a stable probability distribution."""
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def _count_collisions_per_row(arr):
    """Vectorized: size - n_unique for each row. arr shape (..., size)."""
    sorted_a = np.sort(arr, axis=-1)
    n_unique = 1 + np.sum(sorted_a[:, 1:] != sorted_a[:, :-1], axis=-1)
    return arr.shape[-1] - n_unique


def calculate_energy_dynamic(samples, row_weights, col_weights, box_weights):
    """Dynamic EBM: multiplies collisions by evolving penalty weights."""
    M, size, _ = samples.shape
    box_size = int(np.sqrt(size))
    
    # Row collisions: (M, size)
    row_colls = _count_collisions_per_row(samples.reshape(M * size, size)).reshape(M, size)
    col_colls = _count_collisions_per_row(samples.transpose(0, 2, 1).reshape(M * size, size)).reshape(M, size)
    
    # Box collisions: reshape so each box becomes a row
    boxes = samples.reshape(M, box_size, box_size, box_size, box_size).transpose(0, 1, 3, 2, 4).reshape(M, size, size)
    box_colls = _count_collisions_per_row(boxes.reshape(M * size, size)).reshape(M, size)
    
    raw_violations = np.stack([row_colls, col_colls, box_colls], axis=1)
    energies = (row_colls * row_weights).sum(axis=1) + (col_colls * col_weights).sum(axis=1) + (box_colls * box_weights).sum(axis=1)
    return energies, raw_violations

def _generate_samples_numba_core(probs, samples):
    num_samples, size = samples.shape[0], probs.shape[0]
    for i in prange(num_samples):
        for r in range(size):
            p_row = probs[r].copy()
            for c in range(size):
                p_c = p_row[c].copy()
                s = p_c.sum()
                if s <= 1e-8:
                    p_c[:] = 1.0
                    for prev_c in range(c):
                        p_c[samples[i, r, prev_c] - 1] = 0.0
                    s = p_c.sum()
                p_c /= s
                digit_idx = np.searchsorted(np.cumsum(p_c), np.random.random())
                if digit_idx >= size:
                    digit_idx = size - 1
                samples[i, r, c] = digit_idx + 1
                p_row[:, digit_idx] = 0.0
    return samples


if _NUMBA_AVAILABLE:
    _generate_samples_numba = njit(cache=True, parallel=True)(_generate_samples_numba_core)


def _generate_samples_numpy(probs, samples):
    """Pure numpy fallback when Numba unavailable or disabled."""
    num_samples, size = samples.shape[0], probs.shape[0]
    rng = np.random.default_rng()
    for i in range(num_samples):
        for r in range(size):
            p_row = probs[r].copy()
            for c in range(size):
                p_c = p_row[c].copy()
                s = p_c.sum()
                if s <= 1e-8:
                    p_c[:] = 1.0
                    for prev_c in range(c):
                        p_c[samples[i, r, prev_c] - 1] = 0.0
                    s = p_c.sum()
                p_c /= s
                digit_idx = np.searchsorted(np.cumsum(p_c), rng.random())
                if digit_idx >= size:
                    digit_idx = size - 1
                samples[i, r, c] = digit_idx + 1
                p_row[:, digit_idx] = 0.0
    return samples


def generate_samples(logits, num_samples):
    """STC row-permutation sampler. Row energy = 0 by construction."""
    size = logits.shape[0]
    samples = np.zeros((num_samples, size, size), dtype=np.int32)
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = (exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)).astype(np.float64)
    if _NUMBA_AVAILABLE:
        try:
            return _generate_samples_numba(probs, samples)
        except Exception:
            return _generate_samples_numpy(probs, samples)
    return _generate_samples_numpy(probs, samples)

def update_tensor(logits, samples, energies, learning_rate, fixed_mask, given_puzzle):
    """STC Top-K Elite: average only best samples. Skips incomplete samples (zeros) from SMC."""
    size = logits.shape[0]
    complete = np.all(samples >= 1, axis=(1, 2))
    if np.any(complete):
        comp_energies = np.where(complete, energies, np.inf)
        k = max(1, int(len(samples) * 0.1))
        elite_idx = np.argpartition(comp_energies, min(k, comp_energies.size - 1))[:k]
        elite_idx = elite_idx[complete[elite_idx]]
        if len(elite_idx) == 0:
            return logits
        elite = samples[elite_idx]
    else:
        return logits
    k = len(elite)
    weighted_counts = np.zeros((size, size, size))
    inv_k = 1.0 / k
    for r in range(size):
        for c in range(size):
            bc = np.bincount(elite[:, r, c] - 1, minlength=size) * inv_k
            weighted_counts[r, c, :] = bc
    
    epsilon = 1e-8
    target_logits = np.log(weighted_counts + epsilon)
    
    # Update the tensor
    new_logits = (1 - learning_rate) * logits + learning_rate * target_logits
    
    if given_puzzle is not None and np.any(fixed_mask):
        r_idx, c_idx = np.where(fixed_mask)
        val = given_puzzle[fixed_mask] - 1
        new_logits[r_idx, c_idx, :] = -1e4
        new_logits[r_idx, c_idx, val] = 1e4
    return new_logits


def update_tensor_elite(logits, samples, energies, learning_rate, fixed_mask, given_puzzle, elite_fraction):
    """Elite update with dynamic elite_fraction for parallel tempering."""
    size = logits.shape[0]
    k = max(1, int(len(samples) * elite_fraction))
    elite = samples[np.argsort(energies)[:k]]
    weighted_counts = np.zeros((size, size, size))
    inv_k = 1.0 / k
    for r in range(size):
        for c in range(size):
            bc = np.bincount(elite[:, r, c] - 1, minlength=size) * inv_k
            weighted_counts[r, c, :] = bc
    epsilon = 1e-8
    target_logits = np.log(weighted_counts + epsilon)
    new_logits = (1 - learning_rate) * logits + learning_rate * target_logits
    if given_puzzle is not None and np.any(fixed_mask):
        r_idx, c_idx = np.where(fixed_mask)
        val = given_puzzle[fixed_mask] - 1
        new_logits[r_idx, c_idx, :] = -1e4
        new_logits[r_idx, c_idx, val] = 1e4
    return new_logits


def apply_message_passing(logits, given_puzzle, fixed_mask, strength=0.5):
    """Soft constraint propagation: cells broadcast probs to suppress conflicts in row/col/box."""
    size = logits.shape[0]
    box_size = int(np.sqrt(size))
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    penalties = np.zeros_like(logits)
    penalties += probs.sum(axis=1, keepdims=True)
    penalties += probs.sum(axis=0, keepdims=True)
    for br in range(box_size):
        for bc in range(box_size):
            r0, r1 = br * box_size, (br + 1) * box_size
            c0, c1 = bc * box_size, (bc + 1) * box_size
            box_sum = probs[r0:r1, c0:c1, :].sum(axis=(0, 1))
            penalties[r0:r1, c0:c1, :] += box_sum
    penalties -= 3 * probs

    new_logits = logits - (strength * penalties)
    if given_puzzle is not None and np.any(fixed_mask):
        r_idx, c_idx = np.where(fixed_mask)
        val = given_puzzle[fixed_mask] - 1
        new_logits[r_idx, c_idx, :] = -1e4
        new_logits[r_idx, c_idx, val] = 1e4
    return new_logits


def generate_samples_smc(logits, num_samples, given_puzzle=None, fixed_mask=None):
    """SMC with forward masking: only valid digits per row/col/box. Returns (samples, energies)."""
    size = logits.shape[0]
    box_size = int(np.sqrt(size))
    samples = np.zeros((num_samples, size, size), dtype=np.int32)
    energies = np.zeros(num_samples)
    if fixed_mask is None and given_puzzle is not None:
        fixed_mask = given_puzzle != 0
    elif fixed_mask is None:
        fixed_mask = np.zeros((size, size), dtype=bool)

    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    base_probs = exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-12)
    rng = np.random.default_rng()

    for i in range(num_samples):
        col_used = np.zeros((size, size), dtype=bool)
        box_used = np.zeros((size, size), dtype=bool)
        dead_end = False

        for r in range(size):
            if dead_end:
                break
            row_used = np.zeros(size, dtype=bool)

            for c in range(size):
                box_idx = (r // box_size) * box_size + (c // box_size)

                if given_puzzle is not None and fixed_mask[r, c]:
                    digit = given_puzzle[r, c]
                    samples[i, r, c] = digit
                    d = digit - 1
                    row_used[d] = True
                    col_used[c, d] = True
                    box_used[box_idx, d] = True
                    continue

                p_c = base_probs[r, c].copy()
                p_c[row_used] = 0
                p_c[col_used[c]] = 0
                p_c[box_used[box_idx]] = 0
                sum_p = np.sum(p_c)

                if sum_p <= 1e-8:
                    dead_end = True
                    energies[i] = (size - r) * size + (size - c)
                    break

                p_c /= sum_p
                digit_idx = np.searchsorted(np.cumsum(p_c), rng.random())
                if digit_idx >= size:
                    digit_idx = size - 1
                samples[i, r, c] = digit_idx + 1
                row_used[digit_idx] = True
                col_used[c, digit_idx] = True
                box_used[box_idx, digit_idx] = True

    return samples, energies


def sinkhorn_projection(logits, iterations=10, given_puzzle=None, fixed_mask=None):
    """Synchronize tensor to row/col/box constraints in continuous space. Re-applies fixed clues if provided."""
    size = logits.shape[0]
    box_size = int(np.sqrt(size))
    P = np.exp(np.clip(logits - np.max(logits, axis=-1, keepdims=True), -500, 500))
    P = P / (np.sum(P, axis=-1, keepdims=True) + 1e-12)

    for _ in range(iterations):
        P = P / (np.sum(P, axis=0, keepdims=True) + 1e-12)
        P = P / (np.sum(P, axis=1, keepdims=True) + 1e-12)
        for br in range(box_size):
            for bc in range(box_size):
                r0, r1 = br * box_size, (br + 1) * box_size
                c0, c1 = bc * box_size, (bc + 1) * box_size
                box_sum = np.sum(P[r0:r1, c0:c1, :], axis=(0, 1), keepdims=True)
                P[r0:r1, c0:c1, :] /= (box_sum + 1e-12)
        P = P / (np.sum(P, axis=-1, keepdims=True) + 1e-12)

    out = np.log(P + 1e-12)
    if given_puzzle is not None and fixed_mask is not None and np.any(fixed_mask):
        r_idx, c_idx = np.where(fixed_mask)
        val = given_puzzle[fixed_mask] - 1
        out[r_idx, c_idx, :] = -1e4
        out[r_idx, c_idx, val] = 1e4
    return out


def targeted_cluster_swap(board, fixed_mask, verbose=True):
    """Last-mile local search: try swapping pairs of cells within each row to clear remaining collisions."""
    size = board.shape[0]
    box_size = int(np.sqrt(size))
    test_board = board.copy()
    for r in range(size):
        for c1 in range(size):
            if fixed_mask[r, c1]:
                continue
            for c2 in range(c1 + 1, size):
                if fixed_mask[r, c2]:
                    continue
                test_board[r, c1], test_board[r, c2] = test_board[r, c2], test_board[r, c1]
                solved = True
                for i in range(size):
                    if len(set(test_board[:, i])) != size:
                        solved = False
                        break
                if solved:
                    for br in range(box_size):
                        for bc in range(box_size):
                            box = test_board[br*box_size:(br+1)*box_size, bc*box_size:(bc+1)*box_size]
                            if len(set(box.flatten())) != size:
                                solved = False
                                break
                if solved:
                    if verbose:
                        print(f"\n   [CLUSTER SWAP] Row {r}, cols ({c1}, {c2})", flush=True)
                    return test_board
                test_board[r, c1], test_board[r, c2] = test_board[r, c2], test_board[r, c1]
    return board


def _board_energy(board, size):
    """Total collision count for a single board (for cluster-swap check)."""
    ones = np.ones(size)
    e, _ = calculate_energy_dynamic(board.reshape(1, size, size), ones, ones, ones)
    return e[0]


def run_parallel_tempering_stc(size, given_puzzle=None, iterations=1000, batch_per_replica=1000):
    """Parallel tempering with multiple elite-fraction replicas and replica exchange."""
    print(f"\nStarting {size}x{size} Parallel Tempering STC Solver...", flush=True)
    elite_fractions = [0.02, 0.05, 0.15, 0.40]
    num_replicas = len(elite_fractions)
    learning_rate = 0.3

    replicas_logits = []
    fixed_masks = []
    for _ in range(num_replicas):
        l, fm = init_stc_state(size, given_puzzle)
        replicas_logits.append(l)
        fixed_masks.append(fm)

    ones = np.ones(size)
    best_samples = [None] * num_replicas

    for step in range(iterations):
        replica_min_energies = []

        for i in range(num_replicas):
            samples = generate_samples(replicas_logits[i], batch_per_replica)
            energies, _ = calculate_energy_dynamic(samples, ones, ones, ones)

            min_e = np.min(energies)
            replica_min_energies.append(min_e)
            best_samples[i] = samples[np.argmin(energies)]

            if min_e == 0:
                print(f"\n>>> Perfect Solution Found by Replica {i} (Elite Frac {elite_fractions[i]}) at step {step}! <<<", flush=True)
                return best_samples[i]

            replicas_logits[i] = update_tensor_elite(
                replicas_logits[i], samples, energies,
                learning_rate, fixed_masks[i], given_puzzle, elite_fractions[i]
            )

        if step % 20 == 0:
            print(f"Step {step:3d} | Min Energies [Cold -> Hot]: {replica_min_energies}", flush=True)

        if step > 0 and step % 10 == 0:
            for i in range(num_replicas - 1):
                E_cold = replica_min_energies[i]
                E_hot = replica_min_energies[i + 1]
                delta_E = E_hot - E_cold
                if delta_E <= 0:
                    swap_prob = 1.0
                else:
                    swap_prob = np.exp(-delta_E / (elite_fractions[i + 1] * 20))
                if np.random.rand() < swap_prob:
                    replicas_logits[i], replicas_logits[i + 1] = replicas_logits[i + 1], replicas_logits[i]
                    if delta_E < 0:
                        print(f"   [SWAP] Replica {i} and {i+1} swapped! Greedy solver teleported to better minimum.", flush=True)

    print(">>> Failed to converge to 0 energy. Returning best guess. <<<", flush=True)
    best_idx = np.argmin(replica_min_energies)
    return best_samples[best_idx]


def run_stc_solver(size, given_puzzle=None, iterations=1500, batch_size=4000, learning_rate=0.2, msg_pass_strength=0.5):
    print(f"\nStarting {size}x{size} Dynamic GLS STC Solver...", flush=True)
    logits, fixed_mask = init_stc_state(size, given_puzzle)
    
    # Initialize penalty weights to 1.0 (Standard Sudoku Rules)
    row_weights = np.ones(size)
    col_weights = np.ones(size)
    box_weights = np.ones(size)
    
    best_min_energy = float('inf')
    stagnation_counter = 0
    
    for step in range(iterations):
        samples = generate_samples(logits, batch_size)
        
        # Calculate dynamic energy
        energies, raw_violations = calculate_energy_dynamic(samples, row_weights, col_weights, box_weights)
        
        min_energy = np.min(energies)
        mean_energy = np.mean(energies)
        best_idx = np.argmin(energies)
        
        # To know if we are ACTUALLY solved, we must check if raw violations are 0
        # (Because min_energy could be 0 if weights somehow broke, though they only increase)
        actual_collisions = np.sum(raw_violations[best_idx])
        
        if actual_collisions < best_min_energy:
            best_min_energy = actual_collisions
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            
        if step % 50 == 0 or actual_collisions == 0:
            print(f"Step {step:3d} | Actual Collisions: {actual_collisions:3.0f} | Weighted E: {min_energy:5.1f} | Patience: {stagnation_counter}", flush=True)
            
        if actual_collisions == 0:
            print(">>> Perfect Solution Found! <<<", flush=True)
            return samples[best_idx]

        if actual_collisions > 0 and actual_collisions <= 4:
            fixed_board = targeted_cluster_swap(samples[best_idx].copy(), fixed_mask, verbose=True)
            if _board_energy(fixed_board, size) == 0:
                print(">>> Perfect Solution Found via Cluster Swap! <<<", flush=True)
                return fixed_board
            
        # ==========================================
        # THE GLOBAL TRICK: Dynamic Constraint Warping
        # ==========================================
        if stagnation_counter > 25:
            print(f"Step {step:3d} | Stuck at {actual_collisions} collisions (patience {stagnation_counter}). Warping...", flush=True)
            
            # Find exactly which rules the best current sample is breaking
            best_violations = raw_violations[best_idx]
            
            # Increase the penalty for THOSE SPECIFIC rules
            # This forces the STC to prioritize fixing them in the next batch
            row_weights += best_violations[0] * 2.0
            col_weights += best_violations[1] * 2.0
            box_weights += best_violations[2] * 2.0
            
            # We still melt the logits slightly so the tensor has the flexibility to change
            logits = logits * 0.8
            
            # Reset patience
            stagnation_counter = 0
            
        else:
            logits = update_tensor(logits, samples, energies, learning_rate, fixed_mask, given_puzzle)
            logits = apply_message_passing(logits, given_puzzle, fixed_mask, strength=msg_pass_strength)
            
    print(">>> Failed to converge to 0 energy. Returning best guess. <<<", flush=True)
    return samples[np.argmin(energies)]
