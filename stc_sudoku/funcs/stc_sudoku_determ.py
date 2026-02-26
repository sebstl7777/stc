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
    """STC Top-K Elite: average only best samples."""
    size = logits.shape[0]
    k = max(1, int(len(samples) * 0.1))
    elite = samples[np.argpartition(energies, k)[:k]]
    
    # Vectorized digit counts via bincount
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
