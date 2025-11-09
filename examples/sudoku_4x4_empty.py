import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# 4x4 mini-sudoku setup (symbols 1..4, 2x2 boxes)
# 0 denotes empty cell
puzzle = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
], dtype=int)

N = 4
nums = range(1, N+1)

# Probabilities per cell and digit: P[r,c,n]
P = np.ones((N, N, N), dtype=float) / N

# Fix given clues as one-hot
for r in range(N):
    for c in range(N):
        if puzzle[r, c] != 0:
            val = puzzle[r, c] - 1
            P[r, c, :] = 0.0
            P[r, c, val] = 1.0

# Helper: compute soft constraints cost for a candidate assignment distribution P
# We'll compute, for each constraint, a local penalty per (r,c,n)

def compute_penalties(P):
    # Targets: each cell exactly one number; each row/col/box each number appears exactly once
    # We convert these equalities into squared penalties on expectations.
    # For differentiability-like behavior, we distribute gradients as local penalties.

    local_pen = np.zeros_like(P)

    # 1) Cell constraint: sum_n P[r,c,n] = 1
    cell_sum = P.sum(axis=2, keepdims=True)  # (N,N,1)
    cell_err = (cell_sum - 1.0)  # (N,N,1)
    # distribute equally to each n in the cell
    local_pen += np.repeat(cell_err / N, N, axis=2)

    # 2) Row constraint: for each (r,n), sum_c P[r,c,n] = 1
    row_sum = P.sum(axis=1, keepdims=True)  # (N,1,N)
    row_err = (row_sum - 1.0)  # (N,1,N)
    # broadcast to columns
    local_pen += np.repeat(row_err / N, N, axis=1)

    # 3) Col constraint: for each (c,n), sum_r P[r,c,n] = 1
    col_sum = P.sum(axis=0, keepdims=True)  # (1,N,N)
    col_err = (col_sum - 1.0)
    local_pen += np.repeat(col_err / N, N, axis=0)

    # 4) Box constraint: 2x2 boxes; for each box b and n, sum_{(r,c) in b} P[r,c,n] = 1
    box_size = 2
    for br in range(0, N, box_size):
        for bc in range(0, N, box_size):
            box = P[br:br+box_size, bc:bc+box_size, :]  # (2,2,N)
            bsum = box.sum(axis=(0,1), keepdims=True)  # (1,1,N)
            berr = (bsum - 1.0)  # (1,1,N)
            # distribute to the 4 cells
            local_pen[br:br+box_size, bc:bc+box_size, :] += berr / (box_size*box_size)

    return local_pen


def project_clues(P):
    # Renormalize per cell and enforce clues
    for r in range(N):
        for c in range(N):
            if puzzle[r, c] != 0:
                val = puzzle[r, c] - 1
                P[r, c, :] = 0.0
                P[r, c, val] = 1.0
            else:
                s = P[r, c, :].sum()
                if s <= 1e-9:
                    P[r, c, :] = 1.0 / N
                else:
                    P[r, c, :] /= s
    return P


def initialize_probabilities():
    P_init = np.ones((N, N, N), dtype=float) / N
    # For empty puzzles, add small random perturbations to break symmetry
    if np.count_nonzero(puzzle) == 0:
        noise = np.random.uniform(0.1, 0.9, (N, N, N))
        P_init = P_init * 0.5 + noise * 0.5
        # Renormalize per cell
        for r in range(N):
            for c in range(N):
                P_init[r, c, :] /= P_init[r, c, :].sum()
    # Apply clue constraints
    for r in range(N):
        for c in range(N):
            if puzzle[r, c] != 0:
                val = puzzle[r, c] - 1
                P_init[r, c, :] = 0.0
                P_init[r, c, val] = 1.0
    return P_init


# Annealing parameters
T0 = 1.0
Tmin = 0.001
steps = 1000  # increased steps for better convergence
alpha = (Tmin / T0) ** (1.0 / steps)
eta = 0.3  # reduced step size for stability

# Visualization setup
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax_grid, ax_cost = axes
plt.tight_layout()

im_list = []
text_grid = [[None for _ in range(N)] for __ in range(N)]

# Heatmap for per-cell entropy (uncertainty): high entropy = uncertain

def cell_entropy(P):
    # entropy over digits at each cell
    eps = 1e-12
    ent = -(P * np.log(P + eps)).sum(axis=2)
    # normalize to [0,1]
    ent /= np.log(N)
    return ent

# Cost monitor
costs = []

# Better initialization
P = initialize_probabilities()

# Initialize plots
ent = cell_entropy(P)
im = ax_grid.imshow(ent, vmin=0, vmax=1, cmap='viridis')
ax_grid.set_title('Cell uncertainty (entropy)')
ax_grid.set_xticks(range(N))
ax_grid.set_yticks(range(N))
for r in range(N):
    for c in range(N):
        if text_grid[r][c] is None:
            text_grid[r][c] = ax_grid.text(c, r, '', ha='center', va='center', color='w', fontsize=14)

line_cost, = ax_cost.plot([], [], lw=2)
ax_cost.set_xlim(0, steps)
ax_cost.set_ylim(0, 30)
ax_cost.set_title('Energy (penalty) over steps')
ax_cost.set_xlabel('step')
ax_cost.set_ylabel('penalty')


def total_penalty(P):
    # scalar objective: sum of squared constraint residuals
    # cell
    cell_res = P.sum(axis=2) - 1.0
    # row
    row_res = P.sum(axis=1) - 1.0
    # col
    col_res = P.sum(axis=0) - 1.0
    # box
    box_size = 2
    box_res_acc = 0.0
    for br in range(0, N, box_size):
        for bc in range(0, N, box_size):
            bsum = P[br:br+box_size, bc:bc+box_size, :].sum(axis=(0,1)) - 1.0
            box_res_acc += (bsum**2).sum()
    return (cell_res**2).sum() + (row_res**2).sum() + (col_res**2).sum() + box_res_acc


def is_valid_solution(P):
    """Check if current P represents a valid sudoku solution"""
    # Get deterministic assignment
    assignment = P.argmax(axis=2)
    
    # Check cell constraints
    for r in range(N):
        for c in range(N):
            if P[r, c, assignment[r, c]] < 0.9:  # Not confident enough
                return False
    
    # Check row constraints
    for r in range(N):
        if len(set(assignment[r, :])) != N:
            return False
            
    # Check column constraints
    for c in range(N):
        if len(set(assignment[:, c])) != N:
            return False
            
    # Check box constraints
    box_size = 2
    for br in range(0, N, box_size):
        for bc in range(0, N, box_size):
            box_vals = assignment[br:br+box_size, bc:bc+box_size].flatten()
            if len(set(box_vals)) != N:
                return False
                
    return True


T = T0

def update(frame):
    global P, T
    # compute local penalties (acting like gradients)
    pen = compute_penalties(P)
    
    # Modified update rule with adaptive noise
    noise_scale = min(T, 0.1)  # Reduce noise as temperature decreases
    noise = np.random.normal(scale=noise_scale, size=P.shape)
    P = P - eta * pen + noise * 0.05  # Reduced noise contribution
    P = np.clip(P, 1e-6, None)
    P = project_clues(P)

    # compute diagnostics
    ent = cell_entropy(P)
    E = total_penalty(P)
    costs.append(E)

    # choose current MAP digits per cell for display
    guess = P.argmax(axis=2) + 1

    # update visuals
    im.set_data(ent)
    for r in range(N):
        for c in range(N):
            ch = str(guess[r, c]) if puzzle[r, c] == 0 else str(puzzle[r, c])
            # color fixed clues differently
            color = 'w' if puzzle[r, c] == 0 else 'cyan'
            text_grid[r][c].set_text(ch)
            text_grid[r][c].set_color(color)

    line_cost.set_data(range(len(costs)), costs)
    # auto-scale y as it descends
    ymin = 0
    ymax = max(5.0, np.max(costs[-50:]) * 1.2) if costs else 30
    ax_cost.set_ylim(ymin, ymax)

    # Check for early convergence
    if len(costs) > 10 and np.mean(costs[-10:]) < 0.01:
        print(f"Converged at step {frame}")
        # Stop animation early
        ani.event_source.stop()

    # cool down
    T *= alpha
    ax_grid.set_title(f'Cell uncertainty (T={T:.3f})')
    return [im, line_cost] + [t for row in text_grid for t in row]

ani = animation.FuncAnimation(fig, update, frames=steps, interval=100, blit=False, repeat=False)
plt.show()