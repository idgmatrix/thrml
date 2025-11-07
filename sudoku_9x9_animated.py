import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 9x9 sudoku setup (symbols 1..9, 3x3 boxes)
# 0 denotes empty cell
puzzle = np.array([
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
], dtype=int)

# Known correct solution for verification
solution = np.array([
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9]
], dtype=int)

N = 9

# Function to check if a solution is valid
def is_valid_solution(grid):
    # Check rows
    for r in range(N):
        if len(set(grid[r, :])) != N or 0 in grid[r, :]:
            return False
    
    # Check columns
    for c in range(N):
        if len(set(grid[:, c])) != N or 0 in grid[:, c]:
            return False
    
    # Check 3x3 boxes
    for br in range(0, N, 3):
        for bc in range(0, N, 3):
            box = grid[br:br+3, bc:bc+3]
            if len(set(box.flatten())) != N or 0 in box.flatten():
                return False
    
    # Check clues
    for r in range(N):
        for c in range(N):
            if puzzle[r, c] != 0 and grid[r, c] != puzzle[r, c]:
                return False
    
    return True

# Function to count constraint violations
def count_violations(grid):
    violations = 0
    
    # Row violations
    for r in range(N):
        row_vals = [grid[r, c] for c in range(N) if grid[r, c] != 0]
        violations += len(row_vals) - len(set(row_vals))
    
    # Column violations
    for c in range(N):
        col_vals = [grid[r, c] for r in range(N) if grid[r, c] != 0]
        violations += len(col_vals) - len(set(col_vals))
    
    # Box violations
    for br in range(0, N, 3):
        for bc in range(0, N, 3):
            box_vals = []
            for r in range(br, br+3):
                for c in range(bc, bc+3):
                    if grid[r, c] != 0:
                        box_vals.append(grid[r, c])
            violations += len(box_vals) - len(set(box_vals))
    
    return violations

# Function to initialize a valid grid
def initialize_grid():
    grid = puzzle.copy()
    
    # Fill each 3x3 box with valid values (1-9 without duplicates in box)
    for br in range(0, N, 3):
        for bc in range(0, N, 3):
            # Get numbers already in box
            box_values = set()
            empty_positions = []
            for r in range(br, br+3):
                for c in range(bc, bc+3):
                    if grid[r, c] != 0:
                        box_values.add(grid[r, c])
                    else:
                        empty_positions.append((r, c))
            
            # Get numbers needed
            needed = list(set(range(1, N+1)) - box_values)
            np.random.shuffle(needed)
            
            # Fill empty cells
            for i, (r, c) in enumerate(empty_positions):
                if i < len(needed):
                    grid[r, c] = needed[i]
    
    return grid

# Function to get all empty positions grouped by 3x3 boxes
def get_empty_positions_by_box():
    boxes = []
    for br in range(0, N, 3):
        for bc in range(0, N, 3):
            box_positions = []
            for r in range(br, br+3):
                for c in range(bc, bc+3):
                    if puzzle[r, c] == 0:
                        box_positions.append((r, c))
            boxes.append(box_positions)
    return boxes

# Visualization function to draw Sudoku grid
def draw_sudoku_grid(ax, grid, title="Sudoku"):
    ax.clear()
    ax.set_xlim(-0.5, N-0.5)
    ax.set_ylim(-0.5, N-0.5)
    ax.invert_yaxis()
    
    # Draw cells
    for r in range(N):
        for c in range(N):
            # Draw cell border
            rect = plt.Rectangle((c-0.5, r-0.5), 1, 1, fill=False, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            
            # Add number
            if grid[r, c] != 0:
                color = 'blue' if puzzle[r, c] != 0 else 'red'
                ax.text(c, r, str(grid[r, c]), ha='center', va='center', fontsize=16, color=color)
    
    # Draw bold lines for 3x3 boxes
    for i in range(0, N+1, 3):
        ax.axhline(y=i-0.5, color='black', linewidth=3)
        ax.axvline(x=i-0.5, color='black', linewidth=3)
    
    ax.set_title(title)
    ax.axis('off')

# Simulated annealing to solve Sudoku with visualization
def solve_sudoku_with_visualization():
    # Initialize grid
    current_grid = initialize_grid()
    current_violations = count_violations(current_grid)
    
    # Get all empty positions grouped by boxes
    boxes = get_empty_positions_by_box()
    
    print(f"Initial violations: {current_violations}")
    
    # Annealing parameters
    temperature = 5.0
    min_temperature = 0.0001
    cooling_rate = 0.9995
    max_iterations = 10000
    
    # Track best solution
    best_grid = current_grid.copy()
    best_violations = current_violations
    
    # Data for plotting
    iteration_data = []
    violations_data = []
    temperature_data = []
    best_violations_data = []
    
    # Create figure and axes for visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    plt.ion()  # Turn on interactive mode
    
    # Draw initial state
    draw_sudoku_grid(ax1, current_grid, "Initial Sudoku")
    
    # Setup for violations/temperature plot
    ax2_twin = ax2.twinx()
    line1, = ax2.plot([], [], 'b-', label='Violations')
    line2, = ax2_twin.plot([], [], 'r-', label='Temperature')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Violations', color='b')
    ax2_twin.set_ylabel('Temperature', color='r')
    ax2.set_title('Violations and Temperature')
    
    for iteration in range(max_iterations):
        # Check if we have a valid solution
        if current_violations == 0 and is_valid_solution(current_grid):
            print(f"Solution found at iteration {iteration}!")
            draw_sudoku_grid(ax1, current_grid, "Solution Found!")
            plt.draw()
            plt.pause(0.1)
            return current_grid, iteration
        
        # Choose a random box
        box_positions = [box for box in boxes if len(box) > 1]
        if not box_positions:
            continue
            
        selected_box = box_positions[np.random.randint(0, len(box_positions))]
        
        # Try swapping two random positions in the same box
        if len(selected_box) < 2:
            continue
            
        pos1, pos2 = np.random.choice(len(selected_box), 2, replace=False)
        r1, c1 = selected_box[pos1]
        r2, c2 = selected_box[pos2]
        
        # Perform swap
        new_grid = current_grid.copy()
        new_grid[r1, c1], new_grid[r2, c2] = new_grid[r2, c2], new_grid[r1, c1]
        
        # Check if swap maintains box validity (this is automatic with our approach)
        # But we need to check if it improves overall violations
        new_violations = count_violations(new_grid)
        
        # Decide whether to accept the new state
        if new_violations < current_violations:
            # Accept better solution
            current_grid = new_grid
            current_violations = new_violations
        else:
            # Accept worse solution with some probability
            delta = new_violations - current_violations
            if delta == 0:
                probability = 0.5
            else:
                probability = np.exp(-delta / temperature)
            if np.random.random() < probability:
                current_grid = new_grid
                current_violations = new_violations
        
        # Update best solution
        if current_violations < best_violations:
            best_grid = current_grid.copy()
            best_violations = current_violations
        
        # Cool down
        temperature = max(min_temperature, temperature * cooling_rate)
        
        # Update visualization every 100 iterations
        if iteration % 100 == 0:
            # Update Sudoku grid
            draw_sudoku_grid(ax1, current_grid, f"Sudoku (Iteration {iteration})")
            
            # Update data for plotting
            iteration_data.append(iteration)
            violations_data.append(current_violations)
            temperature_data.append(temperature)
            best_violations_data.append(best_violations)
            
            # Update violations/temperature plot
            line1.set_data(iteration_data, violations_data)
            line2.set_data(iteration_data, temperature_data)
            ax2.relim()
            ax2.autoscale_view()
            ax2_twin.relim()
            ax2_twin.autoscale_view()
            
            # Redraw
            plt.draw()
            plt.pause(0.01)
        
        # Print progress
        if iteration % 1000 == 0:
            print(f"Iteration {iteration}: Violations = {current_violations}, Temperature = {temperature:.6f}, Best = {best_violations}")
    
    print(f"Best solution found after {max_iterations} iterations with {best_violations} violations")
    draw_sudoku_grid(ax1, best_grid, "Best Solution Found")
    plt.draw()
    #plt.ioff()  # Turn off interactive mode
    return best_grid, max_iterations

# Solve the puzzle with visualization
print("Solving Sudoku puzzle with visualization...")
solution_grid, iterations = solve_sudoku_with_visualization()

print(f"\nSolution grid:")
print(solution_grid)
print(f"Valid solution: {is_valid_solution(solution_grid)}")

if is_valid_solution(solution_grid):
    print("SUCCESS: Valid Sudoku solution found!")
else:
    print("FAILED: Could not find valid solution")

plt.show()