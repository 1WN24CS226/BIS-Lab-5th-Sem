import numpy as np
import random
import math
import matplotlib.pyplot as plt

# WSN Optimization Parameters
NUM_SENSORS = 15      # Number of sensor nodes to deploy
SENSING_RANGE = 15.0  # Sensing radius (Rc) of each sensor in meters
FIELD_WIDTH = 100.0   # Width of the WSN deployment area (meters)
FIELD_HEIGHT = 100.0  # Height of the WSN deployment area (meters)

def calculate_coverage(nest):
    """
    Calculate the ratio of the field covered by the sensors.
    `nest` is a 1D array of size 2 * NUM_SENSORS: [x1, y1, x2, y2, ...]
    """
    # Reshape to (NUM_SENSORS, 2) to get (x, y) coordinates
    coords = nest.reshape((NUM_SENSORS, 2))

    # Use a grid-based estimation to calculate coverage area
    grid_size = 2.0  # Grid step size (smaller means more accurate but slower)
    x_grid = np.arange(0, FIELD_WIDTH, grid_size)
    y_grid = np.arange(0, FIELD_HEIGHT, grid_size)

    total_points = len(x_grid) * len(y_grid)
    covered_points = 0

    # Check which grid points are within SENSING_RANGE of at least one sensor
    for x in x_grid:
        for y in y_grid:
            for sx, sy in coords:
                if (x - sx)**2 + (y - sy)**2 <= SENSING_RANGE**2:
                    covered_points += 1
                    break

    coverage_rate = covered_points / total_points
    return coverage_rate

def fitness_function(nest):
    """
    Fitness function to maximize coverage rate.
    Forces penalties if sensors are placed outside the field boundaries.
    """
    coords = nest.reshape((NUM_SENSORS, 2))
    penalty = 0.0
    for sx, sy in coords:
        if sx < 0 or sx > FIELD_WIDTH or sy < 0 or sy > FIELD_HEIGHT:
            penalty += 1.0  # Penalize out-of-bounds deployment

    coverage_rate = calculate_coverage(nest)
    return coverage_rate - penalty

def levy_flight(Lambda=1.5):
    """
    Generate a step from a Levy distribution.
    """
    sigma_u = (math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) /
              (math.gamma((1 + Lambda) / 2) * Lambda * 2**((Lambda - 1) / 2)))**(1 / Lambda)
    u = np.random.normal(0, sigma_u)
    v = np.random.normal(0, 1)
    step = u / (abs(v)**(1 / Lambda))
    return step

def print_puzzle(state):
    """
    Plot the optimal WSN node deployment.
    """
    coords = state.reshape((NUM_SENSORS, 2))
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, FIELD_WIDTH)
    ax.set_ylim(0, FIELD_HEIGHT)

    # Plot each sensor and its coverage range
    for i, (sx, sy) in enumerate(coords):
        circle = plt.Circle((sx, sy), SENSING_RANGE, color='blue', alpha=0.15)
        ax.add_patch(circle)
        ax.plot(sx, sy, 'ro', label='Sensor Node' if i == 0 else "")

    plt.title(f"WSN Sensor Placement (Optimized Coverage)")
    plt.xlabel("X Coordinate (meters)")
    plt.ylabel("Y Coordinate (meters)")
    plt.grid(True)
    plt.legend()
    plt.show()

# Cuckoo Search Parameters
N = 15          # Number of host nests (candidate topologies)
Pa = 0.25       # Probability of discarding worst nests
max_iter = 30   # Iterations (keep modest for performance with grid-based evaluation)
dim = 2 * NUM_SENSORS  # Two parameters (x, y) per sensor

# 1. Initialize population of N host nests randomly within the deployment field
nests = np.random.uniform(0.0, max(FIELD_WIDTH, FIELD_HEIGHT), (N, dim))

# 2. Evaluate fitness of each nest
print("Evaluating initial topologies...")
fitness = np.array([fitness_function(nest) for nest in nests])

# 3. Find the best nest
best_index = np.argmax(fitness)
best_nest = nests[best_index].copy()
best_fitness = fitness[best_index]

# 4. REPEAT until maximum iterations are reached
for iteration in range(max_iter):
    print(f"Iteration {iteration + 1}/{max_iter} (Best Coverage: {best_fitness:.2%})")

    # a. Generate a new solution using Levy Flight
    for i in range(N):
        step_size = levy_flight() * 0.5 * (nests[i] - best_nest)
        new_solution = nests[i] + step_size * np.random.normal(0, 1, dim)

        # Keep within boundaries
        new_solution = np.clip(new_solution, 0.0, max(FIELD_WIDTH, FIELD_HEIGHT))

        # b. Evaluate the fitness of the new solution
        new_fitness = fitness_function(new_solution)

        # c. Select a random nest
        random_index = random.randint(0, N - 1)

        # d. IF new solution is better than selected nest, replace it
        if new_fitness > fitness[random_index]:
            nests[random_index] = new_solution
            fitness[random_index] = new_fitness

    # e. Abandon a fraction (Pa) of the worst nests
    sorted_indices = np.argsort(fitness)
    worst_count = int(N * Pa)

    # f. Generate new solutions to replace abandoned nests
    for i in range(worst_count):
        worst_index = sorted_indices[i]
        nests[worst_index] = np.random.uniform(0.0, max(FIELD_WIDTH, FIELD_HEIGHT), dim)
        # g. Evaluate fitness of new nests
        fitness[worst_index] = fitness_function(nests[worst_index])

    # h. Update the best solution
    current_best_index = np.argmax(fitness)
    if fitness[current_best_index] > best_fitness:
        best_fitness = fitness[current_best_index]
        best_nest = nests[current_best_index].copy()

# 5. Display the best topology and its coverage value
print(f"Optimal WSN Coverage Achieved: {best_fitness:.2%}")
print_puzzle(best_nest)