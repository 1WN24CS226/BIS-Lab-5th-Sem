import numpy as np
import matplotlib.pyplot as plt
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

num_tasks = 20
num_machines = 5

# Generate task workloads (e.g., millions of instructions or arbitrary units)
task_workloads = np.random.uniform(10, 100, size=num_tasks)

# Generate machine processing speeds (units/sec)
machine_speeds = np.random.uniform(5, 15, size=num_machines)

# Calculate the processing time matrix T where T[i][j] is the time task i takes on machine j
T = np.zeros((num_tasks, num_machines))
for i in range(num_tasks):
    for j in range(num_machines):
        T[i][j] = task_workloads[i] / machine_speeds[j]

print("Task workloads:", np.round(task_workloads, 1))
print("Machine speeds:", np.round(machine_speeds, 1))

class ResourceSchedulingACO:
    def __init__(self, time_matrix, num_ants=20, num_iterations=100, alpha=1.0, beta=2.0, decay=0.2, Q=10.0):
        self.T = time_matrix  # Time matrix T[i][j]
        self.num_tasks, self.num_machines = time_matrix.shape
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.decay = decay
        self.Q = Q
        
        # Initialize pheromones: pheromones[i][j] represents the pheromone level of assigning task i to machine j
        self.pheromones = np.ones((self.num_tasks, self.num_machines))
        # Heuristic value: inverse of execution time
        self.eta = 1.0 / (self.T + 1e-10)

    def run(self):
        best_schedule = None
        best_makespan = float('inf')
        history = []

        for iteration in range(self.num_iterations):
            all_schedules = []
            all_makespans = []

            for ant in range(self.num_ants):
                # Construct solution: map each task to a machine
                schedule = self._construct_schedule()
                makespan = self._evaluate_schedule(schedule)
                
                all_schedules.append(schedule)
                all_makespans.append(makespan)

                # Keep track of global best
                if makespan < best_makespan:
                    best_makespan = makespan
                    best_schedule = schedule
            
            history.append(best_makespan)
            
            # Evaporate pheromones
            self.pheromones *= (1.0 - self.decay)
            
            # Deposit pheromones based on quality of schedules
            self._deposit_pheromones(all_schedules, all_makespans)

        return best_schedule, best_makespan, history

    def _construct_schedule(self):
        schedule = np.zeros(self.num_tasks, dtype=int)
        for i in range(self.num_tasks):
            # Calculate selection probabilities for task i across all machines
            tau = self.pheromones[i] ** self.alpha
            eta_val = self.eta[i] ** self.beta
            probabilities = tau * eta_val
            
            prob_sum = np.sum(probabilities)
            if prob_sum == 0:
                probabilities = np.ones(self.num_machines) / self.num_machines
            else:
                probabilities /= prob_sum
            
            # Choose machine for task i
            selected_machine = np.random.choice(self.num_machines, p=probabilities)
            schedule[i] = selected_machine
        return schedule

    def _evaluate_schedule(self, schedule):
        # Calculate time spent on each machine
        machine_times = np.zeros(self.num_machines)
        for task_idx, machine_idx in enumerate(schedule):
            machine_times[machine_idx] += self.T[task_idx][machine_idx]
        # Makespan is the maximum time taken by any machine
        return np.max(machine_times)

    def _deposit_pheromones(self, all_schedules, all_makespans):
        for schedule, makespan in zip(all_schedules, all_makespans):
            pheromone_to_add = self.Q / makespan
            for task_idx, machine_idx in enumerate(schedule):
                self.pheromones[task_idx][machine_idx] += pheromone_to_add

# Run the ACO Scheduler
scheduler = ResourceSchedulingACO(T, num_ants=30, num_iterations=120, alpha=1.0, beta=2.0, decay=0.1, Q=50.0)
best_schedule, best_makespan, history = scheduler.run()

print(f"Best Makespan obtained: {best_makespan:.2f} seconds\n")
for m in range(num_machines):
    assigned_tasks = np.where(best_schedule == m)[0]
    total_time_on_m = sum(T[task][m] for task in assigned_tasks)
    print(f"Machine {m} (Speed={machine_speeds[m]:.1f}): Assigned tasks {assigned_tasks} | Total Time: {total_time_on_m:.2f}s")

# Visualization: Plot workload distribution on machines and learning curve
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 1. Gantt chart / Workload bar plot
machine_bars = [[] for _ in range(num_machines)]
machine_times = np.zeros(num_machines)

for task_idx, machine_idx in enumerate(best_schedule):
    time_needed = T[task_idx][machine_idx]
    machine_bars[machine_idx].append((task_idx, time_needed))

for m in range(num_machines):
    start_time = 0
    for task_idx, duration in machine_bars[m]:
        axes[0].barh(f"Machine {m}", duration, left=start_time, edgecolor='black', alpha=0.8, label=f"Task {task_idx}" if num_tasks <= 10 else "")
        axes[0].text(start_time + duration/2, m, f"T{task_idx}", ha='center', va='center', color='white', fontweight='bold', fontsize=8)
        start_time += duration

axes[0].axvline(best_makespan, color='red', linestyle='--', label=f'Makespan Limit ({best_makespan:.2f}s)')
axes[0].set_xlabel('Execution Time (seconds)')
axes[0].set_ylabel('Resource Allocations')
axes[0].set_title('Resource Scheduling Assignment (Best Schedule)')
axes[0].grid(axis='x', linestyle='--', alpha=0.5)

# 2. Convergence plot
axes[1].plot(history, color='purple', linewidth=2)
axes[1].set_title('ACO Convergence Curve')
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Best Makespan')
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()