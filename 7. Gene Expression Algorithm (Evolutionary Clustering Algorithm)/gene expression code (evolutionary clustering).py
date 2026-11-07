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
