
import random

# ==========================================
# 1. PROBLEM DATA
# ==========================================

exams = [
    "Mathematics",
    "DBMS",
    "Operating Systems",
    "Computer Networks",
    "Software Engineering",
    "Artificial Intelligence",
    "Machine Learning",
    "Python"
]

# Students enrolled in each examination
students = {
    "Mathematics": ["S1", "S2", "S3", "S4"],
    "DBMS": ["S1", "S2", "S5"],
    "Operating Systems": ["S2", "S3", "S6"],
    "Computer Networks": ["S1", "S4", "S7"],
    "Software Engineering": ["S3", "S5", "S8"],
    "Artificial Intelligence": ["S4", "S6", "S8"],
    "Machine Learning": ["S5", "S7", "S8"],
    "Python": ["S1", "S6", "S7"]
}

NUM_EXAMS = len(exams)
NUM_SLOTS = 5

# GA parameters
POPULATION_SIZE = 100
GENERATIONS = 300
MUTATION_RATE = 0.10
ELITE_SIZE = 2

random.seed(42)


# ==========================================
# 2. FIND CONFLICTING EXAM PAIRS
# ==========================================

conflict_pairs = []

for i in range(NUM_EXAMS):
    for j in range(i + 1, NUM_EXAMS):

        common_students = (
            set(students[exams[i]]) &
            set(students[exams[j]])
        )

        if common_students:
            conflict_pairs.append((i, j))


# ==========================================
# 3. INITIAL POPULATION
# ==========================================

def create_population():

    population = []

    for _ in range(POPULATION_SIZE):

        chromosome = [
            random.randint(0, NUM_SLOTS - 1)
            for _ in range(NUM_EXAMS)
        ]

        population.append(chromosome)

    return population


# ==========================================
# 4. FITNESS FUNCTION
# ==========================================

def fitness(chromosome):

    clashes = 0
    adjacent = 0
    spread = 0

    for i, j in conflict_pairs:

        difference = abs(
            chromosome[i] - chromosome[j]
        )

        # Hard constraint: no simultaneous exams
        if difference == 0:
            clashes += 1

        # Soft constraint: avoid consecutive exams
        elif difference == 1:
            adjacent += 1

        # Reward examination spacing
        spread += difference

    score = (
        -1000 * clashes
        - 10 * adjacent
        + spread
    )

    return score


# ==========================================
# 5. SELECTION
# Tournament Selection
# ==========================================

def selection(population):

    candidates = random.sample(population, 5)

    return max(candidates, key=fitness)


# ==========================================
# 6. CROSSOVER
# Single Point Crossover
# ==========================================

def crossover(parent1, parent2):

    point = random.randint(1, NUM_EXAMS - 1)

    child = (
        parent1[:point] +
        parent2[point:]
    )

    return child


# ==========================================
# 7. MUTATION
# Random Slot Assignment
# ==========================================

def mutation(chromosome):

    for i in range(NUM_EXAMS):

        if random.random() < MUTATION_RATE:

            chromosome[i] = random.randint(
                0, NUM_SLOTS - 1
            )

    return chromosome


# ==========================================
# 8. GENETIC ALGORITHM
# ==========================================

def genetic_algorithm():

    population = create_population()

    best_solution = None
    best_fitness = float("-inf")

    for generation in range(GENERATIONS):

        population.sort(
            key=fitness,
            reverse=True
        )

        current_best = population[0]

        if fitness(current_best) > best_fitness:

            best_solution = current_best.copy()
            best_fitness = fitness(best_solution)

        # Preserve best chromosomes
        new_population = [
            chromosome.copy()
            for chromosome in population[:ELITE_SIZE]
        ]

        # Create new generation
        while len(new_population) < POPULATION_SIZE:

            parent1 = selection(population)
            parent2 = selection(population)

            child = crossover(parent1, parent2)

            child = mutation(child)

            new_population.append(child)

        population = new_population

        if generation % 50 == 0:

            print(
                f"Generation {generation}: "
                f"Fitness = {best_fitness}"
            )

    return best_solution


# ==========================================
# 9. CHECK CONSTRAINTS
# ==========================================

def check_clashes(chromosome):

    clashes = 0

    for i, j in conflict_pairs:

        if chromosome[i] == chromosome[j]:
            clashes += 1

    return clashes


# ==========================================
# 10. DISPLAY FINAL TIMETABLE
# ==========================================

best = genetic_algorithm()

print("\n========== FINAL TIMETABLE ==========")

for slot in range(NUM_SLOTS):

    print(f"\nTIME SLOT {slot + 1}:")

    for i in range(NUM_EXAMS):

        if best[i] == slot:

            print("  -", exams[i])

print("\n========== PERFORMANCE ==========")

print("Fitness:", fitness(best))
print("Number of clashes:", check_clashes(best))

if check_clashes(best) == 0:
    print("Status: FEASIBLE TIMETABLE")
else:
    print("Status: CLASHES FOUND")

print("\nBest Chromosome:", best)
