import numpy as np
import random
import matplotlib.pyplot as plt
import math

# พิกัดของเมือง (x, y)
cities = [
    (60, 200), (180, 200), (80, 180), (140, 180), (20, 160),
    (100, 160), (200, 160), (140, 140), (40, 120), (100, 120),
    (180, 100), (60, 80), (120, 80), (180, 60), (20, 40),
    (100, 40), (200, 40), (20, 20), (60, 20), (160, 20)
]
num_cities = len(cities)

# คำนวณระยะทางระหว่างเมือง
def calculate_distance(city1, city2):
    return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

# คำนวณระยะทางทั้งหมดของเส้นทาง
def calculate_total_distance(route):
    distance = 0
    for i in range(len(route)):
        city1 = cities[route[i]]
        city2 = cities[route[(i + 1) % len(route)]]
        distance += calculate_distance(city1, city2)
    return distance

# สร้างประชากรเริ่มต้น
def create_initial_population(pop_size):
    population = []
    for _ in range(pop_size):
        individual = list(range(num_cities))
        random.shuffle(individual)
        population.append(individual)
    return population

# คำนวณความเหมาะสม (fitness คือ 1/ระยะทาง เพราะเราต้องการระยะทางน้อยที่สุด)
def calculate_fitness(population):
    fitness_values = []
    for individual in population:
        distance = calculate_total_distance(individual)
        fitness = 1 / distance
        fitness_values.append(fitness)
    return fitness_values

# การคัดเลือก
def selection(population, fitness_values):
    sorted_population = [x for _, x in sorted(zip(fitness_values, population), reverse=True)]
    return sorted_population[:2]  # เลือก 2 อันดับแรก

# การผสมพันธุ์แบบ Order Crossover (OX)
def order_crossover(parent1, parent2):
    start, end = sorted(random.sample(range(num_cities), 2))
    child = [None] * num_cities

    # คัดลอกช่วงหนึ่งจากพ่อแม่ตัวที่ 1
    child[start:end+1] = parent1[start:end+1]

    # เติมเมืองจากพ่อแม่ตัวที่ 2
    pointer = 0
    for city in parent2:
        if city not in child:
            while child[pointer] is not None:
                pointer += 1
            child[pointer] = city
    return child

# การกลายพันธุ์แบบ Swap Mutation
def swap_mutation(chromosome, mutation_rate):
    for i in range(num_cities):
        if random.random() < mutation_rate:
            j = random.randint(0, num_cities - 1)
            chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
    return chromosome

# อัลกอริทึมพันธุกรรม
def genetic_algorithm(pop_size=100, max_generations=500, crossover_rate=0.8, mutation_rate=0.2):
    population = create_initial_population(pop_size)
    best_distances = []
    best_route = None
    best_distance = float('inf')

    for generation in range(max_generations):
        fitness_values = calculate_fitness(population)
        new_population = []

        # บันทึกเส้นทางที่ดีที่สุด
        current_best = population[np.argmax(fitness_values)]
        current_best_distance = calculate_total_distance(current_best)
        if current_best_distance < best_distance:
            best_distance = current_best_distance
            best_route = current_best
        best_distances.append(best_distance)

        while len(new_population) < pop_size:
            parents = selection(population, fitness_values)
            if random.random() < crossover_rate:
                child = order_crossover(parents[0], parents[1])
            else:
                child = parents[0][:]
            child = swap_mutation(child, mutation_rate)
            new_population.append(child)

        population = new_population

    return best_route, best_distance, best_distances

# แสดงเส้นทางบนกราฟ
def plot_route(route, title="Best Route"):
    x = [cities[i][0] for i in route]
    y = [cities[i][1] for i in route]
    x.append(x[0])
    y.append(y[0])
    
    plt.figure(figsize=(10, 10))
    plt.plot(x, y, 'o-', markersize=10)
    for i, city in enumerate(route):
        plt.annotate(str(city), (cities[city][0], cities[city][1]), 
                    xytext=(5, 5), textcoords='offset points')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()

# ทดสอบอัลกอริทึม
if __name__ == "__main__":
    best_route, best_distance, all_best_distances = genetic_algorithm()
    print(f"เส้นทางที่ดีที่สุด: {best_route}")
    print(f"ระยะทางรวม: {best_distance:.2f}")

    plt.figure(figsize=(10, 6))
    plt.plot(all_best_distances)
    plt.title('Convergence')
    plt.xlabel('Generation')
    plt.ylabel('Best Distance')
    plt.grid(True)
    plt.show()

    plot_route(best_route, f"Best Route (Distance: {best_distance:.2f})")
