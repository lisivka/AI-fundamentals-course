import numpy as np
import matplotlib.pyplot as plt

def total_distance(route):
    total_distance = 0
    for i in range(len(route) - 1):
        city1, city2 = route[i], route[i + 1]
        total_distance += np.linalg.norm(np.array(cities[city1]) - np.array(cities[city2]))
    total_distance += np.linalg.norm(np.array(cities[route[-1]]) - np.array(cities[route[0]]))
    return total_distance

def generate_initial_population(size, cities):
    population = []
    for _ in range(size):
        route = list(cities.keys())
        np.random.shuffle(route)
        population.append(route)
    return population

def selection(population, scores):
    p = scores / sum(scores)
    selected_indices = np.random.choice(len(population), size=len(population), p=scores / sum(scores))
    return [population[i] for i in selected_indices]

def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + [city for city in parent2 if city not in parent1[:crossover_point]]
    child2 = parent2[:crossover_point] + [city for city in parent1 if city not in parent2[:crossover_point]]
    return child1, child2

def mutate(route):
    if np.random.rand() < mutation_rate:
        idx1, idx2 = np.random.choice(len(route), size=2, replace=False)
        route[idx1], route[idx2] = route[idx2], route[idx1]
    return route

def visualize_route(cities, best_route):

    # Разделение координат на x и y
    x = [cities[city][0] for city in best_route]
    y = [cities[city][1] for city in best_route]

    # Визуализация точек
    plt.scatter(x, y, c='red', marker='o', label='Cities')

    # Визуализация маршрута
    for i in range(len(best_route) - 1):
        plt.plot([cities[best_route[i]][0], cities[best_route[i + 1]][0]],
                 [cities[best_route[i]][1], cities[best_route[i + 1]][1]], 'k-')

    # Связывание последней точки с первой
    plt.plot([cities[best_route[-1]][0], cities[best_route[0]][0]],
             [cities[best_route[-1]][1], cities[best_route[0]][1]], 'k-')

    # Подписи точек
    for city, (cx, cy) in cities.items():
        plt.annotate(city, (cx, cy), textcoords="offset points", xytext=(0, 5), ha='center')

    plt.title('Best Route Visualization')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.show()

if __name__ == "__main__":

    # Определение городов и их координат
    # cities = {'A': (0, 0), 'B': (1, 3), 'C': (3, 1), 'D': (5, 5), 'E': (7, 3)}
    cities = {'A': (0, 0), 'B': (2, 4), 'C': (5, 8), 'D': (9, 1), 'E': (3, 6), 'F': (3,5), 'G': (4,7)     }
    # Количество особей в популяции
    population_size = 100
    # Количество поколений
    generations = 100
    # Вероятность мутации
    mutation_rate = 0.1
    # Генерация начальной популяции
    population = generate_initial_population(population_size, cities)

    for generation in range(generations):
        # Расчет фитнес-функции для каждой особи в популяции
        scores = [1 / total_distance(route) for route in population]

        # Отбор особей для скрещивания
        selected_population = selection(population, scores)

        # Создание нового поколения особей
        new_population = []
        while len(new_population) < population_size:
            # parent1, parent2 = np.random.choice(selected_population, size=2, replace=False)
            parent1, parent2 = np.random.choice(len(selected_population), size=2, replace=False)
            parent1, parent2 = selected_population[parent1], selected_population[parent2]

            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.extend([child1, child2])

        population = new_population


    # Нахождение лучшего маршрута в последнем поколении
    best_route = max(population, key=lambda route: 1 / total_distance(route))

    print("Лучший маршрут:", best_route)
    print("Длина маршрута:", total_distance(best_route))

    visualize_route(cities, best_route)

