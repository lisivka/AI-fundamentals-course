import itertools
import math
"""
Задача маршрутизации транспортных средств (VRP) — это классическая задача оптимизации,
 которая включает в себя парк транспортных средств, которым поручено доставлять товары 
 или услуги группе клиентов из центрального склада. 
 У каждого покупателя есть спрос на определенное количество товаров, 
 а возможности транспортных средств для перевозки этих товаров ограничены. 
 Цель состоит в том, чтобы найти оптимальный набор маршрутов для транспортных
  средств такой, чтобы все клиенты посещались ровно один раз, 
  общая потребность каждого маршрута не превышала вместимость транспортного средства, 
  а общее время или расстояние в пути было минимизировано.

Ваша следующая задача — определить функцию optimize_vrp(), 
которая принимает следующие входные данные:

depot: координаты (x, y) депо, где все транспортные средства начинают и заканчивают свой маршрут.
customers: список кортежей, представляющих местоположения клиентов и их потребности, где каждый кортеж содержит (x, y, спрос).
vehicle_capacity: Максимальная вместимость каждого автомобиля.
num_vehicles: Количество автомобилей, имеющихся в автопарке.
Функция optimize_vrp()возвращает оптимизированные маршруты для транспортных средств, 
а также общее расстояние поездки.

Кроме того, вы можете определить функцию calculate_distance()и использовать ее 
для расчета расстояния между двумя местоположениями.

Примечание. Эта функция optimize_vrp()реализует метод грубой силы для решения 
проблемы выбора маршрута транспортных средств (VRP) и поиска оптимизированных 
маршрутов для парка транспортных средств, чтобы минимизировать расстояние поездки. 
Функция принимает в качестве входных данных местоположение депо, местоположение 
и потребности клиентов, ограничение вместимости транспортных средств и 
количество доступных транспортных средств и возвращает оптимизированные маршруты 
для транспортных средств вместе с общим расстоянием поездки. Он использует грубую силу 
для генерации всех возможных вариантов индексов клиентов и оценивает общее расстояние 
перемещения для каждого варианта, чтобы найти лучшее решение.
"""



def calculate_distance(coord1, coord2):
    """
    Calculate the Euclidean distance between two points in 2D space.

    Args:
        coord1 (tuple): The coordinates (x, y) of the first point.
        coord2 (tuple): The coordinates (x, y) of the second point.

    Returns:
        float: The Euclidean distance between the two points.
    """
    x1, y1 = coord1
    x2, y2 = coord2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def optimize_vrp(depot, customers, vehicle_capacity, num_vehicles):
    """
    Optimize the Vehicle Routing Problem to minimize total travel distance using Brute Force.

    Args:
        depot (tuple): The coordinates (x, y) of the depot, where the vehicles start and end their routes.
        customers (list of tuple): A list of tuples representing the coordinates (x, y) of each customer location.
        vehicle_capacity (int): The maximum capacity of each vehicle.
        num_vehicles (int): The number of vehicles available in the fleet.

    Returns:
        list: A list of routes, where each route represents the sequence of customer locations visited by a single vehicle.
    """

    # Add depot location to the list of customer locations
    all_locations = [depot] + customers

    # Generate all possible permutations of customer visits
    customer_permutations = list(itertools.permutations(customers))

    min_distance = float('inf')
    best_route = None

    for permutation in customer_permutations:
        routes = [[] for _ in range(num_vehicles)]
        route_capacity = [0] * num_vehicles
        current_location = [depot] * num_vehicles
        total_distance = 0

        for customer in permutation:
            best_vehicle = None
            best_distance = float('inf')

            for i in range(num_vehicles):
                if route_capacity[i] + customer[2] <= vehicle_capacity:
                    distance = calculate_distance(current_location[i], customer)
                    if distance < best_distance:
                        best_distance = distance
                        best_vehicle = i

            if best_vehicle is not None:
                routes[best_vehicle].append(customer)
                route_capacity[best_vehicle] += customer[2]
                current_location[best_vehicle] = customer
                total_distance += best_distance

        # Check if the total distance is the minimum so far
        if total_distance < min_distance:
            min_distance = total_distance
            best_route = routes

    return best_route





if __name__ == "__main__":

    # Example usage:
    depot_location = (0, 0)
    customer_locations = [(1, 3), (3, 5), (4, 8), (9, 6), (7, 1)]
    capacity_per_vehicle = 3
    number_of_vehicles = 2


    optimized_routes = optimize_vrp(depot_location, customer_locations, capacity_per_vehicle, number_of_vehicles)
    print(optimized_routes)
