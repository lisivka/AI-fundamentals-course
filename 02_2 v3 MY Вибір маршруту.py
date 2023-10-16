import itertools
import math


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

def find_best_vehicle(current_depot, customer_location,
                      remaining_capacity, num_vehicles):
    min_distance = float('inf')
    best_vehicle = -1

    for vehicle in range(num_vehicles):
        if remaining_capacity[vehicle] >= 1:
            distance = calculate_distance(current_depot, customer_location)

            if distance < min_distance:
                min_distance = distance
                best_vehicle = vehicle
    return min_distance, best_vehicle

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
    # Generate all possible permutations of customer visits
    num_customers = len(customers)
    all_permutations = list(itertools.permutations(range(num_customers)))

    # Initialize variables to keep track of the best solution
    best_distance = float('inf')
    best_routes = None

    for permutation in all_permutations:
        routes = [[] for _ in range(num_vehicles)]
        remaining_capacity = [vehicle_capacity] * num_vehicles
        total_distance = 0.0
        current_depot = depot

        for customer_indx in permutation:
            customer_location = customers[customer_indx]
            min_distance, best_vehicle = find_best_vehicle(current_depot,
                                                           customer_location,
                                                           remaining_capacity,
                                                           num_vehicles)

            if best_vehicle != -1:
                routes[best_vehicle].append(customer_location)
                total_distance += min_distance
                remaining_capacity[best_vehicle] -= 1
                current_depot = customer_location

        # Add the return trip to the depot for each vehicle
        for vehicle in range(num_vehicles):
            if routes[vehicle]:
                total_distance += calculate_distance(routes[vehicle][-1],
                                                     depot)
                routes[vehicle].append(depot)

        if total_distance < best_distance:
            best_distance = total_distance
            best_routes = routes

    return best_routes




def print_route(routes, depot, customer_locations):
    all_locations = [depot]+customer_locations
    num_row = 1 + max(location[0] for location in all_locations)
    num_col = 1 + max(location[1] for location in all_locations)
    matrix = [['_' for _ in range(num_col)] for _ in range(num_row)]
    depot_x, depot_y =depot[0], depot[1]
    for row,col in customer_locations:
        matrix[row][col] = 'M'

    matrix[depot_x][depot_y] = 'D'
    print(f"Matrix: {num_row}x{num_col} with customers marked as 'M' and depot as 'D'")
    print(*matrix, sep='\n')

    mark = "R"
    for route in routes:
        count = 1
        copy_matrix = [row.copy() for row in matrix]
        print('\n',f"Route: {route}")
        for row,col in route:
            copy_matrix[row][col] = mark + str(count)
            count += 1
        copy_matrix[depot_x][depot_y] = mark + str(0)
        print(*copy_matrix, sep='\n' )




# Example usage:
depot_location = (0, 0)
customer_locations = [(1, 3), (3, 5), (4, 8), (9, 6), (7, 1)]
# customer_locations = [(2, 1),(2, 0),(4, 0),(1, 3), (1, 5),
#                       (3, 5), (4, 8), (9, 6),(7, 1), ]
capacity_per_vehicle = 3
number_of_vehicles = 2

optimized_routes = optimize_vrp(depot_location, customer_locations,
                                capacity_per_vehicle, number_of_vehicles)
print("optimized_routes: ", *optimized_routes, sep='\n')

print_route(optimized_routes, depot_location, customer_locations)
