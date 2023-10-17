import pulp

# @test_optimize_oim
def optimize_inventory_management(demand, holding_cost, ordering_cost,
                                  initial_inventory, reorder_point):
    # Create a Linear Programming problem
    model = pulp.LpProblem("Inventory_Management", pulp.LpMinimize)

    # Decision variables
    periods = range(len(demand))
    inventory = [pulp.LpVariable(f"inventory_{t}", lowBound=0) for t in
                        periods]
    order_quantity = [pulp.LpVariable(f"order_quantity_{t}", lowBound=0) for t
                      in periods]

    # Objective function: minimize total cost
    model += pulp.lpSum(
        holding_cost * inventory[t] + ordering_cost * order_quantity[t]
        for t in periods)
    # print(f"===============model: {model}")

    # Constraints
    for t in periods:
        # Inventory balance constraint
        if t == 0:
            model += (inventory[t] == initial_inventory +
                      order_quantity[t] -
                      demand[t])
        else:
            model += inventory[t] == (inventory[t - 1] +
                                             order_quantity[t] -
                                             demand[t])

        # Ensure enough is ordered to meet demand
        model += order_quantity[t] >= (demand[t] - inventory[t])
        model += order_quantity[t] >= 0  # Non-negativity constraint

        # Reorder point constraint
        model += inventory[t] <= reorder_point
        model += inventory[t] >= 0  # Non-negativity constraint

    # Solve the Linear Programming problem
    model.solve()

    # Extract the optimal solution
    result = [inventory[t].varValue for t in periods]

    # ==========================
    info = [(f"Period {t}: "
             f"Order quantity = {order_quantity[t].varValue},"
             f"Demand = {demand[t]}",
             f"Ending inventory = {inventory[t].varValue}, "
             ) for t in periods]
    print(f"initial_inventory_level = {initial_inventory}")
    print(*info, sep='\n')

    return result


# Example usage:
demand_forecast = [10, 20, 15, 25, 30]

holding_cost_per_period = 1.5
ordering_cost_per_order = 25.0
initial_inventory_level = 50
reorder_point_level = 50

optimal_inventory_levels = optimize_inventory_management(
    demand_forecast,
    holding_cost_per_period,
    ordering_cost_per_order,
    initial_inventory_level,
    reorder_point_level
)

print("Optimal Inventory Levels:", optimal_inventory_levels)
