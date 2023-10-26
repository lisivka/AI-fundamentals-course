import pulp


# @test_optimize_oim
def optimize_inventory_management(demand, holding_cost, ordering_cost,
                                  initial_inventory, reorder_point):
    # Create a Linear Programming problem
    model = pulp.LpProblem("Inventory_Management", pulp.LpMinimize)

    # Decision variables
    periods = len(demand)+1 #add 0 period
    demand = [0] + demand

    inventory = [pulp.LpVariable(f"inventory_{t}", lowBound=0) for t in
                 range(periods)]
    order_quantity = [pulp.LpVariable(f"order_quantity_{t}", lowBound=0) for t
                      in range(periods)]

    # Objective function: minimize total cost
    model += pulp.lpSum(
        holding_cost * inventory[t] + ordering_cost * order_quantity[t]
        for t in range(1,periods))

    # Constraints
    model += inventory[0] == initial_inventory

    for t in range(1, periods):

        model += inventory[t] == (
                inventory[t - 1]+
                                  order_quantity[t] -
                                  demand[t]
        )
        model += order_quantity[t] >= (demand[t] -
                                           inventory[t - 1])

        model += order_quantity[t] >= 0  # Non-negativity constraint
        model += inventory[t] >= 0  # Non-negativity constraint

        # Reorder point constraint
        model += inventory[t] <= reorder_point

    # Solve the Linear Programming problem
    model.solve()

    # Extract the optimal solution
    result = [inventory[t].varValue for t in range(periods)]

    # ==========================
    info = [(f"Period {t}: "
             f"Starting inventory = {inventory[t-1].varValue}, "
             f"Order quantity = {order_quantity[t].varValue},"
             f"Demand = {demand[t]}",
             f"Ending inventory = {inventory[t].varValue}, "

             f"Total cost = {holding_cost * inventory[t].varValue +  ordering_cost * order_quantity[t].varValue}"
             ) for t in range(1,periods)]

    print(*info, sep='\n')
    print(f"Model TOTAL cost = {pulp.value(model.objective)}")

    return result[1:] #remove 0 period


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
