from collections import deque


# @test_is_valid
def is_valid(cell, grid):
    """
    Check if a cell is a valid open cell in the grid.

    Args:
    cell (tuple): The cell coordinates (row, col).
    grid (list of list of str): A 2D grid of characters.

    Returns:
    bool: True if the cell is a valid open cell, False otherwise.
    """
    row, col = cell
    if 0 <= row < len(grid) and 0 <= col < len(grid[0]) and grid[row][
        col] != 'X':
        return True
    return False


# @test_get_neighbors
def get_neighbors(cell, grid):
    """
    Get neighboring cells that are valid open cells in the grid.

    Args:
    cell (tuple): The cell coordinates (row, col).
    grid (list of list of str): A 2D grid of characters.

    Returns:
    list of tuple: List of neighboring cell coordinates [(row1, col1), (row2, col2), ...].
    """

    row, col = cell
    neighbors = []
    directions = [(1, 0), (-1, 0),  (0, -1), (0, 1)]
    for dx, dy in directions:
        neighbor_row, neighbor_col = row + dx, col + dy
        if is_valid((neighbor_row,neighbor_col), grid):
            neighbors.append((neighbor_row, neighbor_col))


    return neighbors


# @test_find_shortest_path
def find_shortest_path(grid, start, target):
    """
    Find the shortest path from the starting point to the target point on a grid.

    This function uses a breadth-first search (BFS) algorithm to find the shortest path
    from the starting point to the target point on the grid. The grid is represented as
    a 2D list of characters, where 'S' is the starting point, 'E' is the target point,
    'X' are blocked cells, and ' ' (space) are open cells.

    Args:
    grid (list of list of str): A 2D grid of characters.
    start (tuple): The starting point coordinates (row, col).
    target (tuple): The target point coordinates (row, col).

    Returns:
    list of tuple: The shortest path as a list of coordinate tuples [(row1, col1), (row2, col2), ...].
                   An empty list is returned if there is no valid path.
    """
    if not is_valid(start, grid) or not is_valid(target, grid):
        return []

    visited = set([start])
    queue = deque([start])
    path = []

    while queue:
        current = queue.popleft()
        print(f"Visiting current: {current} ")
        neighbors = get_neighbors(current, grid)
        for neighbor in neighbors:
            if neighbor == target:
                return path + [neighbor]
            else:
                queue.append((neighbor, [current]))




    return path



if __name__ == "__main__":


    # Example usage:
    grid = [
        ['S', ' ', ' ', ' ', ' '],
        ['X', 'X', ' ', ' ', 'E'],
        [' ', ' ', 'X', ' ', ' '],
        ['X', 'X', ' ', 'X', ' '],
        [' ', ' ', ' ', ' ', ' ']
    ]
    grid = [
    ['S', ' ', 'X', 'X', 'E'],
    ['X', ' ', ' ', 'X', ' '],
    ['X', 'X', ' ', ' ', ' '],
    [' ', 'X', 'X', 'X', ' '],
    [' ', ' ', ' ', ' ', ' ']
    ]
    expected_result = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (2, 4),
                       (1, 4)]
    start_point = (0, 0)
    end_point = (0, 4)

    # cell = (0, 2)  # (row, col)
    # print(is_valid(cell, grid))
    # cell = (0, 2)  # (row, col)
    # print(get_neighbors(cell, grid))
    print(get_neighbors((0, 1), grid))

    # shortest_path = find_shortest_path(grid, start_point, end_point)
    # print("Shortest Path:", shortest_path)
