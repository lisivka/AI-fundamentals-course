from collections import deque


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
    if 0 <= row < len(grid) and 0 <= col < len(grid[0]) and grid[row][col] != 'X':
        return True
    return False

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
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for dr, dc in directions:
        neighbor_row, neighbor_col = row + dr, col + dc
        if is_valid((neighbor_row, neighbor_col), grid):
            neighbors.append((neighbor_row, neighbor_col))
    return neighbors

def find_shortest_path(grid, start, target):
    """
    Find the shortest path from the starting point to the target point on a grid.

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

    visited = set()
    queue = deque([(start, [])])

    while queue:
        current, path = queue.popleft()
        if current == target:
            return path + [current]
        if current not in visited:
            visited.add(current)
            for neighbor in get_neighbors(current, grid):
                queue.append((neighbor, path + [current]))

    return []

# Example usage:
grid = [
    ['S', ' ', ' ', ' ', ' '],
    ['X', 'X', ' ', ' ', 'E'],
    [' ', ' ', 'X', ' ', ' '],
    ['X', 'X', ' ', 'X', ' '],
    [' ', ' ', ' ', ' ', ' ']
]

start_point = (0, 0)
end_point = (1, 4)

shortest_path = find_shortest_path(grid, start_point, end_point)
print("Shortest Path:", shortest_path)
