def print_path(grid, points=[(None, None), ]):
    """
    Print the grid with the path marked with asterisks.

    """
    for row in range(len(grid)):
        for col in range(len(grid[row])):
            if (row, col) in points:
                dot = f"*"
            else:
                dot = f"{grid[row][col]}" if grid[row][col] != " " else f"-"
            print(dot, end="  ")
        print()
    print()


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


# @test_dfs
def dfs(row, col, visited, solution_path, maze):
    """
    Depth-First Search (DFS) algorithm to explore the maze and find a path from
    the current cell to the exit point 'E'.

    Args:
    row (int): The current row coordinate.
    col (int): The current column coordinate.
    visited (list of list of bool): A 2D grid representing visited cells in the maze.
    solution_path (list of tuple): A list of coordinate tuples representing the solution path.

    Returns:
    bool: True if a valid path is found from the current cell to 'E', False otherwise.
    """
    # Get the dimensions of the maze
    num_rows = len(visited)
    num_cols = len(visited[0])

    # Check if the current cell is out of bounds or visited
    if (row < 0 or row >= num_rows or col < 0 or col >= num_cols or
            visited[row][col] or maze[row][col] == 'X'):
        return False

    # Mark the current cell as visited
    visited[row][col] = True

    # If 'E' is found, add the current cell to the solution path and return True
    if maze[row][col] == 'E':
        solution_path.append((row, col))
        return True

    # Explore the adjacent cells in a specific order (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dr, dc in directions:
        if dfs(row + dr, col + dc, visited, solution_path):
            solution_path.append(
                (row, col))  # Add the current cell to the solution path
            return True

    return False


# @test_solve_maze
def solve_maze(maze):
    """
    Solve a maze using Depth-First Search (DFS) algorithm.

    Args:
    maze (list of list of str): A 2D grid representing the maze.

    Returns:
    list of tuple: The solution path as a list of coordinate tuples [(row1, col1), (row2, col2), ...].
                   An empty list is returned if there is no valid path.
    """
    # Find the starting point

    start_row, start_col = -1, -1
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            if maze[i][j] == 'S':
                start_row, start_col = i, j
                break
        if start_row != -1:
            break

    if start_row == -1:
        return []  # No starting point 'S' found

    # Initialize the visited array and solution_path list
    num_rows = len(maze)
    num_cols = len(maze[0])
    visited = [[False] * num_cols for _ in range(num_rows)]
    solution_path = []

    # Call DFS to find the path
    if dfs(start_row, start_col, visited, solution_path):
        return solution_path[
               ::-1]  # Reverse the solution path to start from 'S'
    else:
        return []  # No valid path found


if __name__ == '__main__':
    # Example usage:
    maze = [
        ['S', ' ', 'X', 'X', 'E'],
        ['X', ' ', ' ', 'X', ' '],
        ['X', 'X', ' ', ' ', ' '],
        [' ', 'X', 'X', 'X', ' '],
        [' ', ' ', ' ', ' ', ' ']
    ]

    path = solve_maze(maze)
    print("Solution Path:", path)

    print_path(maze, path)

    maze = [
        ['S', ' ', ' ', ' ', 'X', ' '],
        ['X', 'X', 'X', ' ', 'X', ' '],
        [' ', ' ', ' ', ' ', ' ', ' '],
        ['X', 'X', ' ', 'X', 'X', 'X'],
        [' ', ' ', ' ', ' ', ' ', 'E']
    ]

    path = solve_maze(maze)
    print("Solution Path:", path)
    print_path(maze, path)
