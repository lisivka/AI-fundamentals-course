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



# @test_dfs
def dfs(row, col, visited, solution_path):
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

    num_rows = len(visited)
    num_cols = len(visited[0])

    if row < 0 or row >= num_rows or col < 0 or col >= num_cols or visited[
        row][col] or maze[row][col] == 'X':
        return False

    visited[row][col] = True


    if maze[row][col] == 'E':
        solution_path.append((row, col))
        return True


    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dr, dc in directions:
        if dfs(row + dr, col + dc, visited, solution_path):
            solution_path.append((row, col))  # Add the current cell to the solution path
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
    start = (-1, -1)
    end = (-1, -1) # Initialize the start and end coordinates

    for i in range(len(maze)):
        for j in range(len(maze[i])):
            if maze[i][j] == 'S':
                start = (i, j)
            if maze[i][j] == 'E':
                end = (i, j)
        if start != (-1, -1) and end != (-1, -1):
            break
    if start == (-1, -1) or end == (-1, -1) or start == end:
        return []

    # Initialize the visited array and solution_path list
    num_rows = len(maze)
    num_cols = len(maze[0])
    visited = [[False] * num_cols for _ in range(num_rows)]
    solution_path = []

    # Call DFS to find the path
    if dfs(start[0], start[1], visited, solution_path):
        return solution_path[::-1]  # Reverse the solution path to start from 'S'
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