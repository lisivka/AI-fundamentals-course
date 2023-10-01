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
    result = False


    return result

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

    for row in range(len(maze)):
        for col in range(len(maze[0])):
            if maze[row][col] == 'S':
                start = (row, col)
            # elif maze[row][col] == 'E':
            #     end = (row, col)


    # Initialize the visited array and solution_path list

    solution_path = []
    num_cols = len(maze[0])
    num_rows = len(maze)
    visited = [ [False]*num_cols  for _ in  range(num_rows) ]
    path = dfs(start[0], start[1], visited, solution_path)

    return path


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
