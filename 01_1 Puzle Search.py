def print_word(grid, points=[(None, None), ]):
    """
    Print a grid of letters.
    """

    for row in range(len(grid)):
        for col in range(len(grid[row])):
            if (row, col) not in points and points != [(None, None),]:
                dot = f"*"
            else:
                dot = f"{grid[row][col]}"

            print(dot, end=" ")
        print()
    print()


# @test_search_word
def search_word(grid, word, view=False):
    """
    Search for a word within a grid of letters.

    This function searches for the given word in the grid by checking all possible
    directions (horizontal, vertical, and diagonal) starting from each cell.

    Args:
    grid (list of list of str): A 2D grid of letters.
    word (str): The word to search for.

    Returns:
    bool: True if the word is found in the grid, False otherwise.
    """
    # Define possible directions (horizontal, vertical, diagonal)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1),
                  (1, 1)]

    def check_cell(x, y, dx, dy):
        points = []
        for char in word:
            if not (0 <= x < len(grid) and 0 <= y < len(grid[0])) or grid[x][y] != char:
                return False, points
            points.append((x, y))
            x, y = x + dx, y + dy
            # print(f"char: {char}, x: {x}, y: {y}")
        return True, points

    # Iterate through each cell in the grid and check if the word can be found
    result = False
    count = 0
    first_letter = word[0]
    if view == True:
        print(f"Original Grid: find word: {word}")
        print_word(grid)

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == first_letter:

                for dx, dy in directions:
                    check, points = check_cell(i, j, dx, dy)
                    if check:
                        result = True
                        count += 1
                        if view == True:
                            print(f"№{count} Word: {word}, points: {points}")
                            print_word(grid, points=points)

    return result


# @test_find_words
def find_words(grid, words, view=False):
    """
    Find words within a grid of letters.

    This function searches for words in the given grid by calling the search_word function
    for each word to be found.

    Args:
    grid (list of list of str): A 2D grid of letters.
    words (list of str): A list of words to find in the grid.

    Returns:
    list of str: A list of words found in the grid.
    """
    result = []
    for word in words:
        if search_word(grid, word, view=view):
            result.append(word)

    return result


# Example usage:
grid = [['A', 'B', 'C', 'D'], ['E', 'F', 'G', 'O'], ['I', 'J', 'K', 'G'], ['M', 'N', 'H', 'P']]
word_list = ["HELLO", "WORLD", "HI", "FOOD", "DOG", "GOD"]
found_words = find_words(grid, word_list, view=True)
print("Found words:", found_words)

print("----------------------")
print("Visualization by search")
words = ["ROCK", "IN", "OF", "OOO", ]
grid = [['R', 'B', 'R', 'R'],
        ['E', 'O', 'O', 'O'],
        ['I', 'C', 'C', 'C'],
        ['K', 'N', 'O', 'K']]
print(find_words(grid, words, view=True))
