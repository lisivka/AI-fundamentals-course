# https://levelup.gitconnected.com/mastering-tic-tac-toe-with-minimax-algorithm-3394d65fa88f
# https://gist.github.com/redwrasse/1d7b0980d5dd5ded1a59ae633c5cbc20
# https://habr.com/ru/articles/329058/

PLAYER_X = "X"
PLAYER_O = "O"
EMPTY = " "
SIZE = 3
COUNTER = [i for i in range(SIZE)]


def print_board(board):
    print(" ", *COUNTER, sep=" ")
    print(" ", "-" * (SIZE + SIZE - 1))
    count = 0
    for row in board:
        print(count, "|".join(row))
        print(" ", "-" * (SIZE + SIZE - 1))
        count += 1


def get_player_move():
    flag = True
    row, col = None, None

    try:
        row = int(input(f"Enter the row {COUNTER}: "))
        col = int(input(f"Enter the column {COUNTER}: "))
    except ValueError as e:
        print("Incorrect value. Try again.")
        print("Possible values are 0, 1, 2.")
        # continue
        flag = False

    except Exception as e:
        print(f"Error.{e}")
        # continue
        flag = False

    return flag, row, col


def check_player_move(row, col, board):
    if row not in range(SIZE) or col not in range(SIZE):
        print(" Invalid row or column. Try again.")
        print(f"Possible values are {COUNTER}")
        return False

    if board[row][col] != EMPTY:
        print("Cell is not empty. Try again.")
        return False

    return True


def check_win_draw(board):
    if evaluate(board) == -1:
        print("You win!")
        return True
    if evaluate(board) == 1:
        print("Computer wins!")
        return True
    if is_full(board):
        print("It's a draw!")
        return True
    return False


def play():
    board = [[EMPTY for _ in range(SIZE)] for _ in range(SIZE)]
    print("Lets play!")
    print_board(board)
    while True:

        # get player move
        correct, row, col = get_player_move()
        if correct == False:
            continue
        if check_player_move(row, col, board) == False:
            # print_board(board)
            continue
        board[row][col] = PLAYER_O
        print("Player move:")
        print_board(board)
        if check_win_draw(board):
            break

        # get computer move
        move = get_best_move(board)
        if move is not None:
            row, col = get_best_move(board)
            board[row][col] = PLAYER_X
            print("Computer move:")
        else:
            print("No move left")

        print_board(board)
        if check_win_draw(board):
            break


def is_full(board):
    for row in board:
        if EMPTY in row:
            return False
    return True


def get_empty_cells(board):
    empty_cells = []
    for row in range(3):
        for col in range(3):
            if board[row][col] == EMPTY:
                empty_cells.append((row, col))
    return empty_cells


# ===============================================
def evaluate(board):
    '''This function evaluates the current state of the board and determines the outcome of the game.

    Parameters:
        board (list): A 3x3 array representing the board state with symbols.

    Returns:
        int:
            -1 if PLAYER_O has won.
            1 if PLAYER_X has won.
            0 if the game is a draw or still ongoing.
            '''

    # Check rows
    for row in board:
        if set(row) == {PLAYER_X}:
            return 1
        if set(row) == {PLAYER_O}:
            return -1
    # Check columns by transposing the board
    transposed_board = list(zip(*board))
    for row in transposed_board:
        if set(row) == {PLAYER_X}:
            return 1
        if set(row) == {PLAYER_O}:
            return -1

    # Check diagonals
    main_diagonal = {board[i][i] for i in range(SIZE)}
    second_diagonal = {board[i][2 - i] for i in range(SIZE)}
    if main_diagonal == {PLAYER_X} or second_diagonal == {PLAYER_X}:
        return 1
    if main_diagonal == {PLAYER_O} or second_diagonal == {PLAYER_O}:
        return -1

    return 0  # the game is a draw or still ongoing


def game_over(board):
    """Returns True if the game is over, False otherwise."""
    if evaluate(board) != 0 or is_full(board):
        return True

    return False


def minimax(board, depth, alpha, beta, is_maximizing):
    '''Implements the Minimax algorithm with Alpha-Beta pruning to determine the optimal score for a given board state.

    Parameters:
        board (list): A 3x3 array representing the current board state with symbols.
        depth (int): The current depth of the recursion.
        alpha (int): The best score found so far for the maximizing player.
        beta (int): The best score found so far for the minimizing player.
        is_maximizing (bool): True if the current player is the maximizing player (Player X), False if the current player is the minimizing player (Player O).

    Returns:
        int:
            -1 if PLAYER_O has won.
            1 if PLAYER_X has won.
            0 if the game is a draw or still ongoing.
            '''
    # Function code here

    if game_over(board) or depth == 0:
        return evaluate(board)

    if is_maximizing:
        best_score = -float('inf')
        for row, cow in available_moves(board):
            board[row][cow] = PLAYER_X
            score = minimax(board, depth - 1, alpha, beta, False)
            board[row][cow] = EMPTY
            best_score = max(score, best_score)
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break

        return best_score
    else:
        best_score = float('inf')
        for row, cow in available_moves(board):
            board[row][cow] = PLAYER_O
            score = minimax(board, depth - 1, alpha, beta, True)
            board[row][cow] = EMPTY
            best_score = min(score, best_score)
            beta = min(beta, best_score)
            if beta <= alpha:
                break

        return best_score


def available_moves(board):
    """Returns a list of available moves for the current board state."""

    available = []
    for row in range(len(board)):
        for col in range(len(board[row])):
            if board[row][col] == EMPTY:
                available.append((row, col))
    # print(f"available: {available}")
    return available


def get_best_move(board):
    '''Returns the best move for the current board state.

    Parameters:
        board (list): A 3x3 array representing the board state with symbols.

    Returns:
        tuple: A tuple of the row and column for the best move.
        '''
    best_score = -float('inf')
    best_move = None
    for row, col in available_moves(board):
        board[row][col] = PLAYER_X
        score = minimax(board, 3, -float("inf"), float("inf"), False)  # Глубина пошуку
        board[row][col] = EMPTY
        if score > best_score:
            best_score = score
            best_move = row, col
    return best_move


if __name__ == "__main__":
    # test_evaluate(evaluate)
    # test_minimax(minimax)
    # test_best_move(get_best_move)

    # Start the game
    play()
