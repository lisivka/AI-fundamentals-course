# https://levelup.gitconnected.com/mastering-tic-tac-toe-with-minimax-algorithm-3394d65fa88f


PLAYER_X = "X"
PLAYER_O = "O"
EMPTY = " "




def print_board(board):
    # print(*board, sep="\n")
    # print()
    print("-" * 5)
    for row in board:
        print("|".join(row))
        print("-" * 5)


def play():
    board = [[EMPTY for _ in range(3)] for _ in range(3)]
    print("Lets play!")
    print_board(board)

    while True:
        try:
            row = int(input("Enter the row (0-2): "))
            col = int(input("Enter the column (0-2): "))
        except ValueError as e:
            print("Incorrect value. Try again.")
            print("Possible values are 0, 1, 2.")
            continue
        except Exception:
            print(f"Error.{e}")
            continue

        if row not in range(3) or col not in range(3):
            print(" Invalid row or column. Try again.")
            print("Possible values are 0, 1, 2.")
            continue

        if board[row][col] != EMPTY:
            print("Cell is not empty. Try again.")
            continue

        board[row][col] = PLAYER_O


        if evaluate(board) == -1:
            print_board(board)
            print("You win!")
            break

        if is_full(board):
            print_board(board)
            print("It's a draw!")
            break

        best_move = get_best_move(board)
        board[best_move[0]][best_move[1]] = PLAYER_X

        if evaluate(board) == 1:
            print_board(board)
            print("Computer wins!")
            break

        if is_full(board):
            print("It's a draw!")
            break


        print_board(board)


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
        # Function code here
        pass

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
        # Function code here
        pass



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

    # Check rows and columns
    for row in board:
        if row.count(PLAYER_X) == 3:
            return 1
        if row.count(PLAYER_O) == 3:
            return -1
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] == PLAYER_X:
            return 1
        if board[0][col] == board[1][col] == board[2][col] == PLAYER_O:
            return -1

    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] == PLAYER_X:
        return 1
    if board[0][0] == board[1][1] == board[2][2] == PLAYER_O:
        return -1

    if board[0][2] == board[1][1] == board[2][0] == PLAYER_X:
        return 1
    if board[0][2] == board[1][1] == board[2][0] == PLAYER_O:
        return -1

    return 0 #the game is a draw or still ongoing



# ===============================================
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
    if evaluate(board) == 1 or evaluate(board) == -1:
        return evaluate(board)

    if is_full(board):
        return 0

    if is_maximizing:
        best_score = -float("inf")
        for row in range(3):
            for col in range(3):
                if board[row][col] == EMPTY:
                    board[row][col] = PLAYER_X
                    score = minimax(board, depth + 1, alpha, beta, False)
                    board[row][col] = EMPTY
                    best_score = max(score, best_score)
                    alpha = max(alpha, best_score)
                    if beta <= alpha:
                        break
        return best_score

    else:
        best_score = float("inf")
        for row in range(3):
            for col in range(3):
                if board[row][col] == EMPTY:
                    board[row][col] = PLAYER_O
                    score = minimax(board, depth + 1, alpha, beta, True)
                    board[row][col] = EMPTY
                    best_score = min(score, best_score)
                    beta = min(beta, best_score)
                    if beta <= alpha:
                        break
        return best_score





def get_best_move(board):
    '''Returns the best move for the current board state.

    Parameters:
        board (list): A 3x3 array representing the board state with symbols.

    Returns:
        tuple: A tuple of the row and column for the best move.
        '''
    best_score = -float("inf")
    best_move = None
    for row in range(3):
         for col in range(3):
            if board[row][col] == EMPTY:
                board[row][col] = PLAYER_X
                score = minimax(board, 0, -float("inf"), float("inf"), False)
                board[row][col] = EMPTY
                if score > best_score:
                    best_score = score
                    best_move = (row, col)


    return best_move



if __name__ == "__main__":
    # test_evaluate(evaluate)
    # test_minimax(minimax)
    # test_best_move(get_best_move)

    # Start the game
    play()