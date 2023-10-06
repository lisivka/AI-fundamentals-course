import random

EMPTY = ' '
PLAYER1 = 'x'
PLAYER2 = 'o'
DIM = 3

def player_wins(state, player):
    anycol = any(all(state[i][j] == player for i in range(DIM)) for j in range(DIM))
    anyrow = any(all(state[j][i] == player for i in range(DIM)) for j in range(DIM))
    anydiag = any([all(state[i][i] == player for i in range(DIM)), all(state[i][DIM - 1 - i] == player for i in range(DIM))])
    return any([anycol, anyrow, anydiag])

# this method can be made more efficient,
# no need to wait until the board is entirely filled
def stalemate(state):
    return not (player_wins(state, PLAYER1) or player_wins(state, PLAYER2)) and all_filled(state)

def game_over(state):
    return player_wins(state, PLAYER1) or player_wins(state, PLAYER2) or all_filled(state)

def game_over_score(state):
    if player_wins(state, PLAYER1):
        return 1
    elif player_wins(state, PLAYER2):
        return -1
    return 0

def all_filled(state):
    return len(open_coordinates(state)) == 0

def open_coordinates(state):
    return [(i, j) for i in range(DIM) for j in range(DIM) if state[i][j] == EMPTY]

def next_states(state, player):
    coords = open_coordinates(state)
    ns = []
    for i, j in coords:
        statecp = [row[:] for row in state]
        statecp[i][j] = player
        ns.append(statecp)
    return ns

def other_player(player):
    return PLAYER1 if player == PLAYER2 else PLAYER2

def state_to_key(state):
    return ''.join([''.join(a for a in row) for row in state])

def key_to_state(key):
    return [[a for a in key[i:i+DIM]] for i in [j * DIM for j in range(DIM)]]

# todo: efficient symmetry states update in minimax
def apply_symmetry_op(op, state):
    statecp = [row[:] for row in state]
    for i in range(DIM):
        for j in range(DIM):
            ip, jp = op(i, j)
            statecp[i][j] = state[ip][jp]
    return statecp

# i = row, j = col
def rotate_right(i, j):
    return j, DIM - i - 1

def flip_coord(i, j):
    return i, DIM - j - 1

def rotate_n(n):
    def f(i, j):
        for m in range(n):
            i, j = rotate_right(i, j)
        return i, j
    return f

# minimax with caching
def minimax(state, player, scores):
    if state_to_key(state) in scores:
        return scores[state_to_key(state)]
    if not game_over(state):
        next_scores = [minimax(ns, other_player(player), scores) for ns in next_states(state, player)]
        if player == PLAYER1:
            best = max(next_scores)
        else:
            best = min(next_scores)
    else:
        best = game_over_score(state)
    scores[state_to_key(state)] = best
    return best

def main():
    state = [[EMPTY] * DIM] * DIM
    scores = {}
    minimax(state, PLAYER1, scores)

# learned state transition
def state_transition(state, player, scores):
    ns = next_states(state, player)
    if player == PLAYER1:
        return max([(next_state, scores[state_to_key(next_state)]) for next_state in ns], key=lambda e: e[1])[0]
    else:
        return min([(next_state, scores[state_to_key(next_state)]) for next_state in ns], key=lambda e: e[1])[0]

def play_game():
    state = [[EMPTY] * DIM] * DIM
    scores = {}
    print('training minimax...')
    best = minimax(state, PLAYER1, scores)
    assert best == 0, "the learned minimax strategy adopted by both players must always end in stalemate."
    player = PLAYER1
    print( "playing game, with player {PLAYER1} using minimax, {PLAYER2} playing randomly.")
    while not game_over(state):
        if player == PLAYER2:
            state = random.choice(next_states(state, player))
            #state = state_transition(state, player, scores) # having player 2 use the minimax strategy
            # as well will always result in stalemate.
        else:
            state = state_transition(state, player, scores)
        print(f"player {player} played: {state_to_key(state)}")
        player = other_player(player)
    if player_wins(state, PLAYER1):
        print(f'player {PLAYER1} wins.')
    elif player_wins(state, PLAYER2):
        print(f'player {PLAYER2} wins')
    else:
        print('stalemate.')

if __name__ == '__main__':
    play_game()




