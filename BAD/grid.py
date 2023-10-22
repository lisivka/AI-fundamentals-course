# grid =[ [1,2,3],[1,2,3],[3,3,3],]
# for row in grid:
#     print(row)
# # print col
# for col in zip(*grid):
#     print(col)
#
# print(list(zip(*grid)))

# # reverse grid
# for row in grid[::-1]:
#     print(row)
#
# # chenge row and col
# for col in zip(*grid):
#     print(col[::-1])
# # transpose grid
# for col in zip(*grid):
#     print(col)
PLAYER_X = "X"
PLAYER_O = "O"
board =[ ["X",1,3],[1,"X",3],[3,1,"X"],]


main_diagonal = {board[i][i] for i in range(3)}
print(main_diagonal)
second_diagonal = [board[i][2 - i] for i in range(3)]
print(second_diagonal)

if set(main_diagonal) == {PLAYER_X} or set(second_diagonal) == {PLAYER_X}:
    print("PLAYER_X")
elif set(main_diagonal) == {PLAYER_O} or set(second_diagonal) == {PLAYER_O}:
    print("PLAYER_O")
else:
    print("X# 0")