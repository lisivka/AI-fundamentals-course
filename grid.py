grid =[ [1,2,3],[1,2,3],[3,3,3],]
for row in grid:
    print(row)
# print col
for col in zip(*grid):
    print(col)

print(list(zip(*grid)))

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

