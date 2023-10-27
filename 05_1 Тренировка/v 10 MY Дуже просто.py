import numpy as np
import matplotlib.pyplot as plt

def act(x):
    return 0 if x <= 0 else 1

def weight_dot(x1,x2, weight1, weight2):
    return x1*weight1 + x2*weight2

def go(C):
    x = np.array([C[0], C[1]])
    print(f"x: {x}")

    # w1 = [1, 1, -1.5]
    # w2 = [1, 1, -0.5]
    w1 = [-1, 1, 1]
    w2 = [1, -1, 1]
    for j in range(len(x)):
        for i in range(len(w1)):
            # print(f"w1 {i}: {w1[i]}", f"x{j}: {x[j]}")
            # print(f"w2 {i}: {w2[i]}", f"x{j}: {x[j]}")
            dot = weight_dot(x[j], x[j], w1[i], w2[i])
            print(f"w1 {i}: {w1[i]}", f"x{j}: {x[j]}")
            print(f"dot: {dot}")


    w_hidden = np.array([w1, w2])
    w_out = np.array([-1, 1, -0.5])

    # print(f"x: {x}")
    # print(f"w_hidden: {w_hidden}", sep="\n")
    # sum = np.dot(w_hidden, x)
    # print(f"sum: {sum}")
    out = [act(x) for x in sum]
    out.append(1)
    out = np.array(out)

    sum = np.dot(w_out, out)
    y = act(sum)
    return y

C1 = [(1,0), (0,1)]
C2 = [(0,0), (1,1)]

# print("Очікую 1" ,go(C1[0]))
print("Очікую 0" ,go(C1[1]))
# print( "Очікую 1" ,go(C1[0]), go(C1[1]) )
# print("Очікую 0" , go(C2[0]), go(C2[1]) )