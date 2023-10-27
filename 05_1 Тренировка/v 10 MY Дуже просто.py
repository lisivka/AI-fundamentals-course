# https://www.youtube.com/watch?v=gDvDwH4dJFI
import numpy as np
import matplotlib.pyplot as plt

def activation(x):
    return 0 if x <= 0 else 1

# def counting_weight(x, w1, w2):
#     a = np.array(w1*x[0])
#     if a.size == 0:
#         a = np.zeros(3)
#     b = np.array(w2*x[1])
#     if b.size == 0:
#         b = np.zeros(3)
#     return np.stack(a+ b)

    # return np.array(w1*x[0] + w2*x[1])
def counting_weight2(x, w):

    res = np.array(w*x)
    # Пустий масив замінимо на нулі
    if res.size == 0:
        res = np.zeros(3)
    # print(f"!!res: {res}")
    return res


def predict(C):
    x = np.array([C[0], C[1]])
    print(f"x: {x}")

    # w1 = [1, 1, -1.5]
    # w2 = [1, 1, -0.5]
    # w3 = [-1, 1, -0.5]
    w1 = [-1, 1, 1]
    w2 = [1, -1, 1]
    w3 = [2, 2, -1]
    # w_hidden = counting_weight(x, w1, w2)
    # print(f"w_hidden: {w_hidden}")
    w_hidden1 = counting_weight2(x[0], w1)
    w_hidden2 = counting_weight2(x[1], w2)
    w_hidden = np.stack(w_hidden1 + w_hidden2) # Сума двох масивів
    # print(f"==w_hidden: {w_hidden}")
    w_hidden_activation =np.array([activation(x) for x in w_hidden])
    # print(f"==w_hidden_activation: {w_hidden_activation}")
    w_out = counting_weight2(w_hidden_activation, w3)

    # print(f"w_out: {w_out}")
    # print(f"sum: {sum(w_out)}")
    out = activation(sum(w_out))
    # print(f"==out: {out}")
    return out
    # print(f"weigth: {w_hidden1}")
    # print(f"weigth: {w_hidden2}")

    # print(f"weigth: {w_hidden}")
    # w_out = counting_weight(w_hidden,w3)
    # print(f"weigth: {w_out}")


    # w_hidden = np.array([w1, w2])
    # w_out = np.array([-1, 1, -0.5])

    # print(f"x: {x}")
    # print(f"w_hidden: {w_hidden}", sep="\n")
    # sum = np.dot(w_hidden, x)
    # print(f"sum: {sum}")
    # out = [act(x) for x in sum]
    # out.append(1)
    # out = np.array(out)
    #
    # sum = np.dot(w_out, out)
    # y = act(sum)
    return

C1 = [(1,0), (0,1)]
C2 = [(0,0), (1,1)]

# print("Очікую 1:" ,go(C1[1]))
# print("Очікую 1:" ,go(C1[0]))
# print("Очікую 0:" ,go(C2[1]))
print( "Очікую 1:" ,predict(C1[0]), predict(C1[1]) )
print("Очікую 0:" , predict(C2[0]), predict(C2[1]) )