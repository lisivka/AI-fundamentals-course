# https://www.youtube.com/watch?v=gDvDwH4dJFI
import numpy as np
# import matplotlib.pyplot as plt

def activation(x):
    return 0 if x <= 0 else 1

def counting_weight2(x, w):

    result = np.array(w*x)
    # Пустий масив замінимо на нулі
    if result.size == 0:
        result = np.zeros(3)
    return result


def predict(X):
    x = np.array([X[0], X[1]])
    print(f"x: {x}")

    w1 = [-1, 1, 1]
    w2 = [1, -1, 1]
    w3 = [2, 2, -1]
    w_hidden1 = counting_weight2(x[0], w1)
    w_hidden2 = counting_weight2(x[1], w2)
    layer_hidden = np.stack(w_hidden1 + w_hidden2) # Сума двох масивів

    layer_hidden_activation =np.array([activation(x) for x in layer_hidden])
    layer_out = counting_weight2(layer_hidden_activation, w3)
    out = activation(sum(layer_out))

    return out

if __name__ == "__main__":
    C1 = [(1, 0), (0, 1)]
    C2 = [(0, 0), (1, 1)]
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])



    print( "Очікую 1: отримав" ,predict([1,0]))
    print("Очікую 0: отримав", predict([1, 1]))
    print( "Очікую 1: отримав" ,predict(C1[0]), predict(C1[1]) )
    print("Очікую 0: отримав" , predict(C2[0]), predict(C2[1]) )
    print("Очікую 1: отримав", predict(X[1]))
    print([predict(x) for x in X])