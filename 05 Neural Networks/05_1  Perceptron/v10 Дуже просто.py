import numpy as np

def act(x):
    return 0 if x <= 0 else 1

def go(C):
    x = np.array([C[0], C[1], 1])

    w1 = [1, 1, -1.5]
    w2 = [1, 1, -0.5]
    w_hidden = np.array([w1, w2])
    w_out = np.array([-1, 1, -0.5])

    sum = np.dot(w_hidden, x)
    out = [act(x) for x in sum]
    out.append(1)
    out = np.array(out)

    sum = np.dot(w_out, out)
    y = act(sum)
    return y


if __name__ == "__main__":
    C1 = [(1, 0), (0, 1)]
    C2 = [(0, 0), (1, 1)]

    print(f"C1 {C1[0]} predict {go(C1[0])} C1: {C1[1]} predict {go(C1[1])}")
    print(f"C2 {C2[0]} predict {go(C2[0])} C2: {C2[1]} predict {go(C2[1])}")