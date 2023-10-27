import numpy as np

class Perceptron:
    def __init__(self):
        self.w_hidden = np.array([[1, 1, -1.5], [1, 1, -0.5]])
        self.w_out = np.array([-1, 1, -0.5])

    def activation(self, x):
        return 0 if x <= 0 else 1

    def forward(self, x):
        x = np.array([x[0], x[1], 1])
        hidden_sum = np.dot(self.w_hidden, x)
        hidden_out = [self.activation(x) for x in hidden_sum]

        hidden_out.append(1)
        hidden_out = np.array(hidden_out)

        output_sum = np.dot(self.w_out, hidden_out)
        y = self.activation(output_sum)

        print("\n", f"x: {x}")
        # print(f"w_hidden: {self.w_hidden}", sep="\n")
        # print(f"hidden_sum: {hidden_sum}")
        # print(f"hidden_out: {hidden_out}")
        # print(f"w_out: {self.w_out}", sep="\n")
        # print(f"output_sum: {output_sum}")
        print(f"y: {y}")


        return y

# Test your Perceptron
C1 = [(1, 0), (0, 1)]
C2 = [(0, 0), (1, 1)]

perceptron = Perceptron()

answer1 = perceptron.forward(C1[0]) # 0
answer2 = perceptron.forward(C1[1]) # 1

answer3 = perceptron.forward(C2[0]) # 0
answer4 = perceptron.forward(C2[1]) # 0

print(f"{'-'*20}")
print(f"answer1: {answer1}", f"answer2: {answer2}", sep=" ")

print(f"answer3: {answer3}", f"answer4: {answer4}", sep=" ")
# print(perceptron.forward(C2[0]), perceptron.forward(C2[1]))
# print(perceptron.forward(C1[0]), perceptron.forward(C1[1]))
