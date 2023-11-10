# list1 = [0, 1]
# list2 = [1, 2, 3]
#
# # Проверка, какой список короче, и использование его длины для итерации
# # if len(list1) < len(list2):
# #     result = sum(list1[i] * list2[i] for i in range(len(list1)))
# # else:
# #     result = sum(list1[i] * list2[i] for i in range(len(list2)))
#
# # print("Результат умножения и суммирования:", result)
# summa = 0
# for i in range(len(list1)):
#     for j in range(len(list2)):
#         print(list1[i], list2[j], list1[i] * list2[j] )
#         summa += list1[i] * list2[j]
# print("Результат умножения и суммирования:", summa)
#
#
# import numpy as np
# list3 = np.array([0, 1]+[0])
# print(list3)
# dot = np.dot(list3, list2)
# print(dot)


print("="*20)
import numpy as np

x = np.array([1, 2, 3])
x = np.array([1, 2,]+[0])
print(x)
w = np.array([2, 4, 6])

result = np.multiply(x, w)  # Используя функцию multiply
# Или можно просто использовать оператор *
# result = x * w

print(result)


import numpy as np

x = np.array([1, 2])
w = np.array([2, 4, 6])
print("O="*20)
result = np.outer(x, w)

print(result)


import numpy as np

x = np.array([1, 2])
w = np.array([2, 4, 6])
print("DO="*20)
# result = x.dot(w)

print(result)


x = [1, 2]
w = [2, 4, 6]

result = []
for i in w:
    result.append(x[0] * i + x[1] * i)

print(result)


import numpy as np

x = np.array([1, 2])
w = np.array([2, 4, 6])

result1 = (x * w[0]).sum()  # Вычисляем сумму x[0]*w[1] + x[1]*w[1]
result2 = (x * w[1]).sum()  # Вычисляем сумму x[0]*w[1] + x[1]*w[1]
result3 = (x * w[2]).sum()  # Вычисляем сумму x[0]*w[2] + x[1]*w[2]
result_list = [result1, result2, result3]

print("Numpy:"*20)
print(result_list)

import numpy as np

x = np.array([1, 2])
w1 = np.array([1, 1, 1])
w2 = np.array([2, 4, 6])

result = x[0] * w1 + x[1] * w2
print("Tolist:"*20)
result_list = result.tolist()
print(result_list)
print(result)
