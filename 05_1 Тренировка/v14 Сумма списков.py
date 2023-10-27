list1 = [0, 1]
list2 = [1, 2, 3]

# Проверка, какой список короче, и использование его длины для итерации
# if len(list1) < len(list2):
#     result = sum(list1[i] * list2[i] for i in range(len(list1)))
# else:
#     result = sum(list1[i] * list2[i] for i in range(len(list2)))

# print("Результат умножения и суммирования:", result)
summa = 0
for i in range(len(list1)):
    for j in range(len(list2)):
        print(list1[i], list2[j], list1[i] * list2[j] )
        summa += list1[i] * list2[j]
print("Результат умножения и суммирования:", summa)


import numpy as np
list3 = np.array([0, 1]+[0])
print(list3)
dot = np.dot(list3, list2)
print(dot)