# https://stackabuse.com/search-algorithms-in-python/
import math
import time
from datetime import timedelta
from random import random, randint


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        time.sleep(1)
        print(f"---------Started {func.__name__}")
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        str_time = str(execution_time)

        # Преобразуем время выполнения в объект timedelta
        timedelta_obj = timedelta(seconds=execution_time, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)

        # Форматируем timedelta в требуемый формат
        formatted_time = str(timedelta_obj)

        print(f"{func.__name__} took {formatted_time} {str_time} to execute")
        return result

    return wrapper


def generate_sorted_random_numbers(num=100):
    random_numbers = []
    previous_number = 0  # Начинаем с 0, чтобы гарантировать возрастание.

    for _ in range(num):
        next_number = previous_number + randint(1, 5)  # Генерируем
        # следующее число в диапазоне [1, 10].
        if next_number > 1000:
            next_number = 1000  # Ограничиваем число 1000, чтобы не превысить верхний предел.
        random_numbers.append(next_number)
        previous_number = next_number

    return random_numbers

@timeit
def BinarySearch(lys, val):

    first = 0
    last = len(lys) - 1
    index = -1
    while (first <= last) and (index == -1):
        mid = (first + last) // 2
        if lys[mid] == val:
            index = mid
        else:
            if val < lys[mid]:
                last = mid - 1
            else:
                first = mid + 1
    return index


@timeit
def JumpSearch(lys, val):
    length = len(lys)
    jump = int(math.sqrt(length))
    left, right = 0, 0
    while left < length and lys[left] <= val:
        right = min(length - 1, left + jump)
        if lys[left] <= val and lys[right] >= val:
            break
        left += jump;
    if left >= length or lys[left] > val:
        return -1
    right = min(length - 1, right)
    i = left
    while i <= right and lys[i] <= val:
        if lys[i] == val:
            return i
        i += 1
    return -1


@timeit
def FibonacciSearch(lys, val):
    fibM_minus_2 = 0
    fibM_minus_1 = 1
    fibM = fibM_minus_1 + fibM_minus_2
    while (fibM < len(lys)):
        fibM_minus_2 = fibM_minus_1
        fibM_minus_1 = fibM
        fibM = fibM_minus_1 + fibM_minus_2
    index = -1;
    while (fibM > 1):
        i = min(index + fibM_minus_2, (len(lys) - 1))
        if (lys[i] < val):
            fibM = fibM_minus_1
            fibM_minus_1 = fibM_minus_2
            fibM_minus_2 = fibM - fibM_minus_1
            index = i
        elif (lys[i] > val):
            fibM = fibM_minus_2
            fibM_minus_1 = fibM_minus_1 - fibM_minus_2
            fibM_minus_2 = fibM - fibM_minus_1
        else:
            return i
    if (fibM_minus_1 and index < (len(lys) - 1) and lys[index + 1] == val):
        return index + 1;
    return -1


@timeit
def ExponentialSearch(lys, val):
    if lys[0] == val:
        return 0
    index = 1
    while index < len(lys) and lys[index] <= val:
        index = index * 2
    return BinarySearch(lys[:min(index, len(lys))], val)

@timeit
def teest_time(atr):
    time.sleep(2)
    print("test")
    return atr


if __name__ == "__main__":

    print(BinarySearch([4, 4, 4, 4, 4], 4))

    print(JumpSearch([1, 2, 3, 4, 5, 6, 7, 8, 9], 5))

    print(FibonacciSearch([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 6))

    print(FibonacciSearch([4, 4, 4, 4, 4], 4))

    print(ExponentialSearch([1, 2, 3, 4, 5, 6, 7, 8], 3))


    lst = generate_sorted_random_numbers(num=100)
    print(f"lst = {lst}")
    print(BinarySearch(lst, 1000))



    print(teest_time(1))
