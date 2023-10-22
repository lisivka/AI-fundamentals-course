"""
У задачі планування ресурсів ми маємо набір завдань, які потрібно виконати,
кожне зі своєю тривалістю та вимогами до ресурсів. Крім того, у нас є набір
доступних ресурсів з обмеженою ємністю. Мета полягає в тому, щоб розподілити
завдання між ресурсами таким чином, щоб усі завдання були виконані в установлені терміни, а ресурси використовувалися ефективно, не перевищуючи своїх можливостей.

Ваше завдання полягає в тому, щоб визначити schedule_tasks()функцію,
яка приймає такі вхідні дані:

tasks: список кортежів, що представляють завдання, де кожен кортеж містить
( task_name, duration, resource_requirements).
resources: словник, що представляє доступні ресурси та їхні потужності,
де ключі — це назви ресурсів, а значення — їхні потужності.
deadline: максимальний час ( deadline), протягом якого мають бути виконані всі завдання.
Функція schedule_tasksповертає словник, який представляє оптимальне призначення
завдань ресурсам разом із часом виконання для кожного завдання.

Примітка : реалізуйте простий алгоритм Greedy Scheduling для оптимізації
завдання планування ресурсів. У цьому алгоритмі завдання призначаються ресурсам
у жадібний спосіб на основі їх тривалості та вимог до ресурсів."""

# @test_schedule_task
def schedule_tasks(tasks, resources, deadline):
    """
    Optimize resource scheduling to complete tasks within the given deadline using Greedy Scheduling.

    Args:
        tasks (list of tuple): A list of tasks, where each tuple contains (task_name, duration, demands).
        resources (dict): A dictionary representing available resources and their capacities.
                          Keys are resource names, and values are their capacities.
        deadline (float): The maximum time (deadline) within which all tasks must be completed.

    Returns:
        dict: A dictionary representing the optimal assignment of tasks to resources.
              The keys are resource names, and the values are lists of tasks assigned to each resource.
              The dictionary also includes the completion time for each task.
              Example: {'Resource1': ['TaskA', 'TaskB'], 'Resource2': ['TaskC'], 'TaskA': 4.5, 'TaskB': 7.2, 'TaskC': 5.0}
    """
    #initialization
    assigned_tasks = {}
    task_times = {}
    current_time = 0.0

    # Sort tasks in ascending order of their durations
    tasks.sort(key=lambda x: x[1])
    print(f"tasks: {tasks}")



        # Sort resources in ascending order of their remaining capacities
    resources = dict(sorted(resources.items(), key=lambda x: x[1]))
    print(f"resources: {resources}")

                # Assign task to the resource
    for task_name, duration, demands in tasks:
        print(f"==task_name: {task_name}, duration: {duration}, demands:"
              f" {demands}")
        for resource in demands:
            if demands[resource] <= resources[resource]:
                assigned_tasks[task_name] = resource
                resources[resource] -= demands[resource]
                task_times[task_name] = duration
                break



                # Update resource capacities
    for task in tasks:
        task_name, duration, demands = task
        for resource in demands:
            if demands[resource] <= resources[resource]:
                resources[resource] -= demands[resource]
                task_times[task_name] = duration
                break


            # If the task couldn't be assigned to any resource, extend the deadline
    if len(assigned_tasks) < len(tasks):
        deadline += 1.0
        return schedule_tasks(tasks, resources, deadline)
    else:
        return {**assigned_tasks, **task_times}
        # return assigned_tasks, task_times


    # Update completion time for each task


def time_result(schedule_tasks):
    result = []
    for task in schedule_tasks:
        result.append(schedule_tasks[task])


    return sum(result)


if __name__ == '__main__':
    # Example usage:
    tasks_list = [
        ('TaskA', 4.5, {'Resource1': 2, 'Resource2': 1}),
        ('TaskB', 7.2, {'Resource2': 3}),
        ('TaskC', 5.0, {'Resource1': 1})
    ]

    resources_dict = {'Resource1': 10, 'Resource2': 15}

    deadline_time = 12.0

    result = schedule_tasks(tasks_list, resources_dict, deadline_time)
    print(f"result: {result}")
    print(f"time result: {time_result(result)}")
