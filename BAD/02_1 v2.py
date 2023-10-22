def schedule_tasks(tasks, resources, deadline):
    assigned_tasks = {}
    task_times = {}
    current_time = 0.0

    # Сортируем задачи по возрастанию продолжительности
    tasks.sort(key=lambda x: x[1])

    # Сортируем ресурсы по возрастанию оставшихся мощностей
    sorted_resources = sorted(resources.items(), key=lambda x: x[1])

    for task_name, duration, requirements in tasks:
        task_assigned = False

        for resource, capacity in sorted_resources:
            if all(requirements.get(res, 0) <= resources[res] for res in requirements):
                # Если ресурс соответствует требованиям задачи, назначаем задачу
                assigned_tasks.setdefault(resource, []).append(task_name)
                task_times[task_name] = current_time
                current_time += duration
                # Обновляем мощность ресурса
                for res, req in requirements.items():
                    resources[res] -= req
                task_assigned = True
                break

        if not task_assigned:
            # Если задачу нельзя назначить ни одному ресурсу, увеличиваем срок
            current_time = max(current_time, deadline)
            task_times[task_name] = current_time

    return {**assigned_tasks, **task_times}

# Пример использования
tasks_list = [
    ('TaskA', 4.5, {'Resource1': 2, 'Resource2': 1}),
    ('TaskB', 7.2, {'Resource2': 3}),
    ('TaskC', 5.0, {'Resource1': 1})
]

resources_dict = {'Resource1': 10, 'Resource2': 15}

deadline_time = 12.0

result = schedule_tasks(tasks_list, resources_dict, deadline_time)
print(result)
