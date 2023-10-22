def schedule_tasks(tasks, resources, deadline):
    assigned_tasks = {}
    task_times = {}
    current_time = 0.0

    # Sort tasks in ascending order of their durations
    tasks.sort(key=lambda x: x[1])

    # Sort resources in ascending order of their remaining capacities
    sorted_resources = sorted(resources.keys(), key=lambda x: resources[x])

    for task in tasks:
        task_name, duration, resource_requirements = task
        assigned = False

        for resource in sorted_resources:
            if resource_requirements.get(resource, 0) > resources[resource]:
                continue

            # Assign the task to the resource
            assigned_tasks[resource] = assigned_tasks.get(resource, []) + [task_name]
            task_times[task_name] = current_time
            current_time += duration

            # Update resource capacities
            for res, capacity in resource_requirements.items():
                resources[res] -= capacity

            assigned = True
            break

        if not assigned:
            # If the task couldn't be assigned to any resource, extend the deadline
            current_time = deadline

    # Update completion time for each task
    for task in tasks:
        task_name, duration, _ = task
        if task_name not in task_times:
            task_times[task_name] = deadline

    # return {**assigned_tasks, **task_times}
    return {'assigned_tasks': assigned_tasks, 'task_times': task_times}

if __name__ == '__main__':

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

    tasks_list = [
        ('TaskA', 4.5, {'Resource1': 2, 'Resource2': 1}),
        ('TaskB', 7.2, {'Resource2': 3}),
        ('TaskC', 5.0, {'Resource1': 1})
    ]

    resources_dict = {'Resource1': 10, 'Resource2': 15}

    deadline_time = 12.0

    result = schedule_tasks(tasks_list, resources_dict, deadline_time)
    print(result)
