def schedule_tasks(tasks, resources, deadline):
    assigned_tasks = {}
    task_times = {}
    current_time = 0.0

    # Sort tasks in ascending order of their durations
    tasks.sort(key=lambda task: task[1])

    while current_time < deadline:
        # Sort resources in ascending order of their remaining capacities
        sorted_resources = sorted(resources.items(), key=lambda item: item[1])

        task_assigned = False

        for task_name, duration, requirements in tasks:
            for resource_name, capacity in sorted_resources:
                if resource_name in requirements and requirements[resource_name] > 0:
                    # Assign the task to the resource
                    if resource_name not in assigned_tasks:
                        assigned_tasks[resource_name] = []
                    assigned_tasks[resource_name].append(task_name)

                    # Update resource capacities
                    resources[resource_name] -= 1
                    requirements[resource_name] -= 1

                    # Update the completion time for the task
                    task_times[task_name] = current_time + duration

                    task_assigned = True
                    break  # Move to the next task

        if not task_assigned:
            # If no tasks can be assigned, extend the deadline
            min_duration = min([task[1] for task in tasks])
            current_time += min_duration
        else:
            current_time = min([v for k, v in task_times.items()])

    # Create a result dictionary
    result = {}
    for resource_name, task_list in assigned_tasks.items():
        result[resource_name] = task_list
    for task_name, task_completion_time in task_times.items():
        result[task_name] = task_completion_time

    return result

# Example usage:
tasks_list = [
    ('TaskA', 4.5, {'Resource1': 2, 'Resource2': 1}),
    ('TaskB', 7.2, {'Resource2': 3}),
    ('TaskC', 5.0, {'Resource1': 1})
]

resources_dict = {'Resource1': 10, 'Resource2': 15}

deadline_time = 12.0

result = schedule_tasks(tasks_list, resources_dict, deadline_time)
print(result)
