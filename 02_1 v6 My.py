# @test_schedule_task
def schedule_tasks(tasks, resources, deadline):
    """
    Optimize resource scheduling to complete tasks within the given deadline using Greedy Scheduling.

    Args:
        tasks (list of tuple): A list of tasks, where each tuple contains (task_name, duration, resource_requirements).
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

    # Sort resources in ascending order of their remaining capacities
    resources = dict(sorted(resources.items(), key=lambda x: x[1]))
    for task in tasks:
        task_name, duration, requirements = task
        assigned = False
        for resource in resources:
            demand = requirements.get(resource, 0)
            # print(f"Task {task_name} demand {demand} for resource {resource}")
            if demand > resources[resource]:
                print("Not enough resources")
                continue
            if demand == 0:
                print("No requirements")
                continue
            if current_time + duration > deadline:
                print("Deadline exceeded")
                assigned_tasks = {}
                task_times = {}
                current_time = 0.0
                break

            # Assign task to the resource

            assigned_tasks[resource] = assigned_tasks.get(resource, []) + [task_name]

            current_time += duration
            task_times[task_name] = current_time
            print(f"Task {task_name} requirements {requirements} assigned to{resource} at time {current_time}")
            # Update resource capacities
            for res, capacity in requirements.items():
                resources[res] -= capacity
            assigned = True
            break


            # If the task couldn't be assigned to any resource, extend the deadline
        if not assigned:
            current_time = deadline
            print(f"Task {task_name} requirements {requirements} not assigned to any resource at time {current_time}")


    # Update completion time for each task
    for task in tasks:
        task_name, duration, _ = task
        if task_name not in task_times:
            task_times[task_name] = deadline

        return {**assigned_tasks, **task_times}
    # return {'assigned_tasks': **assigned_tasks, 'task_times': task_times}


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