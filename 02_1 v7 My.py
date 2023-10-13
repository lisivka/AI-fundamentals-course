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
    tasks = sorted(tasks, key=lambda task: task[1])

    # Sort resources in ascending order of their remaining capacities
    resources = dict(sorted(resources.items(), key=lambda x: x[1]))

    while tasks:
        # Get the next task to be scheduled
        task = tasks.pop(0)
        print(f"==task: {task}")
        print(f"{task[0]}: {task[1]}")
        time_task = task[1]
        # Find a resource that can schedule the task
        resource = None
        # for resource_name, capacity in resources:
        for resource_name, capacity in resources.items():
            if capacity >= task[1]:
                resource = resource_name
                break

        # If no resource can schedule the task, extend the deadline
        if resource is None:
            print('Resource not found for task: {}'.format(task[0]))
            deadline += task[1]
            continue

        # Assign task to the resource
        assigned_tasks.setdefault(resource, []).append(task[0])

        # Update resource capacities
        resources[resource] -= task[1]

        # Update completion time for the task
        task_times[task[0]] = current_time + task[1]

        # Update current time
        current_time += task[1]

    return {'assigned_tasks': assigned_tasks, 'task_times': task_times}

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

    print(result)