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
    """
    # Initialization
    assigned_tasks = {}
    task_times = {}
    current_time = 0.0

    # Sort tasks in ascending order of their durations
    tasks.sort(key=lambda x: x[1], reverse=True)

    # Sort resources in ascending order of their remaining capacities
    resources = dict(sorted(resources.items(), key=lambda x: x[1]))

    for task in tasks:
        task_name, duration, requirements = task
        assigned = False
        for resource in resources:
            demand = requirements.get(resource, 0)

            if demand > resources[resource]:
                print("Not enough resources")
                continue
            if demand == 0:
                print("No requirements")
                continue

            if current_time + duration <= deadline:
                # Assign task to the resource
                assigned_tasks[resource] = assigned_tasks.get(resource, []) + [task_name]

                current_time += duration
                task_times[task_name] = current_time

                print(f"Task {task_name} requirements {requirements} assigned to {resource} at time {current_time}")

                # Update resource capacities
                for res, capacity in requirements.items():
                    resources[res] -= capacity

                assigned = True
                break

        # If the task couldn't be assigned to any resource or exceeds the deadline, extend the deadline
        if not assigned:
            current_time = min(current_time + duration, deadline)
            print(f"Task {task_name} requirements {requirements} not assigned to any resource at time {current_time}")

    # Update completion time for each resource in assigned_tasks
    for resource in assigned_tasks:
        total_time = 0.0
        for schedule in assigned_tasks[resource]:
            total_time += get_time_task(schedule, tasks)
            task_times[schedule] = total_time

    return {'assigned_tasks': assigned_tasks, 'task_times': task_times}

def get_time_task(task_schedule, task_list):
    for task in task_list:
        if task[0] == task_schedule:
            return task[1]
    return 0

# Example usage:
tasks_list = [
    ('TaskA', 4.5, {'Resource1': 2, 'Resource2': 1}),
    ('TaskB', 17.2, {'Resource2': 3}),
    ('TaskC', 5.0, {'Resource1': 1}),
    ('TaskE', 6.0, {'Resource2': 1}),
]

resources_dict = {'Resource1': 10, 'Resource2': 15}

deadline_time = 12.0

result = schedule_tasks(tasks_list, resources_dict, deadline_time)
print(result)
