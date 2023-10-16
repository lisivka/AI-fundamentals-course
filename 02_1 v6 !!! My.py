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
    # initialization
    assigned_tasks = {}
    task_times = {}
    resource_times = {key: 0.0 for key in resources}
    # current_time = 0.0

    # Sort tasks in ascending order of their durations
    tasks.sort(key=lambda x: x[1], reverse=True)
    # tasks.sort(key=lambda x: x[1])

    # Sort resources in ascending order of their remaining capacities
    resources = dict(sorted(resources.items(), key=lambda x: x[1], reverse=True))

    for task in tasks:
        task_name, duration, requirements = task
        assigned = False
        for resource in resources:
            current_time = resource_times[resource]
            demand = requirements.get(resource, 0)

            if demand == 0:
                # print(f"No requirements  for {task_name}")
                continue
            if demand > resources[resource]:
                print(f"Not enough resources {resource} "
                      f"{resources[resource]} for {task_name} demand-{demand}")
                continue

            # Assign task to the resource
            timeline = resource_times[resource] + duration
            if timeline <= deadline:
                assigned_tasks[resource] = assigned_tasks.get(resource, []) + [task_name]
                current_time += duration
                task_times[task_name] = current_time
                resource_times[resource] = current_time
                for res, capacity in requirements.items():
                    resources[res] -= capacity
                assigned = True

                break
            else:
                # print(f"Time {timeline} > the Deadline {deadline} for {task_name} and  {assigned_tasks[resource]}")
                continue

        # If the task couldn't be assigned to any resource, extend the deadline
        if not assigned:
            print(f"Task {task_name} requirements {requirements} not assigned to any resource")
            continue

    # Update completion time for each resource in assigned_tasks
    for resource in assigned_tasks:
        total_time = 0.0
        for shedule in assigned_tasks[resource]:
            total_time += get_time_task(shedule, tasks)
            task_times[shedule] = total_time

    return { **assigned_tasks, **task_times}


def get_time_task(task_shedule, task_list):
    for task in task_list:
        if task[0] == task_shedule:
            return task[1]

    return task[1]

def print_data(data):
    for key, value in data.items():
        if isinstance(value, list):
            value_str = ', '.join(value)
        else:
            value_str = str(value)
        print(f'{key}: {value_str}')


# Example usage:
tasks_list = [
    ('Task_A', 6.0, {'Resource1': 2, 'Resource2': 1}),
    ('Task_B', 7.2, {                'Resource2': 3}),
    # ('TaskC', 6.0,
    ('Task_C', 6.1, {'Resource1': 2, 'Resource2': 1}),
    ('Task_E', 7.0, {                'Resource2': 1}),
    ('Task_D', 7.1, {'Resource1': 2, 'Resource2': 1}),
    ('Task_S', 10.0, {                               'Resource3': 1}),
]
resources_dict = {'Resource1': 10, 'Resource2': 15, 'Resource3': 10}
deadline_time = 15.2

result = schedule_tasks(tasks_list, resources_dict, deadline_time)
print_data(result)
# print(f"result: {result}")
print()

tasks_list = [
    ('TaskA', 4.5, {'Resource1': 2, 'Resource2': 1}),
    ('TaskB', 7.2, {                'Resource2': 3}),
    ('TaskC', 5.0, {'Resource1': 1})
]

resources_dict = {'Resource1': 10, 'Resource2': 15}

deadline_time = 12.0

result = schedule_tasks(tasks_list, resources_dict, deadline_time)
# print(f"result: {result}")
print_data(result)

