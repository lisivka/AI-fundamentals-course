"""
n the resource scheduling task, we have a set of tasks to be performed, each with its own duration and resource requirements. Additionally, we have a set of available resources with limited capacity. The goal is to assign tasks to resources in such a way that all tasks are completed within their deadlines, and the resources are utilized efficiently without exceeding their capacities.

Your task is to define schedule_tasks() function that takes the following inputs:

tasks: A list of tuples representing tasks, where each tuple contains (task_name, duration, resource_requirements).
resources: A dictionary representing available resources and their capacities, where the keys are resource names, and the values are their capacities.
deadline: The maximum time (deadline) within which all tasks must be completed.
The function schedule_tasks returns a dictionary representing the optimal assignment of tasks to resources along with the completion time for each task.

Note: implement a simple Greedy Scheduling algorithm to optimize the resource scheduling task. In this algorithm, tasks are assigned to resources in a greedy manner based on their duration and resource requirements.
"""

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
    print(f"tasks: {tasks}")



        # Sort resources in ascending order of their remaining capacities
    resources = dict(sorted(resources.items(), key=lambda x: x[1]))
    print(f"resources: {resources}")

                # Assign task to the resource
    for task in tasks:
        task_name, duration, resource_requirements = task
        for resource in resource_requirements:
            if resource_requirements[resource] <= resources[resource]:
                assigned_tasks[task_name] = resource
                resources[resource] -= resource_requirements[resource]
                task_times[task_name] = duration
                break



                # Update resource capacities
    for task in tasks:
        task_name, duration, resource_requirements = task
        for resource in resource_requirements:
            if resource_requirements[resource] <= resources[resource]:
                resources[resource] -= resource_requirements[resource]
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
    for task in tasks:
        task_name, duration, resource_requirements = task
        task_times[task_name] = duration
        current_time += duration
        if current_time > deadline:
            deadline += 1.0
            return schedule_tasks(tasks, resources, deadline)
        else:
            return {**assigned_tasks, **task_times}
            # return assigned_tasks, task_times



# Example usage:
# tasks_list = [
#     ('TaskA', 4.5, {'Resource1': 2, 'Resource2': 1}),
#     ('TaskB', 7.2, {'Resource2': 3}),
#     ('TaskC', 5.0, {'Resource1': 1})
# ]
#
# resources_dict = {'Resource1': 10, 'Resource2': 15}
#
# deadline_time = 12.0
#
# result = schedule_tasks(tasks_list, resources_dict, deadline_time)
# print(result)
def time_result(schedule_tasks):
    result = []
    for task in schedule_tasks:
        result.append(schedule_tasks[task])

    print(f"time result: {sum(result)}")
    return result


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
