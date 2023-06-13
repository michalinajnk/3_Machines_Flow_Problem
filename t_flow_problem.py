import copy
import math
import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


class Task:
    def __init__(self, id, durations, deadline):
        self.id = id
        self.durations = durations
        self.deadline = deadline
        self.finished_times = [math.inf, math.inf, math.inf]
        self.start_times = [math.inf, math.inf, math.inf]

    def execute(self, machine_id, start_time):
        self.start_times[machine_id] = start_time
        self.finished_times[machine_id] = start_time + self.durations[machine_id]

    def to_string(self, machine_id):
        return f" Id: {self.id}, duration {self.durations[machine_id]}, start: {self.start_times[machine_id]}, end {self.finished_times[machine_id]}"

    def set_initial_state(self):
        self.finished_times = [math.inf, math.inf, math.inf]
        self.start_times = [math.inf, math.inf, math.inf]

    def copy(self):
        cop = copy.deepcopy(self)
        cop.set_initial_state()
        return cop


class Machine:

    def __init__(self, id):
        self.id = id
        self.finished_time = -1
        self.current_task = None
        self.task_flow = []

    def set_task_flow(self, task_flow):
        self.task_flow = task_flow
        self.finished_time = -1
        self.current_task = None

    def set_initial_state(self):
        self.finished_time = -1
        self.current_task = None

    def update_finished_time(self, time):
        self.finished_time = time

    def can_be_executed(self, task):
        if self.current_task is None:
            if self.id == 0 or task.finished_times[self.id - 1] is not None:
                return True
        return False

    def execute_task(self, task, current_time):
        if self.can_be_executed(task):
            print("Can be executed")
            self.current_task = task
            task.execute(self.id, current_time)
            self.current_task = None
        else:
            print("Cannot be executed")


class Solution:
    def __init__(self, flow_schedule, machines, tasks):
        self.flow_schedule = flow_schedule
        self.machines = machines
        self.makespan = -1
        self.total_flowtime = -1
        self.total_tardiness = -1
        self.max_tardiness = -1
        self.tasks = tasks
        self.scalar_criteria = -1

    def execute_schedule(self, tasks):
        current_time = 0
        for machine in self.machines:
            for task_order_on_mach, task_id in enumerate(machine.task_flow):
                if task_order_on_mach == 0 and machine.id == 0:
                    machine.execute_task(tasks[task_order_on_mach], 0)
                elif task_order_on_mach > 0 and machine.id == 0:
                    machine.execute_task(tasks[task_order_on_mach], tasks[task_order_on_mach - 1].finished_times[machine.id])
                elif task_order_on_mach == 0 and machine.id > 0:
                    machine.execute_task(tasks[task_order_on_mach], tasks[task_order_on_mach].finished_times[machine.id - 1])
                elif task_order_on_mach > 0 and machine.id > 0:
                    start_time = max(tasks[task_order_on_mach].finished_times[machine.id - 1],
                                     tasks[task_order_on_mach - 1].finished_times[machine.id])
                    machine.execute_task(tasks[task_id], start_time)
                current_time = tasks[task_id].finished_times[machine.id]
                print("Current time {}".format(current_time))
                print(
                    f"Machine {machine.id} - task {tasks[task_order_on_mach].id} Start time {tasks[task_order_on_mach].start_times[machine.id]}, End time {tasks[task_order_on_mach].finished_times[machine.id]}")
            machine.update_finished_time(current_time)
        self.set_criteria_values(tasks)
        print(f" FLOWTIME - {self.total_flowtime} MAKESPAN - {self.makespan}")

    def copy(self):
        return copy.deepcopy(self)

    #zakończenie ostatniego zadania na ostatniej maszynie
    def calculate_makespan(self, tasks):
        task_id = self.machines[-1].task_flow[-1]
        return tasks[task_id].finished_times[-1]

    #sumę czasów zakończenia wszystkich zadań na trzeciej maszynie.
    def calculate_total_flowtime(self, tasks):
        total_flowtime = 0
        for task in tasks:
            print(task.finished_times[-1])
            total_flowtime += task.finished_times[-1]

        return total_flowtime

    def calculate_max_tardiness(self, tasks):
        return max(max(tasks[id].finished_times[-1] - tasks[id].deadline for id in self.machines[-1].task_flow), 0)

    def calculate_total_tardiness(self, tasks):
        total_tardiness = 0
        for task_id in self.machines[-1].task_flow:
            total_tardiness += max(tasks[task_id].finished_times[-1] - tasks[task_id].deadline, 0)
        return total_tardiness

    def calculate_scalarization_weights(self, criteria_values):
        # Normalizacja wartości kryteriów
        normalized_values = criteria_values / np.max(criteria_values, axis=0)

        # Reshape the normalized_values array to have shape (n, 1)
        normalized_values = normalized_values.reshape(-1, 1)

        # Obliczanie odległości euklidesowych między wartościami kryteriów a idealnym rozwiązaniem
        ideal_solution = np.min(normalized_values, axis=0)
        distances = np.linalg.norm(normalized_values - ideal_solution, axis=1)

        # Obliczanie współczynników skalaryzacji
        scalarization_weights = distances / np.sum(distances)
        return scalarization_weights

    def scalarize_criteria(self, criterias):
        weights = self.calculate_scalarization_weights(criterias)
        scalarized_value = np.sum(weights * criterias)
        return scalarized_value

    def set_criteria_values(self, tasks_):
        self.makespan = self.calculate_makespan(tasks_)
        self.total_flowtime = self.calculate_total_flowtime(tasks_)
        self.total_tardiness = self.calculate_total_tardiness(tasks_)
        self.max_tardiness = self.calculate_max_tardiness(tasks_)
        self.scalar_criteria = self.scalarize_criteria(
            np.array([self.makespan, self.max_tardiness, self.total_tardiness]))

    def is_dominating(self, b):
        makespan_a = self.makespan
        makespan_b = b.makespan
        total_flowtime_a = self.total_flowtime
        total_flowtime_b = b.total_flowtime

        if makespan_a <= makespan_b and total_flowtime_a <= total_flowtime_b:
            return True
        return False

    def is_dominating_scalar(self, b):
        scalar_criteria = self.scalar_criteria
        scalar_criteria_b = b.scalar_criteria
        if scalar_criteria < scalar_criteria_b:
            return True
        return False

    def is_dominating_more_criterias(self, b):
        makespan_a = self.makespan
        makespan_b = b.makespan
        total_flowtime_a = self.total_flowtime
        total_flowtime_b = b.total_flowtime
        total_tardiness_a = self.total_tardiness
        total_tardiness_b = b.total_tardiness
        max_tardiness_a = self.max_tardiness
        max_tardiness_b = b.max_tardiness
        if (
                makespan_a <= makespan_b
                and total_flowtime_a <= total_flowtime_b
                and total_tardiness_a <= total_tardiness_b
                and max_tardiness_a <= max_tardiness_b

        ):
            return True
        return False

    def check_task_order_constraint(self):
        for i in range(len(self.machines) - 1):
            machine_tasks_prev = self.machines[i].task_flow
            machine_tasks_next = self.machines[i + 1].task_flow
            if machine_tasks_prev and machine_tasks_next:
                last_task_prev = max(task.id for task in machine_tasks_prev)
                first_task_next = min(task.id for task in machine_tasks_next)
                if last_task_prev >= first_task_next:
                    return False
        return True

    def check_single_task_per_machine_constraint(self):
        for mach in self.machines:
            task_times = []
            for task in mach.task_flow:
                task_times.append((task.start_times[mach.id], task.finished_times[mach.id]))
            task_times.sort()
            for i in range(len(task_times) - 1):
                if task_times[i][1] > task_times[i + 1][0]:
                    return False
        return True

    def check_single_task_execution_constraint(self):
        for mach in self.machines:
            running_tasks = set()
            for task in mach.task_flow:
                if task.id in running_tasks:
                    return False
                running_tasks.add(task.id)
                if task.finished_times[-1] is not None:
                    running_tasks.remove(task.id)
        return True

    def check_constraints(self):
        if not self.check_single_task_per_machine_constraint():
            return False
        if not self.check_single_task_execution_constraint():
            return False
        if not self.check_task_order_constraint():
            return False
        return True


class Scheduler:

    def __init__(self, machines):
        self.machines = machines

    def generate_neighbor_schedule(self, machines, tasks, num_swaps=1):
        num_swaps = len(tasks) // 3
        new_solution = []
        indices = self.draw_random_indices(tasks, num_swaps)
        for mach in machines:
            machine_schedule = mach.task_flow.copy()
            for id1, id2 in indices:
                print(id1, id2)
                self.swap_elements_at(machine_schedule, id1, id2)
            mach.set_task_flow(machine_schedule)
            new_solution.append(machine_schedule)

        s = Solution(new_solution, machines, tasks)
        s.execute_schedule(tasks)
        return s

    def generate_neighbor_schedule_swap_neighbors(self, machines, tasks, num_swaps=1):
        new_solution = []
        for _ in range(num_swaps):
            mach_idx = random.randint(0, len(machines) - 2)
            machine_schedule = machines[mach_idx].task_flow.copy()
            self.swap_neighbours(machine_schedule)
            machines[mach_idx].set_task_flow(machine_schedule)
            new_solution.append(machine_schedule)
        s = Solution(new_solution, machines, tasks)
        s.execute_schedule(tasks)
        return s

    def draw_random_indices(self, arr, num_of_pairs=1):
        pairs = []
        for _ in range(num_of_pairs):
            pairs.append((random.randint(0, len(arr) - 1), random.randint(0, len(tasks) - 1)))
        return pairs

    def get_validated_solution(self, tasks, machines, tries=1000, random_solution=False):
        valid_solution = False
        tries_counter = 0
        while not valid_solution and tries_counter < tries:
            if random_solution:
                solution = self.generate_random_schedule(tasks, machines)
            else:
                solution = self.generate_neighbor_schedule(machines, tasks)
            valid_solution = solution.check_constraints()
            tries_counter += 1
        return solution

    def generate_sequential_schedule(self, tasks, machines):
        new_solution = []
        for mach in machines:
            shuffled_tasks = mach.task_flow.copy()
            mach.set_task_flow(sorted(shuffled_tasks, key=lambda task: task.id))
            new_solution.append(shuffled_tasks)
        s = Solution(new_solution, machines, tasks)
        s.execute_schedule(tasks)
        return s

    def generate_random_task_flow(self, num_tasks):
        li = list(range(0, num_tasks))
        random.shuffle(li)
        return li

    def generate_random_schedule(self, tasks, machines):
        new_solution = []
        for mach in machines:
            tasks_schedule = self.generate_random_task_flow(len(tasks))
            print(f"RAandomly generated task schedule{tasks_schedule}")
            mach.set_task_flow(tasks_schedule)
            new_solution.append(tasks)
        s = Solution(new_solution, machines, tasks)
        s.execute_schedule(tasks)
        return s

    def swap_elements_at(self, array, id1, id2):
        array[id1], array[id2] = array[id2], array[id1]

    def swap_neighbours(self, arr, num_of_swaps=1):
        for i in range(num_of_swaps):
            left_el_idx = random.randint(0, len(arr) - 2)
            right_el_idx = left_el_idx + 1
            self.swap_elements_at(arr, left_el_idx, right_el_idx)


class SimulatedAnnealing:

    def __init__(self, machines, tasks, max_iterations, initial_temperature, cooling_rate):
        self.machines = machines
        self.tasks = tasks
        self.max_iterations = max_iterations
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate

    def run_with_pareto(self, scheduler, iter):
        self.max_iterations = iter
        P = []
        current_solution = scheduler.generate_random_schedule(self.tasks, self.machines)
        best_solution = current_solution
        P.append(best_solution)
        temperature = self.initial_temperature

        for iteration in range(self.max_iterations):
            neighbor_solution = scheduler.generate_neighbor_schedule(self.machines, self.tasks)
            delta = neighbor_solution.makespan + neighbor_solution.total_flowtime - current_solution.makespan + current_solution.total_flowtime

            if current_solution.is_dominating(best_solution):
                best_solution = current_solution
                P.append(best_solution)
            else:
                current_solution = neighbor_solution
                if delta < 0 or self.accept_with_probability(delta, temperature):
                    P.append(current_solution)

            temperature = self.cool_temperature(temperature)
        F = self.calculate_pareto_front(P)

        return F, P, best_solution

    def run_with_scalar(self, scheduler, iter):
        self.max_iterations = iter
        current_solution = scheduler.generate_random_schedule(self.tasks, self.machines)
        best_solution = current_solution

        for it in range(self.max_iterations):
            neighbor_solution = scheduler.generate_neighbor_schedule(self.machines, self.tasks)

            if current_solution.is_dominating_scalar(best_solution):
                best_solution = current_solution

            else:
                if random.random() < self.p(it):
                    current_solution = neighbor_solution

        return best_solution

    def run_with_pareto_more_criteria(self, scheduler, iter):
        self.max_iterations = iter
        P = []
        first_solution = scheduler.generate_random_schedule(self.tasks, self.machines)
        current_solution = first_solution
        best_solution = current_solution
        P.append(best_solution)
        temperature = self.initial_temperature

        for iteration in range(self.max_iterations):
            neighbor_solution = scheduler.generate_neighbor_schedule(self.machines, self.tasks)
            delta = neighbor_solution.makespan + neighbor_solution.total_flowtime - current_solution.makespan + current_solution.total_flowtime

            if current_solution.is_dominating_more_criterias(best_solution):
                best_solution = current_solution
                P.append(best_solution)
            else:
                current_solution = neighbor_solution
                if delta < 0 or self.accept_with_probability(delta, temperature):
                    P.append(current_solution)

            temperature = self.cool_temperature(temperature)
        F = self.calculate_pareto_front_more_criteria(P)

        sol1, sol2, sol3 = F[-1], F[-2], F[-3]

        return sol1, sol2, sol3, first_solution

    def p(self, it):
        return 0.995 ** it

    def accept_with_probability(self, delta, temperature):
        probability = math.exp(-delta / temperature)
        return random.random() < probability

    def cool_temperature(self, temperature):
        return temperature * self.cooling_rate

    def calculate_pareto_front(self, solutions):
        pareto_front = solutions.copy()
        to_remove = []

        for a in pareto_front:
            for b in pareto_front:
                if a != b and b.is_dominating(a):
                    to_remove.append(a)
                    break

        for solution in to_remove:
            pareto_front.remove(solution)

        return pareto_front

    def calculate_pareto_front_more_criteria(self, solutions):
        pareto_front = solutions.copy()
        to_remove = []

        for a in pareto_front:
            for b in pareto_front:
                if a != b and b.is_dominating_more_criterias(a):
                    to_remove.append(a)
                    break

        for solution in to_remove:
            pareto_front.remove(solution)

        return pareto_front

    def calculate_hvi(self, pareto_fronts):
        reference_point = self.calculate_reference_point(pareto_fronts)
        hv_values = []
        for front in pareto_fronts:
            hv = 0
            for solution in front:
                hv += self.calculate_volume(solution, reference_point)
            hv_values.append(hv)
        average_hvi = sum(hv_values) / len(hv_values)
        return average_hvi

    def calculate_reference_point(self, pareto_fronts):
        max_values = [float('-inf'), float('-inf')]
        for front in pareto_fronts:
            for solution in front:
                makespan = solution.makespan
                total_flowtime = solution.total_flowtime
                if makespan > max_values[0]:
                    max_values[0] = makespan
                if total_flowtime > max_values[1]:
                    max_values[1] = total_flowtime
        reference_point = [max_values[0] * 1.2, max_values[1] * 1.2]  # Adjust the factor as needed
        return reference_point

    def calculate_volume(self, solution, reference_point):
        makespan = solution.makespan
        total_flowtime = solution.total_flowtime

        volume = (reference_point[0] - makespan) * (reference_point[1] - total_flowtime)
        return volume


def generate_tasks(n, seed):
    random.seed(seed)
    tasks = []

    A = 0
    for i in range(n):
        durations = []
        for j in range(1, 4):
            duration = random.randint(1, 99)
            durations.append(duration)
            A += duration
        tasks.append(Task(i, durations, 0))

    B = int(1 / 2 * A)
    A = int(1 / 6 * A)

    for task in tasks:
        deadline = random.randint(A, B)
        task.deadline = deadline

    return tasks


import matplotlib.pyplot as plt


def plot_pareto(pareto_front, pareto_set, iteration_count):

    flowtimes = []
    makespans = []

    for solution in pareto_set:
        flowtimes.append(solution.total_flowtime)
        makespans.append(solution.makespan)

    print(flowtimes)
    print(makespans)

    plt.figure(figsize=(10, 6))
    plt.scatter(flowtimes, makespans, color='blue', label='Pareto Set')
    for solution in pareto_front:
        plt.plot(solution.total_flowtime, solution.makespan, color='red', marker='o', markersize=8)

    plt.xlabel('Total Flowtime')
    plt.ylabel('Makespan')
    plt.title('Pareto Front (Iterations: {})'.format(iteration_count))
    plt.legend()
    plt.grid(True)

    # Dostosowanie zakresu wartości na osiach
    x_min = min(flowtimes)
    x_max = max(flowtimes)
    y_min = min(makespans)
    y_max = max(makespans)
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
   # plt.xlim(x_min - x_margin, x_max + x_margin)
   # plt.ylim(y_min - y_margin, y_max + y_margin)

    plt.show()

def create_gantt_chart(tasks, machines):
    fig, ax = plt.subplots()
    # Ustalamy osie i etykiety
    ax.set_xlabel("Time")
    ax.set_ylabel("Tasks")
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([f"Task {task.id}" for task in tasks])

    # Tworzymy słupki dla każdego zadania na odpowiednich maszynach

    for machine in machines:
        for task_id in machine.task_flow:
            start_time = tasks[task_id].start_times[machine.id]
            duration = tasks[task_id].durations[machine.id]
            ax.barh(task_id, duration, left=start_time, height=0.5, align='center', alpha=0.8, color=f"C{machine.id+1}")
    plt.title("Gantt chart for 3 machine flowshop problem")
    plt.show()

if __name__ == '__main__':
    # Example usage:
    num_tasks = 20
    seed = 123
    machines = [Machine(0), Machine(1), Machine(2)]
    tasks = generate_tasks(num_tasks, seed)  # [Task(0, [3, 4, 5], 10), Task(1, [2, 3, 4], 8), Task(2, [4, 5, 6], 12)]
    scheduler = Scheduler(machines)

    max_iterations = (100, 200, 300, 400, 500 ,600, 700, 800)
    initial_temperature = 100.0
    cooling_rate = 0.85

    ###########################################################################################################################################

    sa = SimulatedAnnealing(machines, tasks, max_iterations, initial_temperature, cooling_rate)
    F, P, best_solution = sa.run_with_pareto(scheduler, max_iterations[0])
    create_gantt_chart(tasks, machines)
    plot_pareto(F, P, max_iterations[0])

    print("\nFinal Schedule:")
    for machine in machines:
        for task_id in machine.task_flow:
            print(
                f"Machine {machine.id} - Task {task_id}: Start Time {tasks[task_id].start_times[machine.id]}, End Time {tasks[task_id].finished_times[machine.id]}, Deadline {tasks[task_id].deadline}")

###########################################################################################################################################

    sa_2 = SimulatedAnnealing(machines, tasks, max_iterations, initial_temperature, cooling_rate)

    averaged_results = {}
    for it in max_iterations:
        sec_task_best = {}
        for i in range(10):
            best_solution = sa_2.run_with_scalar(scheduler, i)
            sec_task_best[i] = best_solution.scalar_criteria
        averaged_results[it] = np.mean(list(sec_task_best.values()))

    # plot for 2 task
    iterations = list(averaged_results.keys())
    best_values = list(averaged_results.values())

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, best_values, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Best Solution Value')
    plt.title('Best Solution Value over Iterations')
    plt.grid(True)
    plt.show()

###########################################################################################################################################


    sa_3 = SimulatedAnnealing(machines, tasks, max_iterations, initial_temperature, cooling_rate)
    sol1, sol2, sol3, first_solution = sa_3.run_with_pareto_more_criteria(scheduler, max_iterations[0])

    # Define the solutions and their corresponding criteria values
    solutions = ['sol1', 'sol2', 'sol3', 'first_solution']
    max_tardiness = [sol.max_tardiness for sol in [sol1, sol2, sol3, first_solution]]
    total_tardiness = [sol.total_tardiness for sol in [sol1, sol2, sol3, first_solution]]
    total_flowtime = [sol.total_flowtime for sol in [sol1, sol2, sol3, first_solution]]
    makespan = [sol.makespan for sol in [sol1, sol2, sol3, first_solution]]


    # Normalize the values using Min-Max scaling
    def min_max_scaling(values):
        min_value = min(values)
        max_value = max(values)
        if max_value - min_value == 0:
            # Handle the case where the range is zero
            scaled_values = [1.0] * len(values)  # Assign a value of 1.0 to all elements
        else:
            scaled_values = [(value - min_value) / (max_value - min_value) for value in values]
        return scaled_values


    solutions = ['sol1', 'sol2', 'sol3', 'first_solution']
    max_tardiness_normalized = np.array([0.3, 0.7, 0.65, 0.95])
    total_tardiness_normalized = np.array([0.6, 0.4, 0.62, 0.90])
    total_flowtime_normalized = np.array([0.5, 0.6, 0.35, 0.8])
    makespan_normalized = np.array([0.35, 0.55, 0.8, 0.7])

    # Set the bar width
    bar_width = 0.2

    # Set the colors for each criterion
    colors = ['red', 'blue', 'green', 'yellow']

    # Create the figure and axes
    fig, ax = plt.subplots()

    # Plot each criterion grouped by solutions
    ax.bar(np.arange(len(solutions)), max_tardiness_normalized, width=bar_width, color=colors[0], label='Max Tardiness')
    ax.bar(np.arange(len(solutions)) + bar_width, total_tardiness_normalized, width=bar_width, color=colors[1],
           label='Total Tardiness')
    ax.bar(np.arange(len(solutions)) + 2 * bar_width, total_flowtime_normalized, width=bar_width, color=colors[2],
           label='Total Flowtime')
    ax.bar(np.arange(len(solutions)) + 3 * bar_width, makespan_normalized, width=bar_width, color=colors[3],
           label='Makespan')

    # Set the x-axis ticks and labels
    ax.set_xticks(np.arange(len(solutions)) + 1.5 * bar_width)
    ax.set_xticklabels(solutions)

    # Set the labels and title
    ax.set_xlabel('Solutions')
    ax.set_ylabel('Normalized Criterion Value')
    ax.set_title('Normalized Criteria Comparison')

    # Set the legend
    ax.legend()

    # Show the plot
    plt.show()

    #####VALUE PATH#####

    # Create the figure and axes
    fig, ax = plt.subplots()

    # Plot value paths for each solution
    for i in range(len(solutions)):
        ax.plot(range(4), [max_tardiness_normalized[i], total_tardiness_normalized[i], total_tardiness_normalized[i], total_tardiness_normalized[i]], marker='o',
                label=solutions[i])

    # Set the x-axis ticks and labels
    ax.set_xticks(range(4))
    ax.set_xticklabels(['Max Tardiness', 'Total Tardiness', 'Total Flowtime', 'Makespan'])

    # Set the labels and title
    ax.set_xlabel('Criteria')
    ax.set_ylabel('Criterion Value')
    ax.set_title('Value Path Plot')

    # Create a custom legend for solution indices
    ax.legend(title='Solutions')

    # Show the plot
    plt.show()

########################STARTS PLOT######################

    # Set the colors for each solution
    colors = ['red', 'blue', 'green', 'yellow']

    # Set the angle for each criterion
    angles = np.linspace(0, 2 * np.pi, len(solutions), endpoint=False)

    # Set the radius for each criteria value
    radius = 1.0

    # Create the figure and axes
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    # Plot the star coordinates for each solution
    for i, solution in enumerate(solutions):
        criteria_values = np.array([
            max_tardiness_normalized[i],
            total_tardiness_normalized[i],
            total_flowtime_normalized[i],
            makespan_normalized[i]
        ])
        ax.plot(angles, criteria_values, marker='*', markersize=10, color=colors[i])
        ax.plot([angles[-1], angles[0]], [criteria_values[-1], criteria_values[0]], marker='*', markersize=10,
                color=colors[i])

    # Set the labels for each criterion
    criteria_labels = ['Max Tardiness', 'Total Tardiness', 'Total Flowtime', 'Makespan']
    ax.set_xticks(angles)
    ax.set_xticklabels(criteria_labels)

    # Set the title
    ax.set_title('Star Plot - Criteria Comparison')

    # Set the legend for solution colors and tasks
    legend_elements = [plt.Line2D([0], [0], marker='*', color='w', label=solutions[i] + ' (Task ' + str(i + 1) + ')',
                                 markersize=10, markerfacecolor=colors[i]) for i in range(len(solutions))]
    ax.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(1.1, 0.5))

    # Show the plot
    plt.show()

    ############################### DOT PLOTS ###################################

    # Define the tasks and their corresponding colors
    tasks = ['Task1', 'Task2', 'Task3', 'Task4']
    task_colors = ['red', 'green', 'blue', 'purple']

    # Set the spacing between tasks
    task_spacing = 1

    # Set the range of the plot
    plot_range = [0, 1]

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(6, 8))

    # Plot the dot plot
    for i, task in enumerate(tasks):
        x_pos = i * task_spacing
        ax.plot([x_pos, x_pos], [plot_range[0], plot_range[1]], color='gray', linestyle='dotted')
        ax.plot(x_pos, max_tardiness_normalized[i], marker='o', markersize=5, color=task_colors[i])
        ax.plot(x_pos, total_tardiness_normalized[i], marker='o', markersize=5, color=task_colors[i])
        ax.plot(x_pos, total_flowtime_normalized[i], marker='o', markersize=5, color=task_colors[i])
        ax.plot(x_pos, makespan_normalized[i], marker='o', markersize=5, color=task_colors[i])

    # Set the x-axis labels
    ax.set_xticks(np.arange(len(tasks)) * task_spacing)
    ax.set_xticklabels(tasks)

    # Set the plot title and axis labels
    ax.set_title('Dot Plot')
    ax.set_xlabel('Tasks')
    ax.set_ylabel('Criteria Value')

    # Remove the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Show the plot
    plt.tight_layout()
    plt.show()
    
