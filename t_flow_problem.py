import copy
import math
import random
import matplotlib.pyplot as plt

class Task:
    def __init__(self, id, durations, deadline):
        self.id = id
        self.durations = durations
        self.deadline = deadline
        self.finished_times = [-1, -1, -1]
        self.start_times = [-1, -1, -1]

    def execute(self, machine_id, start_time):
        self.start_times[machine_id] = start_time
        self.finished_times[machine_id] = start_time + self.durations[machine_id]

    def to_string(self, machine_id):
        return f" Id: {self.id}, duration {self.durations[machine_id]}, start: {self.start_times[machine_id]}, end {self.finished_times[machine_id]}"

    def set_initial_state(self):
        self.finished_times = [-1, -1, -1]
        self.start_times = [-1, -1, -1]

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
            if self.id == 0 or task.finished_times[self.id - 1] != -1:
                return True
        return False

    def execute_task(self, task, current_time):
        if self.can_be_executed(task):
            self.current_task = task
            task.execute(self.id, current_time)
            self.current_task = None




class Solution:
    def __init__(self, flow_schedule, machines):
        self.flow_schedule = flow_schedule
        self.machines = machines
        self.makespan = -1
        self.total_flowtime = -1

    def execute_schedule(self):
        current_time = 0
        for machine in self.machines:
            for task in machine.task_flow:
                start_time = max(task.finished_times[machine.id - 1], current_time) if machine.id > 0 else current_time
                machine.execute_task(task, start_time)
                current_time = task.finished_times[machine.id]
                print(f"Machine {machine.id} - task {task.id} Start time {task.start_times[machine.id]}, End time {task.finished_times[machine.id]}")
            machine.update_finished_time(current_time)
        self.set_criteria_values()
        print(f" FLOWTIME - {self.total_flowtime} MAKESPAN - {self.makespan}")

    def copy(self):
        return copy.deepcopy(self)
    def calculate_makespan(self):
        return max(task.finished_times[-1] for task  in self.machines[-1].task_flow)

    def calculate_total_flowtime(self):
        return sum(task.finished_times[-1] for task in self.machines[-1].task_flow)

    def set_criteria_values(self):
        self.makespan = self.calculate_makespan()
        self.total_flowtime = self.calculate_total_flowtime()

    def is_dominating(self, b):
        makespan_a = self.makespan
        makespan_b = b.makespan
        total_flowtime_a = self.total_flowtime
        total_flowtime_b = b.total_flowtime

        if makespan_a <= makespan_b and total_flowtime_a <= total_flowtime_b:
            return True
        return False


class Scheduler:

    def __init__(self, machines):
        self.machines = machines

    def generate_neighbor_schedule(self, machines):
        new_solution = []
        for mach in machines:
            machine_schedule = mach.task_flow.copy()
            self.swap(machine_schedule)
            mach.set_task_flow(machine_schedule)
            new_solution.append(machine_schedule)

        s = Solution(new_solution, machines)
        s.execute_schedule()
        return s

    def generate_random_schedule(self, tasks, machines):
        new_solution = []
        for mach in machines:
            shuffled_tasks = tasks.copy()  # Tworzenie kopii listy tasks
            random.shuffle(shuffled_tasks)  # Przetasowanie listy
            mach.set_task_flow(shuffled_tasks)
            new_solution.append(shuffled_tasks)

        s = Solution(new_solution, machines)
        s.execute_schedule()
        return s


    def swap(self, old_schedule):
        new_schedule = []
        for el in old_schedule:
            new_schedule.append(el.copy())
        self.swap_random_elements(new_schedule)
        return new_schedule

    def swap_random_elements(self, array):
        index1 = random.randint(0, len(array) - 1)
        index2 = random.randint(0, len(array) - 1)
        array[index1], array[index2] = array[index2], array[index1]



class SimulatedAnnealing:

    def __init__(self, machines, tasks, max_iterations, initial_temperature, cooling_rate):
        self.machines = machines
        self.tasks = tasks
        self.max_iterations = max_iterations
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate

    def run(self, scheduler, iter):
        self.max_iterations = iter
        P = []
        current_solution = scheduler.generate_random_schedule(self.tasks, self.machines)
        best_solution = current_solution
        P.append(best_solution)
        temperature = self.initial_temperature

        for iteration in range(self.max_iterations):
            neighbor_solution = scheduler.generate_neighbor_schedule(self.machines)
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


        return F, P, best_solution.flow_schedule

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

def create_gantt_chart(flow_schedule, machines, tasks):
    fig, ax = plt.subplots()
    # Ustalamy osie i etykiety
    ax.set_xlabel("Time")
    ax.set_ylabel("Tasks")
    ax.set_yticks(range(len(flow_schedule)))
    ax.set_yticklabels([f"Task {task.id}" for task in tasks])
    # Tworzymy słupki dla każdego zadania na odpowiednich maszynach
    for i, task in enumerate(tasks):
        for j in range(len(machines)):
            start_time = task.start_times[j]
            end_time = task.finished_times[j]
            duration = end_time - start_time
            # Rysujemy słupek dla danego zadania i maszyny
            ax.barh(i, duration, left=start_time, height=0.5, align='center', alpha=0.8, color=f"C{j+1}")

    plt.show()

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

    B = int(1/2 * A)
    A = int(1/6 * A)

    for task in tasks:
        deadline = random.randint(A, B)
        task.deadline = deadline

    return tasks

def plot_pareto(pareto_set, pareto_front):
    set_makespan_values = [solution.makespan for solution in pareto_set]
    set_total_flowtime_values = [solution.total_flowtime for solution in pareto_set]
    front_makespan_values = [solution.makespan for solution in pareto_front]
    front_total_flowtime_values = [solution.total_flowtime for solution in pareto_front]

    plt.scatter(set_makespan_values, set_total_flowtime_values, label='Pareto Set')
    plt.plot(front_makespan_values, front_total_flowtime_values, 'o-', label='Pareto Front')
    plt.xlabel('Makespan')
    plt.ylabel('Total Flowtime')
    plt.title('Pareto Set and Front')
    plt.legend()
    plt.show()

def plot_gantt_chart(flow_schedule, machines, tasks):
    fig, ax = plt.subplots()
    ax.set_xlabel("Time")
    ax.set_ylabel("Tasks")
    ax.set_yticks(range(len(flow_schedule)))
    ax.set_yticklabels([f"Task {task.id}" for task in tasks])

    for i, task in enumerate(flow_schedule):
        for j in range(len(machines)):
            start_time = task.start_times[j]
            end_time = task.finished_times[j]
            duration = end_time - start_time
            ax.barh(i, duration, left=start_time, height=0.5, align='center', alpha=0.8, color=f"C{j+1}")

    plt.show()

if __name__ == '__main__':
    # Example usage:
    num_tasks = 5
    seed = 123
    machines = [Machine(0), Machine(1), Machine(2)]
    tasks = generate_tasks(num_tasks, seed)  # [Task(0, [3, 4, 5], 10), Task(1, [2, 3, 4], 8), Task(2, [4, 5, 6], 12)]
    scheduler = Scheduler(machines)

    max_iterations = (5,10) # (100, 200, 400, 800, 1600)
    initial_temperature = 100.0
    cooling_rate = 0.95

    sa = SimulatedAnnealing(machines, tasks, max_iterations, initial_temperature, cooling_rate)

    pareto_fronts, pareto_sets = {}, {}
    for maxIter in max_iterations:
        pareto_front, pareto_set, final_schedule = sa.run(scheduler, maxIter)  # F -> pareto_front
        pareto_fronts[maxIter] = pareto_front
        pareto_sets[maxIter] = pareto_set
        plot_pareto(pareto_set, pareto_front)
        plot_gantt_chart(final_schedule, machines,tasks)


    print("\nFinal Schedule:")
    for machine in machines:
        for task in final_schedule:
            print(
                f"Machine {machine.id} - Task {task.id}: Start Time {task.start_times[machine.id]}, End Time {task.finished_times[machine.id]}")

    print(f" FLOWTIME - {final_schedule.total_flowtime} MAKESPAN - {final_schedule.makespan}")