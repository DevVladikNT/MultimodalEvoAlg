import copy
import numpy as np
from matplotlib import pyplot as plt
from numpy import random

# The left boundary of the definition area x
left_limit_x = 0
# The right boundary of the definition area x
right_limit_x = 1
# Radius of the landscape change area
sigma = 0.1
# Founded solutions
solutions = []


# Input function
def function(x):
    weight = 1
    for solution in solutions:
        if np.abs(solution - x) < sigma:
            weight = np.abs(solution - x) / sigma
    return np.sin(5 * np.pi * (x ** 0.75 - 0.05)) ** 6 * weight


class Individual(object):

    def __init__(self, mutation_param):
        self.mutation_param = mutation_param
        self.x = 0
        self.func = None  # Function value of this individual
        self.weight = None  # Weight (the probability of getting an individual into the next generation)
        self.x = random.uniform(left_limit_x, right_limit_x)
        self.update_func()

    # Updating the function value for an individual
    def update_func(self):
        self.func = function(self.x)

    # Getting individual parameters
    def get_params(self):
        return {'x': self.x, 'func': self.func, 'weight': self.weight}

    def set_x(self, new_x):
        self.x = new_x
        self.update_func()

    def mutate(self, current_generation, total_generation):
        def delta(individual, cur_gen, total_gen, y):
            return y * (1 - random.random() ** (1 - cur_gen / total_gen) ** individual.mutation_param)

        mode = random.randint(0, 2)
        if mode == 0:
            mutated = self.x + delta(self, current_generation, total_generation,
                                     right_limit_x - self.x)
        else:
            mutated = self.x - delta(self, current_generation, total_generation,
                                     self.x - left_limit_x)
        self.x = mutated
        self.update_func()

    def set_weight(self, min_value, sum_individual):
        if not (sum_individual == 0):
            self.weight = np.abs(self.func - min_value) / sum_individual


class Population(object):

    def __init__(self, population_count, crossing_over_probability, mutation_probability, crossing_over_param,
                 mutation_param):
        self.individuals_list = []
        self.population_count = population_count  # Population size
        self.crossing_over_probability = crossing_over_probability
        self.mutation_probability = mutation_probability
        self.crossing_over_param = crossing_over_param
        self.mutation_param = mutation_param
        self.init_population()

    # Individuals init in this population
    def init_population(self):
        for _ in range(self.population_count):
            self.individuals_list.append(Individual(self.mutation_param))

    def sort_population(self):
        # Sorting by increasing weight (probability of getting an individual into the next generation)
        self.individuals_list.sort(key=lambda individual: individual.get_params()['weight'])

    # Getting x values and functions for all individuals of a given generation
    def get_x_func(self):
        x_arr = []
        func = []
        for individual in self.individuals_list:
            x = individual.get_params()['x']
            x_arr.append(x)
            func.append(individual.get_params()['func'])
        return x_arr, func

    # Search for the module of the maximum value of the function among all individuals
    def find_max(self):
        max_individual = self.individuals_list[0].get_params()['func']
        for individual in self.individuals_list:
            if individual.get_params()['func'] > max_individual:
                max_individual = individual.get_params()['func']
        return np.abs(max_individual)

    # Search for the module of the minimum value of the function among all individuals
    def find_min(self):
        min_individual = self.individuals_list[0].get_params()['func']
        for individual in self.individuals_list:
            if individual.get_params()['func'] < min_individual:
                min_individual = individual.get_params()['func']
        return np.abs(min_individual)

    # Search for the sum of all function values reduced by the modulus
    # of the minimum function value among all individuals
    def find_sum(self, min_individual):
        sum_individual = 0
        for individual in self.individuals_list:
            sum_individual += individual.func - min_individual
        return np.abs(sum_individual)

    def evolution(self, cur_gen, total_gen):
        self.reproduction()
        self.crossing_over_blx()
        self.mutation(cur_gen, total_gen)

    def reproduction(self):
        # Updated list of individuals
        individuals_list_new = []

        max_individual = self.find_max()
        min_individual = self.find_min()
        # The sum of all function values, reduced by the modulus of the maximum function value, among all individuals
        sum_individual = self.find_sum(min_individual)

        for individual in self.individuals_list:
            individual.set_weight(min_individual, sum_individual)
        self.sort_population()
        # Weight reduction (приведение веса)
        for i in range(1, self.population_count):
            self.individuals_list[i].weight += self.individuals_list[i - 1].weight

        # Elitism
        individuals_list_new.append(copy.copy(self.individuals_list[-1]))
        for i in range(self.population_count - 1):
            r = random.random()
            for individual in self.individuals_list:
                # If the weight of the individual is sufficient
                if r <= individual.weight:
                    individuals_list_new.append(copy.copy(individual))
                    break

        if not (len(individuals_list_new) < self.population_count):
            # Updating the list of individuals
            self.individuals_list = individuals_list_new
            self.sort_population()

    def crossing_over_blx(self):
        # Updated list of individuals
        individuals_list_new = []
        # As long as there are pairs to cross
        while len(individuals_list_new) < self.population_count:
            count = 0
            parent_1 = []
            parent_2 = []
            while count < self.population_count:
                # We choose two parents at random
                parent_1 = self.individuals_list.pop(random.randint(0, len(self.individuals_list) - 1))
                parent_2 = self.individuals_list.pop(random.randint(0, abs(len(self.individuals_list) - 1)))
                self.individuals_list.append(parent_1)
                self.individuals_list.append(parent_2)
                count += 1
            # If random says that the crossing should be
            if count < self.population_count and random.random() <= self.crossing_over_probability:
                x_p1 = parent_1.get_params()['x']
                x_p2 = parent_2.get_params()['x']
                x_min = min(x_p1, x_p2)
                x_max = max(x_p1, x_p2)
                i = x_max - x_min
                x_new = -100
                while x_new < left_limit_x or x_new > right_limit_x:
                    x_new = random.uniform(x_min - i * self.crossing_over_param,
                                           x_max + i * self.crossing_over_param)
                child = copy.deepcopy(parent_1)
                child.set_x(x_new)
                individuals_list_new.append(child)
            # Otherwise, adding parents to the updated list of individuals
            else:
                child1 = copy.deepcopy(parent_1)
                child2 = copy.deepcopy(parent_2)
                individuals_list_new.extend([child1, child2])
        # If 2 parents are finally added and the population has become
        # more than 100 individuals, then we remove the second parent
        if len(individuals_list_new) == self.population_count + 1:
            individuals_list_new.pop(self.population_count)
        # Updating the list of individuals
        self.individuals_list = individuals_list_new

    def mutation(self, cur_gen, total_gen):
        for individual in self.individuals_list:
            if random.random() <= self.mutation_probability:
                individual.mutate(cur_gen, total_gen)

    # Search for the best individual in the population
    def find_best(self):
        maximum = -100
        x = []
        for individual in self.individuals_list:
            if individual.get_params()['func'] > maximum:
                maximum = individual.get_params()['func']
                x = individual.get_params()['x']
        return x, maximum


def main():
    population_count = 100
    crossing_over_probability = 0.5
    mutation_probability = 0.001
    crossing_over_param = 0.5
    mutation_param = 2
    total_generation = 50

    fig = plt.figure()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    initial_func = []
    values = []
    for _ in range(5):
        # Current population number
        current_generation = 0
        # Initialization of the initial population
        population = Population(population_count, crossing_over_probability, mutation_probability, crossing_over_param,
                                mutation_param)
        while current_generation < total_generation:
            x = np.arange(0, 1, 0.01)
            y = []
            for xi in x:
                y.append(function(xi))
            initial_func.append([x, y])

            x_values, func_values = population.get_x_func()
            values.append([x_values, func_values])

            population.evolution(current_generation, total_generation)
            current_generation += 1
        # The best individual as a result of evolution
        x_min, minimum = population.find_best()
        solutions.append(x_min)
        # Output the y value of the best individual to the console
        print(f'f({x_min:.3f}) = {minimum:.6f}')

    def animate(i):
        if i >= len(values):
            return
        fig.clear()
        plt.plot(initial_func[i][0], initial_func[i][1])
        plt.plot(values[i][0], values[i][1], 'ro')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title(f'Generation {i}')

    import matplotlib.animation as ani
    animator = ani.FuncAnimation(fig, animate, interval=100)
    plt.show()


if __name__ == '__main__':
    main()
