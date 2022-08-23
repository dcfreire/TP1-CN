from cmath import inf
from random import choices
from secrets import choice
from typing import Callable, List, Tuple

import numpy as np
import sympy as sym

from .functions import add, cos, div, mul, rand_f, sin, sub
from .individual import Individual
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED
import os

def protected_division(x, y):
    if y == 0:
        return 1
    return x/y

class GP:
    def __init__(self, data: np.ndarray, popsize: int, mutation_rate: float, min_depth: int, max_depth: int, tournament_size: int, generations: int, phi: float, n_elites: int) -> None:
        self.phi = phi
        self.functions: List[Callable] = [add, div, sub, mul, sin, cos]
        self.terminals: List = [sym.Symbol(f"x_{i}") for i in range(len(data[0])-1)] + [rand_f()]
        self.data = data
        self.generations = generations
        self.popsize = popsize
        self.pop: List[Tuple[float, Individual]] = []
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.max_mutation = mutation_rate
        self.mutation_rate = mutation_rate
        self.torunament_size = tournament_size
        self.divisor = np.sum((self.data[:, -1] - np.mean(self.data[:, -1]))**2)
        self.it = 0
        self.n_elites = n_elites
        print(self.divisor)

    def _gen_pop(self):
        for i in range(self.popsize):
            ind = Individual.random(
                    (i//(self.popsize/(self.max_depth - 1))) + self.min_depth,
                    "grow" if i % 2 else "full",
                    self.functions,
                    self.terminals
                )
            self.pop.append((self.fitness(ind), ind))

    def tournament(self, subpop: List[Tuple[float, Individual]]) -> Tuple[float, Individual]:
        return min(subpop, key=self.get_fitness)

    def get_fitness(self, ind: Tuple[float, Individual]) -> float:
        return ind[0]

    def get_weighted_fitness(self, ind: Tuple[float, Individual]) -> float:
        progress = self.it/self.generations
        div_p = len([1 for i, _ in self.pop if i == ind[0]])/self.popsize
        div = np.log(self.phi + div_p/progress)/np.log(self.phi)
        return ind[0] * div

    @staticmethod
    def get_result_from_future(future):
        return future.result()

    def get_elites(self):
        sorted_pop = sorted(self.pop, key=self.get_fitness)
        ret = []
        r_fits = []
        for ind in sorted_pop:
            if ind[0] not in r_fits:
                ret.append(ind)
                r_fits.append(ind[0])
            if len(ret) == self.n_elites:
                break

        return ret

    def selection(self):
        with ProcessPoolExecutor(max_workers=os.cpu_count()*3) as executor:
            tasks = []
            psize = self.popsize - self.n_elites
            while len(tasks)/psize < 1 - self.mutation_rate:
                tasks.append(executor.submit(self.crossover_worker))

            while len(tasks) < psize:
                tasks.append(executor.submit(self.mutation_worker))

            offspring, _ = wait(tasks, return_when=ALL_COMPLETED)
            offspring = list(map(self.get_result_from_future, offspring))
            offspring.extend(self.get_elites())



        self.pop = offspring

    def crossover_worker(self):
        tourn1 = self.tournament(choices(self.pop, k=self.torunament_size))
        tourn2 = self.tournament(choices(self.pop, k=self.torunament_size))
        of1, of2 = tourn1[1].crossover(tourn2[1], self.max_depth, self.terminals)
        of1 = (self.fitness(of1), of1)
        of2 = (self.fitness(of2), of2)
        return min([tourn1, tourn2, of1, of2], key=self.get_weighted_fitness)

    def mutation_worker(self):
        old = self.tournament(choices(self.pop, k=self.torunament_size))
        new = old[1].copy()
        new.mutate(self.max_depth, self.functions, self.terminals)
        new = (self.fitness(new), new)
        return min([old, new], key=self.get_weighted_fitness)


    def fitness(self, ind: Individual, data=None) -> float:
        exp = ind.get_func()
        total = 0
        data = self.data if data is None else data

        if isinstance(exp, (float, int)):
            for line in data:
                y = line[-1]
                ev = exp
                total += (y - ev)**2

        else:
            free_symbols = sorted(list(exp.free_symbols), key=lambda s: s.name)
            try:
                lamb = sym.lambdify(free_symbols, exp, modules={'protected_division': protected_division})
            except:
                return inf
            free_symbols = [int(s.name[2:]) for s in free_symbols]
            for line in data:
                y = line[-1]
                ev = lamb(*line[free_symbols])
                total += (y - ev)**2

        return np.sqrt(total/self.divisor)


    def start(self):
        self._gen_pop()

        while self.it < self.generations:
            print("IT", self.it)
            print("CURBEST", min(map(self.get_fitness, self.pop)))
            diversity = len(set(map(self.get_fitness, self.pop)))
            #self.mutation_rate = self.max_mutation * (1 - np.log10(1 + 9*diversity/self.popsize))
            print("MRATE:", self.mutation_rate)
            print("DIV:", diversity)

            self.it += 1
            self.selection()
        return min(self.pop, key=self.get_fitness)
