from cmath import inf
from random import choices
from secrets import choice
from typing import Callable, List, Tuple

import numpy as np
import sympy as sym

from .functions import add, cos, div, mul, rand_f, sin, sub
from .individual import Individual
from multiprocessing import Pool

class GP:
    def __init__(self, data: np.ndarray, popsize: int, mutation_rate: float, min_depth: int, max_depth: int, tournament_size: int) -> None:
        self.functions: List[Callable] = [add, div, sub, cos, mul, sin]
        self.terminals: List = [sym.Symbol(f"x_{i}") for i in range(len(data[0])-1)] + [rand_f(1, 10)]
        self.data = data
        self.popsize = popsize
        self.pop: List[Tuple[float, Individual]] = []
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.mutation_rate = mutation_rate
        self.torunament_size = tournament_size
        self.divisor = np.sum((self.data[:, -1] - np.mean(self.data[:, -1]))**2)
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

    def selection(self):
        offspring = [min(self.pop, key=self.get_fitness)]
        while len(offspring)/self.popsize < 1 - self.mutation_rate:
            tourn1 = self.tournament(choices(self.pop, k=self.torunament_size))
            tourn2 = self.tournament(choices(self.pop, k=self.torunament_size))
            of1, of2 = tourn1[1].crossover(tourn2[1], self.max_depth, self.terminals)
            of1 = (self.fitness(of1), of1)
            of2 = (self.fitness(of2), of2)
            offspring.append(min([tourn1, tourn2, of1, of2], key=self.get_fitness))

        while len(offspring) < self.popsize:
            old = choice(self.pop)
            new = old[1].copy()
            new.mutate(self.max_depth, self.functions, self.terminals)
            new = (self.fitness(new), new)
            offspring.append(min([old, new], key=self.get_fitness))

        self.pop = offspring


    def fitness(self, ind: Individual) -> float:
        exp = ind.get_func()
        total = 0

        if isinstance(exp, (float, int)):
            for line in self.data:
                y = line[-1]
                ev = exp
                total += (y - ev)**2

        else:
            free_symbols = sorted(list(exp.free_symbols), key=lambda s: s.name)
            try:
                lamb = sym.lambdify(free_symbols, exp)
            except:
                return inf
            free_symbols = [int(s.name[2:]) for s in free_symbols]
            for line in self.data:
                y = line[-1]
                ev = lamb(*line[free_symbols])
                total += (y - ev)**2

        return np.sqrt(total/self.divisor)



    def start(self):
        self._gen_pop()

        i = 0
        while i < 100:
            print(i)
            print(min(map(self.get_fitness, self.pop)))
            i += 1
            self.selection()

        return min(self.pop, key=self.get_fitness)
