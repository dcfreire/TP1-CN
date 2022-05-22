import os
from concurrent.futures import ALL_COMPLETED, ProcessPoolExecutor, wait
from time import sleep

import numpy as np
import sympy as sym

from gp import GP, Individual
from gp.functions import *


def test_func(data, popsize, mutation_rate, tournament_size, phi, generations):
    g = GP(data, popsize, mutation_rate, 2, 7, tournament_size, generations, phi)
    var = g.start()[1]
    #print(g.fitness(var))
    print(var.get_func())
    #print(var.get_function_arr())
    return var

if __name__ == "__main__":
    with open('datasets/datasets/synth1/synth1-train.csv', 'r') as f:
        data = np.loadtxt(f, delimiter=",")
    with open('datasets/datasets/synth1/synth1-test.csv', 'r') as f:
        test = np.loadtxt(f, delimiter=",")

    g = GP(data, 200, 1, 2, 7, 3, 100, 5)
    testing_params = [
        {
            "data": data,
            "popsize": 100,
            "mutation_rate": 0.2,
            "tournament_size": 3,
            "phi": 5,
            "generations": 100
        },
        {
            "data": data,
            "popsize": 100,
            "mutation_rate": 0.2,
            "tournament_size": 3,
            "phi": 4,
            "generations": 100
        },
        {
            "data": data,
            "popsize": 200,
            "mutation_rate": 0.2,
            "tournament_size": 3,
            "phi": 5,
            "generations": 100
        },
        {
            "data": data,
            "popsize": 100,
            "mutation_rate": 0.2,
            "tournament_size": 3,
            "phi": 5,
            "generations": 200
        },
        {
            "data": data,
            "popsize": 200,
            "mutation_rate": 0.2,
            "tournament_size": 3,
            "phi": 5,
            "generations": 500
        },
    ]
    for params in testing_params:
        with ProcessPoolExecutor(max_workers=10) as executor:
            tasks = []
            for i in range(30):
                tasks.append(executor.submit(test_func, *params.values()))

            done, _ = wait(tasks, return_when=ALL_COMPLETED)
            sleep(5)
            r = []
            for task in done:
                print("================================================")
                res = task.result()
                r.append(g.fitness(res, data=test))
                print(res)
                print(g.fitness(res))
                print(g.fitness(res, data=test))
            print("MEAN:", np.mean(r))
    os.system('shutdown now')
"""
import numpy as np
import sympy as sym
from gp import Individual, GP
from gp.functions import *

if __name__ == "__main__":
    with open('datasets/datasets/synth1/synth1-train.csv', 'r') as f:
        data = np.loadtxt(f, delimiter=",")
    g = GP(data, 100, 0.1, 2, 7, 3)
    #var = g.start()
    #print(g.fitness(var))
    #print(var.get_func())
    #print(var.get_function_arr())
    #x = sym.Symbol('x')
    i0 = Individual(mul)
    i1 = Individual(sub)
    i2 = Individual(sin)
    i0.left = i1
    i0.right = i2
    i3 = Individual(sym.Symbol('x_1'))
    i4 = Individual(add)
    i1.left = i3
    i1.right = i4
    i5 = Individual(sym.Symbol('x_1'))
    i2.left = i5
    i6 = Individual(sym.Symbol('x_0'))
    i7 = Individual(sym.Symbol('x_0'))
    i4.left = i6
    i4.right = i7

    print(i0.get_depth())
    print(i0.get_function_arr())

    print("================================================")
    #print(g.fitness(var))

"""
