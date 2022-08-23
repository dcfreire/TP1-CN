import numpy as np
from fast_gp import GP
from concurrent.futures import ALL_COMPLETED, ProcessPoolExecutor, wait
import json
import time
import sympy as sym

from fast_gp.individual import Individual

with open("datasets/datasets/synth1/synth1-train.csv", "r") as f:
    data2 = np.loadtxt(f, delimiter=",")
with open("datasets/datasets/synth1/synth1-test.csv", "r") as f:
    test2 = np.loadtxt(f, delimiter=",")

with open("datasets/datasets/synth2/synth2-train.csv", "r") as f:
    data1 = np.loadtxt(f, delimiter=",")
with open("datasets/datasets/synth2/synth2-test.csv", "r") as f:
    test1 = np.loadtxt(f, delimiter=",")

with open("datasets/datasets/concrete/concrete-train.csv", "r") as f:
    data3 = np.loadtxt(f, delimiter=",")
with open("datasets/datasets/concrete/concrete-test.csv", "r") as f:
    test3 = np.loadtxt(f, delimiter=",")


def test_func(_, data, popsize, mutation_rate, tournament_size, phi, generations, test, elitist_operators):
    # print(f"Testing {name}")

    g = GP(data, popsize, mutation_rate, 2, 7, tournament_size, generations, phi, 5, elitist_operators)
    var = g.start(test)

    return var

"""testing_params = [ {
        "name": "tests/phi/1.5",
        "data": data2,
        "popsize": 500,
        "mutation_rate": 0.1,
        "tournament_size": 2,
        "phi": 1.5,
        "generations": 100,
        "test": test2,
        "elitist_operators": True,
},
{
        "name": "tests/phi/1.2",
        "data": data2,
        "popsize": 500,
        "mutation_rate": 0.1,
        "tournament_size": 2,
        "phi": 1.2,
        "generations": 100,
        "test": test2,
        "elitist_operators": True,
},
]"""

testing_params = [
    {
        "name": "tests/final/synth1",
        "data": data2,
        "popsize": 500,
        "mutation_rate": 0.1,
        "tournament_size": 2,
        "phi": 1.5,
        "generations": 100,
        "test": test2,
        "elitist_operators": True,
    },
    {
        "name": "tests/final/synth2",
        "data": data1,
        "popsize": 500,
        "mutation_rate": 0.1,
        "tournament_size": 2,
        "phi": 1.5,
        "generations": 100,
        "test": test1,
        "elitist_operators": True,
    },
    {
        "name": "tests/final/concrete",
        "data": data3,
        "popsize": 500,
        "mutation_rate": 0.1,
        "tournament_size": 2,
        "phi": 1.5,
        "generations": 100,
        "test": test3,
        "elitist_operators": True,
    },
]

g = GP(data2, 1, 1, 2, 7, 1, 1, 1, 10)
for params in testing_params:
    r = []

    with ProcessPoolExecutor(max_workers=30) as executor:
        tasks = []
        for i in range(30):
            tasks.append(executor.submit(test_func, *params.values()))

        done, _ = wait(tasks, return_when=ALL_COMPLETED)

        for task in done:
            print(int(time.time()))
            res = task.result()
            res[1]["func"] = str(res[0][1].get_func())
            r.append(res[1])
        executor.shutdown()
    with open("%s.json" % (params["name"]), "w") as f:
        json.dump(r, f)
