import numpy as np
import sympy as sym
from gp import Individual, GP
from gp.functions import *

if __name__ == "__main__":
    with open('datasets/datasets/synth1/synth1-train.csv', 'r') as f:
        data = np.loadtxt(f, delimiter=",")
    g = GP(data, 100, 0.1, 2, 7, 3)
    var = g.start()[1]
    #print(g.fitness(var))
    print(var.get_func())
    #print(var.get_function_arr())
    x = sym.Symbol('x')


    print("================================================")
    print(g.fitness(var))


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