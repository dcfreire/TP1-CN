from random import randint

import sympy as sym


def cos(x):
    return sym.cos(x)

def sin(x):
    return sym.sin(x)

def mul(x, y):
    return x*y

def add(x, y):
    return x + y

def sub(x, y):
    return x - y

def div(x, y):
    if y == 0:
        return 1
    return x/y

def rand_f(a, b):
    def rand():
        return randint(a, b)
    return rand
