from random import randint

import sympy as sym

class protected_division(sym.Function):
    @classmethod
    def eval(cls, x, y):
        if y.is_Number:
            if y.is_zero:
                return sym.S.One
            else:
                return x/y

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
    return protected_division(x, y)

def rand_f():
    return randint(2, 60)
