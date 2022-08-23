from random import randint

import numpy as np


def cos(x):
    return np.cos(x)


def sin(x):
    return np.sin(x)


def mul(x, y):
    return x * y


def add(x, y):
    return x + y


def sub(x, y):
    return x - y


def rand_f():
    return randint(2, 60)
