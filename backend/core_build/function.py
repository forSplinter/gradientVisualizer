import numpy as np
from fastapi import Fastapi


class MathFunction:
    def __init__(self, name, function):
        self.name = name
        self.function = function

    def compute(self, x, y):
        return self.function(x, y)


class AllFunction:
    def __init__(self):
        self.function = {}

    def register(self, name, function):
        self.function[name] = MathFunction(name, function)

    def get_f(self, name):
        return self.functions.get(name, None)

    def list_f(self):
        return list(self.fucntions.keys())


f_register = AllFunction()

f_register.register("quadratic", lambda x, y: x**2 + y**2)
f_register.register("exp", lambda x, y: np.exp(x) + np.exp(y))
f_register.register("saddle", lambda x, y: x**2 - y**2)
f_register.register("abs", lambda x, y: np.abs(x) + np.abs(y))
f_register.register("sin", lambda x, y: np.sin(x) * np.cos(y))
f_register.register("gaussian", lambda x, y: np.exp(-(x**2 + y**2)))
