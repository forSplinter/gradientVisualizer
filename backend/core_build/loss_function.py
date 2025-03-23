import numpy as np


class LossFunction:
    def __init__(self, name, function):
        self.name = name
        self.function = function

    def compute(self, y_true, y_pred):
        return self.function(y_true, y_pred)


class AllLost:
    def __init__(self):
        self.lost_function = {}

    def register(self, name, function):
        self.lost_function[name] = LossFunction(name, function)

    def get_lf(self, name):
        return self.lost_function.get(name, None)

    def list_lf(self):
        return list(self.lost_function.keys())


def cross_entropy(y_true, y_pred):  # TODO revoir l'entropy ici !!! pas bonne
    y_pred = 1 / (1 + np.exp(-y_pred))
    y_pred = np.clip(y_pred, 1e-9, 1.0 - 1e-9)

    y_true = np.array(y_true, dtype=np.float64)

    if y_true.ndim == 0:
        return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


l_register = AllLost()
l_register.register("MSE", lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2))
l_register.register("MAE", lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred)))
l_register.register("entropy", cross_entropy)
