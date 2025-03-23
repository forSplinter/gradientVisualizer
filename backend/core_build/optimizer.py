import numpy as np


class GradientDescend:
    def __init__(self, function, loss_function, lr=0.01, iterations=100):
        self.function = function
        self.loss_function = loss_function
        self.lr = lr
        self.iterations = iterations
        self.w, self.b = np.random.randn(), np.random.randn()
        self.loss_history = []

    def step(self):
        grad_w = (
            self.function.compute(self.w + 1e-5, self.b)
            - self.function.compute(self.w - 1e-5, self.b)
        ) / (2 * 1e-5)
        grad_b = (
            self.function.compute(self.w, self.b + 1e-5)
            - self.function.compute(self.w, self.b - 1e-5)
        ) / (2 * 1e-5)

        self.w -= self.lr * grad_w  # updating weight
        self.b -= self.lr * grad_b  # updating bias

        loss = self.loss_function.compute(self.function.compute(self.w, self.b), 0)
        self.loss_history.append(loss)

    def optimize(self):
        for _ in range(self.iterations):
            self.step()
        return [self.w, self.b], self.loss_history
