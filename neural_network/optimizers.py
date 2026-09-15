import numpy as np

class SGD:
    def __init__(self,learning_rate=0.01):
        self.learning_rate=learning_rate

    def update(self,params):
        for param, grad in params:
            param -= self.learning_rate * grad

    