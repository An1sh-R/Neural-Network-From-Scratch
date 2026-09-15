import numpy as np

class Model:
    def __init__(self,layers):
        self.layers = layers

    def forward(self,X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self,dZ):
        for layer in reversed(self.layers):
            dZ = layer.backward(dZ)
        return dZ

    def dense_layers(self):
        return [layer for layer in  self.layers if hasattr(layer,'params')]