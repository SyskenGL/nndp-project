#!/usr/bin/env python3
import typing
import numpy as np


class Layer:

    def __init__(self, name: str, in_size: int, width: int, activation: typing.Callable):
        self.name = name
        self.in_size = in_size
        self.width = width
        self.activation = activation
        self.weight = np.random.randn(width, in_size)
        self.bias = np.random.randn(width)

    def compute(self, layer_in: np.ndarray) -> np.ndarray:
        layer_out = np.dot(self.weight, layer_in) + self.bias
        return self.activation(layer_out)


class MultiLayer:

    def __init__(self, layers: list[Layer]):
        self.layers = layers
        self.in_size = layers[0].in_size
        self.out_size = layers[-1].width
        self.depth = len(layers)
        self.dimension = sum([layer.width for layer in layers])
        self.width = max([layer.width for layer in layers])

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.compute(out)
        return out

    def get_hidden_layers(self):
        return self.layers[:-1]
