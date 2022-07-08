#!/usr/bin/env python3
import typing
import numpy as np


class Multilayer:

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
