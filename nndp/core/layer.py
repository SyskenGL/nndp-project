#!/usr/bin/env python3
import numpy as np
from nndp.math.function import Activation


class Layer:

    def __init__(self, name):
        self.name = name
        self.in_data = None
        self.out_data = None

    def feed_forward(self, in_data: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Dense(Layer):

    def __init__(
        self, in_size: int, width: int, activation: Activation, name: str = "N/D"
    ):
        super().__init__(name)
        if in_size <= 0 or width <= 0:
            attr = "in_size" if in_size <= 0 else "width"
            raise ValueError(f"{attr} value must be greater than zero.")
        if isinstance(type(activation), Activation):
            raise ValueError(
                f"expected type {Activation} for activation, received {type(activation)}."
            )
        self.name = name
        self.in_size = in_size
        self.width = width
        self.activation = activation
        self.weights = np.random.randn(width, in_size)
        self.biases = np.random.randn(width, 1)

    def feed_forward(self, in_data: np.ndarray) -> np.ndarray:
        in_data = np.reshape(in_data, (self.in_size, 1))
        self.in_data = in_data
        self.out_data = self.activation.function()(
            np.dot(self.weights, in_data) + self.biases
        )
        return self.out_data

    def __str__(self) -> str:
        return f"Dense layer ({self.name}): " \
               f"[in_size: {self.in_size} - width: {self.width} - " \
               f"activation: {self.activation.function().__name__}]"
