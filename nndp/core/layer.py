#!/usr/bin/env python3
import numpy as np
from nndp.math.function import Function


class Dense:

    def __init__(
        self,
        in_size: int,
        width: int,
        activation: Function,
        name: str = "N/D"
    ):
        if in_size <= 0 or width <= 0:
            attr = "in_size" if in_size <= 0 else "width"
            raise ValueError(f"{attr} value must be greater than zero")
        if isinstance(type(activation), Function):
            raise ValueError(
                f"Expected type {Function} for activation, received {type(activation)}."
            )
        self.name = name
        self.in_data = None
        self.out_data = None
        self.in_size = in_size
        self.width = width
        self.activation = activation.value
        self.weights = np.random.randn(width, in_size)
        self.biases = np.random.randn(width, 1)

    def feed_forward(self, in_data: np.ndarray) -> np.ndarray:
        in_data = np.reshape(in_data, (2, 1))
        self.in_data = in_data
        self.out_data = self.activation["function"](
            np.dot(self.weights, in_data) + self.biases
        )
        return self.out_data

    def __str__(self) -> str:
        activation_name = self.activation["function"].__name__
        return f"Dense layer ({self.name}): " \
               f"[in_size: {self.in_size} - width: {self.width} - activation: {activation_name}]"
