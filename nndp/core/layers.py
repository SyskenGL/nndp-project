#!/usr/bin/env python3
from __future__ import annotations
import uuid
import numpy as np
from tabulate import tabulate
from nndp.errors import NotBuiltLayerError
from nndp.math.functions import Activation


class Layer:

    def _init_(
        self,
        in_size: int,
        width: int,
        activation: Activation = Activation.IDENTITY,
        name: str = None,
    ):
        if in_size <= 0 or width <= 0:
            attribute = "in_size" if in_size <= 0 else "width"
            raise ValueError(f"{attribute} value must be greater than zero.")
        self._in_size = in_size
        self._width = width
        self._activation = activation
        self._name = (
            name if name else f"{self.__class__.__name__}_{str(uuid.uuid4())[:8]}"
        )
        self._in_data = None
        self._out_data = None
        self._delta = None
        self._weights = None
        self._biases = None

    def build(self, weights: np.ndarray = None, biases: np.ndarray = None) -> Layer:
        raise NotImplementedError

    def is_built(self):
        return self._weights is not None and self._biases is not None

    def forward_propagation(self, in_data: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward_propagation(self, expected: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def update(self, delta: np.ndarray, learning_rate: float = 0.001) -> None:
        raise NotImplementedError

    @property
    def in_size(self) -> int:
        return self._in_size

    @property
    def width(self) -> int:
        return self._width

    @property
    def activation(self) -> Activation:
        return self._activation

    @property
    def name(self) -> str:
        return self._name

    @property
    def out_data(self) -> np.ndarray:
        return self._out_data

    @property
    def delta(self) -> np.ndarray:
        return self._delta

    def __str__(self) -> str:
        return str(tabulate([[
                self._name,
                self._in_size,
                self._width,
                self._activation.function().__name__,
                self._in_data if self._in_data else "-",
                self._out_data if self._out_data else "-"
            ]],
            headers=["name", "in_size", "width", "activation", "in_data", "out_data"],
            tablefmt="fancy_grid",
            colalign=["center"]*6
        ))


class Dense(Layer):

    def __init__(
        self,
        in_size: int,
        width: int,
        activation: Activation = Activation.IDENTITY,
        name: str = None,
    ):
        super()._init_(in_size, width, activation, name)

    def build(self, weights: np.ndarray = None, biases: np.ndarray = None) -> Dense:
        if self.is_built():
            return self
        if weights is not None:
            if (self._width, self._in_size) != weights.shape:
                raise ValueError(
                    f"provided shape {weights.shape} is not aligned with expected shape {self._weights.shape}."
                )
            self._weights = weights
        else:
            self._weights = np.random.randn(self._width, self._in_size)
        if biases is not None:
            if (self._width, self._in_size) != biases.shape:
                raise ValueError(
                    f"provided shape {biases.shape} is not aligned with expected shape {self.biases.shape}."
                )
            self._biases = biases
        else:
            self._biases = np.random.randn(self._width, 1)
        return self

    @Layer.require_built
    def forward_propagation(self, in_data: np.ndarray) -> np.ndarray:
        if not self.is_built():
            raise NotBuiltLayerError("undefined weights and biases.")
        in_data = np.reshape(in_data, (self._in_size, 1))
        self._in_data = in_data
        self._out_data = self._activation.function()(
            np.dot(self._weights, in_data) + self._biases
        )
        return self._out_data

    @Layer.require_built
    def backward_propagation(self, expected: np.ndarray) -> np.ndarray:
        if not self.is_built():
            raise NotBuiltLayerError("attempt to operate on an non-built layer.")

    @Layer.require_built
    def update(self, delta: np.ndarray, learning_rate: float = 0.001) -> None:
        if not self.is_built():
            raise NotBuiltLayerError("attempt to operate on an non-built layer.")
