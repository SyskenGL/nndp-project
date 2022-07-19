#!/usr/bin/env python3
from __future__ import annotations
import uuid
import numpy as np
from enum import Enum, auto
from tabulate import tabulate
from nndp.utils.decorators import require_built
from nndp.math.functions import Activation


class Category(Enum):

    HIDDEN = auto()
    OUTPUT = auto()


class Layer:

    def __init__(
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
        self._category = None
        self._in_data = None
        self._in_weighted = None
        self._out_data = None
        self._weights = None
        self._biases = None
        self._delta = None
        self._accumulated_weights_delta = None
        self._accumulated_biases_delta = None

    def build(
        self,
        category: Category = Category.HIDDEN,
        weights: np.ndarray = None,
        biases: np.ndarray = None
    ) -> Layer:
        raise NotImplementedError

    def is_built(self) -> bool:
        return self._weights is not None and self._biases is not None

    def forward_propagation(self, in_data: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward_propagation(self, expected: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def update(self, learning_rate: float = 0.001) -> None:
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
    def in_data(self) -> np.ndarray:
        return self._in_data

    @property
    def in_weighted(self) -> np.ndarray:
        return self._in_weighted

    @property
    def out_data(self) -> np.ndarray:
        return self._out_data

    @property
    def delta(self) -> np.ndarray:
        return self._delta

    def __str__(self) -> str:
        with np.printoptions(formatter={'float': '{: 0.5f}'.format}, suppress=True):
            return str(tabulate([[
                    self._category if self._category else "-",
                    self._name,
                    self._in_size,
                    self._width,
                    self._activation.function().__name__,
                ]],
                headers=[
                    "category", "name", "in_size", "width", "activation"
                ],
                tablefmt="fancy_grid",
                colalign=["center"]*5
            ))


class Dense(Layer):

    def __init__(
        self,
        in_size: int,
        width: int,
        activation: Activation = Activation.SIGMOID,
        name: str = None,
    ):
        super().__init__(in_size, width, activation, name)

    def build(
        self,
        category: Category = Category.HIDDEN,
        weights: np.ndarray = None,
        biases: np.ndarray = None
    ) -> Dense:
        if self.is_built():
            return self
        self._category = category
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
        self._accumulated_weights_delta = np.zeros((self._width, self._in_size))
        self._accumulated_biases_delta = np.zeros((self._width, 1))
        return self

    @require_built
    def forward_propagation(self, in_data: np.ndarray) -> np.ndarray:
        in_data = np.reshape(in_data, (self._in_size, 1))
        self._in_data = in_data
        self._in_weighted = self._weights @ in_data + self._biases
        self._out_data = self._activation.function()(self._in_weighted)
        return self._out_data

    @require_built
    def backward_propagation(self, delta: np.ndarray) -> np.ndarray:
        delta = np.reshape(delta, (self._width, 1))
        self._delta = self._activation.prime()(self._in_weighted) * delta
        return self._weights.T @ delta

    @require_built
    def update(self, learning_rate: float = 0.001) -> None:
        self._accumulated_weights_delta += (self._delta @ self._in_data.T)
        self._accumulated_biases_delta += self._delta
        self._weights -= learning_rate * self._accumulated_weights_delta
        self._biases -= learning_rate * self._accumulated_biases_delta
