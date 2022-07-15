#!/usr/bin/env python3
from __future__ import annotations
import uuid
import numpy as np
from tabulate import tabulate
from nndp.errors import NotBuiltLayerError
from nndp.math.functions import Activation


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
        self.__in_size = in_size
        self.__width = width
        self.__activation = activation
        self.__name = (
            name if name else f"{self.__class__.__name__}_{str(uuid.uuid4())[:8]}"
        )
        self.__in_data = None
        self.__out_data = None
        self._weights = None
        self._biases = None

    def build(self, weights: np.ndarray = None, biases: np.ndarray = None) -> Layer:
        raise NotImplementedError

    def is_built(self):
        return self._weights is not None and self._biases is not None

    def activate(self, in_data: np.ndarray) -> np.ndarray:
        if not self.is_built():
            raise NotBuiltLayerError("undefined weights and biases.")
        in_data = np.reshape(in_data, (self.in_size, 1))
        self.__in_data = in_data
        self.__out_data = self.__activation.function()(
            np.dot(self._weights, in_data) + self._biases
        )
        return self.__out_data

    @property
    def in_size(self) -> int:
        return self.__in_size

    @property
    def width(self) -> int:
        return self.__width

    @property
    def activation(self) -> Activation:
        return self.__activation

    @property
    def name(self) -> str:
        return self.__name

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @property
    def biases(self) -> np.ndarray:
        return self._biases

    @property
    def in_data(self) -> np.ndarray:
        return self.__in_data

    @property
    def out_data(self) -> np.ndarray:
        return self.__out_data

    def __str__(self) -> str:
        return str(tabulate([[
                self.__name,
                self.__in_size,
                self.__width,
                self.__activation.function().__name__,
                self.__in_data if self.__in_data else "-",
                self.__out_data if self.__out_data else "-"
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
        super().__init__(in_size, width, activation, name)

    def build(self, weights: np.ndarray = None, biases: np.ndarray = None) -> Dense:
        if self.is_built():
            return self
        if weights is not None:
            if (self.width, self.in_size) != weights.shape:
                f"provided shape {weights.shape} is not aligned with expected shape {self._weights.shape}."
            self._weights = weights
        else:
            self._weights = np.random.randn(self.width, self.in_size)
        if biases is not None:
            if (self.width, self.in_size) != biases.shape:
                f"provided shape {biases.shape} is not aligned with expected shape {self._biases.shape}."
            self._biases = biases
        else:
            self._biases = np.random.randn(self.width, 1)
        return self
