#!/usr/bin/env python3
from __future__ import annotations
import uuid
import numpy as np
from tabulate import tabulate
from nndp.math.functions import Activation
from nndp.utils.decorators import require_built
from nndp.utils.decorators import require_not_built


class Layer:

    def __init__(
        self,
        width: int,
        activation: Activation = Activation.IDENTITY,
        name: str = None,
    ):
        if width <= 0:
            raise ValueError(
                f"expected width value greater than zero, received {width}."
            )
        self._name = (
            name if name is not None
            else f"{self.__class__.__name__}_{str(uuid.uuid4())[:8]}"
        )
        self._width = width
        self._activation = activation
        self._in_size = None
        self._in_data = None
        self._in_weighted = None
        self._out_data = None
        self._weights = None
        self._biases = None

    def is_built(self) -> bool:
        return (
            self._weights is not None and
            self._biases is not None and
            self._in_size is not None
        )

    @require_not_built
    def build(
        self,
        in_size: int,
        weights: np.ndarray = None,
        biases: np.ndarray = None
    ) -> None:
        raise NotImplementedError

    @require_built
    def forward_propagation(self, in_data: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @require_built
    def backward_propagation(self, expected: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @require_built
    def update(self, learning_rate: float = 0.001) -> None:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self._name

    @property
    def width(self) -> int:
        return self._width

    @property
    def activation(self) -> Activation:
        return self._activation

    @property
    def in_size(self) -> int:
        return self._in_size if self._in_size is not None else 0

    @property
    def in_data(self) -> np.ndarray:
        return self._in_data

    @property
    def in_weighted(self) -> np.ndarray:
        return self._in_weighted

    @property
    def out_data(self) -> np.ndarray:
        return self._out_data

    def __str__(self) -> str:
        return str(tabulate([[
                self.__class__.__name__,
                self.name,
                self.in_size if self.in_size else "-",
                self.width,
                self.activation.function().__name__,
                self.is_built()
            ]],
            headers=[
                "\033[1m TYPE \033[0m",
                "\033[1m NAME \033[0m",
                "\033[1m IN_SIZE \033[0m",
                "\033[1m WIDTH \033[0m",
                "\033[1m ACTIVATION \033[0m",
                "\033[1m BUILT \033[0m"
            ],
            tablefmt="fancy_grid",
            colalign=["center"] * 6
        ))


class Dense(Layer):

    def __init__(
        self,
        width: int,
        activation: Activation = Activation.IDENTITY,
        name: str = None,
    ):
        super().__init__(width, activation, name)
        self._delta = None
        self._accumulated_weights_delta = None
        self._accumulated_biases_delta = None

    @require_not_built
    def build(
        self,
        in_size: int,
        weights: np.ndarray = None,
        biases: np.ndarray = None
    ) -> None:
        if in_size <= 0:
            raise ValueError(
                f"expected in_size value greater than zero, received {in_size}."
            )
        if weights is not None and (self._width, self._in_size) != weights.shape:
            raise ValueError(
                f"provided weights shape {weights.shape} "
                f"is not aligned with expected shape {self._weights.shape}."
            )
        if biases is not None and (self._width, self._in_size) != biases.shape:
            raise ValueError(
                f"provided biases shape {biases.shape} "
                f"is not aligned with expected shape {self.biases.shape}."
            )
        self._in_size = in_size
        self._weights = (
            weights if weights is not None
            else np.random.randn(self._width, self._in_size)
        )
        self._biases = (
            biases if biases is not None
            else np.random.randn(self._width, 1)
        )
        self._accumulated_weights_delta = np.zeros(self._weights.shape)
        self._accumulated_biases_delta = np.zeros(self._biases.shape)

    @require_built
    def forward_propagation(self, in_data: np.ndarray) -> np.ndarray:
        in_data = np.reshape(in_data, (-1, 1))
        self._in_data = in_data
        self._in_weighted = self._weights @ in_data + self._biases
        self._out_data = self._activation.function()(self._in_weighted)
        return self._out_data

    @require_built
    def backward_propagation(self, delta: np.ndarray) -> np.ndarray:
        delta = np.reshape(delta, (-1, 1))
        self._delta = self._activation.prime()(self._in_weighted) * delta
        return self._weights.T @ delta

    @require_built
    def update(self, learning_rate: float = None) -> None:
        self._accumulated_weights_delta += (self._delta @ self._in_data.T)
        self._accumulated_biases_delta += self._delta
        if learning_rate is not None:
            self._weights -= learning_rate * self._accumulated_weights_delta
            self._biases -= learning_rate * self._accumulated_biases_delta
            self._accumulated_weights_delta = np.zeros(self._weights.shape)
            self._accumulated_biases_delta = np.zeros(self._biases.shape)
