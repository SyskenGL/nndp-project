#!/usr/bin/env python3
from __future__ import annotations
import uuid
import numpy as np
from copy import deepcopy
from typing import Optional
from tabulate import tabulate
from nndp.utils.functions import Activation
from nndp.utils.decorators import require_built
from nndp.utils.decorators import require_not_built


class Layer:

    def __init__(
        self,
        width: int,
        activation: Activation = Activation.IDENTITY,
        name: Optional[str] = None,
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
        self._weights_derivative = None
        self._biases_derivative = None
        self._copies = 0

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
        weights: Optional[np.ndarray] = None,
        biases: Optional[np.ndarray] = None
    ) -> None:
        raise NotImplementedError

    @require_built
    def forward_propagation(self, in_data: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @require_built
    def backward_propagation(self, expected: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @require_built
    def update(self, **kwargs) -> None:
        raise NotImplementedError

    @require_built
    def predict(self, in_data: np.ndarray) -> np.ndarray:
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

    @property
    def weights(self) -> np.ndarray:
        return np.copy(self._weights)

    @property
    def biases(self) -> np.ndarray:
        return np.copy(self._biases)

    @property
    def n_trainable(self) -> int:
        return (
            self._weights.size + self._biases.size
            if self.is_built() else 0
        )

    def __str__(self) -> str:
        return str(tabulate([[
                self.__class__.__name__,
                self._name,
                self._in_size if self._in_size else "-",
                self._width,
                self._activation.function().__name__,
                self.n_trainable if self.is_built() else "-",
                self.is_built()
            ]],
            headers=[
                "\033[1m TYPE \033[0m",
                "\033[1m NAME \033[0m",
                "\033[1m IN_SIZE \033[0m",
                "\033[1m WIDTH \033[0m",
                "\033[1m ACTIVATION \033[0m",
                "\033[1m TRAINABLE \033[0m",
                "\033[1m BUILT \033[0m"
            ],
            tablefmt="fancy_grid",
            colalign=["center"] * 7
        ))

    def __deepcopy__(self, memodict: Optional[dict] = None) -> Layer:
        memodict = {} if memodict is None else memodict
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memodict))
        self._copies += 1
        self._name += f"_CL{self._copies}"
        return result


class Dense(Layer):

    def __init__(
        self,
        width: int,
        activation: Activation = Activation.IDENTITY,
        name: Optional[str] = None,
    ):
        super().__init__(width, activation, name)

    @require_not_built
    def build(
        self,
        in_size: int,
        weights: Optional[np.ndarray] = None,
        biases: Optional[np.ndarray] = None
    ) -> None:
        if in_size <= 0:
            raise ValueError(
                f"expected in_size value greater than zero, received {in_size}."
            )
        self._in_size = in_size
        if weights is not None and (self._width, self._in_size) != weights.shape:
            raise ValueError(
                f"provided weights shape {weights.shape} "
                f"is not aligned with expected shape {(self._width, self._in_size)}."
            )
        if biases is not None and (self._width, 1) != biases.shape:
            raise ValueError(
                f"provided biases shape {biases.shape} "
                f"is not aligned with expected shape {(self._width, 1)}."
            )
        self._weights = (
            weights if weights is not None
            else .1 * np.random.randn(self._width, self._in_size)
        )
        self._biases = (
            biases if biases is not None
            else .1 * np.random.randn(self._width, 1)
        )
        self._weights_derivative = np.zeros(self._weights.shape)
        self._biases_derivative = np.zeros(self._biases.shape)

    @require_built
    def forward_propagation(self, in_data: np.ndarray) -> np.ndarray:
        self._in_data = in_data
        self._in_weighted = self._weights @ in_data + self._biases
        self._out_data = self._activation.function()(self._in_weighted)
        return self._out_data

    @require_built
    def backward_propagation(self, delta: np.ndarray) -> np.ndarray:
        delta = self._activation.prime()(self._in_weighted) * delta
        self._weights_derivative += (delta @ self._in_data.T)
        self._biases_derivative += delta
        return self._weights.T @ delta

    @require_built
    def predict(self, in_data: np.ndarray) -> np.ndarray:
        in_weighted = self._weights @ in_data + self._biases
        return self._activation.function()(in_weighted)

    @require_built
    def update(self, **kwargs) -> None:
        eta = kwargs.get("eta", 0.001)
        if not 0 < eta <= 1:
            raise ValueError("eta must be in (0, 1].")
        self._weights -= eta * self._weights_derivative
        self._biases -= eta * self._biases_derivative
        self._weights_derivative = np.zeros(self._weights.shape)
        self._biases_derivative = np.zeros(self._biases.shape)


class ResilientDense(Dense):

    def __init__(
        self,
        width: int,
        activation: Activation = Activation.IDENTITY,
        name: Optional[str] = None,
    ):
        super().__init__(width, activation, name)
        self._last_weights_derivative = None
        self._last_biases_derivative = None
        self._weights_delta = None
        self._biases_delta = None

    @require_built
    def update(self, **kwargs) -> None:

        eta_positive = kwargs.get("eta_positive", 1.2)
        eta_negative = kwargs.get("eta_negative", 0.5)
        delta_max = kwargs.get("delta_max", 50)
        delta_min = kwargs.get("delta_min", 1e-6)
        delta_zero = kwargs.get("delta_zero", 0.0125)

        if not eta_positive > 1:
            raise ValueError("eta_positive must be in (1, inf).")
        if not 0 < eta_negative < 1:
            raise ValueError("eta_negative must be in (0, 1).")

        if (
            self._last_weights_derivative is None
            or self._last_biases_derivative is None
        ):
            self._last_weights_derivative = np.zeros(self._weights_derivative.shape)
            self._last_biases_derivative = np.zeros(self._biases_derivative.shape)
            self._weights_delta = (
                np.ones(self._weights_derivative.shape) * delta_zero
            )
            self._biases_delta = (
                np.ones(self._biases_derivative.shape) * delta_zero
            )

        same_sign = (
            self._weights_derivative *
            self._last_weights_derivative > 0
        )
        self._weights_delta[same_sign] = np.minimum(
            self._weights_delta[same_sign] * eta_positive, delta_max
        )
        self._weights[same_sign] -= (
            np.sign(self._weights_derivative[same_sign]) *
            self._weights_delta[same_sign]
        )
        self._last_weights_derivative[same_sign] = (
            self._weights_derivative[same_sign]
        )

        diff_sign = (
            self._weights_derivative *
            self._last_weights_derivative < 0
        )
        self._weights_delta[diff_sign] = np.maximum(
            self._weights_delta[diff_sign] * eta_negative, delta_min
        )
        self._last_weights_derivative[diff_sign] = 0

        no_sign = (
            self._weights_derivative *
            self._last_weights_derivative == 0
        )
        self._weights[no_sign] -= (
            np.sign(self._weights_derivative[no_sign]) *
            self._weights_delta[no_sign]
        )
        self._last_weights_derivative[no_sign] = (
            self._weights_derivative[no_sign]
        )

        same_sign = (
            self._last_biases_derivative *
            self._biases_derivative > 0
        )
        self._biases[same_sign] -= (
            np.sign(self._biases_derivative[same_sign]) *
            self._biases_delta[same_sign]
        )
        self._biases_delta[same_sign] = np.minimum(
            self._biases_delta[same_sign] * eta_positive, delta_max
        )
        self._last_biases_derivative[same_sign] = (
            self._biases_derivative[same_sign]
        )

        diff_sign = (
            self._biases_derivative *
            self._last_biases_derivative < 0
        )
        self._biases_delta[diff_sign] = np.maximum(
            self._biases_delta[diff_sign] * eta_negative, delta_min
        )
        self._last_biases_derivative[diff_sign] = 0

        no_sign = (
            self._biases_derivative *
            self._last_biases_derivative == 0
        )
        self._biases[no_sign] -= (
            np.sign(self._biases_derivative[no_sign]) *
            self._biases_delta[no_sign]
        )
        self._last_biases_derivative[no_sign] = (
            self._biases_derivative[no_sign]
        )

        self._weights_derivative = np.zeros(self._weights.shape)
        self._biases_derivative = np.zeros(self._biases.shape)
