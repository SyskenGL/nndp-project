#!/usr/bin/env python3
from __future__ import annotations
import uuid
import numpy as np
from tabulate import tabulate
from nndp.utils.functions import Activation
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
    def update(self, update: bool, **kwargs) -> None:
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
        return self._weights.size + self._biases.size if self.is_built() else 0

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
        return self._weights.T @ self._delta

    @require_built
    def predict(self, in_data: np.ndarray) -> np.ndarray:
        in_data = np.reshape(in_data, (-1, 1))
        in_weighted = self._weights @ in_data + self._biases
        return self._activation.function()(in_weighted)

    @require_built
    def update(self, update: bool = False, **kwargs) -> None:
        eta = kwargs.get("eta", 0.001)
        if not 0 < eta <= 1:
            raise ValueError("eta must be in (0, 1].")
        self._accumulated_weights_delta += (self._delta @ self._in_data.T)
        self._accumulated_biases_delta += self._delta
        if update:
            self._weights -= eta * self._accumulated_weights_delta
            self._biases -= eta * self._accumulated_biases_delta
            self._accumulated_weights_delta = np.zeros(self._weights.shape)
            self._accumulated_biases_delta = np.zeros(self._biases.shape)


class ResilientDense(Dense):

    def __init__(
        self,
        width: int,
        activation: Activation = Activation.IDENTITY,
        name: str = None,
    ):
        super().__init__(width, activation, name)
        self._last_accumulated_weights_delta = None
        self._last_accumulated_bias_delta = None
        self._weights_delta = None
        self._biases_delta = None

    @require_built
    def update(self, update: bool = False, **kwargs) -> None:

        eta_positive = kwargs.get("eta_positive", 1.2)
        eta_negative = kwargs.get("eta_negative", 0.5)
        delta_max = kwargs.get("delta_max", 50)
        delta_min = kwargs.get("delta_min", 1e-6)
        delta_zero = kwargs.get("delta_zero", 0.0125)

        if not eta_positive > 1:
            raise ValueError("eta_positive must be in (1, inf).")
        if not 0 < eta_negative < 1:
            raise ValueError("eta_negative must be in (0, 1).")

        self._accumulated_weights_delta += self._delta @ self._in_data.T
        self._accumulated_biases_delta += self._delta

        if update:

            if (
                self._last_accumulated_weights_delta is None
                or self._last_accumulated_bias_delta is None
            ):
                self._last_accumulated_weights_delta = np.zeros(
                    self._accumulated_weights_delta.shape
                )
                self._last_accumulated_bias_delta = np.zeros(
                    self._accumulated_biases_delta.shape
                )
                self._weights_delta = (
                    np.ones(self._accumulated_weights_delta.shape) * delta_zero
                )
                self._biases_delta = (
                    np.ones(self._accumulated_biases_delta.shape) * delta_zero
                )

            same_sign = (
                self._accumulated_weights_delta *
                self._last_accumulated_weights_delta > 0
            )
            self._weights_delta[same_sign] = np.minimum(
                self._weights_delta[same_sign] * eta_positive, delta_max
            )
            self._weights[same_sign] -= (
                np.sign(self._accumulated_weights_delta[same_sign]) *
                self._weights_delta[same_sign]
            )
            self._last_accumulated_weights_delta[same_sign] = (
                self._accumulated_weights_delta[same_sign]
            )

            diff_sign = (
                self._accumulated_weights_delta *
                self._last_accumulated_weights_delta < 0
            )
            self._weights_delta[diff_sign] = np.maximum(
                self._weights_delta[diff_sign] * eta_negative, delta_min
            )
            self._last_accumulated_weights_delta[diff_sign] = 0

            no_sign = (
                self._accumulated_weights_delta *
                self._last_accumulated_weights_delta == 0
            )
            self._weights[no_sign] -= (
                np.sign(self._accumulated_weights_delta[no_sign]) *
                self._weights_delta[no_sign]
            )
            self._last_accumulated_weights_delta[no_sign] = (
                self._accumulated_weights_delta[no_sign]
            )

            same_sign = (
                self._last_accumulated_bias_delta *
                self._accumulated_biases_delta > 0
            )
            self._biases[same_sign] -= (
                np.sign(self._accumulated_biases_delta[same_sign]) *
                self._biases_delta[same_sign]
            )
            self._biases_delta[same_sign] = np.minimum(
                self._biases_delta[same_sign] * eta_positive, delta_max
            )
            self._last_accumulated_bias_delta[same_sign] = (
                self._accumulated_biases_delta[same_sign]
            )

            diff_sign = (
                self._accumulated_biases_delta *
                self._last_accumulated_bias_delta < 0
            )
            self._biases_delta[diff_sign] = np.maximum(
                self._biases_delta[diff_sign] * eta_negative, delta_min
            )
            self._last_accumulated_bias_delta[diff_sign] = 0

            no_sign = (
                self._accumulated_biases_delta *
                self._last_accumulated_bias_delta == 0
            )
            self._biases[no_sign] -= (
                np.sign(self._accumulated_biases_delta[no_sign]) *
                self._biases_delta[no_sign]
            )
            self._last_accumulated_bias_delta[no_sign] = (
                self._accumulated_biases_delta[no_sign]
            )

            self._accumulated_weights_delta = np.zeros(self._weights.shape)
            self._accumulated_biases_delta = np.zeros(self._biases.shape)
