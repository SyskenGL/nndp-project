#!/usr/bin/env python3
from __future__ import annotations
from typing import Optional
import numpy as np
from nndp.math.functions import Loss
from nndp.core.layers import Layer, Category
from nndp.errors import AlreadyBuiltError
from nndp.errors import IncompatibleLayersError
from nndp.utils.decorators import require_built, require_not_built


class NeuralNetwork:

    def __init__(self, layers: list[Layer] = None, loss: Loss = Loss.SSE):
        self._layers = layers if layers else []
        for h in range(1, len(self._layers)):
            if self._layers[h].in_size != self._layers[h - 1].width:
                raise IncompatibleLayersError(
                    f"in_size {self._layers[h].in_size} is not equal "
                    f"to previous layer width {self._layers[h - 1].width}."
                )
        self._loss = loss

    def build(
        self,
        weights: list[np.ndarray] = None,
        biases: list[np.ndarray] = None
    ) -> NeuralNetwork:
        if self.is_built():
            return self
        for h in range(0, self.depth):
            self._layers[h].build(
                Category.HIDDEN if h != (self.depth - 1) else Category.OUTPUT,
                weights[h] if weights else None,
                biases[h] if biases else None
            )
        return self

    def is_built(self) -> bool:
        return any([layer.is_built() for layer in self._layers])

    @require_not_built
    def add(self, layer: Layer) -> None:
        if self.is_built():
            raise AlreadyBuiltError("unable to modify a model already built.")
        if len(self._layers) and (layer.in_size != self._layers[-1].width):
            raise IncompatibleLayersError(
                f"in_size {layer.in_size} is not equal "
                f"to previous layer width {self._layers[-1].width}."
            )
        self._layers.append(layer)

    @require_built
    def forward_propagation(self, in_data: np.ndarray) -> np.ndarray:
        for layer in self._layers:
            in_data = layer.forward_propagation(in_data)
        return in_data

    @require_built
    def backward_propagation(self, expected: np.ndarray) -> list[np.ndarray]:
        expected = np.reshape(expected, (self.out_size, 1))
        delta = self._loss.prime()(self.out_data, expected)
        for layer in reversed(self._layers):
            delta = layer.backward_propagation(delta)
        return [layer.delta for layer in self._layers]

    @require_built
    def update(self, learning_rate: float = 0.001) -> None:
        for layer in self._layers:
            layer.update(learning_rate)

    @property
    def depth(self) -> int:
        return len(self._layers)

    @property
    def width(self) -> int:
        return max([layer.width for layer in self._layers]) if len(self._layers) else 0

    @property
    def in_size(self) -> int:
        return self._layers[0].in_size if len(self._layers) else 0

    @property
    def out_size(self) -> int:
        return self._layers[-1].width if len(self._layers) else 0

    @property
    def in_data(self) -> Optional[np.ndarray]:
        return self._layers[0].in_data if len(self._layers) else None

    @property
    def out_data(self) -> Optional[np.ndarray]:
        return self._layers[-1].out_data if len(self._layers) else None

    @property
    def size(self) -> int:
        return sum([layer.width for layer in self._layers])

    def __str__(self):
        string = ""
        for layer in self._layers:
            string += (str(layer) + "\n")
        return string
