#!/usr/bin/env python3
from typing import Optional
import numpy as np
from nndp.errors import IncompatibleLayersError
from nndp.math.functions import Loss
from nndp.core.layers import Layer


class Serial:

    def __init__(self, layers: list[Layer] = None, loss: Loss = Loss.SSE):
        self.__layers = layers if layers else []
        for h in range(1, len(self.__layers)):
            if self.__layers[h].in_size != self.__layers[h - 1].width:
                raise IncompatibleLayersError(
                    f"in_size {self.__layers[h].in_size} is not equal "
                    f"to previous layer width {self.__layers[h - 1].width}."
                )
        self.__loss = loss

    def add(self, layer: Layer) -> None:
        if len(self.__layers) and (layer.in_size != self.__layers[-1].width):
            raise IncompatibleLayersError(
                f"in_size {layer.in_size} is not equal "
                f"to previous layer width {self.__layers[-1].width}."
            )
        self.__layers.append(layer)

    def forward_propagation(self, in_data: np.ndarray) -> np.ndarray:
        out_data = in_data
        for layer in self.__layers:
            out_data = layer.activate(out_data)
        return out_data

    def backward_propagation(self, expected: np.ndarray) -> list[np.ndarray]:
        expected = np.reshape(expected, (self.out_size, 1))
        delta = [np.zeros(shape=(layer.width, 1)) for layer in self.__layers]
        for h in range((self.depth - 1), -1, -1):
            delta[h] = (
                self.__layers[h].activation.prime()(self.__layers[h].in_data) *
                np.dot(self.__layers[h + 1].weights.T, delta[h + 1])
            ) if h != self.depth - 1 else (
                self.__layers[h].activation.prime()(self.__layers[h].in_data) *
                self.__error.prime()(self.__layers[h].out_data, expected)
            )
        return delta

    def update(self, delta: np.ndarray, learning_rate: float = 0.001) -> None:
        for h in range(0, self.depth):
            if h == 0:
                self.__layers[h].weights -= np.dot(delta[h], self.out_data)
            else:
                pass

    @property
    def loss(self) -> Loss:
        return self.__loss

    @property
    def depth(self) -> int:
        return len(self.__layers)

    @property
    def width(self) -> int:
        return max([layer.width for layer in self.__layers]) if len(self.__layers) else 0

    @property
    def in_size(self) -> int:
        return self.__layers[0].in_size if len(self.__layers) else 0

    @property
    def out_size(self) -> int:
        return self.__layers[-1].width if len(self.__layers) else 0

    @property
    def in_data(self) -> Optional[np.ndarray]:
        return self.__layers[0].in_data if len(self.__layers) else None

    @property
    def out_data(self) -> Optional[np.ndarray]:
        return self.__layers[-1].out_data if len(self.__layers) else None

    @property
    def size(self) -> int:
        return sum([layer.width for layer in self.__layers])

"""

def back_propagation(self, in_data: np.ndarray, t: np.ndarray):
    # Feed Forward
    self.feed_forward(in_data)
    # Delta Calculating
    delta = self.get_delta(t)
    # Update Values
    weights_prime, bias_prime = self.get_weights_bias_prime(delta)

    return weights_prime, bias_prime

def get_delta(self, t: np.ndarray) -> np.ndarray:
    delta = list()

    for layer in self.layers:
        delta.append(np.zeros(layer.width))

    for layer in reversed(self.layers):
        act_fun_prime = Activation.prime(layer.activation)

        if layer == self.layers[-1]:
            # Output Nodes
            error_fun_prime = Error.prime()
            delta = act_fun_prime(layer.in_size) * error_fun_prime(layer.out_data, t)
        else:
            # Hidden Nodes
            delta = act_fun_prime(layer.in_size) * np.dot(np.transpose(layer.weights),
                                                          delta[self.layers.index(layer) - 1])
    return delta

def get_weights_bias_prime(self, delta):
    weights_prime = []
    bias_prime = []

    for layer in self.layers:
        if layer == self.layers[0]:
            weights_prime.append(np.dot(delta[self.layers.index(layer)], np.transpose(layer.in_data)))
        else:
            prev_layer = self.layers[self.layers.index(layer) - 1]
            weights_prime.append(np.dot(delta[self.layers.index(layer)], np.transpose(prev_layer.out_data)))
        bias_prime.append(delta[self.layers.index(layer)])

    return weights_prime, bias_prime



class Sequential:

    def __init__(self, layers: list[Layer] = None):
        self.layers = layers if layers else []
        if len(self.layers):
            for h in range(1, len(self.layers)):
                if self.layers[h].in_size != self.layers[h - 1].width:
                    raise IncompatibleLayersError(
                        f"in_size {self.layers[h].in_size} is not equal "
                        f"to previous layer width {self.layers[h - 1].width}."
                    )

    def add(self, layer: Layer):
        if len(self.layers):
            if layer.in_size != self.layers[-1].width:
                raise IncompatibleLayersError(
                    f"in_size {layer.in_size} is not equal "
                    f"to previous layer width {self.layers[-1].width}."
                )
        self.layers.append(layer)

    def feed_forward(self, in_data: np.ndarray) -> np.ndarray:
        out_data = in_data
        for layer in self.layers:
            out_data = layer.feed_forward(out_data)
        return out_data

    def __str__(self) -> str:
        return (
            " ■ Neural Network: \n" +
            str(tabulate(
                [
                    ["TYPE", self.__class__.__name__],
                    ["INPUT", self.layers[0].in_size if len(self.layers) else 0],
                    ["OUTPUT", self.layers[-1].in_size if len(self.layers) else 0],
                    ["LAYERS", len(self.layers)],
                    ["LOSS", None]
                ],
                tablefmt="fancy_grid",
                colalign=["center"]*2
            )) +
            "\n\n ■ Detailed Neural Network: \n" +
            str(tabulate(
                [[
                    ("I:" if (h == 0) else ("O:" if (h == len(self.layers) - 1) else "H:")) + str(h),
                    self.layers[h].name,
                    self.layers[h].type,
                    self.layers[h].in_size,
                    self.layers[h].width,
                    self.layers[h].activation.function().__name__
                ] for h in range(0, len(self.layers))],
                headers=["DEPTH", "NAME", "TYPE", "IN_SIZE", "WIDTH", "ACTIVATION"],
                tablefmt="fancy_grid",
                colalign=["center"] * 6
            ))
        )
"""

from layers import Dense

s = Serial()
s.add(Dense(5, 13).build())
s.add(Dense(13, 15).build())
s.add(Dense(15, 7).build())
s.add(Dense(7, 3).build())
s.add(Dense(3, 3).build())
print(s.feed_forward(np.array([1, 2, 3, 4, 5])))
print(s.backward_error_propagation(np.array([1, 2, 3]))[1])


