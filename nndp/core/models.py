#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
from typing import Optional
from nndp.math.functions import Loss
from nndp.core.layers import Layer, Category
from nndp.errors import IncompatibleLayersError
from nndp.utils.collections import Set
from nndp.utils.decorators import require_built, require_not_built


class Sequential:

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
        self, weights: list[np.ndarray] = None, biases: list[np.ndarray] = None
    ) -> Sequential:
        if self.is_built():
            return self
        for h in range(0, self.depth):
            if self._layers[h].is_built():
                if h != (self.depth - 1) and self._layers[h].category.value != Category.HIDDEN.value:
                    raise IncompatibleLayersError(
                        f"expected {Category.HIDDEN} layer, received {self._layers[h].category} layer."
                    )
                if h == (self.depth - 1) and self._layers[h].category.value != Category.OUTPUT.value:
                    raise IncompatibleLayersError(
                        f"expected {Category.OUTPUT} layer, received {self._layers[h].category} layer."
                    )
            self._layers[h].build(
                Category.HIDDEN if h != (self.depth - 1) else Category.OUTPUT,
                weights[h] if weights else None,
                biases[h] if biases else None
            )
        return self

    def is_built(self) -> bool:
        return all([layer.is_built() for layer in self._layers])

    @require_not_built
    def add(self, layer: Layer) -> None:
        if len(self._layers) and (layer.in_size != self._layers[-1].width):
            raise IncompatibleLayersError(
                f"in_size {layer.in_size} is not equal "
                f"to previous layer width {self._layers[-1].width}."
            )
        self._layers.append(layer)

    @require_built
    def predict(self, in_data: np.ndarray) -> np.ndarray:
        self._forward_propagation(in_data)
        return self.out_data

    def fit(
        self,
        training_set: Set,
        learning_rate: float = 0.001,
        n_batches: int = 1,
        epochs: int = 500
    ) -> None:
        if not 0 <= learning_rate <= 1:
            raise ValueError("learning rate must be in [0, 1].")
        if not (0 <= n_batches <= training_set.size):
            raise ValueError(f"n_batches must be in (0, {training_set.size}].")
        batches = [
            Set(data, labels)
            for data, labels in
            zip(
                np.array_split(training_set.data, n_batches),
                np.array_split(training_set.labels, n_batches)
            )
        ] if n_batches not in [0, 1] else [training_set]
        for epoch in range(0, epochs):
            print(f"Epoch #{epoch}.")
            for batch in batches:
                for instance in range(0, batch.size):
                    self._forward_propagation(batch.data[instance])
                    self._backward_propagation(batch.labels[instance])
                    self._update(
                        learning_rate
                        if (n_batches == 0 or instance == batch.size - 1)
                        else None
                    )

    @require_built
    def _forward_propagation(self, in_data: np.ndarray) -> None:
        for layer in self._layers:
            in_data = layer.forward_propagation(in_data)

    @require_built
    def _backward_propagation(self, expected: np.ndarray) -> None:
        expected = np.reshape(expected, (-1, 1))
        delta = self._loss.prime()(self.out_data, expected)
        for layer in reversed(self._layers):
            delta = layer.backward_propagation(delta)

    @require_built
    def _update(self, learning_rate) -> None:
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
    def loss(self) -> Loss:
        return self._loss

    @property
    def size(self) -> int:
        return sum([layer.width for layer in self._layers])

    # TODO
    def __str__(self):
        string = ""
        for layer in self._layers:
            string += (str(layer) + "\n")
        return string

"""
from mnist import MNIST
from layers import Dense
from nndp.math.functions import Activation, Loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def get_mnist_data(valid=None):
    mdata = MNIST('../data/mnist', return_type='numpy')

    train_data, train_labels = mdata.load_training()
    test_data, test_labels = mdata.load_testing()
    train_data = train_data.reshape((60000, 1, 28, 28))
    test_data = test_data.reshape((10000, 1, 28, 28))

    if valid is not None:
        n_valid = int(valid * 60000)
        indexes = np.array(range(60000))
        np.random.shuffle(indexes)

        valid_X = train_data[indexes[0:n_valid]]
        valid_y = train_labels[indexes[0:n_valid]]
        train_X = train_data[indexes[n_valid:]]
        train_y = train_labels[indexes[n_valid:]]
        return train_X, train_y, valid_X, valid_y, test_data, test_labels

    return train_data, train_labels, test_data, test_labels


train_X, train_y, valid_X, valid_y, test_X, test_y = get_mnist_data(valid=.25)
test_X = test_X / 255 - .5
train_X = train_X / 255 - .5
valid_X = valid_X / 255 - .5


nn = Sequential(
    [
        Dense(784, 10, Activation.SIGMOID),
        Dense(10, 10, Activation.SIGMOID).build(),
        Dense(10, 10, Activation.IDENTITY),
    ],
    Loss.CROSS_ENTROPY
)
nn.build()

nn.fit(Set(X_train, t_train), n_batches=0, learning_rate=0.001, epochs=500)
y_test = np.squeeze(np.array([nn.predict(test) for test in X_test]))

print(nn.predict(X_train[0]), t_train[0])
"""