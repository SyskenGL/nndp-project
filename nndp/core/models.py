#!/usr/bin/env python3
from __future__ import annotations
import uuid
import numpy as np
from tabulate import tabulate
from nndp.core.layers import Layer
from nndp.errors import EmptyModelError
from nndp.utils.collections import Set
from nndp.utils.functions import Loss
from nndp.utils.decorators import require_built
from nndp.utils.decorators import require_not_built
from nndp.utils.metrics import accuracy_score


class MLP:

    def __init__(
        self,
        layers: list[Layer] = None,
        loss: Loss = Loss.SSE,
        name: str = None,
    ):
        self._name = (
            name if name is not None
            else f"{self.__class__.__name__}_{str(uuid.uuid4())[:8]}"
        )
        self._layers = layers if layers else []
        self._loss = loss

    def is_built(self) -> bool:
        return (
            len(self._layers) != 0 and
            all([layer.is_built() for layer in self._layers])
        )

    @require_not_built
    def build(
        self,
        in_size: int,
        weights: list[np.ndarray] = None,
        biases: list[np.ndarray] = None
    ) -> None:
        if len(self._layers) == 0:
            raise EmptyModelError(
                f"attempt to build an empty model {self.__class__.__name__}"
            )
        if in_size <= 0:
            raise ValueError(
                f"expected in_size value greater than zero, received {in_size}."
            )
        if weights is not None and len(weights) != len(self._layers):
            raise ValueError(
                f"expected a list of weights of "
                f"length {len(self._layers)}, received {len(weights)}."
            )
        if biases is not None and len(biases) != len(self._layers):
            raise ValueError(
                f"expected a list of biases of "
                f"length {len(self._layers)}, received {len(biases)}."
            )
        for h in range(0, self.depth):
            self._layers[h].build(
                self._layers[h - 1].width if h != 0 else in_size,
                weights[h] if weights else None,
                biases[h] if biases else None
            )

    @require_not_built
    def push(self, layer: Layer) -> None:
        self._layers.append(layer)

    @require_not_built
    def pop(self):
        self._layers.pop()

    @require_built
    def predict(self, in_data: np.ndarray) -> np.ndarray:
        out_data = in_data
        for layer in self._layers:
            out_data = layer.predict(out_data)
        return out_data

    @require_built
    def fit(
        self,
        training_set: Set,
        validation_set: Set = None,
        learning_rate: float = .001,
        n_batches: int = 1,
        epochs: int = 500,
        target_training_accuracy: float = None,
        target_validation_accuracy: float = None
    ) -> np.ndarray:

        if training_set.size == 0:
            raise ValueError("provided an empty training set.")
        if validation_set and validation_set.size == 0:
            raise ValueError("provided an empty validation set.")
        if not validation_set and target_validation_accuracy:
            raise ValueError(f"expected a validation set.")

        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1].")
        if not 0 <= n_batches <= training_set.size:
            raise ValueError(f"n_batches must be in [0, {training_set.size}].")
        if epochs <= 0:
            raise ValueError(f"epochs must be greater than 0.")

        if target_training_accuracy and not 0 < target_training_accuracy < 1:
            raise ValueError("target_training_accuracy must be in (0, 1).")
        if target_validation_accuracy and not 0 < target_validation_accuracy < 1:
            raise ValueError("target_validation_accuracy must be in (0, 1).")

        stats = []
        batches = [
            Set(data, labels)
            for data, labels in
            zip(
                np.array_split(training_set.data, n_batches),
                np.array_split(training_set.labels, n_batches)
            )
        ] if n_batches not in [0, 1] else [training_set]

        for epoch in range(0, epochs):

            for batch in batches:
                for instance in range(0, batch.size):
                    self._forward_propagation(batch.data[instance])
                    self._backward_propagation(batch.labels[instance])
                    self._update(
                        learning_rate
                        if (n_batches == 0 or instance == batch.size - 1)
                        else None
                    )

            training_predictions = np.array([self.predict(x) for x in training_set.data])
            validation_predictions = np.array([
                self.predict(x) for x in validation_set.data
            ]) if validation_set else None

            training_loss = self._loss.function()(training_predictions, training_set.labels)
            validation_loss = self._loss.function()(
                validation_predictions, validation_set.labels
            ) if validation_set else None

            training_accuracy = accuracy_score(training_predictions, training_set.labels)
            validation_accuracy = accuracy_score(validation_predictions, validation_set.labels)

            stats.append((
                epoch,
                training_loss,
                validation_loss,
                training_accuracy,
                validation_accuracy
            ))

            print({
                "epoch": epoch,
                "training_loss": training_loss,
                "validation_loss": validation_loss,
                "training_accuracy": round(training_accuracy, 2),
                "validation_accuracy": round(validation_accuracy, 2)
            })

            if (
                (target_training_accuracy and target_training_accuracy <= training_accuracy) or
                (target_validation_accuracy and target_validation_accuracy <= validation_accuracy)
            ):
                break

        return np.array(stats)

    @require_built
    def _forward_propagation(self, in_data: np.ndarray) -> None:
        out_data = in_data
        for layer in self._layers:
            out_data = layer.forward_propagation(out_data)

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
    def name(self) -> str:
        return self._name

    @property
    def depth(self) -> int:
        return len(self._layers)

    @property
    def size(self) -> int:
        return sum([layer.width for layer in self._layers]) + self.in_size

    @property
    def width(self) -> int:
        return (
            max([layer.width for layer in self._layers] + [self.in_size])
            if len(self._layers) else 0
        )

    @property
    def loss(self) -> Loss:
        return self._loss

    @property
    def in_size(self) -> int:
        return self._layers[0].in_size if len(self._layers) else 0

    @property
    def in_data(self) -> np.ndarray:
        return self._layers[0].in_data if len(self._layers) else None

    @property
    def out_size(self) -> int:
        return self._layers[-1].width if len(self._layers) else 0

    @property
    def out_data(self) -> np.ndarray:
        return self._layers[-1].out_data if len(self._layers) else None

    @property
    def layers(self) -> tuple[Layer]:
        return tuple(self._layers)

    @property
    def n_trainable(self) -> tuple[Layer]:
        return (
            sum([layer.weights.size + layer.biases.size for layer in self._layers])
            if self.is_built() else 0
        )

    def __str__(self):
        details = str(tabulate(
            [
                ["\033[1m TYPE \033[0m", self.__class__.__name__],
                ["\033[1m NAME \033[0m", self._name],
                ["\033[1m DEPTH \033[0m", self.depth],
                ["\033[1m WIDTH \033[0m", self.width],
                ["\033[1m SIZE \033[0m", self.size],
                [
                    "\033[1m IN_SIZE \033[0m",
                    self.in_size if self.in_size else "-"
                ],
                [
                    "\033[1m OUT_SIZE \033[0m",
                    self.out_size if self.out_size else "-"
                ],
                ["\033[1m LOSS \033[0m", self._loss.function().__name__],
                [
                    "\033[1m # TRAINABLE \033[0m",
                    self.n_trainable if self.is_built() else "-"
                ],
                ["\033[1m BUILT\033[0m", self.is_built()]
            ],
            tablefmt="fancy_grid",
            colalign=["center"] * 2
        ))
        layers = "\n".join([str(layer) for layer in self._layers])
        structure = str(tabulate([[
                self._layers[h].name if h != -1 else "-",
                ("HIDDEN" if h != self.depth - 1 else "OUTPUT")
                if h != -1 else "INPUT",
                self._layers[h].width if h != -1
                else (self.in_size if self.in_size else "-"),
            ] for h in range(-1, self.depth)],
            tablefmt="fancy_grid",
            colalign=["center"] * 3
        )) if len(self._layers) != 0 else ""
        return str(tabulate(
            [[
                details,
                layers if layers != "" else "-",
                structure if structure != "" else "-"]],
            headers=[
                "\033[1m DETAILS \033[0m",
                "\033[1m LAYERS \033[0m",
                "\033[1m STRUCTURE \033[0m"
            ],
            tablefmt="fancy_grid",
            colalign=["center"] * 3
        ))
