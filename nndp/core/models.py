#!/usr/bin/env python3
from __future__ import annotations
import uuid
import logging
import numpy as np
from tabulate import tabulate
from nndp.core.layers import Layer
from nndp.errors import EmptyModelError
from nndp.utils.collections import Set
from nndp.utils.functions import Loss
from nndp.utils.decorators import require_built
from nndp.utils.decorators import require_not_built
from nndp.utils.metrics import accuracy_score, f1_score


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
        self._logger = logging.getLogger(self._name)

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
        for h in range(self.depth):
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
        n_batches: int = 1,
        epochs: int = 500,
        target_loss: float = None,
        target_accuracy: float = None,
        target_f1: float = None,
        weak_target: bool = True,
        **kwargs
    ) -> np.ndarray:

        if training_set.size == 0:
            raise ValueError("provided an empty training set.")
        if validation_set and validation_set.size == 0:
            raise ValueError("provided an empty validation set.")
        if not validation_set and (target_loss or target_accuracy or target_f1):
            raise ValueError(f"expected a validation set.")

        if not 0 <= n_batches <= training_set.size:
            raise ValueError(f"n_batches must be in [0, {training_set.size}].")
        if epochs <= 0:
            raise ValueError(f"epochs must be greater than 0.")

        if target_accuracy and not 0 < target_accuracy < 1:
            raise ValueError("target_accuracy must be in (0, 1).")
        if target_f1 and not 0 < target_f1 < 1:
            raise ValueError("target_f1 must be in (0, 1).")

        stats = []
        batches = [
            Set(data, labels)
            for data, labels in
            zip(
                np.array_split(training_set.data, n_batches),
                np.array_split(training_set.labels, n_batches)
            )
        ] if n_batches not in [0, 1] else [training_set]

        for epoch in range(epochs):

            for batch in batches:
                for instance in range(batch.size):
                    data = batch.data[:, instance].reshape(-1, 1)
                    label = batch.labels[:, instance].reshape(-1, 1)
                    self._forward_propagation(data)
                    self._backward_propagation(label)
                    self._update(n_batches == 0 or instance == batch.size - 1, **kwargs)

            training_predictions = self.predict(training_set.data)
            validation_predictions = self.predict(validation_set.data)

            training_loss = self._loss.function()(
                training_predictions, training_set.labels
            )
            validation_loss = self._loss.function()(
                validation_predictions, validation_set.labels
            ) if validation_set else None

            training_accuracy = accuracy_score(
                training_predictions, training_set.labels
            )
            validation_accuracy = accuracy_score(
                validation_predictions, validation_set.labels
            ) if validation_set else None

            training_f1 = f1_score(
                training_predictions, training_set.labels
            )
            validation_f1 = f1_score(
                validation_predictions, validation_set.labels
            ) if validation_set else None

            stats.append((
                epoch,
                training_loss, validation_loss,
                training_accuracy, validation_accuracy,
                training_f1, validation_f1
            ))

            self._logger.info(
                f"\033[1m Epoch \033[0m{epoch + 1}/{epochs}\n\n"
                f"\033[1m   • Training loss:\033[0m {training_loss:.3f}\n"
                f"\033[1m   • Training accuracy:\033[0m {training_accuracy:.3f}\n"
                f"\033[1m   • Training F1:\033[0m {training_f1:.3f}\n\n"
                f"\033[1m   • Validation loss:\033[0m "
                f"{format(validation_loss, '.3f') if validation_loss else '-'}\n"
                f"\033[1m   • Validation accuracy:\033[0m "
                f"{format(validation_accuracy, '.3f') if validation_accuracy else '-'}\n"
                f"\033[1m   • Validation F1:\033[0m "
                f"{format(validation_f1, '.3f') if validation_f1 else '-'}\n\n"
                f"\033[1m   • Target loss:\033[0m "
                f"{format(target_loss, '.3f') if target_loss else '-'}\n"
                f"\033[1m   • Target accuracy:\033[0m "
                f"{format(target_accuracy, '.3f') if target_accuracy else '-'}\n"
                f"\033[1m   • Target F1:\033[0m "
                f"{format(target_f1, '.3f') if target_f1 else '-'}"
            )

            if (
                (weak_target and (
                    (target_loss and target_loss >= validation_loss) or
                    (target_accuracy and target_accuracy <= validation_accuracy) or
                    (target_f1 and target_f1 <= validation_f1)
                )) or
                (not weak_target and (
                    (not target_loss or (target_loss and target_loss >= validation_loss)) and
                    (not target_accuracy or (target_accuracy and target_accuracy <= validation_accuracy)) and
                    (not target_f1 or (target_f1 and target_f1 <= validation_f1)) and
                    (target_loss or target_accuracy or target_f1)
                ))
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
        delta = self._loss.prime()(self.out_data, expected)
        for layer in reversed(self._layers):
            delta = layer.backward_propagation(delta)

    @require_built
    def _update(self, update: bool, **kwargs) -> None:
        for layer in self._layers:
            layer.update(update, **kwargs)

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
                    "\033[1m TRAINABLE \033[0m",
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