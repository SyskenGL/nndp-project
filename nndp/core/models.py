#!/usr/bin/env python3
from __future__ import annotations
import os
import uuid
import logging
import pickle
import pickletools
import nndp.data
import numpy as np
from copy import deepcopy
from typing import Optional
from tabulate import tabulate
from nndp.core.layers import Layer
from nndp.errors import EmptyModelError
from nndp.utils.collections import Dataset
from nndp.utils.functions import Loss
from nndp.utils.metrics import Target, Metric
from nndp.utils.decorators import require_built
from nndp.utils.decorators import require_not_built


class MLP:

    def __init__(
        self,
        layers: list[Layer] = None,
        loss: Loss = Loss.SSE,
        name: Optional[str] = None
    ):
        self._name = (
            name if name is not None
            else f"{self.__class__.__name__}_{str(uuid.uuid4())[:8]}"
        )
        self._layers = layers if layers else []
        self._loss = loss
        self._copies = 0
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
        weights: Optional[list[np.ndarray]] = None,
        biases: Optional[list[np.ndarray]] = None
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
    def validate(
        self,
        validation_set: Dataset,
        metrics: list[Metric] = (Metric.LOSS,)
    ) -> dict:

        if validation_set.size == 0:
            raise ValueError("provided an empty validation set.")
        if metrics is None or len(metrics) == 0:
            raise ValueError("no metrics provided.")
        if len(metrics) != len(set(metrics)):
            raise ValueError(f"multiple same metric provided.")

        data = {}
        validation_predictions = self.predict(validation_set.data)

        for metric in metrics:
            if metric != Metric.LOSS:
                metric_function = metric.score()
                metric_name = metric.name.lower()
            else:
                metric_function = self._loss.function()
                metric_name = "loss"
            data[metric_name] = metric_function(
                validation_predictions,
                validation_set.labels
            )
        return data

    @require_built
    def cross_validate(
        self,
        dataset: Dataset,
        n_splits: int = 5,
        metrics: list[Metric] = (Metric.LOSS,),
        epochs: int = 30,
        n_batches: int = 1
    ) -> dict:

        if metrics is None or len(metrics) == 0:
            raise ValueError("no metrics provided.")
        if len(metrics) != len(set(metrics)):
            raise ValueError(f"multiple same metric provided.")

        scores = []
        k_fold = dataset.k_fold(n_splits)

        for k, (training_set, validation_set) in enumerate(k_fold):
            model = deepcopy(self)
            self._logger.info(
                f"\033[1m • Cross validation of the model {model._name}"
                f" - {k + 1} of {n_splits}\033[0m"
            )
            model.fit(
                training_set, n_batches=n_batches, epochs=epochs, stats=None
            )
            scores.append(model.validate(validation_set, metrics))
        scores = {
            metric: [score.get(metric) for score in scores]
            for metric in set().union(*scores)
        }

        result = {}
        for key in scores.keys():
            values = np.array(scores[key])
            result[key] = (values.mean(), values.std())
        return result

    @require_built
    def fit(
        self,
        training_set: Dataset,
        validation_set: Optional[Dataset] = None,
        n_batches: int = 1,
        epochs: int = 500,
        targets: Optional[list[Target]] = None,
        weak_stop: bool = True,
        stats: Optional[list[Metric]] = (Metric.LOSS,),
        **kwargs
    ) -> list:

        targets = [] if targets is None else targets
        stats = [] if stats is None else stats

        if training_set.size == 0:
            raise ValueError("provided an empty training set.")
        if validation_set and validation_set.size == 0:
            raise ValueError("provided an empty validation set.")
        if not validation_set and len(targets) != 0:
            raise ValueError(f"expected a validation set.")

        if not 0 <= n_batches <= training_set.size:
            raise ValueError(f"n_batches must be in [0, {training_set.size}].")
        if epochs <= 0:
            raise ValueError(f"epochs must be greater than 0.")

        target_metrics = [target.metric for target in targets]
        if len(target_metrics) != len(set(target_metrics)):
            raise ValueError(f"multiple targets with same metric provided.")
        if len(stats) != len(set(stats)):
            raise ValueError(f"multiple stats with same metric provided.")

        training_stats = []
        batches = [
            Dataset(data, labels)
            for data, labels in
            zip(
                np.array_split(training_set.data, n_batches, axis=1),
                np.array_split(training_set.labels, n_batches, axis=1)
            )
        ] if n_batches not in [0, 1] else [training_set]

        for epoch in range(epochs):

            for batch in batches:
                for instance in range(batch.size):
                    data = batch.data[:, instance].reshape(-1, 1)
                    label = batch.labels[:, instance].reshape(-1, 1)
                    self._forward_propagation(data)
                    self._backward_propagation(label)
                    if n_batches == 0 or instance == batch.size - 1:
                        self._update(**kwargs)

            training_predictions = self.predict(training_set.data)
            validation_predictions = (
                self.predict(validation_set.data)
                if validation_set is not None else None
            )

            targets_satisfied = []
            for target in targets:
                if target.metric != Metric.LOSS:
                    metric_function = target.metric.score()
                else:
                    metric_function = self._loss.function()
                current_target = metric_function(
                    validation_predictions,
                    validation_set.labels
                )
                targets_satisfied.append(target.is_satisfied(current_target))

            if len(stats) != 0:

                epoch_stats = {"epoch": epoch, "training": {}, "validation": {}}
                for metric in stats:
                    if metric != Metric.LOSS:
                        metric_function = metric.score()
                        metric_name = metric.name.lower()
                    else:
                        metric_function = self._loss.function()
                        metric_name = "loss"
                    epoch_stats["training"][metric_name] = metric_function(
                        training_predictions,
                        training_set.labels
                    )
                    if validation_set is not None:
                        epoch_stats["validation"][metric_name] = metric_function(
                            validation_predictions,
                            validation_set.labels
                        )
                training_stats.append(epoch_stats)

                learning_method = (
                    "on-line" if (n_batches == 0 or n_batches == training_set.size) else
                    ("full-batch" if n_batches == 1 else "mini-batch")
                )
                log = f"\033[1m Epoch \033[0m{epoch + 1} of {epochs} - [{learning_method}]\n"
                for metric, value in epoch_stats["training"].items():
                    log += f"\n\033[1m   • Training {metric}:\033[0m {value:.3f}"
                log += "\n"
                for metric, value in epoch_stats["validation"].items():
                    log += f"\n\033[1m   • Validation {metric}:\033[0m {value:.3f}"
                log += "\n"
                for target in targets:
                    log += (
                        f"\n\033[1m   • Target {target.metric.name.lower()}:"
                        f"\033[0m {target.target:.3f}"
                    )
                self._logger.info(log)

            if (
                len(targets_satisfied) != 0 and (
                    weak_stop and any(targets_satisfied) or
                    not weak_stop and all(targets_satisfied)
                )
            ):
                break

        return training_stats

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
    def _update(self, **kwargs) -> None:
        for layer in self._layers:
            layer.update(**kwargs)

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
    def layers(self) -> list[Layer]:
        return list(self._layers)

    @property
    def n_trainable(self) -> int:
        return (
            sum([layer.weights.size + layer.biases.size for layer in self._layers])
            if self.is_built() else 0
        )

    def save(self, path: str = None) -> None:
        if path is None:
            path = os.path.join(os.path.dirname(nndp.data.__file__), "saved")
            if not os.path.isdir(path):
                os.mkdir(path)
            path = os.path.join(path, f"{self._name}.net")
        with open(path, "wb") as file:
            pickled = pickle.dumps(self)
            file.write(pickletools.optimize(pickled))

    @staticmethod
    def load(path: str) -> MLP:
        with open(path, "rb") as file:
            unpickled = pickle.Unpickler(file)
            return unpickled.load()

    def __str__(self) -> str:
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

    def __deepcopy__(self, memodict: Optional[dict] = None) -> MLP:
        memodict = {} if memodict is None else memodict
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memodict))
        self._copies += 1
        result._name += f"_CL{self._copies}"
        return result
