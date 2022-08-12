#!/usr/bin/env python3
import numpy as np
from enum import Enum
from sklearn import metrics


def accuracy_score(y: np.ndarray, t: np.ndarray):
    y = np.argmax(y, axis=0)
    t = np.argmax(t, axis=0)
    return metrics.accuracy_score(y, t)


def precision_score(y: np.ndarray, t: np.ndarray):
    y = np.argmax(y, axis=0)
    t = np.argmax(t, axis=0)
    return metrics.precision_score(
        y, t, average="macro", zero_division=0
    )


def recall_score(y: np.ndarray, t: np.ndarray):
    y = np.argmax(y, axis=0)
    t = np.argmax(t, axis=0)
    return metrics.recall_score(
        y, t, average="macro", zero_division=0
    )


def f1_score(y: np.ndarray, t: np.ndarray):
    y = np.argmax(y, axis=0)
    t = np.argmax(t, axis=0)
    return metrics.f1_score(
        y, t, average="macro", zero_division=0
    )


class Metric(Enum):

    ACCURACY = 0
    PRECISION = 1
    RECALL = 2
    F1 = 3
    LOSS = 4

    def score(self):
        return [
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            lambda *args: None
        ][self.value]


class Target:

    def __init__(self, metric: Metric, target: float):
        if metric != Metric.LOSS and not 0 < target < 1:
            raise ValueError(
                f"target for metric {metric.name.lower()} must be in (0, 1)."
            )
        self._metric = metric
        self._target = target

    def is_satisfied(self, current_target: float):
        if self._metric != Metric.LOSS:
            return current_target > self._target
        else:
            return current_target < self._target

    @property
    def metric(self):
        return self._metric

    @property
    def target(self):
        return self._target
