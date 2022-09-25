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


class EarlyStop:

    def __init__(
        self,
        metric: Metric,
        trigger: float,
        greedy: bool = False
    ):
        if greedy and not 0 < trigger < 100:
            raise ValueError(
                f"trigger for metric {metric.name.lower()} must be in (0, 100)."
            )
        elif not greedy and metric != Metric.LOSS and not 0 < trigger < 1:
            raise ValueError(
                f"target for metric {metric.name.lower()} must be in (0, 1)."
            )
        self._metric = metric
        self._trigger = trigger
        self._greedy = greedy
        self._best = None

    def is_satisfied(self, value: float):
        if self._greedy:
            self._best = value if self._best is None else self._best
            variation = (value - self._best) / self._best * 100.0
            if self._metric != Metric.LOSS:
                self._best = value if value > self._best else self._best
                return variation <= (-1 * self._trigger)
            else:
                self._best = value if value < self._best else self._best
                return variation >= self._trigger
        else:
            if self._metric != Metric.LOSS:
                return value >= self._trigger
            else:
                return value <= self._trigger

    @property
    def metric(self):
        return self._metric

    @property
    def trigger(self):
        return self._trigger

    @property
    def greedy(self):
        return self._greedy
