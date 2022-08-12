#!/usr/bin/env python3
import numpy as np
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
