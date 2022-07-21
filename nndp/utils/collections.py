#!/usr/bin/env python3
import numpy as np


class Set:

    def __init__(self, data: np.ndarray, labels: np.ndarray):
        if data.shape[0] != labels.shape[0]:
            raise ValueError(f"data and label sizes do not match ({data.shape[0]} != {labels.shape[0]}).")
        self._data = data
        self._labels = labels

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    @property
    def size(self) -> int:
        return self._labels.shape[0]
