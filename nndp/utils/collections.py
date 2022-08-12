#!/usr/bin/env python3
from __future__ import annotations
import numpy as np


class Set:

    def __init__(self, data: np.ndarray, labels: np.ndarray):
        if data.shape[1] != labels.shape[1]:
            raise ValueError(
                f"data and label sizes do not match "
                f"({data.shape[1]} != {labels.shape[1]})."
            )
        self._data = data
        self._labels = labels

    @staticmethod
    def split(dataset: Set, portion: float = 0.25) -> tuple[Set, Set]:
        if not 0 < portion < 1:
            raise ValueError("portion must be in (0, 1).")
        choices = np.random.permutation(dataset.size)
        rt_dataset_size = int(dataset.size * portion)
        lt_dataset_size = dataset.size - rt_dataset_size
        rt_data = dataset.data[:, choices[lt_dataset_size:]]
        rt_labels = dataset.labels[:, choices[lt_dataset_size:]]
        lt_data = dataset.data[:, choices[:lt_dataset_size]]
        lt_labels = dataset.labels[:, choices[:lt_dataset_size]]
        return Set(lt_data, lt_labels), Set(rt_data, rt_labels)

    def get_random_dataset(self, instances: int = 10000, scaled: bool = True):
        if instances >= self.size:
            raise ValueError(
                f"too many instances requested - "
                f"dataset size: {self.size}."
            )
        choices = np.random.choice(
            self.data.shape[1], size=instances, replace=False
        )
        data = self.data[:, choices]
        labels = self.labels[:, choices]
        data = data / 255.0 if scaled else data
        return Set(data, labels)

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    @property
    def size(self) -> int:
        return self._labels.shape[1]
