#!/usr/bin/env python3
from __future__ import annotations
import numpy as np


class Dataset:

    def __init__(self, data: np.ndarray, labels: np.ndarray):
        if data.shape[1] != labels.shape[1]:
            raise ValueError(
                f"data and label sizes do not match "
                f"({data.shape[1]} != {labels.shape[1]})."
            )
        self._data = data
        self._labels = labels

    def split(self, portion: float = 0.25) -> tuple[Dataset, Dataset]:
        if not 0 < portion <= self.size:
            raise ValueError(f"portion must be in (0, {self.size}].")
        choices = np.random.permutation(self.size)
        rt_dataset_size = (
            int(self.size * portion)
            if (0 < portion < 1) else portion
        )
        lt_dataset_size = self.size - rt_dataset_size
        rt_data = self.data[:, choices[lt_dataset_size:]]
        rt_labels = self.labels[:, choices[lt_dataset_size:]]
        lt_data = self.data[:, choices[:lt_dataset_size]]
        lt_labels = self.labels[:, choices[:lt_dataset_size]]
        return Dataset(lt_data, lt_labels), Dataset(rt_data, rt_labels)

    def k_fold(self, n_splits: int = 2, shuffle: bool = False):
        if not 1 < n_splits < self.data.shape[1]:
            raise ValueError(
                f"n_split must be greater in [2, {self.data.shape[1]}]."
            )
        shuffle = (
            np.random.permutation(self.size)
            if shuffle else np.arange(0, self.size)
        )
        data = np.array(np.array_split(self.data[:, shuffle], n_splits, axis=1))
        labels = np.array(np.array_split(self.labels[:, shuffle], n_splits, axis=1))
        return [
            (
                Dataset(
                    np.concatenate(np.delete(data, (k,), axis=0), axis=1),
                    np.concatenate(np.delete(labels, (k,), axis=0), axis=1)
                ),
                Dataset(data[k], labels[k])
            ) for k in range(n_splits)
        ]

    def random(self, instances: int = 10000):
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
        return Dataset(data, labels)

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    @property
    def size(self) -> int:
        return self._labels.shape[1]

    @property
    def shape(self) -> tuple:
        return self._labels.shape

    def __str__(self) -> str:
        return f"{{data: {str(self.data.shape)}, labels: {str(self.labels.shape)}}}"
