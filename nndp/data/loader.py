#!/usr/bin/env python3
import os
import numpy as np
from mnist import MNIST
from nndp.utils.collections import Set


class Loader:

    @staticmethod
    def encode(labels: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def decode(labels: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def split(dataset: Set, portion: float = 0.25) -> tuple[Set, Set]:
        raise NotImplementedError

    def get_random_dataset(self, instances: int, scaled: bool):
        raise NotImplementedError

    @property
    def dataset(self) -> Set:
        raise NotImplementedError


class MNISTLoader(Loader):

    def __init__(self):
        self._mnist_data = MNIST(os.path.join(os.path.dirname(__file__), "mnist"))
        data, labels = self._mnist_data.load_training()
        self._dataset = Set(
            np.expand_dims(np.array(data), axis=2),
            MNISTLoader.encode(np.array(labels))
        )

    @staticmethod
    def encode(labels: np.ndarray) -> np.ndarray:
        encoded_labels = np.zeros(shape=(labels.shape[0], 10, 1))
        for n in range(labels.shape[0]):
            encoded_labels[n][labels[n]] = 1
        return encoded_labels

    @staticmethod
    def decode(labels: np.ndarray) -> np.ndarray:
        return np.array([
            np.argmax(labels[n]) for n in range(labels.shape[0])
        ])

    @staticmethod
    def split(dataset: Set, portion: float = 0.25) -> tuple[Set, Set]:
        if not 0 < portion < 1:
            raise ValueError("portion must be in (0, 1).")
        choices = np.random.permutation(dataset.size)
        rt_dataset_size = int(dataset.size * portion)
        lt_dataset_size = dataset.size - rt_dataset_size
        rt_data = dataset.data[choices[lt_dataset_size:]]
        rt_labels = dataset.labels[choices[lt_dataset_size:]]
        lt_data = dataset.data[choices[:lt_dataset_size]]
        lt_labels = dataset.labels[choices[:lt_dataset_size]]
        return Set(lt_data, lt_labels), Set(rt_data, rt_labels)

    def get_random_dataset(self, instances: int = 10000, scaled: bool = True):
        if instances >= self._dataset.size:
            raise ValueError(
                f"too many instances requested - "
                f"dataset size: {self._dataset.size}."
            )
        choices = np.random.choice(
            self._dataset.data.shape[0], size=instances, replace=False
        )
        data = self._dataset.data[choices]
        labels = self._dataset.labels[choices]
        data = data / 255.0 if scaled else data
        return Set(data, labels)

    @property
    def dataset(self) -> Set:
        return self._dataset
