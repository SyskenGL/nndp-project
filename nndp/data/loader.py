#!/usr/bin/env python3
import os
import mnist
import numpy as np
from nndp.utils.collections import Dataset


class Loader:

    def __init__(self):
        self._dataset = None

    @staticmethod
    def encode(labels: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def decode(labels: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    @property
    def scaled_dataset(self) -> Dataset:
        raise NotImplementedError


class MNIST(Loader):

    def __init__(self):
        super().__init__()
        self._mnist_data = mnist.MNIST(os.path.join(os.path.dirname(__file__), "mnist"))
        data, labels = self._mnist_data.load_training()
        data = np.array(data).T
        labels = MNIST.encode(np.array(labels))
        self._dataset = Dataset(data, labels)

    @staticmethod
    def encode(labels: np.ndarray) -> np.ndarray:
        encoded_labels = np.zeros(shape=(10, labels.shape[0]))
        for n in range(labels.shape[0]):
            encoded_labels[labels[n]][n] = 1
        return encoded_labels

    @staticmethod
    def decode(labels: np.ndarray) -> np.ndarray:
        return np.array([
            np.argmax(labels[:, n]) for n in range(labels.shape[1])
        ])

    @property
    def scaled_dataset(self) -> Dataset:
        return Dataset(
            self.dataset.data / 255.0, self.dataset.labels
        )