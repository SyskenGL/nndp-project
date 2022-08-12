#!/usr/bin/env python3
import numpy as np
from enum import Enum


def identity(x: np.ndarray) -> np.ndarray:
    return x


def identity_prime(x: np.ndarray) -> np.ndarray:
    return np.ones(x.shape)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x: np.ndarray) -> np.ndarray:
    x = sigmoid(x)
    return x * (1 - x)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def relu_prime(x: np.ndarray) -> np.ndarray:
    return 1 * (x > 0)


def tanh(x: np.ndarray) -> np.ndarray:
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def tanh_prime(x: np.ndarray) -> np.ndarray:
    return 1 - tanh(x)**2


def softmax(x: np.ndarray) -> np.ndarray:
    return np.exp(
        x - np.max(x, axis=0, keepdims=True)
    ) / np.sum(
        np.exp(x - np.max(x, axis=0, keepdims=True)),
        axis=0, keepdims=True
    )


def sse(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    return 0.5 * np.sum(np.square(y - t))


def sse_prime(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    return y - t


def cross_entropy(y: np.ndarray, t: np.ndarray, epsilon: float = 1e-15) -> np.ndarray:
    y = np.clip(y, epsilon, 1. - epsilon)
    return - np.sum(t * np.log(y))


def cross_entropy_prime(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    return - t / y


def softmax_cross_entropy(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    return cross_entropy(softmax(y), t)


def softmax_cross_entropy_prime(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    return softmax(y) - t


class Activation(Enum):

    IDENTITY = 0
    SIGMOID = 1
    RELU = 2
    TANH = 3

    def function(self):
        return [identity, sigmoid, relu, tanh][self.value]

    def prime(self):
        return [identity_prime, sigmoid_prime, relu_prime, tanh_prime][self.value]


class Loss(Enum):

    SSE = 0
    CROSS_ENTROPY = 1
    SOFTMAX_CROSS_ENTROPY = 2

    def function(self):
        return [sse, cross_entropy, softmax_cross_entropy][self.value]

    def prime(self):
        return [sse_prime, cross_entropy_prime, softmax_cross_entropy_prime][self.value]
