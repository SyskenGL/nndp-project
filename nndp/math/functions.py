#!/usr/bin/env python3
import numpy as np
from enum import Enum
from scipy.special import softmax


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


def sse(y: np.ndarray, t: np.ndarray) -> float:
    return 0.5 * np.sum(np.power(y - t, 2))


def sse_prime(y: np.ndarray, t: np.ndarray):
    return y - t


def cross_entropy(y: np.ndarray, t: np.ndarray, epsilon=1e-12) -> float:
    y = np.clip(y, epsilon, 1. - epsilon)
    return - np.sum(t * np.log(y))


def cross_entropy_prime(y: np.ndarray, t: np.ndarray) -> float:
    return (- t / y) + (1 - t) / (1 - y)


def softmax_cross_entropy(y: np.ndarray, t: np.ndarray) -> float:
    return cross_entropy(softmax(y), t)


def softmax_cross_entropy_prime(y: np.ndarray, t: np.ndarray) -> float:
    ce = cross_entropy_prime(y, t)
    return ce - t


class Activation(Enum):

    IDENTITY = 0
    SIGMOID = 1
    RELU = 2
    TANH = 3

    def function(self):
        return [
            identity,
            sigmoid,
            relu,
            tanh
        ][self.value]

    def prime(self):
        return [
            identity_prime,
            sigmoid_prime,
            relu_prime,
            tanh_prime
        ][self.value]


class Error(Enum):

    SSE = 0
    CROSS_ENTROPY = 1
    SOFTMAX_CROSS_ENTROPY = 2

    def function(self):
        return [
            sse,
            cross_entropy,
            softmax_cross_entropy
        ][self.value]

    def prime(self):
        return [
            sse_prime,
            cross_entropy_prime,
            softmax_cross_entropy_prime
        ][self.value]