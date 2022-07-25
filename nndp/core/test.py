#!/usr/bin/env python3
import numpy as np


def tanh(Z):
    tanh.derivative = tanh_derivative
    return np.tanh(Z)


def tanh_derivative(Z):
    return 1.0 - np.tanh(Z) ** 2


def leaky_relu(Z):
    leaky_relu.derivative = leaky_relu_derivative
    return np.where(Z > 0, Z, Z * 0.01)


def leaky_relu_derivative(Z):
    ret = np.ones(Z.shape)
    ret[Z < 0] = .01
    return ret


# ----------------------------------------------------------

def relu(Z):
    relu.derivative = relu_derivative
    return np.maximum(Z, 0)


def sigmoid(Z):
    sigmoid.derivative = sigmoid_derivative
    return 1 / (1 + np.power(np.e, -Z))


def relu_derivative(Z):
    dZ = np.zeros(Z.shape)
    dZ[Z > 0] = 1
    return dZ


def sigmoid_derivative(Z):
    f = 1 / (1 + np.exp(-Z))
    return f * (1 - f)


def softMaxCe(x):
    softMaxCe.derivative = derivSoftMaxCe
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


def derivSoftMaxCe(x):
    return 1


# %%


def crossEntropy(y, t):
    crossEntropy.derivative = derivCrossEntropy
    return np.sum(y * np.log(t))


def crossEntropySM(y, t):
    crossEntropySM.derivative = derivCrossEntropySM
    a = np.log(y + 1e-10)  # evitiamo che ci siano zeri
    return np.sum(t * a)


def sumOfSquare(y, t):
    sumOfSquare.derivative = derivSumOfSquare
    return np.sum(np.square(y - t))


def derivCrossEntropySM(y, t):
    return y - t


def derivCrossEntropy(y, t):
    return -y / t + (1 - t) / (1 - y)


def derivSumOfSquare(y, t):
    return y - t



class FullyConnectedLayer:
    def __init__(self, input_neuroids, n_neuroids, activation):
        self.tag = "hidden"
        self.shape = n_neuroids  # numero neuroni del layer
        self.weights = .1 * np.random.randn(n_neuroids, input_neuroids)  # init pesi
        self.bias = .1 * np.random.randn()  # init bias
        self.actifun = activation  # funzione di attivazione

        # caches [salva i dati per backpropagation e update]
        self._layer_in = None
        self._weighted_in = None
        self._delta = None
        self._cumuled_delta = np.zeros((n_neuroids, input_neuroids))
        self._cumuled_bias_delta = 0

    def forw_prop(self, layer_in):
        self._layer_in = layer_in
        self._weighted_in = self.weights @ layer_in + self.bias  # prodotto matrice-vettore
        return self.actifun(self._weighted_in)

    def back_prop(self, delta):  # ricevo dal layer successivo il delta. Lo moltiplico per la derivata della funz
        # di attivazione
        self._delta = delta * self.actifun.derivative(self._weighted_in)  # gradiente = [ Error'(z) * actifun'(z) ]
        return self.weights.T @ self._delta  # ritorno la trasposta dei pesi per il delta

    def update(self, eta=.1, mustUpdate=True, batch_size=1):
        self._cumuled_delta += self._delta @ self._layer_in.T  # accumula
        self._cumuled_bias_delta += np.sum(self._delta)

        if mustUpdate or batch_size == 1:  # se è online oppure fine del batch
            self._cumuled_delta /= batch_size  # dividi per batch_size
            self._cumuled_bias_delta /= batch_size
            self.weights -= eta * self._cumuled_delta  # aggiorna
            self.bias -= eta * self._cumuled_bias_delta
            self._cumuled_delta = np.zeros(self._cumuled_delta.shape)  # resetta accumulatori
            self._cumuled_bias_delta = 0

    def reset(self):
        self.weights = .1 * np.random.randn(self.weights.shape[0], self.weights.shape[1])


d = FullyConnectedLayer(2, 3, sigmoid)
print(d.forw_prop(np.array([[1], [2]])))
print(d.back_prop(np.array([[1], [2], [3]])))
print(d._delta)
print(d.weights)
print(d.bias)