#!/usr/bin/env python3
import numpy as np
from time import time
import sys

# ----------------------------------------------------------

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

# ----------------------------------------------------------

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

# ----------------------------------------------------------

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

# ----------------------------------------------------------

class NeuralNetwork:

    def __init__(self, eta=.1, target_loss=None, target_epochs=None, target_acc=None, mode='on-line', batch_size=1,
                 loss_fun='sum-of-square'):
        self.layers = []
        self.eta = eta  # learning rate  0 < eta <= 1
        assert (target_loss is not None or target_epochs is not None or target_acc is not None)
        # almeno una condizione di stop ci deve essere
        self.target_epochs = target_epochs
        self.target_loss = target_loss
        self.target_acc = target_acc
        self.mode = mode  # [full-batch/mini-batch/on-line]
        self.loss_metric = loss_fun
        self.learn_curve = []
        self.curr_loss = None

        if mode == 'on-line':
            self.batch_size = 1
        else:
            self.batch_size = batch_size

        if loss_fun == 'sum-of-square':
            self.loss_fun = sumOfSquare
        else:
            self.loss_fun = crossEntropySM

    def add_layer(self, layer):
        self.layers.append(layer)

    def add_all(self, layers):
        for layer in layers:
            self.add_layer(layer)

    def fit(self, X, y, valid_X=None, valid_y=None):
        n_sample = X.shape[0]

        if self.mode == 'full-batch':
            self.batch_size = n_sample

        target = self._one_hot(y)

        epoch = 0
        train_acc = 0
        valid_acc = 0
        err = 1
        start = time()
        ret = []
        try:
            while (self.target_epochs is None or self.target_epochs > epoch) and \
                    (self.target_loss is None or err > self.target_loss) and \
                    (self.target_acc is None or train_acc < self.target_acc):  # fino alla condizione di stop
                epoch += 1
                err = 0
                for i in range(n_sample):  # per ogni dato nel t.s.
                    # bisogna aggiornare i pesi se ci troviamo alla fine del batch
                    to_update = (i % self.batch_size + 1 == self.batch_size)
                    self._back_prop(X[i, :], target[i, :], self.batch_size, to_update)

                    err += np.sum(np.abs(self.curr_loss))
                    sys.stdout.write(f"epoch {epoch} processing sample {i + 1}/{n_sample} curr loss:{err / (i + 1)}\r")
                    sys.stdout.flush()

                err /= n_sample
                print()
                sys.stdout.write(f"calculating training accuracy...\r")
                sys.stdout.flush()

                train_acc = (n_sample - np.count_nonzero(self.predict(X) - y)) / n_sample  # accuracy
                print(f"epoch {epoch} training accuracy: {train_acc}")
                if valid_X is not None:
                    sys.stdout.write(f"calculating validation accuracy...\r")
                    sys.stdout.flush()
                    valid_acc = (valid_X.shape[0] - np.count_nonzero(self.predict(valid_X) - valid_y)) / valid_X.shape[
                        0]
                    print(f"epoch {epoch} validation accuracy: {valid_acc}")

                ret.append([epoch, err, train_acc, valid_acc])  # dati sul training
        except KeyboardInterrupt:
            print("\ntraining stopped by user\n")

        print(f"elapsed time: {time() - start} s")
        return np.array(ret)

    def predict(self, x):
        ret = []
        for i in range(x.shape[0]):
            ret.append(self._forw_prop(x[i, :]))
        return np.array(ret)

    def _forw_prop(self, x):  # propago x per ogni layer
        for layer in self.layers:
            x = layer.forw_prop(x)
        return x.T

    def _back_prop(self, x, t):  # se necessario, upgrado i pesi. Per aggiornarli mi serve
        curr = x  # batch_size.
        for layer in self.layers:  # propaga avanti
            curr = layer.forw_prop(curr)

        self.curr_loss = self.loss_fun(curr, t)  # calcolo perdita
        curr = self.loss_fun.derivative(curr, t)  # calcolo gradiente

        if self.loss_metric == 'cross-entropy':  # normalizza perdita
            self.curr_loss /= np.log(t.shape[0])

        for layer in reversed(self.layers):  # calcola i delta
            curr = layer.back_prop(curr)


    def _one_hot(self, y):  # 1 in corrispondenza della 'parola', 0 altrimenti
        n_class = np.max(y) + 1
        oh = np.zeros((y.shape[0], n_class))
        for i in range(y.shape[0]):
            oh[i, y[i]] = 1
        return oh

    def print(self):
        for i in range(len(self.layers)):
            print(f" {i}) {self.layers[i].tag} layer shape {self.layers[i].shape}")

    def reset(self):
        for layer in self.layers:
            layer.reset()


nn = NeuralNetwork(target_epochs=500)
nn.add_all([
    FullyConnectedLayer(3, 2, sigmoid),
    FullyConnectedLayer(2, 3, sigmoid),
    FullyConnectedLayer(3, 5, sigmoid)
])

print(nn.predict(np.array([[1, 2, 3], [1, 1, 1], [1, 4, 3]])))
