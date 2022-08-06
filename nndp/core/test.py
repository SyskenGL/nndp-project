#!/usr/bin/env python3
import numpy as np
from scipy.special import softmax

from nndp.core.layers import Dense
from nndp.core.models import MLP
from nndp.utils.functions import Activation, Loss

np.seterr(over='ignore')


# funzioni di attivazione
def identity(x):
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(x, 0)


# derivate funzioni di attivazione
def identity_deriv(x):
    return np.ones(x.shape)


def sigmoid_deriv(x):
    z = sigmoid(x)
    return z * (1 - z)


def relu_deriv(x):
    return x > 0


# funzioni di errore
def sum_of_squares(y, t):
    return 0.5 * np.sum(np.power(y - t, 2))


def cross_entropy(y, t, epsilon=1e-15):
    y = np.clip(y, epsilon, 1. - epsilon)
    return - np.sum(t * np.log(y))


def cross_entropy_softmax(y, t):
    softmax_y = softmax(y, axis=0)
    return cross_entropy(softmax_y, t)


# derivate funzioni di errore
def sum_of_squares_deriv(y, t):
    return y - t


# da verificare
def cross_entropy_deriv(y, t):
    return - t / y


def cross_entropy_softmax_deriv(y, t):
    softmax_y = softmax(y, axis=0)
    return softmax_y - t


activation_functions = [sigmoid, relu, identity]
activation_functions_deriv = [sigmoid_deriv, relu_deriv, identity_deriv]

error_functions = [cross_entropy, cross_entropy_softmax, sum_of_squares]
error_functions_deriv = [cross_entropy_deriv, cross_entropy_softmax_deriv, sum_of_squares_deriv]

import sys
import os

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

n_activation_functions = 3
n_error_functions = 3


def __types_of_activation_functions():
    print('\n   Types of activation functions:')
    print('   1] sigmoid')
    print('   2] identity')
    print('   3] ReLU\n')


def __types_of_error_functions():
    print('\n   Types of error functions:')
    print('   1] Cross Entropy')
    print('   2] Cross Entropy Soft Max')
    print('   3] Sum of Squares\n')


def __check_int_input(value, min_value, max_value):
    if not value.isnumeric() or (not int(value) >= min_value or not int(value) <= max_value):
        raise ValueError


def __get_int_input(string, min_value=0, max_value=sys.maxsize):
    flag = False
    value = None
    while not flag:
        try:
            value = input(string)
            __check_int_input(value, min_value, max_value)
            flag = True
        except ValueError:
            print('invalid input!\n')

    return int(value)


def is_standard_conf():
    choice = __get_int_input('Do you want to use the default configuration? (Y=1 / N=0): ', 0, 1)
    os.system('clear')
    return choice


def get_nn_type():
    text = ("\nSelect a neural network type: \n"
            "1] Multilayer Neural Network \n"
            "2] Convolutional Neural Network \n\n? ")
    nn_type = __get_int_input(text, min_value=1, max_value=2)

    if nn_type != 1 and nn_type != 2:
        raise ValueError('Invalid choice')

    os.system('clear')
    return 'fc_nn' if nn_type == 1 else 'cv_nn'


def get_conf_ml_net():
    _, columns = os.popen('stty size', 'r').read().split()
    columns = int(columns)

    title = 'NEURAL NETWORK PROJECT'
    sub_title = 'creation of a multilayer neural network'

    title_space = int((columns - (len(title) + 2)) / 2)
    sub_title_space = int((columns - len(sub_title)) / 2)

    print('\n\n')
    print('-' * title_space, title, '-' * title_space, ' ' * sub_title_space, sub_title, '\n')

    n_hidden_layers = __get_int_input('define the number of hidden layers (min value = 1): ', min_value=1)

    n_hidden_nodes_per_layer = list()
    act_fun_codes = list()

    __types_of_activation_functions()

    for i in range(n_hidden_layers):
        print('hidden layer', i + 1)

        n_nodes = __get_int_input('-  number of nodes: ', 1)
        n_hidden_nodes_per_layer.append(int(n_nodes))

        act_fun_code = 0
        act_fun_codes.append(act_fun_code)

        print('\n')

    print('output layer :')
    act_fun_code = 0
    act_fun_codes.append(act_fun_code)

    __types_of_error_functions()
    error_fun_code = 0

    os.system('clear')

    return n_hidden_layers, n_hidden_nodes_per_layer, act_fun_codes, error_fun_code


def get_conf_cv_net():
    _, columns = os.popen('stty size', 'r').read().split()
    columns = int(columns)

    title = 'NEURAL NETWORK PROJECT'
    sub_title = 'creation of a convolutional neural network'

    title_space = int((columns - (len(title) + 2)) / 2)
    sub_title_space = int((columns - len(sub_title)) / 2)

    print('\n\n')
    print('-' * title_space, title, '-' * title_space, ' ' * sub_title_space, sub_title, '\n')

    n_cv_layers = __get_int_input('define the number of convolutional layers (min value = 1): ', min_value=1)

    n_kernels_per_layer = list()
    act_fun_codes = list()

    print('\n')
    for i in range(n_cv_layers):
        print('convlutional layer', i + 1)

        n_kernels = __get_int_input('-  number of kernels: ', 1)
        n_kernels_per_layer.append(int(n_kernels))

    __types_of_activation_functions()

    print('hidden layer :')
    act_fun_code = __get_int_input('-  activaction function: ', 1, n_activation_functions) - 1
    act_fun_codes.append(act_fun_code)

    n_hidden_nodes = __get_int_input('-  number of nodes: ', 1)

    print('\noutput layer :')
    act_fun_code = __get_int_input('-  activaction function: ', 1, n_activation_functions) - 1
    act_fun_codes.append(act_fun_code)

    __types_of_error_functions()
    error_fun_code = __get_int_input('-  define the error function: ', 1, n_error_functions) - 1

    os.system('clear')

    return n_cv_layers, n_kernels_per_layer, n_hidden_nodes, act_fun_codes, error_fun_code


def get_mnist_data(data):
    data = np.array(data)
    data = np.transpose(data)
    return data


def get_mnist_labels(labels):
    labels = np.array(labels)
    one_hot_labels = np.zeros((10, labels.shape[0]), dtype=int)

    for n in range(labels.shape[0]):
        label = labels[n]
        one_hot_labels[label][n] = 1

    return one_hot_labels


def get_random_dataset(X, t, n_samples=10000):
    if X.shape[1] < n_samples:
        raise ValueError

    n_tot_samples = X.shape[1]
    n_samples_not_considered = n_tot_samples - n_samples

    new_dataset = np.array([1] * n_samples + [0] * n_samples_not_considered)
    np.random.shuffle(new_dataset)

    index = np.where(new_dataset == 1)
    index = np.reshape(index, -1)

    new_X = X[:, index]
    new_t = t[:, index]

    return new_X, new_t


def get_scaled_data(X):
    X = X.astype('float32')
    X = X / 255.0
    return X


def train_test_split(X, t, test_size=0.25):
    n_samples = X.shape[1]
    test_size = int(n_samples * test_size)
    train_size = n_samples - test_size

    dataset = np.array([1] * train_size + [0] * test_size)
    np.random.shuffle(dataset)

    train_index = np.where(dataset == 1)
    train_index = np.reshape(train_index, -1)

    X_train = X[:, train_index]
    t_train = t[:, train_index]

    test_index = np.where(dataset == 0)
    test_index = np.reshape(test_index, -1)

    X_test = X[:, test_index]
    t_test = t[:, test_index]

    return X_train, X_test, t_train, t_test


def convert_to_cnn_input(X, image_size):
    n_instances = X.shape[1]
    new_X = np.empty(shape=(n_instances, image_size, image_size))

    for i in range(n_instances):
        new_X[i] = X[:, i].reshape(image_size, image_size)

    return new_X


def get_metric_value(y, t, metric):
    print(t.shape)
    print(y.shape)
    pred = np.argmax(y, axis=0)
    target = np.argmax(t, axis=0)

    pred = pred.tolist()
    target = target.tolist()

    if metric == 'accuracy':
        return accuracy_score(pred, target)
    elif metric == 'precision':
        return precision_score(pred, target, average='macro', zero_division=0)
    elif metric == 'recall':
        return recall_score(pred, target, average='macro', zero_division=0)
    elif metric == 'f1':
        return f1_score(pred, target, average='macro', zero_division=0)

    raise ValueError()


def print_result(y_test, t_test):
    accuracy = get_metric_value(y_test, t_test, 'accuracy')
    precision = get_metric_value(y_test, t_test, 'precision')
    recall = get_metric_value(y_test, t_test, 'recall')
    f1 = get_metric_value(y_test, t_test, 'f1')

    print('\n')
    print('-' * 63)
    print('Performance on test set\n')
    print(y_test)
    print('     accuracy: {:.2f} - precision: {:.2f} - recall: {:.2f} - f1: {:.2f}\n\n'.format(accuracy, precision,
                                                                                               recall, f1))


import numpy as np

from copy import deepcopy


def __get_delta(net, t, layers_input, layers_output):
    delta = list()
    for i in range(net.n_layers):
        delta.append(np.zeros(net.nodes_per_layer[i]))

    for i in range(net.n_layers - 1, -1, -1):
        act_fun_deriv = activation_functions_deriv[net.act_fun_code_per_layer[i]]

        if i == net.n_layers - 1:
            # calcolo delta nodi di output
            error_fun_deriv = error_functions_deriv[net.error_fun_code]
            delta[i] = act_fun_deriv(layers_input[i]) * error_fun_deriv(layers_output[i], t)

        else:
            # calcolo delta nodi interni
            delta[i] = act_fun_deriv(layers_input[i]) * np.dot(np.transpose(net.weights[i + 1]), delta[i + 1])

    return delta


def __get_weights_bias_deriv(net, x, delta, layers_output):
    weights_deriv = []
    bias_deriv = []

    for i in range(net.n_layers):
        if i == 0:
            weights_deriv.append(np.dot(delta[i], np.transpose(x)))
        else:
            weights_deriv.append(np.dot(delta[i], np.transpose(layers_output[i - 1])))
        bias_deriv.append(delta[i])

    return weights_deriv, bias_deriv


def __standard_gradient_descent(net, weights_deriv, bias_deriv, eta):
    for i in range(net.n_layers):
        net.weights[i] = net.weights[i] - (eta * weights_deriv[i])
        net.bias[i] = net.bias[i] - (eta * bias_deriv[i])
    return net


def batch_learning(net, X_train, t_train, X_val, t_val, eta=0.001, n_epochs=500):
    train_errors, val_errors = list(), list()

    error_fun = error_functions[net.error_fun_code]

    best_net, min_error = None, None
    tot_weights_deriv, tot_bias_deriv = None, None

    n_instances = X_train.shape[1]

    for epoch in range(n_epochs):
        # somma delle derivate
        print('Epoch {} / {}'.format(epoch + 1, n_epochs))
        for n in range(n_instances):
            # si estrapolano singole istanze come vettori colonna
            x = X_train[:, n].reshape(-1, 1)
            t = t_train[:, n].reshape(-1, 1)
            weights_deriv, bias_deriv = __back_propagation(net, x, t)

            if n == 0:
                tot_weights_deriv = deepcopy(weights_deriv)
                tot_bias_deriv = deepcopy(bias_deriv)
            else:
                for i in range(net.n_layers):
                    tot_weights_deriv[i] = np.add(tot_weights_deriv[i], weights_deriv[i])
                    tot_bias_deriv[i] = np.add(tot_bias_deriv[i], bias_deriv[i])

        net = __standard_gradient_descent(net, tot_weights_deriv, tot_bias_deriv, eta)

        y_train = net.sim(X_train)
        y_val = net.sim(X_val)

        train_error = error_fun(y_train, t_train)
        val_error = error_fun(y_val, t_val)

        train_errors.append(train_error)
        val_errors.append(val_error)

        train_accuracy = get_metric_value(y_train, t_train, 'accuracy')
        val_accuracy = get_metric_value(y_val, t_val, 'accuracy')

        print('     train loss: {:.2f} - val loss: {:.2f}'.format(train_error, val_error))
        print('     train accuracy: {:.2f} - val accuracy: {:.2f}\n'.format(train_accuracy, val_accuracy))

        if best_net is None or val_error < min_error:
            min_error = val_error
            best_net = deepcopy(net)

    return best_net


def __back_propagation(net, x, t):
    # x: singola istanza
    layers_input, layers_output = net.forward_step(x)
    delta = __get_delta(net, t, layers_input, layers_output)
    weights_deriv, bias_deriv = __get_weights_bias_deriv(net, x, delta, layers_output)

    return weights_deriv, bias_deriv


import os


class MultilayerNet:
    def __init__(self, n_hidden_layers, n_hidden_nodes_per_layer, act_fun_codes, error_fun_code):
        self.n_input_nodes = 784  # dipende dal dataset: mnist_in = 784
        self.n_output_nodes = 10  #  dipende dal dataset: mnist_out = 10
        self.n_layers = n_hidden_layers + 1

        self.error_fun_code = error_fun_code
        self.act_fun_code_per_layer = act_fun_codes.copy()

        self.nodes_per_layer = n_hidden_nodes_per_layer.copy()
        self.nodes_per_layer.append(self.n_output_nodes)

        self.weights = list()
        self.bias = list()

        self.__initialize_weights_and_bias()

    def __initialize_weights_and_bias(self):
        mu, sigma = 0, 0.1

        for i in range(self.n_layers):
            if i == 0:
                self.weights.append(np.random.normal(mu, sigma, size=(self.nodes_per_layer[i], self.n_input_nodes)))
            else:
                self.weights.append(
                    np.random.normal(mu, sigma, size=(self.nodes_per_layer[i], self.nodes_per_layer[i - 1])))

            self.bias.append(np.random.normal(mu, sigma, size=(self.nodes_per_layer[i], 1)))

    def forward_step(self, x):
        layers_input = list()
        layers_output = list()

        for i in range(self.n_layers):
            if i == 0:
                input = np.dot(self.weights[i], x) + self.bias[i]
                layers_input.append(input)

            else:
                input = np.dot(self.weights[i], layers_output[i - 1]) + self.bias[i]
                layers_input.append(input)

            act_fun = activation_functions[self.act_fun_code_per_layer[i]]
            output = act_fun(input)
            layers_output.append(output)

        return layers_input, layers_output

    def sim(self, x):
        for i in range(self.n_layers):
            if i == 0:
                input = np.dot(self.weights[i], x) + self.bias[i]
            else:
                input = np.dot(self.weights[i], output) + self.bias[i]

            act_fun = activation_functions[self.act_fun_code_per_layer[i]]
            output = act_fun(input)

        return output

    def print_config(self):

        print("• input layer: {:>11} nodes".format(self.n_input_nodes))

        error_fun = error_functions[self.error_fun_code]
        error_fun = error_fun.__name__

        for i in range(self.n_layers):
            act_fun = activation_functions[self.act_fun_code_per_layer[i]]
            act_fun = act_fun.__name__

            if i != self.n_layers - 1:
                print("• hidden layer {}: {:>8} nodes, ".format(i + 1, self.nodes_per_layer[i]),
                      "{:^10} \t (activation function)".format(act_fun))

            else:
                print("• output layer: {:>10} nodes, ".format(self.n_output_nodes),
                      "{:^10} \t (activation function)".format(act_fun))

        print("\n {} (error function)".format(error_fun))

        print('\n')


from mnist import MNIST
from nndp.utils.collections import Set


def run():


    mndata = MNIST('../data/mnist')
    X, t = mndata.load_training()
    X = get_mnist_data(X)
    t = get_mnist_labels(t)
    X, t = get_random_dataset(X, t, n_samples=1000)
    X = get_scaled_data(X)

    X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.25)
    X_train, X_val, t_train, t_val = train_test_split(X_train, t_train, test_size=0.25)

    X_train, X_val, t_train, t_val = X_train.T, X_val.T, t_train.T, t_val.T


    nn = MLP(
        [
            Dense(10, Activation.SIGMOID),
            Dense(10, Activation.SIGMOID),
            Dense(10, Activation.IDENTITY)
        ],
        Loss.SSE
    )

    nn.build(784)

    t_train = np.reshape(t_train, (*t_train.shape, 1))
    t_val = np.reshape(t_val, (*t_val.shape, 1))
    print(X_train.shape)
    print(t_train.shape)

    nn.fit(Set(X_train, t_train), Set(X_val, t_val), learning_rate=0.005, n_batches=0, epochs=30000, target_validation_accuracy=0.9)

def run1():
    # parametri di default
    n_hidden_layers = 2
    n_hidden_nodes_per_layer = [10, 10]
    act_fun_codes = [1, 1, 2]
    error_fun_code = 1

    # caricamento dataset
    mndata = MNIST('../data/mnist')
    X, t = mndata.load_training()
    X = get_mnist_data(X)
    t = get_mnist_labels(t)

    X, t = get_random_dataset(X, t, n_samples=1000)
    X = get_scaled_data(X)

    X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.25)
    X_train, X_val, t_train, t_val = train_test_split(X_train, t_train, test_size=0.25)

    net = MultilayerNet(n_hidden_layers=n_hidden_layers, n_hidden_nodes_per_layer=n_hidden_nodes_per_layer,
                        act_fun_codes=act_fun_codes, error_fun_code=error_fun_code)

    net.print_config()

    net = batch_learning(net, X_train, t_train, X_val, t_val)
    y_test = net.sim(X_test)
    print_result(y_test, t_test)

run()
