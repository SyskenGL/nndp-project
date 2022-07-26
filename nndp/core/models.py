#!/usr/bin/env python3
from __future__ import annotations
import uuid
import numpy as np
from tabulate import tabulate
from nndp.math.functions import Loss
from nndp.core.layers import Layer
from nndp.utils.collections import Set
from nndp.errors import EmptyModelError
from nndp.utils.decorators import require_built
from nndp.utils.decorators import require_not_built


class MLP:

    def __init__(
        self,
        layers: list[Layer] = None,
        loss: Loss = Loss.SSE,
        name: str = None,
    ):
        self._name = (
            name if name is not None
            else f"{self.__class__.__name__}_{str(uuid.uuid4())[:8]}"
        )
        self._layers = layers if layers else []
        self._loss = loss

    def is_built(self) -> bool:
        return (
            len(self._layers) != 0 and
            all([layer.is_built() for layer in self._layers])
        )

    @require_not_built
    def build(
        self,
        in_size: int,
        weights: list[np.ndarray] = None,
        biases: list[np.ndarray] = None
    ) -> None:
        if len(self._layers) == 0:
            raise EmptyModelError(
                f"attempt to build an empty model {self.__class__.__name__}"
            )
        if in_size <= 0:
            raise ValueError(
                f"expected in_size value greater than zero, received {in_size}."
            )
        if weights is not None and len(weights) != len(self._layers):
            raise ValueError(
                f"expected a list of weights of "
                f"length {len(self._layers)}, received {len(weights)}."
            )
        if biases is not None and len(biases) != len(self._layers):
            raise ValueError(
                f"expected a list of biases of "
                f"length {len(self._layers)}, received {len(biases)}."
            )
        for h in range(0, self.depth):
            self._layers[h].build(
                self._layers[h - 1].width if h != 0 else in_size,
                weights[h] if weights else None,
                biases[h] if biases else None
            )

    @require_not_built
    def add(self, layer: Layer) -> None:
        self._layers.append(layer)

    @require_not_built
    def add_all(self, layers: list[Layer]) -> None:
        self._layers.extend(layers)

    @require_not_built
    def pop(self):
        self._layers.pop()

    """
    @require_built
    def predict(self, in_data: np.ndarray) -> np.ndarray:
        self._forward_propagation(in_data)
        return self.out_data

    def fit(
        self,
        training_set: Set,
        learning_rate: float = .001,
        n_batches: int = 1,
        epochs: int = 500
    ) -> None:
        if not 0 <= learning_rate <= 1:
            raise ValueError("learning rate must be in [0, 1].")
        if not (0 <= n_batches <= training_set.size):
            raise ValueError(f"n_batches must be in [0, {training_set.size}].")
        batches = [
            Set(data, labels)
            for data, labels in
            zip(
                np.array_split(training_set.data, n_batches),
                np.array_split(training_set.labels, n_batches)
            )
        ] if n_batches not in [0, 1] else [training_set]
        for epoch in range(0, epochs):
            print(f"Epoch #{epoch}.")
            for batch in batches:
                for instance in range(0, batch.size):
                    self._forward_propagation(batch.data[instance])
                    self._backward_propagation(batch.labels[instance])
                    self._update(
                        learning_rate
                        if (n_batches == 0 or instance == batch.size - 1)
                        else None
                    )
    
    @require_built
    def _update(self, learning_rate) -> None:
        for layer in self._layers:
            layer.update(learning_rate)

    """

    @require_built
    def _forward_propagation(self, in_data: np.ndarray) -> None:
        for layer in self._layers:
            in_data = layer.forward_propagation(in_data)

    @require_built
    def _backward_propagation(self, expected: np.ndarray) -> None:
        expected = np.reshape(expected, (-1, 1))
        delta = self._loss.prime()(self.out_data, expected)
        for layer in reversed(self._layers):
            delta = layer.backward_propagation(delta)

    @property
    def name(self) -> str:
        return self._name

    @property
    def depth(self) -> int:
        return len(self._layers)

    @property
    def size(self) -> int:
        return sum([layer.width for layer in self._layers]) + self.in_size

    @property
    def width(self) -> int:
        return (
            max([layer.width for layer in self._layers] + [self.in_size])
            if len(self._layers) else 0
        )

    @property
    def loss(self) -> Loss:
        return self._loss

    @property
    def in_size(self) -> int:
        return self._layers[0].in_size if len(self._layers) else 0

    @property
    def in_data(self) -> np.ndarray:
        return self._layers[0].in_data if len(self._layers) else None

    @property
    def out_size(self) -> int:
        return self._layers[-1].width if len(self._layers) else 0

    @property
    def out_data(self) -> np.ndarray:
        return self._layers[-1].out_data if len(self._layers) else None

    @property
    def layers(self) -> tuple[Layer]:
        return tuple(self._layers)

    @property
    def n_trainable(self) -> tuple[Layer]:
        return (
            sum([layer.weights.size + layer.biases.size for layer in self._layers])
            if self.is_built() else 0
        )

    def __str__(self):
        details = str(tabulate(
            [
                ["\033[1m TYPE \033[0m", self.__class__.__name__],
                ["\033[1m NAME \033[0m", self._name],
                ["\033[1m DEPTH \033[0m", self.depth],
                ["\033[1m WIDTH \033[0m", self.width],
                ["\033[1m SIZE \033[0m", self.size],
                [
                    "\033[1m IN_SIZE \033[0m",
                    self.in_size if self.in_size else "-"
                ],
                [
                    "\033[1m OUT_SIZE \033[0m",
                    self.out_size if self.out_size else "-"
                ],
                ["\033[1m LOSS \033[0m", self._loss.function().__name__],
                [
                    "\033[1m # TRAINABLE \033[0m",
                    self.n_trainable if self.is_built() else "-"
                ],
                ["\033[1m BUILT\033[0m", self.is_built()]
            ],
            tablefmt="fancy_grid",
            colalign=["center"] * 2
        ))
        layers = "\n".join([str(layer) for layer in self._layers])
        structure = str(tabulate([[
                self._layers[h].name if h != -1 else "-",
                ("HIDDEN" if h != self.depth - 1 else "OUTPUT")
                if h != -1 else "INPUT",
                self._layers[h].width if h != -1
                else (self.in_size if self.in_size else "-"),
            ] for h in range(-1, self.depth)],
            tablefmt="fancy_grid",
            colalign=["center"] * 3
        )) if len(self._layers) != 0 else ""
        return str(tabulate(
            [[
                details,
                layers if layers != "" else "-",
                structure if structure != "" else "-"]],
            headers=[
                "\033[1m DETAILS \033[0m",
                "\033[1m LAYERS \033[0m",
                "\033[1m STRUCTURE \033[0m"
            ],
            tablefmt="fancy_grid",
            colalign=["center"] * 3
        ))


"""

from layers import Dense
from nndp.math.functions import Activation

mlp = MLP([
    Dense(2, Activation.SIGMOID),
    Dense(3, Activation.SIGMOID)
])
weights = [
    np.array([[-0.02231122, -0.10919746,  0.07450828],
 [-0.0025994,  -0.03248617, -0.02875975]]
),
    np.array([[-0.04484072,  0.05372628],
 [ 0.05604744,  0.11775328],
 [ 0.09757671,  0.03080518]]
)
]
biases = [
    np.array([[-0.0469785359818356], [-0.0469785359818356]]),
    np.array([[-0.144365218079866], [-0.144365218079866], [-0.144365218079866]])
]
mlp.build(3, weights, biases)
mlp._forward_propagation(np.array([1, 2, 3]))
mlp._backward_propagation(np.array([7,8,7]))
print(mlp.out_data)
for layer in mlp.layers:
    print(layer._delta)

from mnist import MNIST
from layers import Dense
from nndp.math.functions import Activation, Loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

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

def get_metric_value(y, t, metric):
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
    print('-'*63)
    print('Performance on test set\n')
    print('     accuracy: {:.2f} - precision: {:.2f} - recall: {:.2f} - f1: {:.2f}\n\n'.format(accuracy, precision, recall, f1))

mndata = MNIST('../data/mnist')
X, t = mndata.load_training()
X = get_mnist_data(X)
t = get_mnist_labels(t)

X, t = get_random_dataset(X, t, n_samples=1000)
X = get_scaled_data(X)

X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.25)
X_train, X_test, t_train, t_test = X_train.T, X_test.T, t_train.T, t_test.T

nn = MLP(
    [
        Dense(10, Activation.SIGMOID),
        Dense(10, Activation.SIGMOID),
        Dense(10, Activation.IDENTITY),
    ],
    Loss.CROSS_ENTROPY
)
nn.build(784)

nn.fit(Set(X_train, t_train), n_batches=0, learning_rate=0.001, epochs=5000)
y_test = np.squeeze(np.array([nn.predict(test) for test in X_test]))
print_result(y_test, t_test)
print(nn.predict(X_train[0]), t_train[0])
"""