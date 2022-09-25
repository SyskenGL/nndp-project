#!/usr/bin/env python3
import logging
import csv
from nndp.data.loader import MNIST
from nndp.utils.metrics import Metric, EarlyStop
from nndp.utils.functions import Activation, Loss
from layers import ResilientDense
from nndp.core.models import MLP


if __name__ == "__main__":

    # Abilitazione logging
    logging.basicConfig(format='\n\n%(message)s\n', level=logging.DEBUG)

    # Creazione del caricatore dataset MNIST
    loader = MNIST()

    # Suddivisione dataset in training set, validation test e test set
    dataset = loader.scaled_dataset.random(instances=1000)
    training_set, validation_set = dataset.split(0.25)

    mlp = MLP(
        [
            ResilientDense(100, Activation.SIGMOID),
            ResilientDense(10, Activation.IDENTITY),
        ],
        Loss.SOFTMAX_CROSS_ENTROPY
    )
    mlp.build(784)

    stats = mlp.fit(
        training_set,
        validation_set,
        epochs=300,
        stats=[Metric.LOSS, Metric.ACCURACY, Metric.F1],
        early_stops=[EarlyStop(Metric.LOSS, 0.8, greedy=True)],
        eta_negative=0.25,
        eta_positive=1.05
    )
