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
    dataset = loader.scaled_dataset.random(instances=50000)
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
        epochs=5000,
        early_stops=[EarlyStop(Metric.LOSS, 5, True)],
        stats=[Metric.LOSS, Metric.ACCURACY, Metric.F1],
        eta_negative=0.25,
        eta_positive=1.05
    )

    epochs = stats["epochs"]
    training_loss = [stats["training"]["loss"][epoch] for epoch in range(stats["epochs"])]
    validation_loss = [stats["validation"]["loss"][epoch] for epoch in range(stats["epochs"])]
    training_accuracy = [stats["training"]["accuracy"][epoch] for epoch in range(stats["epochs"])]
    validation_accuracy = [stats["validation"]["accuracy"][epoch] for epoch in range(stats["epochs"])]
    training_f1 = [stats["training"]["f1"][epoch] for epoch in range(stats["epochs"])]
    validation_f1 = [stats["validation"]["f1"][epoch] for epoch in range(stats["epochs"])]

    with open("100_0-25_1-05.csv", "w") as file:
        writer = csv.DictWriter(file, [
            "epoch",
            "training_loss",
            "validation_loss",
            "training_accuracy",
            "validation_accuracy",
            "training_f1",
            "validation_f1"
        ], delimiter=';')
        writer.writeheader()
        for epoch in range(epochs):
            writer.writerow({
                "epoch": epoch,
                "training_loss": training_loss[epoch],
                "validation_loss": validation_loss[epoch],
                "training_accuracy": training_accuracy[epoch],
                "validation_accuracy": validation_accuracy[epoch],
                "training_f1": training_f1[epoch],
                "validation_f1": validation_f1[epoch]
            })