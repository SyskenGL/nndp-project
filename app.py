#!/usr/bin/env python3
import logging
import matplotlib.pyplot as plt
from nndp.data.loader import MNIST
from nndp.utils.metrics import Metric, EarlyStop
from nndp.utils.functions import Activation, Loss
from nndp.core.layers import Dense
from nndp.core.models import MLP


if __name__ == "__main__":

    logging.basicConfig(format='\n\n%(message)s\n', level=logging.DEBUG)
    loader = MNIST()

    dataset = loader.scaled_dataset.random(instances=10000)
    dataset, validation_test = dataset.split(2500)
    training_set, test_set = dataset.split(2500)

    mlp = MLP(
        [
            Dense(10, Activation.RELU),
            Dense(10, Activation.RELU),
            Dense(10, Activation.IDENTITY),
        ],
        Loss.SOFTMAX_CROSS_ENTROPY
    )

    mlp.build(784)

    print(mlp)
    input("\n > Press enter to start... ")

    stats = mlp.fit(
        training_set,
        validation_test,
        epochs=10,
        early_stops=[
            EarlyStop(Metric.LOSS, 50),
            EarlyStop(Metric.ACCURACY, 0.9),
            EarlyStop(Metric.F1, 0.9)
        ],
        weak_stop=True,
        stats=[Metric.LOSS, Metric.ACCURACY, Metric.F1],
        eta=0.0001
    )

    figure, axis = plt.subplots(3)

    training_loss = [stats["training"]["loss"][epoch] for epoch in range(stats["epochs"])]
    validation_loss = [stats["validation"]["loss"][epoch] for epoch in range(stats["epochs"])]
    axis[0].set_title("Loss")
    axis[0].plot(range(stats["epochs"]), training_loss, label="Training")
    axis[0].plot(range(stats["epochs"]), validation_loss, label="Validation")

    training_accuracy = [stats["training"]["accuracy"][epoch] for epoch in range(stats["epochs"])]
    validation_accuracy = [stats["validation"]["accuracy"][epoch] for epoch in range(stats["epochs"])]
    axis[1].set_title("Accuracy")
    axis[1].plot(range(stats["epochs"]), training_accuracy, label="Training")
    axis[1].plot(range(stats["epochs"]), validation_accuracy, label="Validation")

    training_f1 = [stats["training"]["f1"][epoch] for epoch in range(stats["epochs"])]
    validation_f1 = [stats["validation"]["f1"][epoch] for epoch in range(stats["epochs"])]
    axis[2].set_title("F1-Score")
    axis[2].plot(range(stats["epochs"]), training_f1, label="Training")
    axis[2].plot(range(stats["epochs"]), validation_f1, label="Validation")

    plt.legend()
    plt.show()

    print(
        f"Performance on test set: "
        f"{mlp.validate(test_set, metrics=[Metric.LOSS, Metric.ACCURACY, Metric.F1])}"
    )

    mlp.save()