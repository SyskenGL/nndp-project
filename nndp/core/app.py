#!/usr/bin/env python3
import logging
import matplotlib.pyplot as plt
from nndp.data.loader import MNIST
from nndp.utils.metrics import Metric, Target
from nndp.utils.functions import Activation, Loss
from layers import Dense
from nndp.core.models import MLP


if __name__ == "__main__":

    # Abilitazione logging
    logging.basicConfig(format='\n\n%(message)s\n', level=logging.DEBUG)

    # Creazione del caricatore dataset MNIST
    loader = MNIST()

    # Suddivisione dataset in training set, validation test e test set
    dataset = loader.scaled_dataset.random(instances=10000)
    dataset, validation_test = dataset.split(1000)
    training_set, test_set = dataset.split(1000)

    # Creazione di una rete MLP deep con 2 strati interni
    mlp = MLP(
        [
            Dense(10, Activation.SIGMOID),
            Dense(10, Activation.SIGMOID),
            Dense(10, Activation.IDENTITY),
        ],
        Loss.SOFTMAX_CROSS_ENTROPY
    )

    # Costruzione della rete - 784 è la dimensione
    # di un'istanza del dataset MNIST
    mlp.build(784)

    print(mlp)
    input("\n > Press enter to start... ")

    # Addestramento della rete con 500 epoche
    # Condizioni di early-stop (debole):
    #   - Loss <= 50
    #   - Accuracy >= 0.9
    #   - F1-Score >= 0.9
    # Statistiche desiderate [LOSS, ACCURACY, F1]
    stats = mlp.fit(
        training_set,
        validation_test,
        epochs=300,
        targets=[
            Target(Metric.LOSS, 50),
            Target(Metric.ACCURACY, 0.9),
            Target(Metric.F1, 0.9)
        ],
        weak_stop=True,
        stats=[Metric.LOSS, Metric.ACCURACY, Metric.F1]
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

    # Validazione k-fold
    print(mlp.validate(test_set, metrics=[Metric.LOSS, Metric.ACCURACY, Metric.F1]))

    # Salvataggio della rete
    mlp.save()
