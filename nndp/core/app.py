#!/usr/bin/env python3
import logging
from nndp.data.loader import MNIST
from nndp.utils.metrics import Metric, Target
from nndp.utils.functions import Activation, Loss
from layers import Dense
from nndp.core.models import MLP


if __name__ == "__main__":

    logging.basicConfig(format='\n\n%(message)s\n', level=logging.DEBUG)

    # Creazione del caricatore dataset MNIST
    loader = MNIST()

    # Suddivisione dataset in training set e validation test
    dataset = loader.scaled_dataset.random(instances=10000)
    training_set, validation_test = dataset.split(0.25)

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
        epochs=500,
        targets=[
            Target(Metric.LOSS, 50),
            Target(Metric.ACCURACY, 0.9),
            Target(Metric.F1, 0.9)
        ],
        weak_stop=True,
        stats=[Metric.LOSS, Metric.ACCURACY, Metric.F1]
    )

    # Validazione k-fold
    cross_validate = mlp.cross_validate(
        training_set, metrics=[Metric.LOSS, Metric.ACCURACY, Metric.F1]
    )

    # Salvataggio della rete
    mlp.save()
