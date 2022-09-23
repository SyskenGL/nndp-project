#!/usr/bin/env python3
import logging
import csv
from nndp.data.loader import MNIST
from nndp.utils.metrics import Metric
from nndp.utils.functions import Activation, Loss
from layers import ResilientDense
from nndp.core.models import MLP


if __name__ == "__main__":

    # Abilitazione logging
    logging.basicConfig(format='\n\n%(message)s\n', level=logging.DEBUG)

    # Creazione del caricatore dataset MNIST
    loader = MNIST()

    # Suddivisione dataset in training set, validation test e test set
    dataset = loader.scaled_dataset.random(instances=5000)

    with open("k_fold.csv", mode="w") as file:
        writer = csv.DictWriter(file, ["neurons", "eta_negative", "eta_positive", "accuracy"])
        writer.writeheader()
        for neurons in [20, 40, 60, 80, 100]:
            for eta_negative in [0.1, 0.3, 0.5, 0.7, 0.9]:
                for eta_positive in [1.1, 1.3, 1.5, 1.7, 1.9]:
                    mlp = MLP(
                        [
                            ResilientDense(neurons, Activation.SIGMOID),
                            ResilientDense(10, Activation.IDENTITY),
                        ],
                        Loss.SOFTMAX_CROSS_ENTROPY
                    )
                    mlp.build(784)
                    results = mlp.cross_validate(
                        dataset,
                        5,
                        metrics=[Metric.ACCURACY],
                        epochs=50,
                        eta_negative=eta_negative,
                        eta_positive=eta_positive
                    )
                    print(
                        f"neurons: {neurons} - eta negative: {eta_negative} - "
                        f"eta_positive {eta_positive} - results: {results}"
                    )
                    writer.writerow({
                        "neurons": neurons,
                        "eta_negative": eta_negative,
                        "eta_positive": eta_positive,
                        "accuracy": f"avg: {results['accuracy'][0]} - std: {results['accuracy'][1]}"
                    })
