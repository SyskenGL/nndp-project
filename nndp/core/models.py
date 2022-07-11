#!/usr/bin/env python3
import numpy as np
from tabulate import tabulate
from nndp.errors import IncompatibleLayersError
from nndp.core.layers import Layer


class Sequential:

    def __init__(self, layers: list[Layer] = None):
        self.layers = layers if layers else []
        if len(self.layers):
            for h in range(1, len(self.layers)):
                if self.layers[h].in_size != self.layers[h - 1].width:
                    raise IncompatibleLayersError(
                        f"in_size {self.layers[h].in_size} is not equal "
                        f"to previous layer width {self.layers[h - 1].width}."
                    )

    def add(self, layer: Layer):
        if len(self.layers):
            if layer.in_size != self.layers[-1].width:
                raise IncompatibleLayersError(
                    f"in_size {layer.in_size} is not equal "
                    f"to previous layer width {self.layers[-1].width}."
                )
        self.layers.append(layer)

    def feed_forward(self, in_data: np.ndarray) -> np.ndarray:
        out_data = in_data
        for layer in self.layers:
            out_data = layer.feed_forward(out_data)
        return out_data

    def __str__(self) -> str:
        return (
            " ■ Neural Network: \n" +
            str(tabulate(
                [
                    ["TYPE", self.__class__.__name__],
                    ["INPUT", self.layers[0].in_size if len(self.layers) else 0],
                    ["OUTPUT", self.layers[-1].in_size if len(self.layers) else 0],
                    ["LAYERS", len(self.layers)],
                    ["LOSS", None]
                ],
                tablefmt="fancy_grid",
                colalign=["center"]*2
            )) +
            "\n\n ■ Detailed Neural Network: \n" +
            str(tabulate(
                [[
                    ("I:" if (h == 0) else ("O:" if (h == len(self.layers) - 1) else "H:")) + str(h),
                    self.layers[h].name,
                    self.layers[h].type,
                    self.layers[h].in_size,
                    self.layers[h].width,
                    self.layers[h].activation.function().__name__
                ] for h in range(0, len(self.layers))],
                headers=["DEPTH", "NAME", "TYPE", "IN_SIZE", "WIDTH", "ACTIVATION"],
                tablefmt="fancy_grid",
                colalign=["center"] * 6
            ))
        )
