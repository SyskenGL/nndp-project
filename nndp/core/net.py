#!/usr/bin/env python3
import numpy as np
from nndp.error import IncompatibleLayersError
from nndp.core.layer import Dense


class Multilayer:

    def __init__(self, layers: list[Dense] = None):
        if len(layers) > 1:
            for h in range(1, len(layers)):
                if layers[h].in_size != layers[h - 1].width:
                    raise IncompatibleLayersError(
                        f"in_size {layers[h].in_size} is not equal "
                        f"to previous layer width {layers[h - 1].width}."
                    )
        self.layers = layers if layers else []

    def add(self, layer: Dense):
        if len(self.layers) > 1:
            if layer.in_size != self.layers[-1].width:
                raise IncompatibleLayersError(
                    f"in_size {layer.in_size} is not equal "
                    f"to previous layer width {self.layers[-1].width}."
                )
        self.layers.append(layer)

    def feed_forward(self, in_data: np.ndarray) -> np.ndarray:
        out_data = in_data
        for layer in self.layers:
            print(layer)
            out_data = layer.feed_forward(out_data)
        return out_data
