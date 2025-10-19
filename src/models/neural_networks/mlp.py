from typing import List, Tuple

import keras
import numpy as np
import tensorflow as tf


class MLP:
    def __init__(self, observation_shape: int, hidden_shapes: Tuple = (64, 64), output_shape: int = 1, lr: float | np.float32 = 1e-4):
        inputs = keras.layers.Input(shape=(observation_shape,))

        x = inputs
        for shape in hidden_shapes:
            x = keras.layers.Dense(shape, activation='relu', kernel_initializer='he_uniform')(x)

        outputs = keras.layers.Dense(output_shape, activation='linear', kernel_initializer='he_uniform')(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.optimizer = keras.optimizers.Adam(learning_rate=lr, global_clipnorm=0.5) # type: ignore

    def forward(self, x: List | np.ndarray | tf.Tensor) -> tf.Tensor:
        output = self.model(x)
        return output
