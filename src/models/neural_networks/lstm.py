import keras
import tensorflow as tf


class LSTM(tf.Module):
    def __init__(self, observation_shape: int, hidden_shape: int, output_shape: int, lr: float = 1e-5, name = None):
        super().__init__(name)

        inputs = keras.layers.LSTM(
            units=hidden_shape,
            kernel_initializer='he_uniform'
        )

        outputs = keras.layers.Dense(
            units=output_shape,
            activation='linear',
            kernel_initializer='he_uniform'
        )

        self.model = keras.Sequential(
            [inputs, outputs]
        )

        self.optimizer = keras.optimizers.Adam(learning_rate=lr, global_clipnorm=0.5)

    def forward(self, x: tf.Tensor) -> tf.Tensor:
        shape = (x.shape[0], 1, x.shape[1])
        inputs = tf.reshape(x, shape=shape)
        return self.model(inputs)

    @property
    def trainable_variables(self) -> list[str]:
        return self.model.trainable_variables

    def save(self, path: str):
        ckpt = tf.train.Checkpoint(model=self)
        ckpt.write(path)

    def load(self, path: str):
        ckpt = tf.train.Checkpoint(model=self)
        ckpt.restore(path).expect_partial()