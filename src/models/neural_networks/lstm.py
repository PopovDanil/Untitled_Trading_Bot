import keras
import tensorflow as tf


class LSTM(tf.Module):
    def __init__(self, timestamps: int = 3, features: int = 6, output_shape: int = 3, hidden_layers: int = 32, hidden_units: int = 32, lr: float = 1e-5, name = None):
        super().__init__(name)

        self.model = keras.Sequential([
            keras.Input(shape=(timestamps, features)),
            *[
                keras.layers.LSTM(hidden_units, kernel_initializer='orthogonal', dropout=0.2, return_sequences=True, implementation=2)
                for _ in range(hidden_layers)
            ],
            keras.layers.GRU(max(hidden_units // 2, 4)),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(output_shape, activation='linear', kernel_initializer='he_uniform'),
        ])

        # self.model.summary()

        self.optimizer = keras.optimizers.Adam(learning_rate=lr, global_clipnorm=0.5)

    @tf.function
    def forward(self, x: tf.Tensor) -> tf.Tensor:
        return self.model(x)

    @property
    def trainable_variables(self) -> list[str]:
        return self.model.trainable_variables

    def save(self, path: str):
        ckpt = tf.train.Checkpoint(model=self.model)
        ckpt.write(path)

    def load(self, path: str):
        ckpt = tf.train.Checkpoint(model=self.model)
        ckpt.restore(path).expect_partial()

    def __del__(self):
        del self.model
        keras.backend.clear_session()