import keras
import tensorflow as tf


class CNN(tf.Module):
    """
    Creates sequential model consisting of CNNs.
    """
    def __init__(
        self,
        timestamps: int = 3,
        features: int = 6,
        output_shape: int = 3,
        hidden_layers: int = 32,
        hidden_units: int = 32,
        lr: float = 1e-5,
        kernel_size: int = 3,
        clipnorm: float | None = None,
        name = None
        )-> None:
        """
        Initializes instance.

        Args:
            timestamps (int, optional): number of observations passed together. Defaults to 3.
            features (int, optional): number of features. Defaults to 6.
            output_shape (int, optional): output shape. Defaults to 3.
            hidden_layers (int, optional): number of CNNs in model. Defaults to 32.
            hidden_units (int, optional): dimensionality of CNNs outputs, doubles with each layer. Defaults to 32.
            lr (float, optional): learning rate. Defaults to 1e-5.
            kernel_size (int, optional): size of the convolution window. Defaults to 3.
            clipnorm (float | None, optional): gradient clipping. Defaults to None.
            name (_type_, optional): model name. Defaults to None.
        """
        super().__init__(name)

        self.model = keras.Sequential([
            keras.Input(shape=(timestamps, features)),
            *[
                keras.layers.Conv1D(
                    filters=(hidden_units*(2**i)),
                    kernel_size=kernel_size,
                    padding='same',
                    activation='relu',
                    kernel_initializer='he_uniform'
                )
                for i in range(hidden_layers)
            ],
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(hidden_units*(2**(hidden_layers-1)), activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(output_shape, activation='linear', kernel_initializer='he_uniform'),
        ])

        # OPTIONAL (FOR DEBUG ONLY!)
        self.model.summary()

        self.optimizer = keras.optimizers.Adam(learning_rate=lr, global_clipnorm=clipnorm)


    @tf.function
    def forward(self, x: tf.Tensor) -> tf.Tensor:
        return self.model(x)


    @property
    def trainable_variables(self) -> list[str]:
        return self.model.trainable_variables


    def save(self, path: str) -> None:
        ckpt = tf.train.Checkpoint(model=self.model)
        ckpt.write(path)


    def load(self, path: str) -> None:
        ckpt = tf.train.Checkpoint(model=self.model)
        ckpt.restore(path).expect_partial()


    def __del__(self) -> None:
        del self.model
        keras.backend.clear_session()
