"""
Modelo CIFAR-10 para clasificación de imágenes.

Permite usar el model.fit() de Keras.
"""

import tensorflow as tf


class Cifar10Model(tf.keras.Model):
    def __init__(self, gray=True, conv=False):
        """
        Inicializa el modelo CIFAR-10.

        Args:
            gray (bool): Si es True, usa imágenes en escala de grises.
            conv (bool): Si es True, usa una red convolucional.
        """
        super().__init__()

        if conv:
            self.net = tf.keras.Sequential(
                [
                    tf.keras.layers.Conv2D(
                        32, 3, padding="same", input_shape=(32, 32, 1 if gray else 3)
                    ),
                    tf.keras.layers.LeakyReLU(0.01),
                    tf.keras.layers.SpatialDropout2D(0.3),
                    tf.keras.layers.MaxPooling2D(2),
                    tf.keras.layers.Conv2D(64, 3, padding="same"),
                    tf.keras.layers.LeakyReLU(0.01),
                    tf.keras.layers.MaxPooling2D(2),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(128),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.LeakyReLU(0.01),
                    tf.keras.layers.Dense(10),
                ]
            )
        else:
            self.net = tf.keras.Sequential(
                [
                    tf.keras.layers.Flatten(input_shape=(32, 32, 1 if gray else 3)),
                    tf.keras.layers.Dense(128),
                    tf.keras.layers.LeakyReLU(0.01),
                    tf.keras.layers.Dropout(0.4),
                    tf.keras.layers.Dense(32),
                    tf.keras.layers.LeakyReLU(0.01),
                    tf.keras.layers.Dense(10),
                ]
            )

        # Métrica de grad norm para exponer en el log de model.fit()
        self._loss_tracker = tf.keras.metrics.Mean(name="loss")
        self._acc_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        self._gnorm_tracker = tf.keras.metrics.Mean(name="gnorm")

    def call(self, x, training=None):
        return self.net(x, training=training)

    def train_step(self, data):
        images, labels = data

        with tf.GradientTape() as tape:
            predictions = self(images, training=True)
            loss = self.compiled_loss(labels, predictions)

        grads = tape.gradient(loss, self.trainable_variables)
        gnorm = tf.linalg.global_norm(grads)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Actualizar métricas propias
        self._loss_tracker.update_state(loss)
        self._acc_tracker.update_state(labels, predictions)
        self._gnorm_tracker.update_state(gnorm)

        return {
            "loss": self._loss_tracker.result(),
            "accuracy": self._acc_tracker.result(),
            "gnorm": self._gnorm_tracker.result(),
        }

    def test_step(self, data):
        """Controla qué aparece en val_ (logs en el fit)."""
        images, labels = data
        predictions = self(images, training=False)
        loss = self.compiled_loss(labels, predictions)

        self._loss_tracker.update_state(loss)
        self._acc_tracker.update_state(labels, predictions)

        return {
            "loss": self._loss_tracker.result(),
            "accuracy": self._acc_tracker.result(),
        }

    @property
    def metrics(self):
        # Keras llama reset_state() sobre estas métricas al inicio de cada epoch
        return [self._loss_tracker, self._acc_tracker, self._gnorm_tracker]
