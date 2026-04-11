import os

import pandas as pd
import tensorflow as tf

from cifar10.load_data import cifar10_classes
from utils import format_elapsed, plot_confusion_matrix, plot_grid


class EpochMetrics:
    """
    Clase para mantener las métricas de una época durante el entrenamiento.
    """

    def __init__(self, is_chief: bool):
        """
        Inicializa las métricas de la época.

        keras: usar media y sparse para calcular la pérdida y la precisión.

        Args:
            is_chief (bool): Indica si el proceso es principal worker-id == 1 o no.
        """
        self.loss = tf.keras.metrics.Mean()
        self.acc = tf.keras.metrics.SparseCategoricalAccuracy()
        self.gnorm = tf.keras.metrics.Mean()
        self.total_samples = 0

        self.is_chief = is_chief

        # solo el coordinador (chief) guarda los resultados del test data
        self.df = pd.DataFrame(
            columns=[
                "epoch",
                "loss",
                "acc",
                "eval_loss",
                "eval_acc",
                "gnorm",
                "throughput",
                "elapsed",
            ]
            if is_chief
            else [
                "epoch",
                "loss",
                "acc",
                "gnorm",
                "throughput",
                "elapsed",
            ],
        )

    def reset(self) -> None:
        """
        Resetea las métricas de la época.
        """
        self.loss.reset_state()
        self.acc.reset_state()
        self.gnorm.reset_state()
        self.total_samples = 0

    def update(self, loss, logits_list, y_list, gnorm, bs) -> None:
        """
        Minibatch usa solo una parte del conjunto de datos
        Acomular los valores de la época
        """
        self.loss.update_state(loss)
        self.gnorm.update_state(gnorm)
        self.total_samples += int(bs.numpy())

        y_local = tf.concat(y_list, axis=0)
        logits_local = tf.concat(logits_list, axis=0)
        self.acc.update_state(y_local, logits_local)

    def results(self) -> dict:
        """
        Devuelve los resultados de la época.
        """
        return {
            "loss": self.loss.result().numpy(),
            "acc": self.acc.result().numpy(),
            "gnorm": self.gnorm.result().numpy(),
            "n": self.total_samples,
        }

    def add(
        self,
        epoch: int,
        epoch_time: float,
        throughput: float,
        eval_loss: float | None = None,
        eval_acc: float | None = None,
    ) -> None:
        """
        Agrega los resultados de la época al DataFrame.
        """
        r = self.results()

        if self.is_chief:
            self.df.loc[epoch] = [
                epoch + 1,
                r["loss"],
                r["acc"],
                eval_loss,
                eval_acc,
                r["gnorm"],
                throughput,
                epoch_time,
            ]
        else:
            self.df.loc[epoch] = [
                epoch + 1,
                r["loss"],
                r["acc"],
                r["gnorm"],
                throughput,
                epoch_time,
            ]

    def print_epoch(self, epochs=int):
        """
        Imprime los resultados de la época.
        Resultado de test y train
        """
        epoch = self.df.index[-1]
        throughput = self.df["throughput"].iloc[-1]
        epoch_time = self.df["elapsed"].iloc[-1]

        # solo el coordinador imprime los resultados de test
        val_str = (
            f"eval_loss {self.df['eval_loss'].iloc[-1]:.4f} eval_acc {self.df['eval_acc'].iloc[-1]:.4f} | "
            if self.is_chief
            else ""
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"loss {self.df['loss'].iloc[-1]:.4f} acc {self.df['acc'].iloc[-1]:.4f} | "
            f"{val_str}"
            f"gnorm {self.df['gnorm'].iloc[-1]:.4f} | "
            f"{throughput:.0f} samp/s | {format_elapsed(epoch_time)}"
        )

    def save(
        self,
        save_dir: str | None = None,
        worker_index: int = 0,
        conf_matrix=None,
    ):
        """
        Guarda los resultados del entrenamiento.
            - Estadística simple con pd.describe()
            - Gráficas de loss/acc y matriz de confusión (si se proporciona)

        Args:
            save_dir (str | None): Directorio donde guardar los resultados.
            worker_index (int): Índice del worker que guarda los resultados.
            conf_matrix (np.ndarray | None): Matriz de confusión a guardar (sklearn.confusion_matrix).
        """
        description = self.df.describe(percentiles=[0.1, 0.5, 0.9])

        # guardar excel
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

            self.df.to_excel(os.path.join(save_dir, f"metrics_{worker_index}.xlsx"))

            description.to_excel(
                os.path.join(save_dir, f"description_{worker_index}.xlsx"),
                index=True,
            )

        if self.is_chief:
            plot_grid(
                history=[
                    (
                        # unir train/test en una sola gráfica
                        (self.df["loss"][i], self.df["eval_loss"][i]),
                        ((self.df["acc"][i], self.df["eval_acc"][i])),
                        self.df["gnorm"][i],
                    )
                    for i in range(len(self.df))
                ],
                labels=[
                    ("Loss", "Train", "Test"),
                    ("Accuracy", "Train", "Test"),
                    "Grad Norm",
                ],
                n_cols=1,
                save_path=save_dir,
            )

            plot_confusion_matrix(conf_matrix, class_names=cifar10_classes)
