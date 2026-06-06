import os

import pandas as pd
import tensorflow as tf

from utils import format_elapsed, plot_grid


class EpochMetrics:
    """
    Gestionar métricas locales, acomular, limpiar y guardar métricas entre épocas.
    """

    def __init__(self, is_chief: bool):
        self.loss = tf.keras.metrics.Mean()
        self.acc = tf.keras.metrics.SparseCategoricalAccuracy()
        self.top5 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)
        self.grad_norm = tf.keras.metrics.Mean()
        self.total_samples = 0
        self.is_chief = is_chief

        columns = ["epoch", "loss", "acc", "top5", "grad_norm", "throughput", "elapsed"]
        if is_chief:
            columns += ["eval_loss", "eval_acc", "eval_top5"]

        self.df = pd.DataFrame(columns=columns)

    def reset(self) -> None:
        self.loss.reset_state()
        self.acc.reset_state()
        self.top5.reset_state()
        self.grad_norm.reset_state()
        self.total_samples = 0

    def update(self, loss, logits_list, labels_list, grad_norm, batch_size) -> None:
        """
        En cada batch se debe actualizar las métricas
        """

        labels = tf.concat(labels_list, axis=0)
        logits = tf.concat(logits_list, axis=0)

        self.loss.update_state(loss)
        self.acc.update_state(labels, logits)
        self.top5.update_state(labels, logits)
        self.grad_norm.update_state(grad_norm)
        self.total_samples += int(batch_size.numpy())

    def results(self) -> dict:
        return {
            "loss": float(self.loss.result().numpy()),
            "acc": float(self.acc.result().numpy()),
            "top5": float(self.top5.result().numpy()),
            "grad_norm": float(self.grad_norm.result().numpy()),
            "n": self.total_samples,
        }

    def add(
        self,
        epoch: int,
        epoch_time: float,
        throughput: float,
        eval_loss: float | None = None,
        eval_acc: float | None = None,
        eval_top5: float | None = None,
    ) -> None:
        r = self.results()
        row = [
            epoch + 1,
            r["loss"],
            r["acc"],
            r["top5"],
            r["grad_norm"],
            throughput,
            epoch_time,
        ]

        if self.is_chief:
            row += [eval_loss, eval_acc, eval_top5]

        self.df.loc[epoch] = row

    def print_epoch(self, epochs: int) -> None:
        row = self.df.iloc[-1]
        val = ""
        if self.is_chief:
            val = (
                f"eval_loss {row['eval_loss']:.4f} "
                f"eval_acc {row['eval_acc']:.4f} eval_top5 {row['eval_top5']:.4f} | "
            )

        print(
            f"Epoch {int(row['epoch'])}/{epochs} | "
            f"loss {row['loss']:.4f} acc {row['acc']:.4f} top5 {row['top5']:.4f} | "
            f"{val}"
            f"gnorm {row['grad_norm']:.4f} | "
            f"{row['throughput']:.0f} samp/s | {format_elapsed(row['elapsed'])}"
        )

    def save(self, save_dir: str | None, worker_index: int) -> None:
        if not save_dir:
            return

        os.makedirs(save_dir, exist_ok=True)
        self.df.to_excel(os.path.join(save_dir, f"metrics_{worker_index}.xlsx"))
        self.df.describe(percentiles=[0.1, 0.5, 0.9]).to_excel(
            os.path.join(save_dir, f"description_{worker_index}.xlsx")
        )

        if self.is_chief:
            history = [
                (
                    (self.df["loss"][i], self.df["eval_loss"][i]),
                    (self.df["acc"][i], self.df["eval_acc"][i]),
                    (self.df["top5"][i], self.df["eval_top5"][i]),
                    self.df["grad_norm"][i],
                )
                for i in range(len(self.df))
            ]
            plot_grid(
                history=history,
                labels=[
                    ("Loss", "Train", "Valid"),
                    ("Accuracy", "Train", "Valid"),
                    ("Top5", "Train", "Valid"),
                    "Grad Norm",
                ],
                n_cols=2,
                save_path=save_dir,
            )
