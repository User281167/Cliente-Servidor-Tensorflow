import argparse

import tensorflow as tf

from evaluate import evaluate_train, get_confusion_matrix
from train import train_minibatch
from utils import format_elapsed, plot_confusion_matrix, plot_grid, time_wrapper

from .load_data import load_cifar10_data, plot_cifar10_images


@time_wrapper
def train(
    gray=True,
    normalize=False,
    batch_size=128,
    buffer_size=10000,
    ram=False,
    lr=0.001,
    epochs=20,
):
    train_dataset, test_dataset = load_cifar10_data(
        gray=gray,
        normalize=normalize,
        ram=ram,
        buffer_size=buffer_size,
        batch_size=batch_size,
    )

    plot_cifar10_images(train_dataset, gray=gray, normalize=normalize)

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(32, 32, 1 if gray else 3)),
            tf.keras.layers.Dense(128),
            tf.keras.layers.LeakyReLU(negative_slope=0.01),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32),
            tf.keras.layers.LeakyReLU(negative_slope=0.01),
            tf.keras.layers.Dense(10),
        ]
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    log_interval = epochs // 10 or 1
    history = []

    for epoch in range(epochs):
        loss, acc, gnorm, elapsed, throughput = train_minibatch(
            model, optimizer, loss_fn, train_dataset
        )
        loss_test, acc_test = evaluate_train(model, loss_fn, test_dataset)
        history.append(((loss, loss_test), (acc, acc_test), gnorm))

        if epoch % log_interval == 0 or epoch == epochs - 1 or epoch == 0:
            print(
                f"Epoch {epoch + 1:02d}/{epochs} | "
                f"Loss: {loss:.4f} | Acc: {acc * 100:.2f}% | "
                f"Test Loss: {loss_test:.4f} | Test Acc: {acc_test * 100:.2f}% | "
                f"GNorm: {gnorm:.4f} | "
                f"Throughput: {throughput:.0f} samples/s | "
                f"Time: {format_elapsed(elapsed)}"
            )

    conf_maxtrix = get_confusion_matrix(model, test_dataset)
    plot_grid(
        history,
        labels=[("Loss", "Train", "Test"), ("Acc", "Train", "Test"), "Grad Norm"],
    )
    plot_confusion_matrix(conf_maxtrix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar CIFAR-10 en local")
    parser.add_argument("--rgb", action="store_true", help="Usar imagenes en RGB")
    parser.add_argument(
        "--normalize", action="store_true", help="Normalizar imagenes [-1, 1]"
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--buffer-size", type=int, default=10000, help="Buffer size para shuffle"
    )
    parser.add_argument("--ram", action="store_true", help="Cargar datos en RAM")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="Número de epochs")
    args = parser.parse_args()

    train(
        gray=not args.rgb,
        normalize=args.normalize,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        ram=args.ram,
        lr=args.lr,
        epochs=args.epochs,
    )
