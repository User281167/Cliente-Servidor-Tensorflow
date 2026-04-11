import argparse

import tensorflow as tf

from evaluate import get_confusion_matrix
from utils import plot_confusion_matrix, plot_grid, time_wrapper

from .load_data import load_cifar10_data, plot_cifar10_images
from .model import Cifar10Model


@time_wrapper
def train(
    conv=False,
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

    model = Cifar10Model(gray=gray, conv=conv)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    hist = model.fit(train_dataset, epochs=epochs, validation_data=test_dataset)

    # Extraer valores
    history = list(
        zip(
            zip(hist.history["loss"], hist.history["val_loss"]),
            zip(hist.history["accuracy"], hist.history["val_accuracy"]),
            hist.history["gnorm"],
        )
    )

    plot_grid(
        list(history),
        labels=[("Loss", "Train", "Test"), ("Acc", "Train", "Test"), "Grad Norm"],
    )
    conf_maxtrix = get_confusion_matrix(model, test_dataset)
    plot_confusion_matrix(conf_maxtrix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar CIFAR-10 en local")
    parser.add_argument("--conv", action="store_true", help="Usar modelo convolucional")
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
        conv=args.conv,
        gray=not args.rgb,
        normalize=args.normalize,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        ram=args.ram,
        lr=args.lr,
        epochs=args.epochs,
    )
