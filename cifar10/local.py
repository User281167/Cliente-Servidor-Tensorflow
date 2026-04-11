import argparse

from .load_data import load_cifar10_data, plot_cifar10_images


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar CIFAR-10 en local")
    parser.add_argument("--rgb", action="store_true", help="Usar imagenes en RGB")
    parser.add_argument(
        "--normalize", action="store_true", help="Normalizar imagenes [-1, 1]"
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--buffer-size", type=int, default=10000, help="Buffer size")
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
