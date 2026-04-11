import matplotlib.pyplot as plt
import tensorflow as tf

cifar10_classes = [
    "avión",
    "automóvil",
    "pájaro",
    "gato",
    "ciervo",
    "perro",
    "rana",
    "caballo",
    "barco",
    "camión",
]


def load_cifar10_data(
    gray=True,
    normalize=False,
    ram=False,
    distributed=False,
    batch_size=128,
    buffer_size=10000,
):
    """
    Carga y preprocesa el dataset CIFAR-10.

    Args:
        batch_size (int): Tamaño del batch.
        normalize (bool): Si se debe normalizar las imágenes.
        buffer_size (int): Tamaño del buffer para el shuffle.
        ram (bool): Si se debe cargar todo el dataset en RAM.
        distributed (bool): Si se debe configurar el dataset para DDP.

    Returns:
        train_ds, test_ds: Datasets de entrenamiento y prueba.
    """

    # Cargar dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Crear datasets tipo tf.data
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # Normalización: (x - 0.5) / 0.5  → rango [-1, 1]
    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0

        if gray:
            image = tf.image.rgb_to_grayscale(image)

        if not normalize:
            return image, label

        image = (image - 0.5) / 0.5
        return image, label

    if distributed:
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.DATA
        )
        train_ds = train_ds.with_options(options)

    # Pipeline entrenamiento
    train_ds = (
        train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(buffer_size)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Pipeline test (sin shuffle)
    test_ds = (
        test_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    if ram:
        train_ds = train_ds.cache()
        test_ds = test_ds.cache()

    return train_ds, test_ds


def plot_cifar10_images(ds, num_images=10, gray=True, normalize=False):
    plt.figure(figsize=(10, 6))
    collected = []

    # obtener datos del batch
    for images, labels in ds:
        for img, lbl in zip(images.numpy(), labels.numpy()):
            collected.append((img, lbl[0]))

            if len(collected) >= num_images:
                break

        if len(collected) >= num_images:
            break

    cols = 5
    rows = (num_images + cols - 1) // cols

    for i, (image, label) in enumerate(collected):
        # Deshacer normalización
        if normalize:
            # Rango [-1, 1]  →  [0, 1]
            image = (image.astype("float32") * 0.5 + 0.5).clip(0.0, 1.0)
        else:
            image = image.astype("float32")

        plt.subplot(rows, cols, i + 1)

        if gray:
            plt.imshow(image.squeeze(), cmap="gray", vmin=0, vmax=1)
        else:
            plt.imshow(image.clip(0, 1))

        plt.title(f"{cifar10_classes[label]} ({label})", fontsize=9)
        plt.axis("off")

    plt.tight_layout()
    plt.show()
