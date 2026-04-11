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


def load_cifar10_ddp(
    buffer_size,
    gray=True,
    normalize=False,
    ram=False,
    global_batch_size=None,
):
    """
    Carga y preprocesa el dataset CIFAR-10 para DDP.

    Args:
        buffer_size (int): Tamaño del buffer para el shuffle.
        gray (bool): Si se debe convertir las imágenes a escala de grises.
        normalize (bool): Si se debe normalizar las imágenes. [0, 1] -> [-1, 1]
        ram (bool): Si se debe cargar todo el dataset en RAM.
        global_batch_size (int): Tamaño del batch global para DDP.

    Returns:
        train_ds, test_ds: Datasets de entrenamiento y prueba.
    """

    def dataset_fn(input_context):
        (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
        y_train = y_train.flatten()

        def preprocess(image, label):
            if gray:
                image = tf.image.rgb_to_grayscale(image)

            image = tf.cast(image, tf.float32) / 255.0

            if normalize:
                image = (image - 0.5) / 0.5

            return image, label

        ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))

        ds = ds.shard(
            input_context.num_input_pipelines,
            input_context.input_pipeline_id,
        )

        ds = ds.repeat()
        ds = ds.shuffle(buffer_size, reshuffle_each_iteration=True)
        ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

        ds = ds.batch(
            input_context.get_per_replica_batch_size(global_batch_size),
            drop_remainder=True,
        )

        if ram:
            ds = ds.cache()

        ds = ds.prefetch(tf.data.AUTOTUNE)

        return ds

    return dataset_fn


def load_cifar10_eval(
    gray=True,
    normalize=False,
    batch_size=256,
):
    """
    Carga el conjunto de datos de evaluación CIFAR-10.

    Args:
        gray (bool): Si True, convierte las imágenes a escala de grises.
        normalize (bool): Si True, normaliza las imágenes.
        batch_size (int): Tamaño del batch.

    Returns:
        tf.data.Dataset: Dataset de evaluación.
    """
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_test = y_test.flatten()

    def preprocess(image, label):
        if gray:
            image = tf.image.rgb_to_grayscale(image)

        image = tf.cast(image, tf.float32) / 255.0

        if normalize:
            image = (image - 0.5) / 0.5

        return image, label

    ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def plot_cifar10_images(ds, num_images=10, gray=True, normalize=False):
    plt.figure(figsize=(10, 6))
    collected = []

    # obtener datos del batch
    for images, labels in ds:
        for img, lbl in zip(images.numpy(), labels.numpy()):
            collected.append((img, int(lbl)))

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
