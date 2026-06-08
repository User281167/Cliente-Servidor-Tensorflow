from pathlib import Path

import numpy as np
import tensorflow as tf
from datasets import load_dataset
from PIL import Image

DATASET_ID = "zh-plus/tiny-imagenet"
NUM_CLASSES = 200
TRAIN_SAMPLES = 100_000
VALID_SAMPLES = 10_000
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _pil_to_uint8(image) -> np.ndarray:
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    if image.mode != "RGB":
        image = image.convert("RGB")

    return np.asarray(image, dtype=np.uint8)


def _load_hf_split(dataset_id: str, split: str, keep_in_memory: bool = False):
    return load_dataset(dataset_id, split=split, keep_in_memory=keep_in_memory)


def get_label_names(dataset_id: str = DATASET_ID) -> list[str]:
    hf_ds = load_dataset(dataset_id, split="valid")
    return hf_ds.features["label"].names


def _prepare_source(
    dataset_id: str,
    split: str,
    num_shards: int | None = None,
    shard_index: int | None = None,
    ram: bool = False,
):
    """
    Cargar dataset solo cargar el dataset necesario por worker
    """

    hf_ds = _load_hf_split(dataset_id, split=split, keep_in_memory=False)

    if num_shards is not None and shard_index is not None:
        hf_ds = hf_ds.shard(num_shards=num_shards, index=shard_index, contiguous=True)

    if not ram:
        return hf_ds

    images = []
    labels = []

    print(f"Precargando {len(hf_ds)} imagenes Tiny ImageNet ({split}) en RAM...")

    for sample in hf_ds:
        images.append(_pil_to_uint8(sample["image"]))
        labels.append(int(sample["label"]))

    print("Precarga Tiny ImageNet completa.")
    return list(zip(images, labels))


def _sample_generator(source):
    """
    En ram ya se tiene uint8 en disco se tiene imágen
    """

    for sample in source:
        if isinstance(sample, tuple):
            yield sample
        else:
            yield _pil_to_uint8(sample["image"]), int(sample["label"])


def _tf_preprocess(image, label):
    """
    Normalizar datos de entrenamiento

    1. resnet entrada de 224x224 resize necesario para tinyimagenet 64x64
    2. valores de [0, 255] -> [0, 1]
    3. valores normalizados mean | std
    """
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE), antialias=True)
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD

    return image, tf.cast(label, tf.int32)


def _make_tf_dataset(source):
    output_signature = (
        tf.TensorSpec(shape=(64, 64, 3), dtype=tf.uint8),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    return tf.data.Dataset.from_generator(
        lambda: _sample_generator(source),
        output_signature=output_signature,
    )


def load_tiny_imagenet_ddp(
    global_batch_size: int,
    buffer_size: int = 10_000,
    dataset_id: str = DATASET_ID,
    split: str = "train",
    ram: bool = False,
):
    """
    Dataset function para MultiWorkerMirroredStrategy.
    Aplicar suffle entre workers y épocas.
    """

    def dataset_fn(input_context):
        source = _prepare_source(
            dataset_id=dataset_id,
            split=split,
            num_shards=input_context.num_input_pipelines,
            shard_index=input_context.input_pipeline_id,
            ram=ram,
        )

        ds = _make_tf_dataset(source)

        if split == "train":
            ds = ds.repeat()
            ds = ds.shuffle(buffer_size, reshuffle_each_iteration=True)

        ds = ds.map(_tf_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(
            input_context.get_per_replica_batch_size(global_batch_size),
            drop_remainder=True,
        )

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.OFF
        )
        ds = ds.with_options(options)

        return ds.prefetch(tf.data.AUTOTUNE)

    return dataset_fn


def load_tiny_imagenet_eval(
    batch_size: int = 256,
    dataset_id: str = DATASET_ID,
    split: str = "valid",
    ram: bool = False,
):
    source = _prepare_source(dataset_id=dataset_id, split=split, ram=ram)
    ds = _make_tf_dataset(source)
    ds = ds.map(_tf_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)

    return ds.prefetch(tf.data.AUTOTUNE)


def save_index0_sample(
    save_dir: str,
    dataset_id: str = DATASET_ID,
    split: str = "train",
) -> None:
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hf_ds = _load_hf_split(dataset_id, split=split, keep_in_memory=False)
    sample = hf_ds[0]
    image = Image.fromarray(_pil_to_uint8(sample["image"]))
    label = int(sample["label"])

    image.save(out_dir / f"{split}_index0.png")
    (out_dir / f"{split}_index0.txt").write_text(
        f"dataset: {dataset_id}\nsplit: {split}\nindex: 0\nlabel: {label}\n",
        encoding="utf-8",
    )
