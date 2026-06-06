import numpy as np
import tensorflow as tf
from huggingface_hub import snapshot_download

from .load_data import IMAGE_SIZE, NUM_CLASSES


def _download_resnet18_weights() -> str:
    repo_dir = snapshot_download(repo_id="tfimm/resnet18", allow_patterns=["model/*"])
    return f"{repo_dir}/model"


def create_resnet18_classifier(
    num_classes: int = NUM_CLASSES,
    preset: str = "resnet18",
    pretrained: bool = True,
    weights_path: str | None = None,
    train_backbone: bool = False,
    dropout: float = 0.2,
) -> tf.keras.Model:
    model_path = weights_path

    if pretrained and model_path is None:
        model_path = _download_resnet18_weights()

    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="image")

    # Keras 3: cargar SavedModel como TFSMLayer
    backbone = tf.keras.layers.TFSMLayer(
        model_path,
        call_endpoint="serving_default",
    )
    backbone.trainable = train_backbone

    features = backbone(inputs)

    # TFSMLayer devuelve un dict — extraer el tensor de features
    if isinstance(features, dict):
        features = list(features.values())[0]

    if len(features.shape) == 4:
        features = tf.keras.layers.GlobalAveragePooling2D(name="pool")(features)

    x = features
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout, name="dropout")(x)

    logits = tf.keras.layers.Dense(num_classes, name="classifier")(x)
    return tf.keras.Model(inputs=inputs, outputs=logits, name="tiny_imagenet_resnet18")


def save_resnet_weights(model, path="clasificador_head.npy"):
    weights_to_save = {}

    for layer in model.layers:
        if layer.trainable and not isinstance(layer, tf.keras.layers.TFSMLayer):
            for var in layer.variables:
                weights_to_save[var.name] = var.numpy()

    # Guardar como archivo .npy (NumPy) - Instantáneo y sin bloqueos
    np.save(path, weights_to_save)
    print(f"Guardado exitoso: {len(weights_to_save)} tensores.")


def load_resnet_weights(
    weights_path="clasificador_head.npy", train_backbone=False, dropout=0.2
):
    model = create_resnet18_classifier(
        num_classes=NUM_CLASSES,
        train_backbone=train_backbone,
        dropout=dropout,
    )

    weights = np.load(weights_path, allow_pickle=True).item()

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.TFSMLayer):
            continue
        if layer.trainable:
            for v in layer.variables:
                if v.name in weights:
                    v.assign(weights[v.name])

    return model
