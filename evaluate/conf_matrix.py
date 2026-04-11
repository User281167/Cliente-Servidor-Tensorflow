import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix


def get_confusion_matrix(model, dataset) -> np.ndarray:
    """recorre el dataset y acumula predicciones."""
    all_preds = []
    all_labels = []

    for images, labels in dataset:
        predictions = model(images, training=False)
        preds_classes = tf.argmax(predictions, axis=1).numpy()
        all_preds.extend(preds_classes)
        all_labels.extend(labels.numpy())

    return confusion_matrix(all_labels, all_preds)
