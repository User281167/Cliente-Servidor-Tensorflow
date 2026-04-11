import time

import tensorflow as tf

from .grad_norm import grad_norm


@tf.function  # compila el step a TF graph
def _train_batch(model, optimizer, loss_fn, images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, predictions, grads


def train_minibatch(model, optimizer, loss_fn, dataset):
    """
    Minibatch estándar con métricas completas:
    loss, acc, grad_norm, elapsed, throughput
    """
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    n_batches = 0
    last_gnorm = 0.0

    start_time = time.perf_counter()

    for images, labels in dataset:
        loss, predictions, grads = _train_batch(
            model, optimizer, loss_fn, images, labels
        )

        last_gnorm = grad_norm(grads)
        total_loss += loss.numpy()
        preds_classes = tf.argmax(predictions, axis=1, output_type=tf.int32)
        total_correct += tf.reduce_sum(
            tf.cast(preds_classes == tf.cast(labels, tf.int32), tf.int32)
        ).numpy()
        total_samples += images.shape[0]
        n_batches += 1

    elapsed = time.perf_counter() - start_time
    throughput = total_samples / elapsed
    avg_loss = total_loss / n_batches
    avg_acc = total_correct / total_samples

    return avg_loss, avg_acc, last_gnorm, elapsed, throughput
