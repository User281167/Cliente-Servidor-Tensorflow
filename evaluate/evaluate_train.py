import tensorflow as tf


def evaluate_train(model, loss_fn, dataset):
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    n_batches = 0

    for images, labels in dataset:
        predictions = model(images, training=False)
        loss = loss_fn(labels, predictions)
        total_loss += loss.numpy()
        preds_classes = tf.argmax(predictions, axis=1, output_type=tf.int32)
        total_correct += tf.reduce_sum(
            tf.cast(preds_classes == tf.cast(labels, tf.int32), tf.int32)
        ).numpy()

        total_samples += images.shape[0]
        n_batches += 1

    return total_loss / n_batches, total_correct / total_samples
