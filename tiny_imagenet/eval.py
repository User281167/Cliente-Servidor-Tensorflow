import tensorflow as tf


def create_eval_step(model, loss_fn):
    @tf.function
    def eval_step(batch):
        x, y = batch
        logits = model(x, training=False)
        loss = loss_fn(y, logits)
        return loss, y, logits

    return eval_step


def run_eval(eval_step, eval_dataset):
    """
    Run para acomular métricas de manera distribuida
    """

    loss_m = tf.keras.metrics.Mean()
    acc_m = tf.keras.metrics.SparseCategoricalAccuracy()
    top5_m = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)

    for batch in eval_dataset:
        loss, y, logits = eval_step(batch)
        loss_m.update_state(loss)
        acc_m.update_state(y, logits)
        top5_m.update_state(y, logits)

    return (
        float(loss_m.result().numpy()),
        float(acc_m.result().numpy()),
        float(top5_m.result().numpy()),
    )
