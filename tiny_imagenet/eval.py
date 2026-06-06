import tensorflow as tf


def create_eval_step(strategy, model, loss_fn):
    @tf.function
    def eval_step(dist_batch):
        def step_fn(batch):
            x, y = batch
            logits = model(x, training=False)
            loss = loss_fn(y, logits)
            return loss, y, logits

        return strategy.run(step_fn, args=(dist_batch,))

    return eval_step


def run_eval(strategy, eval_step, eval_dataset):
    """
    Ejecutar evaluación en el conjunto de validación de manera distribuida.
    """

    loss_m = tf.keras.metrics.Mean()
    acc_m = tf.keras.metrics.SparseCategoricalAccuracy()
    top5_m = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)
    batch_count = 0

    for batch in eval_dataset:
        loss, y, logits = eval_step(batch)

        for l, yi, li in zip(
            strategy.experimental_local_results(loss),
            strategy.experimental_local_results(y),
            strategy.experimental_local_results(logits),
        ):
            loss_m.update_state(l)
            acc_m.update_state(yi, li)
            top5_m.update_state(yi, li)

        if batch_count % 10 == 0:
            print(
                f"Eval step {batch_count} | "
                f"loss {loss_m.result().numpy():.4f} | "
                f"acc {acc_m.result().numpy():.4f} | "
                f"top5 {top5_m.result().numpy():.4f}",
                end="\r",
            )

    return (
        float(loss_m.result().numpy()),
        float(acc_m.result().numpy()),
        float(top5_m.result().numpy()),
    )
