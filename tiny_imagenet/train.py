import tensorflow as tf


def create_train_step(strategy, model, optimizer, loss_fn):
    @tf.function
    def train_step(dist_inputs):
        def step_fn(inputs):
            x, y = inputs

            with tf.GradientTape() as tape:
                logits = model(x, training=True)
                loss = loss_fn(y, logits)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            grad_norm = tf.linalg.global_norm(grads)
            batch_size = tf.cast(tf.shape(x)[0], tf.float32)
            return loss, logits, batch_size, grad_norm

        per_replica = strategy.run(step_fn, args=(dist_inputs,))
        loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica[0], axis=None)
        logits = strategy.experimental_local_results(per_replica[1])
        batch_size = strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica[2], axis=None
        )
        grad_norm = strategy.reduce(
            tf.distribute.ReduceOp.MEAN, per_replica[3], axis=None
        )

        return loss, logits, batch_size, grad_norm

    return train_step
