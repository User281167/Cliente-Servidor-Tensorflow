import tensorflow as tf


def create_train_step(strategy, model, optimizer, loss_fn):
    # loss_fn debe usar reduction="none" para control manual
    loss_fn_no_reduce = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction="none",
    )

    @tf.function
    def train_step(dist_inputs):
        def step_fn(inputs):
            x, y = inputs
            global_batch_size = tf.cast(
                strategy.num_replicas_in_sync * tf.shape(x)[0], tf.float32
            )
            with tf.GradientTape() as tape:
                logits = model(x, training=True)
                per_sample_loss = loss_fn_no_reduce(y, logits)

                # Normalizar sobre el batch global, no el local
                loss = tf.reduce_sum(per_sample_loss) / global_batch_size

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            grad_norm = tf.linalg.global_norm(grads)
            batch_size = tf.cast(tf.shape(x)[0], tf.float32)
            return loss, logits, batch_size, grad_norm

        per_replica = strategy.run(step_fn, args=(dist_inputs,))

        # SUM cada réplica ya normalizó por global_batch_size
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
