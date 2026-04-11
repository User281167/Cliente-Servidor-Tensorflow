import tensorflow as tf


def create_train_step(strategy, model, optimizer, loss_fn):
    """
    Se necesita hacer override de model.fit
    Esto para evitar Perreplica en tensores

    train_step
    Entrenamiento minibatch
        - Calculo de metricas (loss, accuracy, norma L2 de los gradientes)
    """

    @tf.function
    def train_step(dist_inputs):
        def step_fn(inputs):
            x, y = inputs

            with tf.GradientTape() as tape:
                logits = model(x, training=True)
                loss = loss_fn(y, logits)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # norma de pesos
            # wright_norm = tf.linalg.global_norm(model.trainable_variables)

            gnorm = tf.linalg.global_norm(grads)

            return loss, logits, tf.cast(tf.shape(x)[0], tf.float32), gnorm

        # reduce evitar el Perreplica en tensores
        # Necesario para evitar el error en model.fit
        per_replica = strategy.run(step_fn, args=(dist_inputs,))
        loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica[0], axis=None)

        # Concatenar logits y labels para métricas en Python
        logits = strategy.experimental_local_results(per_replica[1])
        bs = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica[2], axis=None)
        gnorm = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica[3], axis=None)

        return loss, logits, bs, gnorm

    return train_step
