import tensorflow as tf


def grad_norm(grads):
    """Norma L2 de todos los gradientes"""
    total = sum(tf.reduce_sum(g**2).numpy() for g in grads if g is not None)
    return total**0.5
