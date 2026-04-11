import tensorflow as tf
from sklearn.metrics import confusion_matrix


def create_eval_step(model, loss_fn):
    """
    Crea una función de evaluación que se puede usar con tf.function.

    Args:
        model (tf.keras.Model): Modelo a evaluar.
        loss_fn (tf.keras.losses.Loss): Función de pérdida.

    Returns:
        function: Función de evaluación que toma un batch y devuelve la pérdida, las etiquetas y las predicciones.
    """

    @tf.function
    def eval_step(batch):
        x, y = batch
        logits = model(x, training=False)
        loss = loss_fn(y, logits)
        return loss, y, logits

    return eval_step


def run_eval(eval_step, eval_dataset, confusion=False):
    """
    Ejecuta la evaluación del modelo en el conjunto de datos de evaluación.

    Args:
        eval_step (function): Función de evaluación creada con create_eval_step.
        eval_dataset (tf.data.Dataset): Dataset de evaluación.
        confusion (bool): Si True, devuelve la matriz de confusión.

    Returns:
        tuple: (loss, accuracy)
    """
    loss_m = tf.keras.metrics.Mean()
    acc_m = tf.keras.metrics.SparseCategoricalAccuracy()
    all_y = []
    all_logits = []

    for batch in eval_dataset:
        loss, y, logits = eval_step(batch)
        loss_m.update_state(loss)
        acc_m.update_state(y, logits)

        if confusion:
            all_y.append(y)
            all_logits.append(logits)

    if confusion:
        y_true = tf.concat(all_y, axis=0).numpy()
        y_pred = tf.argmax(tf.concat(all_logits, axis=0), axis=1).numpy()

        return (
            loss_m.result().numpy(),
            acc_m.result().numpy(),
            confusion_matrix(y_true, y_pred),
        )

    return loss_m.result().numpy(), acc_m.result().numpy()
