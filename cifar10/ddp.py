import os
import time

import tensorflow as tf

from .eval import create_eval_step, run_eval
from .load_data import load_cifar10_ddp, load_cifar10_eval, plot_cifar10_images
from .metrics import EpochMetrics
from .model import create_model
from .train import create_train_step


def _setup_tf_config(worker_ips, worker_index):
    """
    Configura la variable de entorno TF_CONFIG para el entorno distribuido.

    Args:
        worker_ips (list): Lista de IPs de los trabajadores.
        worker_index (int): Índice del trabajador actual.
    """
    import json

    os.environ["TF_CONFIG"] = json.dumps(
        {
            "cluster": {"worker": worker_ips},
            "task": {"type": "worker", "index": worker_index},
        }
    )


def train(
    worker_ips,
    worker_index,
    conv=False,
    gray=True,
    normalize=False,
    batch_size=128,
    buffer_size=10000,
    ram=False,
    lr=0.001,
    epochs=20,
    save_dir: str | None = None,
):
    # configuración de TF_CONFIG y Data Distributed Parallel
    _setup_tf_config(worker_ips, worker_index)
    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    # cantidad de datos por batch de acuerdo a la cantidad de workers
    global_batch_size = batch_size * strategy.num_replicas_in_sync
    steps_per_epoch = 50000 // global_batch_size

    # dataset distribuido
    train_dataset = strategy.distribute_datasets_from_function(
        load_cifar10_ddp(
            buffer_size=buffer_size,
            gray=gray,
            normalize=normalize,
            ram=ram,
            global_batch_size=global_batch_size,
        )
    )

    with strategy.scope():
        model = create_model(gray=gray, conv=conv)
        model.summary()

        optimizer = tf.keras.optimizers.Adam(lr)

        # sum_over_batch_size con reduce y minibatch
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="sum_over_batch_size"
        )

    train_step = create_train_step(strategy, model, optimizer, loss_fn)

    is_chief = worker_index == 0
    if is_chief:
        eval_dataset = load_cifar10_eval(gray=gray, normalize=normalize, batch_size=512)
        eval_step = create_eval_step(model, loss_fn)

        plot_cifar10_images(eval_dataset, num_images=10, gray=gray, normalize=normalize)

    metrics = EpochMetrics(is_chief)

    for epoch in range(epochs):
        # ejecución de la época
        # evitando el perreplicas de MultiWorkerMirroredStrategy
        metrics.reset()
        t0 = time.perf_counter()

        # ejecución del minibatch
        for step, batch in enumerate(train_dataset):
            loss, logits_list, bs, gnorm = train_step(batch)
            y_list = strategy.experimental_local_results(batch[1])
            metrics.update(loss, logits_list, y_list, gnorm, bs)

            if step % 10 == 0:
                r = metrics.results()
                print(
                    f"Epoch {epoch + 1} step {step} | "
                    f"loss {r['loss']:.4f} acc {r['acc']:.4f} gnorm {r['gnorm']:.4f}",
                    end="\r",
                )

            if step >= steps_per_epoch:
                break

        epoch_time = time.perf_counter() - t0
        r = metrics.results()
        throughput = r["n"] / epoch_time

        eval_loss = eval_acc = None

        if is_chief:
            eval_loss, eval_acc = run_eval(eval_step, eval_dataset)

        metrics.add(
            epoch=epoch,
            epoch_time=epoch_time,
            throughput=throughput,
            eval_loss=eval_loss,
            eval_acc=eval_acc,
        )
        metrics.print_epoch(epochs=epochs)

    if save_dir:
        with open(os.path.join(save_dir, "train_params.txt"), "w") as f:
            f.write(f"epochs: {epochs}\n")
            f.write(f"lr: {lr}\n")
            f.write(f"workers: {len(worker_ips)}\n")
            f.write(f"gray: {gray}\n")
            f.write(f"normalize: {normalize}\n")
            f.write(f"conv: {conv}\n")
            f.write(f"batch_size: {batch_size}\n")
            f.write(f"Final accuracy: {metrics.df['accuracy'].iloc[-1]}")

    conf_matrix = None

    if is_chief:
        _, _, conf_matrix = run_eval(eval_step, eval_dataset, confusion=True)

    metrics.save(save_dir, worker_index, conf_matrix)
