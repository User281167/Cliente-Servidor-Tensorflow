import os
import time
from pathlib import Path

import tensorflow as tf

from .eval import create_eval_step, run_eval
from .load_data import (
    DATASET_ID,
    NUM_CLASSES,
    TRAIN_SAMPLES,
    load_tiny_imagenet_ddp,
    load_tiny_imagenet_eval,
    load_tiny_imagenet_eval_ddp,
    save_index0_sample,
)
from .metrics import EpochMetrics
from .model import (
    create_resnet18_classifier,
    load_resnet_weights,
    save_resnet_weights,
)
from .report import compute_confusion_matrix_and_accuracy, excel_report
from .train import create_train_step


def train(
    worker_ips: list[str],
    worker_index: int,
    batch_size: int = 64,
    buffer_size: int = 10_000,
    lr: float = 1e-3,
    epochs: int = 10,
    ram: bool = False,
    save_dir: str | None = "runs/tiny_imagenet",
    train_backbone: bool = False,
    dropout: float = 0.2,
    eval_batch_size: int = 256,
):
    # ================================================
    # Inicialización de TensorFlow y estrategia de distribución
    # Datos distribuidos y modelo
    # ================================================

    assert "TF_CONFIG" in os.environ, (
        "TF_CONFIG debe estar seteado antes de importar TensorFlow"
    )

    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    global_batch_size = batch_size * strategy.num_replicas_in_sync
    steps_per_epoch = TRAIN_SAMPLES // global_batch_size
    is_chief = worker_index == 0

    train_dataset = strategy.distribute_datasets_from_function(
        load_tiny_imagenet_ddp(
            global_batch_size=global_batch_size,
            buffer_size=buffer_size,
            ram=ram,
        )
    )

    with strategy.scope():
        model = create_resnet18_classifier(
            num_classes=NUM_CLASSES,
            train_backbone=train_backbone,
            dropout=dropout,
        )
        model.summary()

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction="sum_over_batch_size",
        )

    train_step = create_train_step(strategy, model, optimizer, loss_fn)
    metrics = EpochMetrics(is_chief=is_chief)

    if is_chief:
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            save_index0_sample(save_dir)

    eval_step = create_eval_step(strategy, model, loss_fn)
    eval_dataset = strategy.distribute_datasets_from_function(
        load_tiny_imagenet_eval_ddp(
            batch_size=eval_batch_size,
            ram=ram,
        )
    )

    # ================================================
    # Bucle de entrenamiento distribuido
    # Test distribuido cada epoch
    # ================================================

    for epoch in range(epochs):
        metrics.reset()
        t0 = time.perf_counter()

        for step, batch in enumerate(train_dataset):
            loss, logits_list, batch_count, grad_norm = train_step(batch)
            labels_list = strategy.experimental_local_results(batch[1])
            metrics.update(loss, logits_list, labels_list, grad_norm, batch_count)

            if step % 10 == 0:
                r = metrics.results()
                print(
                    f"Epoch {epoch + 1} step {step}/{steps_per_epoch} | "
                    f"loss {r['loss']:.4f} acc {r['acc']:.4f} "
                    f"top5 {r['top5']:.4f} gnorm {r['grad_norm']:.4f}",
                    end="\r",
                )

            if step + 1 >= steps_per_epoch:
                break

        epoch_time = time.perf_counter() - t0
        r = metrics.results()
        throughput = r["n"] / epoch_time if epoch_time > 0 else 0.0

        eval_loss = eval_acc = eval_top5 = None
        eval_loss, eval_acc, eval_top5 = run_eval(strategy, eval_step, eval_dataset)

        metrics.add(
            epoch=epoch,
            epoch_time=epoch_time,
            throughput=throughput,
            eval_loss=eval_loss,
            eval_acc=eval_acc,
            eval_top5=eval_top5,
        )
        metrics.print_epoch(epochs)

    metrics.save(save_dir, worker_index)

    # ================================================
    # Guardar modelo y reporte de entrenamiento
    # ================================================
    if is_chief and save_dir:
        save_resnet_weights(model, Path(save_dir) / "clasificador_head.npy")
        model = load_resnet_weights(
            weights_path=Path(save_dir) / "clasificador_head.npy",
            train_backbone=train_backbone,
            dropout=dropout,
        )

        Path(save_dir, "train_params.txt").write_text(
            "\n".join(
                [
                    f"dataset_id: {DATASET_ID}",
                    "train_split: train",
                    "valid_split: valid",
                    "preset: resnet18",
                    "pretrained: True",
                    f"train_backbone: {train_backbone}",
                    f"epochs: {epochs}",
                    f"lr: {lr}",
                    f"workers: {len(worker_ips)}",
                    f"batch_size_per_worker: {batch_size}",
                    f"global_batch_size: {global_batch_size}",
                    f"ram: {ram}",
                ]
            ),
            encoding="utf-8",
        )

        print("Generando reporte de clases...")

        report_dataset = load_tiny_imagenet_eval(
            batch_size=eval_batch_size,
            ram=False,
        )

        conf, per_class_acc, per_class_top5_acc = compute_confusion_matrix_and_accuracy(
            model, report_dataset, NUM_CLASSES
        )

        excel_report(
            per_class_acc=per_class_acc,
            conf=conf,
            loader=report_dataset,
            save_path=save_dir,
            per_class_top5_acc=per_class_top5_acc,
        )

        print("Reporte de clases generado.")
