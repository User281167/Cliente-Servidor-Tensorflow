import os
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from PIL import Image

from tiny_imagenet.load_data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_label_names,
)

from .wnids import tiny_imagenet_classes


def compute_confusion_matrix_and_accuracy(model, loader, num_classes):
    """
    Calcular matriz de confusión, accuracy por clase y top5 accuracy por clase.

    Args:
        model: Modelo de TensorFlow.
        loader: Dataset de TensorFlow.
        num_classes: Número de clases.

    Returns:
        conf: Matriz de confusión (num_classes, num_classes)
        per_class_acc: Accuracy por clase (top1)
        per_class_top5_acc: Accuracy por clase (top5)
    """
    conf = np.zeros((num_classes, num_classes), dtype=np.int32)
    per_class_correct = np.zeros(num_classes, dtype=np.int32)
    per_class_total = np.zeros(num_classes, dtype=np.int32)
    per_class_top5_correct = np.zeros(num_classes, dtype=np.int32)

    for batch in loader:
        imgs, labels = batch
        logits = model(imgs, training=False)
        preds = tf.argmax(logits, axis=-1)

        # Top5 predictions
        top5_preds = tf.math.top_k(logits, k=5).indices

        for img, label, pred, top5 in zip(imgs, labels, preds, top5_preds):
            label = int(label.numpy())
            pred = int(pred.numpy())
            top5 = [int(p) for p in top5.numpy()]

            conf[label][pred] += 1
            per_class_total[label] += 1

            if pred == label:
                per_class_correct[label] += 1

            if label in top5:
                per_class_top5_correct[label] += 1

    # Calcular accuracy por clase
    per_class_acc = per_class_correct / np.maximum(per_class_total, 1)
    per_class_top5_acc = per_class_top5_correct / np.maximum(per_class_total, 1)

    return conf, per_class_acc, per_class_top5_acc


def clean_name(name: str):
    return name.split(",")[0].strip().replace(" ", "_")


def get_samples(dataset):
    """Generador para obtener muestras de TF dataset."""
    for batch in dataset:
        imgs, labels = batch

        for img, label in zip(imgs, labels):
            lbl = int(label.numpy()) if hasattr(label, "numpy") else int(label)
            yield img, lbl


def extract_images_per_class(save_path, dataset, label_names):
    os.makedirs(save_path, exist_ok=True)
    saved = set()
    n_classes = len(label_names)

    for img, label in get_samples(dataset):
        if label in saved:
            continue

        if isinstance(img, tf.Tensor):
            img = img.numpy()  # float32, shape (H, W, 3)

        # Revertir normalización ImageNet → [0, 1] → uint8
        img = img * IMAGENET_STD + IMAGENET_MEAN
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

        class_name = clean_name(label_names[label])
        Image.fromarray(img).save(os.path.join(save_path, f"{class_name}.png"))
        saved.add(label)

        if len(saved) == n_classes:
            break

    print(f"Guardadas {len(saved)}/{n_classes} clases en {save_path}")


def save_class_report(
    per_class_acc, conf, label_names, save_path, per_class_top5_acc=None
):
    """
    Guardar un informe de las clases en un excel.
    Ordenar clases por precisión descendente.

    Args:
        per_class_acc (ndarray): Array con la precisión por clase (top1).
        conf (ndarray): Matriz de confusión.
        label_names (list): Lista de nombres de las clases.
        save_path (str): Directorio donde se guardará el informe.
        per_class_top5_acc (ndarray): Precisión top5 por clase (opcional).
    """

    # Ordenar clases por accuracy descendente
    sorted_idx = np.argsort(per_class_acc)[::-1]

    rows = []
    classes = tiny_imagenet_classes

    for cls in sorted_idx.tolist():
        acc = float(per_class_acc[cls])

        # Obtener top5 accuracy para esta clase
        top5_acc = None
        if per_class_top5_acc is not None:
            top5_acc = float(per_class_top5_acc[cls])

        row_conf = conf[cls].copy()
        row_conf[cls] = 0

        top_conf = None
        if row_conf.sum() > 0:
            top_conf = int(np.argmax(row_conf))

        class_name = clean_name(label_names[cls])
        top_confused_class = (
            clean_name(label_names[top_conf]) if top_conf is not None else None
        )

        total_errors = float(row_conf.sum())

        acc_confused = (
            float(row_conf[top_conf]) / total_errors
            if top_conf is not None and total_errors > 0
            else 0
        )

        rows.append(
            {
                "class_name": classes.get(class_name, class_name),
                "accuracy": acc,
                "top5_accuracy": top5_acc,
                "top_confused_class": classes.get(
                    top_confused_class, top_confused_class
                )
                if top_confused_class
                else None,
                "acc_confused": acc_confused,
                "img": f"{class_name}.png",
                "img_confused": f"{top_confused_class}.png"
                if top_confused_class
                else None,
            }
        )

    df = pd.DataFrame(rows)
    df.to_excel(os.path.join(save_path, "class_report.xlsx"), index=False)

    return df


def export_to_excel(df, save_path, img_dir):
    """
    Cambiar labels por imágenes en el informe de clases.

    Args:
        df (DataFrame): DataFrame con los datos de las clases.
        save_path (str): Directorio donde se guardará el informe.
        img_dir (str): Directorio donde se encuentran las imágenes.
    """

    wb = Workbook()
    ws = wb.active
    ws.title = "report"

    # HEADERS
    ws.append(list(df.columns))

    # DATA ROWS
    for _, row in df.iterrows():
        ws.append(list(row.values))

    # STYLE: columnas
    for col in ["A", "B", "C", "D", "E", "F", "G"]:
        ws.column_dimensions[col].width = 20

    # columnas de imagen
    ws.column_dimensions["F"].width = 18
    ws.column_dimensions["G"].width = 18

    # INSERT IMAGES + ROW HEIGHT
    for i, row in df.iterrows():
        excel_row = i + 2  # header offset

        # altura de fila
        ws.row_dimensions[excel_row].height = 50

        # imagen principal (columna F)
        img_path = os.path.join(img_dir, str(row["img"]))
        if os.path.exists(img_path):
            img = XLImage(img_path)
            img.width = 64
            img.height = 64
            ws.add_image(img, f"F{excel_row}")

        # imagen confundida (columna G)
        if row["img_confused"] is not None:
            img_conf_path = os.path.join(img_dir, str(row["img_confused"]))

            if os.path.exists(img_conf_path):
                img2 = XLImage(img_conf_path)
                img2.width = 64
                img2.height = 64
                ws.add_image(img2, f"G{excel_row}")

    # SAVE
    output_file = os.path.join(save_path, "report.xlsx")
    wb.save(output_file)

    print(f"Excel guardado en: {output_file}")


def excel_report(per_class_acc, conf, loader, save_path, per_class_top5_acc=None):
    """
    Generar un informe de clases en formato Excel.
    Funciona con Dataset de TensorFlow.

    Args:
        per_class_acc (ndarray): Precisión top1 por clase.
        conf (ndarray): Matriz de confusión.
        loader (Dataset): Dataset de TensorFlow con el dataset de validación.
        save_path (str): Directorio donde se guardará el informe.
        per_class_top5_acc (ndarray): Precisión top5 por clase (opcional).
    """
    # Obtener dataset del loader
    img_dir = os.path.join(save_path, "img")
    label_names = get_label_names()

    if not os.path.exists(img_dir):
        extract_images_per_class(img_dir, loader, label_names)

    df = save_class_report(
        per_class_acc, conf, label_names, save_path, per_class_top5_acc
    )

    export_to_excel(df, save_path, img_dir)
    shutil.rmtree(img_dir, ignore_errors=True)
