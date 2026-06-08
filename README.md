# Tiny ImageNet ResNet18 DDP

Entrenamiento distribuido con `tf.distribute.MultiWorkerMirroredStrategy`, dataset Hugging Face `zh-plus/tiny-imagenet`, backbone `tfimm` ResNet18 ImageNet y un clasificador `Dense(200)`.

## Dependencias

```bash
uv sync
```

## Ejemplo 2 workers

Worker 0:

```bash
python -m tiny_imagenet.main --worker-ips 127.0.0.1:9090 127.0.0.1:9091 --worker-index 0 --epochs 10 --batch-size 64 --save-dir results
```

Worker 1:

```bash
python -m tiny_imagenet.main --worker-ips 127.0.0.1:9090 127.0.0.1:9091 --worker-index 1 --epochs 10 --batch-size 64 --save-dir results
```

## Salidas

- `metrics_<worker>.xlsx`
- `description_<worker>.xlsx`
- `grid.png`
- `clasificador_head.npy`
- `train_params.txt`
- `train_index0.png`
- `train_index0.txt`

No se guarda matriz de confusion: 200 clases no es legible.

## Notas

- `--ram` precarga imagenes `uint8` de 64x64 por shard del worker.
- Preprocesado TensorFlow: resize 224x224, escala `[0, 1]`, normalizacion ImageNet.
- Usa `tf.keras` legacy (`TF_USE_LEGACY_KERAS=1`) porque `tfimm` aun usa nombres de capas con `/`.
- Backbone congelado por defecto; usa `--train-backbone` para fine-tuning completo.
- Preentrenado activo por defecto; baja SavedModel `tfimm/resnet18/model` desde Hugging Face.
- Modelo con parametros en numpy para evitar problemas con tfimm
