# Entrenamiento de red con Tensorflow

## Instalacion con `uv`

Desde la raiz del proyecto:

```bash
uv sync
```

## Instalacion con `pip`

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```


### CIFAR-10 local

```bash
python -m cifar10.local --epochs 20 --lr 0.001 --conv --rgb --normalize --batch-size 128
```
