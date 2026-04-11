# Cliente-Servidor PyTorch

Proyecto de pruebas con PyTorch.

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


### CIFAR-10 Distributed Data Parallel (DDP)

```bash
# Terminal 1
python -m cifar10.main --worker-ips 127.0.0.1:9090 127.0.0.1:9091 --worker-index 0 --epochs 20 --normalize --rgb --conv --save-dir "results"

# Terminal 2
python -m cifar10.main --worker-ips 127.0.0.1:9090 127.0.0.1:9091 --worker-index 1 --epochs 20 --normalize --rgb --conv --save-dir "results"
```
