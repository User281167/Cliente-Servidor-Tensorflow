import argparse
import json
import os


def setup_tf_config(worker_ips: list[str], worker_index: int) -> None:
    """
    Configurar entorno de tensorflow.
    Debe configurarse antes de cualquier importanción de tensorflow.
    """

    os.environ["TF_CONFIG"] = json.dumps(
        {
            "cluster": {"worker": worker_ips},
            "task": {"type": "worker", "index": worker_index},
        }
    )

    print("-" * 50)
    print(f"TF_CONFIG: {os.environ['TF_CONFIG']}")
    print("-" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Entrenar Tiny ImageNet con ResNet18 DDP"
    )
    parser.add_argument("--worker-ips", nargs="+", required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--ram", action="store_true")
    parser.add_argument("--save-dir", type=str, default="results")
    parser.add_argument("--train-backbone", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.2)
    args = parser.parse_args()

    setup_tf_config(args.worker_ips, args.worker_index)

    from .ddp import train

    train(
        worker_ips=args.worker_ips,
        worker_index=args.worker_index,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        lr=args.lr,
        epochs=args.epochs,
        ram=args.ram,
        save_dir=args.save_dir,
        train_backbone=args.train_backbone,
        dropout=args.dropout,
        eval_batch_size=args.eval_batch_size,
    )


if __name__ == "__main__":
    main()
