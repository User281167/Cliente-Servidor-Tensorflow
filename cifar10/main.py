import argparse

from .ddp import train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--conv", action="store_true")
    p.add_argument("--rgb", action="store_true")
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--buffer-size", type=int, default=10000)
    p.add_argument("--ram", action="store_true")
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--worker-ips", nargs="+", default=None)
    p.add_argument("--worker-index", type=int, default=0)
    p.add_argument("--save-dir", type=str, default=None)
    args = p.parse_args()

    train(
        worker_ips=args.worker_ips,
        worker_index=args.worker_index,
        conv=args.conv,
        gray=not args.rgb,
        normalize=args.normalize,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        ram=args.ram,
        lr=args.lr,
        epochs=args.epochs,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
