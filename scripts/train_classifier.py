"""Train the KL-grade classifier.

Examples (run from the project root):
    python scripts/train_classifier.py --mask-source none
    python scripts/train_classifier.py --mask-source unet
    python scripts/train_classifier.py --mask-source unetpp
"""
import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score

from kl_common import (list_split, get_transforms, KLDataset, build_classifier,
                       load_seg_model, add_mask_channel, run_eval, run_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask-source", choices=["none", "unet", "unetpp"], default="none")
    ap.add_argument("--data-root", default="dataset")
    ap.add_argument("--ckpt-dir", default="outputs/checkpoints")
    ap.add_argument("--results-dir", default="outputs/results")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    mask_source = None if args.mask_source == "none" else args.mask_source
    name = run_name(mask_source)
    print(f"Device: {device} | Run: {name}")
    if device.type == "cpu":
        print("WARNING: no GPU found - training will be very slow. Check your PyTorch CUDA install.")

    # Data
    train_tf, eval_tf = get_transforms()
    train_p, train_y = list_split(args.data_root, "train")
    val_p, val_y = list_split(args.data_root, "val")
    print(f"Train: {len(train_p)} | Val: {len(val_p)}")
    pin = device.type == "cuda"
    train_loader = DataLoader(KLDataset(train_p, train_y, train_tf), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, pin_memory=pin)
    val_loader = DataLoader(KLDataset(val_p, val_y, eval_tf), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers, pin_memory=pin)

    # Class weights for the imbalance (rarer grade -> larger weight)
    counts = np.bincount(train_y, minlength=5)
    class_w = torch.tensor(len(train_y) / (5 * counts), dtype=torch.float32).to(device)
    print("Train counts per grade:", counts.tolist())
    print("Class weights:", [round(w, 2) for w in class_w.tolist()])

    # Models
    seg_model = load_seg_model(mask_source, args.ckpt_dir, device) if mask_source else None
    model = build_classifier(4 if mask_source else 3).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Training loop
    best_qwk, no_improve, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        total_loss = 0.0
        for imgs, y in train_loader:
            imgs = add_mask_channel(imgs.to(device), seg_model)
            y = y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = loss_fn(model(imgs), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * y.size(0)
        scheduler.step()

        vp, vy = run_eval(model, val_loader, seg_model, device, use_amp)
        val_acc = float((vp == vy).mean())
        val_qwk = float(cohen_kappa_score(vy, vp, weights="quadratic"))
        train_loss = total_loss / len(train_y)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc, "val_qwk": val_qwk})
        print(f"Epoch {epoch}/{args.epochs} | loss: {train_loss:.4f} | val acc: {val_acc:.4f} "
              f"| val QWK: {val_qwk:.4f} | {time.time() - start:.0f}s")

        if val_qwk > best_qwk:
            best_qwk, no_improve = val_qwk, 0
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, f"{name}_best.pt"))
            print("  -> new best, saved")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"No improvement for {args.patience} epochs, stopping early.")
                break

    pd.DataFrame(history).round(4).to_csv(os.path.join(args.results_dir, f"{name}_training_log.csv"), index=False)
    print(f"Done. Best val QWK: {best_qwk:.4f}")
    print(f"Checkpoint: {os.path.join(args.ckpt_dir, name + '_best.pt')}")


if __name__ == "__main__":
    main()
