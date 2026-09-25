"""Evaluate a trained KL classifier on the unseen test set.

Example:
    python scripts/evaluate_classifier.py --mask-source none
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")   # save figures to file, no pop-up window
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, cohen_kappa_score, classification_report,
                             confusion_matrix, f1_score)

from kl_common import (list_split, get_transforms, KLDataset, build_classifier,
                       load_seg_model, run_eval_probs, run_name, GRADES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask-source", choices=["none", "unet", "unetpp"], default="none")
    ap.add_argument("--data-root", default="dataset")
    ap.add_argument("--ckpt-dir", default="outputs/checkpoints")
    ap.add_argument("--results-dir", default="outputs/results")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    mask_source = None if args.mask_source == "none" else args.mask_source
    name = run_name(mask_source)

    _, eval_tf = get_transforms()
    test_p, test_y = list_split(args.data_root, "test")
    test_loader = DataLoader(KLDataset(test_p, test_y, eval_tf), batch_size=args.batch_size,
                             shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")

    seg_model = load_seg_model(mask_source, args.ckpt_dir, device) if mask_source else None
    model = build_classifier(4 if mask_source else 3, pretrained=False).to(device)
    model.load_state_dict(torch.load(os.path.join(args.ckpt_dir, f"{name}_best.pt"), map_location=device))

    tp, ty, tprobs = run_eval_probs(model, test_loader, seg_model, device, use_amp)
    acc = accuracy_score(ty, tp)
    qwk = cohen_kappa_score(ty, tp, weights="quadratic")
    macro_f1 = f1_score(ty, tp, average="macro")
    report = classification_report(ty, tp, target_names=GRADES, digits=4, output_dict=True)

    print(f"=== {name} on {len(ty)} unseen test images ===")
    print(f"Accuracy: {acc:.4f} | QWK: {qwk:.4f} | Macro F1: {macro_f1:.4f}\n")
    print(classification_report(ty, tp, target_names=GRADES, digits=4))
    pd.DataFrame(report).T.round(4).to_csv(os.path.join(args.results_dir, f"{name}_test_report.csv"))

    # Per-image predictions (test_loader uses shuffle=False, so tp/ty/tprobs line up with test_p in order)
    pred_df = pd.DataFrame({
        "image_path": test_p,
        "true_grade": ty,
        "pred_grade": tp,
        **{f"prob_KL{g}": tprobs[:, g] for g in range(5)},
    }).round(4)
    pred_path = os.path.join(args.results_dir, f"{name}_test_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"Per-image predictions saved to {pred_path}")

    # 5x5 confusion matrix
    cm = confusion_matrix(ty, tp, labels=range(5))
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for r in range(5):
        for c in range(5):
            ax.text(c, r, f"{cm[r, c]}\n({pct[r, c]:.1f}%)", ha="center", va="center",
                    color="white" if pct[r, c] > 50 else "black", fontsize=9)
    ax.set_xticks(range(5)); ax.set_xticklabels(GRADES)
    ax.set_yticks(range(5)); ax.set_yticklabels(GRADES)
    ax.set_xlabel("Predicted grade"); ax.set_ylabel("Actual grade")
    ax.set_title(f"{name}: test confusion matrix")
    plt.tight_layout()
    cm_path = os.path.join(args.results_dir, f"{name}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    print(f"Confusion matrix saved to {cm_path}")

    # Comparison table across runs (one row per run)
    summary_path = os.path.join(args.results_dir, "kl_comparison.csv")
    row = pd.DataFrame([{"Run": name, "Accuracy": acc, "QWK": qwk, "Macro F1": macro_f1,
                         "Recall KL3": report["KL3"]["recall"], "Recall KL4": report["KL4"]["recall"]}]).round(4)
    if os.path.exists(summary_path):
        old = pd.read_csv(summary_path)
        row = pd.concat([old[old["Run"] != name], row], ignore_index=True)
    row.to_csv(summary_path, index=False)
    print("\nComparison so far:")
    print(row.to_string(index=False))


if __name__ == "__main__":
    main()
