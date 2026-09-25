"""Shared pieces for KL-grade classification: data loading, models, and evaluation."""
import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMG_SIZE = 224          # classifier input size
SEG_SIZE = 256          # size the segmentation models were trained at
GRADES = ["KL0", "KL1", "KL2", "KL3", "KL4"]
KL_TEXT = {0: "No osteoarthritis", 1: "Doubtful", 2: "Minimal", 3: "Moderate", 4: "Severe"}
SEG_CKPTS = {"unet": "unet_best.pt", "unetpp": "unetpp_best.pt"}


def run_name(mask_source):
    return f"kl_{mask_source}" if mask_source in SEG_CKPTS else "kl_baseline"


def list_split(data_root, split):
    """Returns image paths and KL labels from data_root/split/<grade>/*."""
    paths, labels = [], []
    for g in range(5):
        files = sorted(glob.glob(os.path.join(data_root, split, str(g), "*")))
        paths += files
        labels += [g] * len(files)
    if not paths:
        raise FileNotFoundError(f"No images found in {os.path.join(data_root, split)} - check --data-root")
    return paths, labels


def get_transforms():
    train_tf = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.HorizontalFlip(p=0.5), A.Rotate(limit=10, p=0.5),
                          A.RandomBrightnessContrast(0.2, 0.2, p=0.5), A.Normalize(MEAN, STD), ToTensorV2()])
    eval_tf = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.Normalize(MEAN, STD), ToTensorV2()])
    return train_tf, eval_tf


class KLDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = paths, labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(self.paths[i], cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image: {self.paths[i]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(image=img)["image"], self.labels[i]


def build_classifier(in_ch, pretrained=True):
    """EfficientNet-B0 with a 5-class head. in_ch=4 adds a channel for the bone mask."""
    m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
    if in_ch == 4:
        old = m.features[0][0]
        new = nn.Conv2d(4, old.out_channels, kernel_size=old.kernel_size, stride=old.stride,
                        padding=old.padding, bias=False)
        with torch.no_grad():
            new.weight[:, :3] = old.weight
            new.weight[:, 3:] = old.weight.mean(dim=1, keepdim=True)
        m.features[0][0] = new
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 5)
    return m


def load_seg_model(kind, ckpt_dir, device):
    """Loads a trained U-Net or U-Net++ (from the Colab runs), frozen."""
    cls = smp.Unet if kind == "unet" else smp.UnetPlusPlus
    m = cls(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    path = os.path.join(ckpt_dir, SEG_CKPTS[kind])
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - download it from Google Drive into {ckpt_dir}")
    m.load_state_dict(torch.load(path, map_location=device))
    m.to(device).eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


def add_mask_channel(imgs, seg_model):
    """Adds the segmentation model's bone-probability map as a 4th input channel."""
    if seg_model is None:
        return imgs
    with torch.no_grad():
        x = F.interpolate(imgs, size=(SEG_SIZE, SEG_SIZE), mode="bilinear", align_corners=False)
        prob = torch.sigmoid(seg_model(x).float())
        prob = F.interpolate(prob, size=imgs.shape[-2:], mode="bilinear", align_corners=False)
    return torch.cat([imgs, prob.to(imgs.dtype)], dim=1)


def run_eval(model, loader, seg_model, device, use_amp):
    """Returns predicted grades and true grades for a whole loader."""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for imgs, y in loader:
            imgs = add_mask_channel(imgs.to(device), seg_model)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(imgs)
            preds.append(logits.argmax(1).cpu())
            labels.append(y)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


def run_eval_probs(model, loader, seg_model, device, use_amp):
    """Like run_eval, but also returns the per-class softmax probabilities."""
    model.eval()
    preds, labels, probs = [], [], []
    with torch.no_grad():
        for imgs, y in loader:
            imgs = add_mask_channel(imgs.to(device), seg_model)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(imgs)
            p = torch.softmax(logits.float(), dim=1)
            preds.append(p.argmax(1).cpu())
            labels.append(y)
            probs.append(p.cpu())
    return torch.cat(preds).numpy(), torch.cat(labels).numpy(), torch.cat(probs).numpy()
