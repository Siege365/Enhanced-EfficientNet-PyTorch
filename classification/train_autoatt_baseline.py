"""
Baseline Replication -- EfficientNetAutoAttB4 (End-to-End Training)

Replicates the architecture from:
  Jain et al. (2024). "Securing Visual Integrity: An Efficient NetB4-Based Solution
  with Attention Layers and Siamese Training for Face Manipulation Detection in Videos."
  IJISAE, 12(3), 1573-1580.

Architecture:
  - EfficientNet-B4 backbone (pretrained ImageNet)
  - AutoAtt module at Stage 3 output (28x28x56 for 224x224 input):
      Conv2d(56,56,k=1) -> Sigmoid -> element-wise multiply back into feature map
  - Global Average Pooling -> Dropout -> FC(1792,2)

Usage on COARE:
  python train_autoatt_baseline.py \
      --data_dirs /scratch1/nathaniel.merka/datasets/extracted/videos-zip/dfdc_part00_faces \
                  /scratch1/nathaniel.merka/datasets/extracted/videos-zip/celebdf_v1_faces \
      --output_dir /scratch1/nathaniel.merka/datasets/output_autoatt_baseline \
      --epochs 15 --batch_size 32 --lr 0.001
"""
import os, sys, json, argparse, glob
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
from torchvision import transforms
from PIL import Image

try:
    from efficientnet_pytorch import EfficientNet
except ImportError:
    raise ImportError("Install with: pip install efficientnet_pytorch")

# ==============================================================================
CONSOLE_LOG_FILE = None

def log_info(msg=""):
    ts = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    formatted = "{} {}".format(ts, msg) if msg else ""
    print(formatted, flush=True)
    if CONSOLE_LOG_FILE and formatted:
        with open(CONSOLE_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")


# ==============================================================================
# AutoAtt Module -- exactly as in Jain et al. (2024) Fig 1
# Tapped at Stage 3 output: (B, 56, 28, 28) for 224x224 input.
# Verified by block-shape inspection: blocks 0-9 produce 28x28x56.
# ==============================================================================
class AutoAttModule(nn.Module):
    def __init__(self, channels=56):
        super(AutoAttModule, self).__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attn(x)


# ==============================================================================
# EfficientNetAutoAttB4 -- Full Model
#
# Verified block layout (224x224 input):
#   Stem      : (1, 48, 112, 112)
#   blocks 00-01 (Stage 1): (1, 24, 112, 112)
#   blocks 02-05 (Stage 2): (1, 32,  56,  56)
#   blocks 06-09 (Stage 3): (1, 56,  28,  28) <-- AutoAtt applied here
#   blocks 10-15 (Stage 4): (1,112,  14,  14)
#   blocks 16-21 (Stage 5): (1,160,  14,  14)
#   blocks 22-29 (Stage 6): (1,272,   7,   7)
#   blocks 30-31 (Stage 7): (1,448,   7,   7)
#   conv_head : (1, 1792, 7, 7)
#   GAP+FC    : (1, 2)
# ==============================================================================
class EfficientNetAutoAttB4(nn.Module):
    STAGE3_END = 10   # _blocks[:10] = stages 1-3, output is 28x28x56
    STAGE3_CH  = 56

    def __init__(self, num_classes=2, dropout=0.4, pretrained=True):
        super(EfficientNetAutoAttB4, self).__init__()
        if pretrained:
            base = EfficientNet.from_pretrained('efficientnet-b4')
        else:
            base = EfficientNet.from_name('efficientnet-b4')

        self._conv_stem = base._conv_stem
        self._bn0       = base._bn0
        self._swish     = base._swish

        n = self.STAGE3_END
        self._blocks_pre  = nn.ModuleList(base._blocks[:n])   # stages 1-3
        self._blocks_post = nn.ModuleList(base._blocks[n:])   # stages 4-7

        self._conv_head  = base._conv_head
        self._bn1        = base._bn1
        self._avg_pooling = base._avg_pooling

        self.auto_att = AutoAttModule(channels=self.STAGE3_CH)
        self._dropout = nn.Dropout(p=dropout)
        self._fc      = nn.Linear(base._fc.in_features, num_classes)

    def forward(self, x):
        # Stem
        x = self._swish(self._bn0(self._conv_stem(x)))
        # Stages 1-3
        for blk in self._blocks_pre:
            x = blk(x, drop_connect_rate=0.0)
        # AutoAtt at 28x28x56
        x = self.auto_att(x)
        # Stages 4-7
        for blk in self._blocks_post:
            x = blk(x, drop_connect_rate=0.0)
        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        x = self._avg_pooling(x).flatten(1)
        x = self._dropout(x)
        return self._fc(x)


# ==============================================================================
# Dataset -- loads pre-extracted face crops from real/ and fake/ subdirectories
# ==============================================================================
class FaceDataset(torch.utils.data.Dataset):
    EXT = ('.jpg', '.jpeg', '.png', '.webp')

    def __init__(self, data_dirs, transform=None):
        self.transform = transform
        self.samples   = []
        for d in data_dirs:
            for folder, label in [(os.path.join(d, 'real'), 0),
                                   (os.path.join(d, 'fake'), 1)]:
                if not os.path.isdir(folder):
                    print("  [WARN] Missing: {}".format(folder))
                    continue
                for root, _, files in os.walk(folder):
                    for f in files:
                        if f.lower().endswith(self.EXT):
                            self.samples.append((os.path.join(root, f), label))
        n_real = sum(1 for _, l in self.samples if l == 0)
        n_fake = sum(1 for _, l in self.samples if l == 1)
        print("  [Dataset] {:,} samples -- real:{:,} fake:{:,}".format(
            len(self.samples), n_real, n_fake))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224))
        if self.transform:
            img = self.transform(img)
        return img, label


# ==============================================================================
# Transforms -- Jain et al. augmentations: HF, BC, HSV, GAUS, DS
# ==============================================================================
def get_transforms(img_size=224):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomApply([
            transforms.Resize((int(img_size * 0.8), int(img_size * 0.8))),
            transforms.Resize((img_size, img_size)),
        ], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
        transforms.RandomApply([
            transforms.Lambda(lambda t: t + torch.randn_like(t) * 0.02)
        ], p=0.3),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


# ==============================================================================
# Training loop
# ==============================================================================
def train_one_epoch(model, loader, criterion, optimizer,
                    scaler, use_amp, device, accum_steps=1):
    model.train()
    running_loss = 0.0
    all_preds  = []
    all_labels = []
    optimizer.zero_grad()

    for i, (imgs, labels) in enumerate(tqdm(loader, desc="Train", leave=False)):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            logits = model(imgs)                          # ONE forward pass
            loss   = criterion(logits, labels) / accum_steps

        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * accum_steps * imgs.size(0)
        # Reuse logits from the SAME forward pass -- no second call to model()
        preds = torch.argmax(logits.detach(), dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc  = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, use_amp, device):
    model.eval()
    running_loss = 0.0
    all_preds  = []
    all_labels = []
    all_probs  = []

    for imgs, labels in tqdm(loader, desc="Val", leave=False):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        running_loss += loss.item() * imgs.size(0)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n_classes = len(set(all_labels))
    return {
        'loss': running_loss / len(loader.dataset),
        'acc':  accuracy_score(all_labels, all_preds),
        'prec': precision_score(all_labels, all_preds, zero_division=0),
        'rec':  recall_score(all_labels, all_preds, zero_division=0),
        'f1':   f1_score(all_labels, all_preds, zero_division=0),
        'auc':  roc_auc_score(all_labels, all_probs) if n_classes > 1 else 0.5,
    }


# ==============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description='EfficientNetAutoAttB4 - Jain et al. 2024 replication')
    p.add_argument('--data_dirs',    nargs='+', required=True)
    p.add_argument('--output_dir',   type=str,   default='output_autoatt_baseline')
    p.add_argument('--img_size',     type=int,   default=224)
    p.add_argument('--epochs',       type=int,   default=15)
    p.add_argument('--batch_size',   type=int,   default=32)
    p.add_argument('--accum_steps',  type=int,   default=1)
    p.add_argument('--lr',           type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--dropout',      type=float, default=0.4)
    p.add_argument('--val_split',    type=float, default=0.15)
    p.add_argument('--patience',     type=int,   default=5)
    p.add_argument('--num_workers',  type=int,   default=4)
    p.add_argument('--no_amp',       action='store_true')
    p.add_argument('--resume',       type=str,   default=None)
    return p.parse_args()


# ==============================================================================
def main():
    args    = parse_args()
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = not args.no_amp and device.type == 'cuda'

    print("\n" + "=" * 65)
    print("  BASELINE: EfficientNetAutoAttB4 (Jain et al., IJISAE 2024)")
    print("  Device: {} | AMP: {} | Input: {}x{}".format(
        device, use_amp, args.img_size, args.img_size))
    print("=" * 65 + "\n")

    # Output directory
    if args.resume:
        save_dir = args.resume
    else:
        run_id   = "autoatt_b4_{}".format(datetime.now().strftime('%Y%m%d_%H%M%S'))
        save_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(save_dir, exist_ok=True)

    global CONSOLE_LOG_FILE
    CONSOLE_LOG_FILE = os.path.join(save_dir, 'console.log')
    log_info("Output: {}".format(save_dir))

    # Transforms
    train_tf, val_tf = get_transforms(args.img_size)

    # Datasets
    log_info("Loading face crops from {} director(ies)...".format(len(args.data_dirs)))
    full_ds  = FaceDataset(args.data_dirs, transform=train_tf)
    val_full = FaceDataset(args.data_dirs, transform=val_tf)

    if len(full_ds) == 0:
        raise RuntimeError(
            "No images found! Check --data_dirs and ensure real/ and fake/ subdirs exist.")

    val_size   = int(len(full_ds) * args.val_split)
    train_size = len(full_ds) - val_size

    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    train_sub, _  = random_split(full_ds,  [train_size, val_size], generator=g1)
    _, val_sub    = random_split(val_full, [train_size, val_size], generator=g2)

    # Class-balanced weights
    labels_tr = [full_ds.samples[i][1] for i in train_sub.indices]
    n_r = labels_tr.count(0)
    n_f = labels_tr.count(1)
    tot = n_r + n_f
    w_r = tot / (2.0 * n_r) if n_r > 0 else 1.0
    w_f = tot / (2.0 * n_f) if n_f > 0 else 1.0
    class_w = torch.tensor([w_r, w_f], dtype=torch.float32).to(device)
    log_info("Class weights -- Real:{:.3f} Fake:{:.3f} | Train:{:,} Val:{:,}".format(
        w_r, w_f, train_size, val_size))

    sample_w = torch.tensor([w_r if l == 0 else w_f for l in labels_tr])
    sampler  = WeightedRandomSampler(sample_w, num_samples=len(sample_w),
                                     replacement=True)

    train_loader = DataLoader(
        train_sub, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(
        val_sub, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    # Model
    log_info("Building EfficientNetAutoAttB4 (pretrained ImageNet)...")
    model = EfficientNetAutoAttB4(
        num_classes=2, dropout=args.dropout, pretrained=True).to(device)
    total_p = sum(p.numel() for p in model.parameters())
    log_info("Total params: {:,}".format(total_p))

    # Sanity-check forward pass
    with torch.no_grad():
        dummy = torch.zeros(2, 3, args.img_size, args.img_size).to(device)
        out   = model(dummy)
        log_info("Sanity check OK -- output shape: {}".format(tuple(out.shape)))
        assert out.shape == (2, 2), "Expected (2,2) output, got {}".format(out.shape)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(weight=class_w)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=use_amp)

    # Resume
    start_epoch = 1
    best_auc    = 0.0
    no_improve  = 0
    history     = []
    resume_path = os.path.join(save_dir, 'resume.pth')

    if args.resume and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scaler.load_state_dict(ckpt['scaler_state'])
        start_epoch = ckpt['epoch'] + 1
        best_auc    = ckpt.get('best_auc', 0.0)
        no_improve  = ckpt.get('no_improve', 0)
        history     = ckpt.get('history', [])
        log_info("[Resume] Epoch {}, best AUC so far: {:.4f}".format(
            start_epoch, best_auc))

    # Training loop
    log_info("Training epochs {}-{} (patience={})...".format(
        start_epoch, args.epochs, args.patience))

    for epoch in range(start_epoch, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer,
            scaler, use_amp, device, args.accum_steps)

        val = validate(model, val_loader, criterion, use_amp, device)
        scheduler.step()

        log_info(
            "Epoch [{:02d}/{}] Train Loss:{:.4f} Acc:{:.1f}% | "
            "Val Loss:{:.4f} Acc:{:.1f}% F1:{:.4f} AUC:{:.4f}".format(
                epoch, args.epochs,
                tr_loss, tr_acc * 100,
                val['loss'], val['acc'] * 100, val['f1'], val['auc']))

        if val['auc'] > best_auc:
            best_auc   = val['auc']
            no_improve = 0
            torch.save(model.state_dict(),
                       os.path.join(save_dir, 'best_model.pth'))
            log_info("  --> Best model saved! (AUC:{:.4f})".format(best_auc))
        else:
            no_improve += 1
            log_info("  No improvement ({}/{})".format(no_improve, args.patience))

        history.append({
            'epoch': epoch,
            'train_loss': tr_loss,
            'train_acc':  tr_acc,
            **val
        })
        torch.save({
            'epoch':           epoch,
            'model_state':     model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scaler_state':    scaler.state_dict(),
            'best_auc':        best_auc,
            'no_improve':      no_improve,
            'history':         history,
        }, resume_path)

        with open(os.path.join(save_dir, 'metrics.json'), 'w') as mf:
            json.dump(history, mf, indent=2)

        if no_improve >= args.patience:
            log_info("Early stopping at epoch {}.".format(epoch))
            break

    log_info("\nDone. Best AUC: {:.4f}".format(best_auc))
    log_info("Best model: {}".format(os.path.join(save_dir, 'best_model.pth')))


if __name__ == '__main__':
    main()