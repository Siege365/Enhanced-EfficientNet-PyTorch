"""
Training Pipeline - Phase 3: TSM + MHSA Video Deepfake Detection

Supports two video training modes:
  1. Standard Video Training:
       python train_video.py --video_dirs E:\\Thesis_Datasets\\videos\\FaceForensics++_C23 \\
           --pretrained_image_checkpoint output/efficientnet_b4_.../best_model.pth \\
           --epochs 15 --batch_size 6 --num_frames 8

  2. Continuous Video Learning / Experience Replay:
       python train_video.py --continuous \\
           --old_video_dirs E:\\Thesis_Datasets\\videos\\FaceForensics++_C23 \\
           --video_dirs E:\\Thesis_Datasets\\videos\\wilddeepfake E:\\Thesis_Datasets\\videos\\kling_scraped\\frames \\
           --pretrained_image_checkpoint output_video/best_video_model.pth \\
           --replay_buffer_size 400 --epochs 10 --batch_size 6

Author: Multi-Model Comparative Study Project
"""
import os
import sys
import json
import time
import argparse
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from network.video_models import video_model_selection
from dataset.video_dataset import VideoFrameSequenceDataset, VideoReplayBufferDataset


CONSOLE_LOG_FILE = None

def log_info(msg=""):
    ts = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    formatted = f"{ts} {msg}" if msg else ""
    print(formatted, flush=True)
    if CONSOLE_LOG_FILE and formatted:
        with open(CONSOLE_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")


DEFAULT_VIDEO_DIRS = [
    r'E:\Thesis_Datasets\videos\FaceForensics++_C23_frames',
    r'E:\Thesis_Datasets\videos\dfd_frames',
    r'E:\Thesis_Datasets\videos\wilddeepfake\deepfake_in_the_wild',
    r'E:\Thesis_Datasets\videos\synth_vid_detect',
    r'E:\Thesis_Datasets\videos\kling_scraped\frames',
    r'E:\Thesis_Datasets\videos\hailuo_scraped\frames',
    r'E:\Thesis_Datasets\videos\civitai_scraped\frames'
]


def parse_args():
    p = argparse.ArgumentParser(description='Train Phase 3 TSM + MHSA Video Deepfake Detector')
    p.add_argument('--video_dirs', nargs='+', default=DEFAULT_VIDEO_DIRS, help='Directories containing video frame sequences')
    p.add_argument('--old_video_dirs', nargs='+', default=None, help='Old video directories for replay buffer')
    p.add_argument('--pretrained_image_checkpoint', type=str, default=None, help='Path to Phase 1/2 best_model.pth')
    p.add_argument('--num_frames', type=int, default=8, help='Number of frames per video sequence (T)')
    p.add_argument('--batch_size', type=int, default=2, help='Batch size (video clips per batch)')
    p.add_argument('--accum_steps', type=int, default=3, help='Gradient accumulation steps (default 3 * batch 2 = effective batch 6)')
    p.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    p.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    p.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    p.add_argument('--dropout', type=float, default=0.5, help='Dropout probability')
    p.add_argument('--val_split', type=float, default=0.15, help='Validation split fraction')
    p.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    p.add_argument('--patience', type=int, default=5, help='Early stopping patience (epochs without improvement)')
    p.add_argument('--freeze_backbone_epochs', type=int, default=3, help='Freeze backbone for first N epochs, then unfreeze for fine-tuning')
    p.add_argument('--continuous', action='store_true', default=False, help='Enable continual learning experience replay')
    p.add_argument('--replay_buffer_size', type=int, default=400, help='Max video clips in replay buffer')
    p.add_argument('--output_dir', type=str, default='output_video', help='Directory to save checkpoints')
    p.add_argument('--no_amp', action='store_true', default=False, help='Disable mixed precision training')
    p.add_argument('--resume', type=str, default=None, help='Path to a previous run output directory (e.g. output_video/video_tsm_mhsa_20260716_053744) to resume training from')
    return p.parse_args()


def get_transforms(img_size=380):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    return train_transform, val_transform


def build_datasets(args, train_tf, val_tf):
    print(f"\n[Dataset] Loading primary video dataset from {len(args.video_dirs)} directories...")
    # Build TWO separate dataset instances so train/val transforms are independent
    train_full_ds = VideoFrameSequenceDataset(args.video_dirs, num_frames=args.num_frames, transform=train_tf)
    val_full_ds = VideoFrameSequenceDataset(args.video_dirs, num_frames=args.num_frames, transform=val_tf)
    
    if len(train_full_ds) == 0:
        raise RuntimeError("No video frame sequences found! Check paths.")

    val_size = int(len(train_full_ds) * args.val_split)
    train_size = len(train_full_ds) - val_size
    
    # Use the same seed so both splits select the same indices
    generator = torch.Generator().manual_seed(42)
    train_ds, _ = random_split(train_full_ds, [train_size, val_size], generator=generator)
    generator2 = torch.Generator().manual_seed(42)
    _, val_ds = random_split(val_full_ds, [train_size, val_size], generator=generator2)

    # Count class distribution in the training split for loss weighting
    train_labels = [train_full_ds.video_clips[i][1] for i in train_ds.indices]
    n_real = train_labels.count(0)
    n_fake = train_labels.count(1)
    total = n_real + n_fake
    # Inverse-frequency weights: rare class gets higher penalty
    w_real = total / (2.0 * n_real) if n_real > 0 else 1.0
    w_fake = total / (2.0 * n_fake) if n_fake > 0 else 1.0
    class_weights = torch.tensor([w_real, w_fake], dtype=torch.float32)
    print(f"  [ClassWeights] Real: {n_real:,} | Fake: {n_fake:,} | w_real={w_real:.3f} | w_fake={w_fake:.3f}")

    # Per-sample weights for WeightedRandomSampler (balanced mini-batches)
    per_sample_w = [w_real if l == 0 else w_fake for l in train_labels]
    sample_weights = torch.tensor(per_sample_w, dtype=torch.float32)

    if args.continuous and args.old_video_dirs:
        print(f"\n[Continual Replay] Building replay buffer from old video directories...")
        old_ds = VideoFrameSequenceDataset(args.old_video_dirs, num_frames=args.num_frames, transform=train_tf)
        train_ds = VideoReplayBufferDataset(train_ds, old_ds, buffer_size=args.replay_buffer_size)

    return train_ds, val_ds, class_weights, sample_weights


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, use_amp, device, accum_steps=1):
    model.train()
    if isinstance(dataloader.dataset, VideoReplayBufferDataset):
        dataloader.dataset.refresh_buffer()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for idx, (inputs, labels) in enumerate(pbar):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_accum = loss / accum_steps

        if use_amp:
            scaler.scale(loss_accum).backward()
            if (idx + 1) % accum_steps == 0 or (idx + 1) == len(dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss_accum.backward()
            if (idx + 1) % accum_steps == 0 or (idx + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()

        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, dataloader, criterion, use_amp, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    for inputs, labels in tqdm(dataloader, desc="Validating", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = torch.argmax(outputs, dim=1)

        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / len(dataloader.dataset)
    val_acc = accuracy_score(all_labels, all_preds)
    val_prec = precision_score(all_labels, all_preds, zero_division=0)
    val_rec = recall_score(all_labels, all_preds, zero_division=0)
    val_f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        val_auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        val_auc = 0.5

    metrics = {
        'loss': val_loss, 'acc': val_acc, 'prec': val_prec,
        'rec': val_rec, 'f1': val_f1, 'auc': val_auc
    }
    return metrics


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  PHASE 3 TSM + MHSA VIDEO DEEPFAKE TRAINING PIPELINE")
    print(f"  Device: {device} | Frames per clip (T): {args.num_frames} | Batch: {args.batch_size}")
    print(f"{'='*60}")

    run_id = f"video_tsm_mhsa_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(save_dir, exist_ok=True)
    global CONSOLE_LOG_FILE
    CONSOLE_LOG_FILE = os.path.join(save_dir, 'console.log')
    log_info(f"Video training started. Output directory: {save_dir}")

    model, img_size, _ = video_model_selection(
        num_frames=args.num_frames,
        pretrained_checkpoint=args.pretrained_image_checkpoint,
        dropout=args.dropout
    )
    model = model.to(device)

    train_tf, val_tf = get_transforms(img_size=img_size)
    train_ds, val_ds, class_weights, sample_weights = build_datasets(args, train_tf, val_tf)

    # WeightedRandomSampler: balanced mini-batches (Fix #3)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    use_pw = args.num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True, persistent_workers=use_pw, prefetch_factor=2 if use_pw else None)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=use_pw, prefetch_factor=2 if use_pw else None)

    # Weighted loss to fix class imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Use AUC as primary scheduler metric (Fix #2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    scaler = torch.amp.GradScaler('cuda', enabled=not args.no_amp)

    best_auc = 0.0
    patience_ctr = 0
    start_epoch = 1
    history = []

    # --- Resume from checkpoint ---
    if args.resume:
        resume_ckpt = os.path.join(args.resume, 'resume.pth')
        if not os.path.exists(resume_ckpt):
            # Fall back to best model if resume.pth not found
            resume_ckpt = os.path.join(args.resume, 'best_video_model.pth')
        if os.path.exists(resume_ckpt):
            log_info(f"[Resume] Loading checkpoint: {resume_ckpt}")
            ckpt = torch.load(resume_ckpt, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            if 'scaler_state_dict' in ckpt and not args.no_amp:
                scaler.load_state_dict(ckpt['scaler_state_dict'])
            best_auc = ckpt.get('best_auc', 0.0)
            patience_ctr = ckpt.get('patience_ctr', 0)
            start_epoch = ckpt.get('epoch', 0) + 1
            history = ckpt.get('history', [])
            log_info(f"[Resume] Resuming from Epoch {start_epoch} | Best AUC so far: {best_auc:.4f} | Patience: {patience_ctr}/{args.patience}")
        else:
            log_info(f"[Resume] WARNING: No checkpoint found at {args.resume}. Starting fresh.")

    # Fix #4: Two-stage training — freeze backbone for first N epochs
    # Only apply freeze if we haven't passed the freeze window yet
    if args.freeze_backbone_epochs > 0 and start_epoch <= args.freeze_backbone_epochs:
        for name, param in model.named_parameters():
            if 'mhsa_head' not in name:
                param.requires_grad = False
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log_info(f"[Stage 1] Backbone FROZEN. Training only MHSA head ({n_trainable:,} params) for {args.freeze_backbone_epochs} epoch(s).")
    elif start_epoch > args.freeze_backbone_epochs:
        # Already past freeze window — ensure backbone is unfrozen
        for param in model.parameters():
            param.requires_grad = True
        log_info(f"[Resume] Already past freeze window. Backbone is UNFROZEN.")

    log_info(f"Starting from epoch {start_epoch} up to {args.epochs} total epochs (patience={args.patience})...")
    for epoch in range(start_epoch, args.epochs + 1):

        # Fix #4: Unfreeze backbone after freeze_backbone_epochs
        if args.freeze_backbone_epochs > 0 and epoch == args.freeze_backbone_epochs + 1:
            for param in model.parameters():
                param.requires_grad = True
            # Reduce LR for fine-tuning to not overwrite learned temporal features
            for pg in optimizer.param_groups:
                pg['lr'] = args.lr * 0.1
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            log_info(f"[Stage 2] Backbone UNFROZEN. Fine-tuning all {n_trainable:,} params with LR={args.lr*0.1:.2e}")

        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, not args.no_amp, device, accum_steps=args.accum_steps)
        val_m = validate(model, val_loader, criterion, not args.no_amp, device)
        elapsed = time.time() - t0

        # Fix #2: Use AUC for LR scheduling
        scheduler.step(val_m['auc'])

        log_info(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
                 f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | "
                 f"Val Loss: {val_m['loss']:.4f} Acc: {val_m['acc']*100:.1f}% F1: {val_m['f1']:.4f} AUC: {val_m['auc']:.4f}")

        record = {'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc, **val_m}
        history.append(record)

        # Fix #2: Save best model and early-stop based on AUC (not F1)
        if val_m['auc'] > best_auc:
            best_auc = val_m['auc']
            patience_ctr = 0
            ckpt_path = os.path.join(save_dir, "best_video_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_auc': best_auc,
                'patience_ctr': patience_ctr,
                'history': history,
                'args': vars(args)
            }, ckpt_path)
            log_info(f"  --> Best Video Model saved! (AUC: {best_auc:.4f})")
        else:
            patience_ctr += 1
            log_info(f"  No improvement in AUC ({patience_ctr}/{args.patience})")

        # Save a resume checkpoint after every epoch (allows safe stop + resume)
        resume_path = os.path.join(save_dir, "resume.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_auc': best_auc,
            'patience_ctr': patience_ctr,
            'history': history,
            'args': vars(args)
        }, resume_path)
        log_info(f"  [Checkpoint] resume.pth saved (Epoch {epoch}/{args.epochs})")

        if patience_ctr >= args.patience:
            log_info(f"Early stopping triggered at epoch {epoch}.")
            break

    with open(os.path.join(save_dir, "training_history.json"), 'w') as f:
        json.dump(history, f, indent=2)

    log_info(f"Training Complete! Best AUC: {best_auc:.4f} | Saved to: {save_dir}")


if __name__ == '__main__':
    main()
