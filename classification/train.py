"""
Training Pipeline - EfficientNet-B4 Comparative Study

Supports two training modes:
  1. Standard Training (Phase 1):
       python train.py --model efficientnet_b4 --epochs 30 --batch_size 16

  2. Continuous Learning / Experience Replay (Phase 2):
       python train.py --model efficientnet_b4 --continuous \\
           --old_checkpoint output/efficientnet_b4_YYYYMMDD_HHMMSS/best_model.pth \\
           --new_datasets updated_data_5 updated_data_6 \\
           --replay_buffer_size 15000 --replay_ratio 0.25 \\
           --epochs 15 --lr 0.00002

Author: Multi-Model Comparative Study Project
"""
import os, sys, csv, json, time, argparse
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from network.models import model_selection
from dataset.transform import (
    efficientnet_default_data_transforms,
    efficientnet_enhanced_data_transforms,
    mobilenet_default_data_transforms,
    mobilenet_enhanced_data_transforms
)
from dataset.image_dataset import create_dataset, ReplayBufferDataset


# Default data directory — E: drive datasets root
DEFAULT_DATA_DIR = r'E:\Thesis_Datasets\images'

CONSOLE_LOG_FILE = None

def log_info(msg=""):
    ts = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    formatted = f"{ts} {msg}" if msg else ""
    print(formatted, flush=True)
    if CONSOLE_LOG_FILE and formatted:
        with open(CONSOLE_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")


def parse_args():
    p = argparse.ArgumentParser(description='Train Deep Learning Models for AI Image Detection')
    p.add_argument('--model', type=str, default='efficientnet_b4',
                   choices=['mobilenet_v3', 'efficientnet_b4', 'efficientnet_b4_cbam', 'efficientnet_b4_spatial'])
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR)
    p.add_argument('--val_split', type=float, default=0.15)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=0.0002)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=7)
    p.add_argument('--min_lr', type=float, default=1e-7)
    p.add_argument('--no_amp', action='store_true', default=False)
    p.add_argument('--output_dir', type=str, default='output')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--fine_tune', action='store_true', default=False,
                   help='Fine-tune from --resume checkpoint with lower LR')
    p.add_argument('--subset_fraction', type=float, default=1.0,
                   help='Fraction of the dataset to use for training (0.0 to 1.0)')
    p.add_argument('--enhanced_aug', action='store_true', default=False,
                   help='Use enhanced transforms (JPEG compression, blur, resizing) for in-the-wild robustness')

    # ── Continuous Learning arguments ──────────────────────────────────────────
    p.add_argument('--continuous', action='store_true', default=False,
                   help='Enable Continuous Learning (Experience Replay) mode')
    p.add_argument('--old_checkpoint', type=str, default=None,
                   help='Path to Phase 1 best_model.pth (required for --continuous)')
    p.add_argument('--new_datasets', nargs='+', default=None,
                   help='New dataset folder names to train on (e.g. updated_data_5 updated_data_6)')
    p.add_argument('--replay_buffer_size', type=int, default=15000,
                   help='Number of old samples in replay buffer per epoch. '
                        'Class-balanced: half real, half fake. (default: 15000)')
    p.add_argument('--replay_ratio', type=float, default=0.25,
                   help='Fraction of each epoch that comes from old replay data. '
                        '0.25 = 25%% old, 75%% new (default: 0.25, unused in indexing but '
                        'informational — actual ratio set by buffer_size vs new data size)')

    # ── Joint Training argument ────────────────────────────────────────────────
    p.add_argument('--joint_extra_paths', nargs='+', default=None,
                   help='Absolute paths to extra datasets (e.g. D5, D6) to include alongside '
                        'D1-D3 in a single joint training run. Each path must contain real/ '
                        'and fake/ subdirectories (folder-based dataset format).')
    return p.parse_args()


# ==============================================================================
# PHASE 1 — Standard Training Dataset Builder
# ==============================================================================

def build_datasets(args, transforms_dict):
    """
    Builds combined train + val splits from all available image datasets.

    Dataset sources (E:\\Thesis_Datasets\\images\\):
      - updated_data_1 : Shutterstock real + AI-paired, labels via train.csv
      - updated_data_2 : DeepDetect-2025 faces (StyleGAN3, DALL-E 3, Midjourney, SD3)
      - updated_data_3 : OpenFake subset (Flux, GPT Image 1, Imagen 4, MJ7, Grok-2, ...)
      - updated_data_4 : Test-only dataset — skipped during training

    Joint Training (--joint_extra_paths):
      Pass absolute paths to D5, D6 etc. to include them alongside D1-D3 in a single
      combined training run, eliminating catastrophic forgetting by design.
    """
    root = args.data_dir
    d1 = os.path.join(root, 'updated_data_1')
    d2 = os.path.join(root, 'updated_data_2')
    d3 = os.path.join(root, 'updated_data_3')
    # updated_data_4 has NO train split — it is used exclusively for testing
    cfgs = []
    if os.path.exists(d1):
        cfgs.append({'type': 'csv', 'path': d1, 'csv_file': 'train.csv'})
        print(f"  [OK] Dataset 1 (Shutterstock+AI): {d1}")
    if os.path.exists(d2):
        cfgs.append({'type': 'folder', 'path': d2})
        print(f"  [OK] Dataset 2 (DeepDetect-2025): {d2}")
    if os.path.exists(d3):
        cfgs.append({'type': 'folder', 'path': d3})
        print(f"  [OK] Dataset 3 (OpenFake 2025-26): {d3}")

    # ── Joint extra datasets (D5, D6, or any absolute path) ───────────────────
    if args.joint_extra_paths:
        for extra_path in args.joint_extra_paths:
            if os.path.exists(extra_path):
                cfgs.append({'type': 'folder', 'path': extra_path})
                print(f"  [OK] Joint Extra Dataset: {extra_path}")
            else:
                print(f"  [WARN] Joint extra dataset not found, skipping: {extra_path}")

    if not cfgs:
        raise ValueError(f"No datasets found in {root}. Check E:\\Thesis_Datasets\\images\\ exists.")
    ds = create_dataset(cfgs, split='train', transform=transforms_dict['train'])
    total = len(ds)

    if args.subset_fraction < 1.0:
        subset_sz = int(total * args.subset_fraction)
        ignore_sz = total - subset_sz
        ds, _ = random_split(ds, [subset_sz, ignore_sz], generator=torch.Generator().manual_seed(42))
        total = len(ds)
        print(f"  [!] Using subset: {total} images ({args.subset_fraction * 100:.1f}% of total data)")

    val_sz = int(total * args.val_split)
    train_sz = total - val_sz
    train_sub, val_sub = random_split(ds, [train_sz, val_sz],
                                       generator=torch.Generator().manual_seed(42))
    print(f"\nPhase 1 Train: {train_sz} | Val: {val_sz}")
    return train_sub, val_sub


# ==============================================================================
# PHASE 2 — Continuous Learning Dataset Builder
# ==============================================================================

def build_continuous_datasets(args, transforms_dict):
    """
    Builds datasets for Phase 2 Continuous Learning using Experience Replay.

    Architecture:
      - NEW data (D5, D6, ...): Full training split — the model learns these
      - OLD data (D1, D2, D3): Random subset (replay buffer) — prevents forgetting
      - Combined via ReplayBufferDataset

    Dual Validation:
      - val_new: 15% split from NEW data — measures learning progress
      - val_old: Fixed 2000-sample split from OLD data — measures forgetting
        (using the val_old split ensures we never train on these old samples)

    Returns:
        train_ds    : ReplayBufferDataset (new + replay buffer)
        val_new_ds  : Validation from NEW datasets
        val_old_ds  : Validation from OLD datasets
    """
    root = args.data_dir

    # ── Load NEW datasets (what we're teaching the model) ─────────────────────
    print("\n  Loading NEW datasets (Phase 2 training data)...")
    new_cfgs = []
    for ds_name in args.new_datasets:
        ds_path = os.path.join(root, ds_name)
        if os.path.exists(ds_path):
            new_cfgs.append({'type': 'folder', 'path': ds_path})
            print(f"  [OK] New dataset: {ds_path}")
        else:
            print(f"  [WARN] New dataset not found, skipping: {ds_path}")

    if not new_cfgs:
        raise ValueError(
            f"No new datasets found in {root}. "
            f"Check --new_datasets argument: {args.new_datasets}"
        )
    new_ds_full = create_dataset(new_cfgs, split='train', transform=transforms_dict['train'])

    # Count class distribution in new data for class weighting
    new_real_count = 0
    new_fake_count = 0
    if hasattr(new_ds_full, 'datasets'):  # ConcatDataset
        for sub_ds in new_ds_full.datasets:
            if hasattr(sub_ds, 'samples'):
                for _, lbl in sub_ds.samples:
                    if lbl == 0:
                        new_real_count += 1
                    else:
                        new_fake_count += 1
            elif hasattr(sub_ds, 'df'):
                new_real_count += int((sub_ds.df['label'] == 0).sum())
                new_fake_count += int((sub_ds.df['label'] == 1).sum())
    print(f"  New data class distribution: {new_real_count:,} real | {new_fake_count:,} fake")

    # Split new data into train and val_new
    new_total = len(new_ds_full)
    new_val_sz = int(new_total * args.val_split)
    new_train_sz = new_total - new_val_sz
    new_train_ds, val_new_ds = random_split(
        new_ds_full, [new_train_sz, new_val_sz],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"  New data — Train: {new_train_sz:,} | Val: {new_val_sz:,}")

    # ── Load OLD datasets (what we DON'T want to forget) ──────────────────────
    print("\n  Loading OLD datasets (replay buffer source)...")
    d1 = os.path.join(root, 'updated_data_1')
    d2 = os.path.join(root, 'updated_data_2')
    d3 = os.path.join(root, 'updated_data_3')
    old_cfgs = []
    if os.path.exists(d1):
        old_cfgs.append({'type': 'csv', 'path': d1, 'csv_file': 'train.csv'})
        print(f"  [OK] D1 (Shutterstock+AI): {d1}")
    if os.path.exists(d2):
        old_cfgs.append({'type': 'folder', 'path': d2})
        print(f"  [OK] D2 (DeepDetect-2025): {d2}")
    if os.path.exists(d3):
        old_cfgs.append({'type': 'folder', 'path': d3})
        print(f"  [OK] D3 (OpenFake 2025-26): {d3}")

    if not old_cfgs:
        raise ValueError("No old datasets (D1/D2/D3) found. Cannot build replay buffer.")

    old_ds_full = create_dataset(old_cfgs, split='train', transform=transforms_dict['train'])

    # Carve out a small, FIXED val_old split from old data (never used in replay)
    old_val_sz = min(2000, int(len(old_ds_full) * 0.02))
    old_train_sz = len(old_ds_full) - old_val_sz
    old_train_ds, val_old_ds = random_split(
        old_ds_full, [old_train_sz, old_val_sz],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"\n  Old data — Replay pool: {old_train_sz:,} | Val (anti-forgetting check): {old_val_sz:,}")

    # ── Wrap in ReplayBufferDataset ────────────────────────────────────────────
    train_ds = ReplayBufferDataset(
        new_dataset=new_train_ds,
        old_dataset=old_train_ds,
        buffer_size=args.replay_buffer_size,
        seed=42
    )
    ratio = train_ds.actual_buffer_size / len(train_ds) * 100
    print(f"\n  [Replay] Final combined train size: {len(train_ds):,} images")
    print(f"  [Replay] Replay ratio: {ratio:.1f}% old / {100 - ratio:.1f}% new per epoch")

    # Compute effective class counts for weighted loss
    # Train split keeps ~85% of new data, plus balanced replay buffer
    train_ratio = new_train_sz / new_total
    eff_real = int(new_real_count * train_ratio) + train_ds._per_class
    eff_fake = int(new_fake_count * train_ratio) + train_ds._per_class
    class_counts = (eff_real, eff_fake)
    print(f"  [Class Weights] Effective training: {eff_real:,} real | {eff_fake:,} fake")

    return train_ds, val_new_ds, val_old_ds, class_counts


# ==============================================================================
# TRAINING LOOPS
# ==============================================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp,
                    replay_dataset=None):
    """
    Train for one epoch.
    If replay_dataset is provided, calls refresh_buffer() at the start to
    re-sample the replay buffer with a fresh random subset of old data.
    """
    model.train()
    if replay_dataset is not None:
        replay_dataset.refresh_buffer()

    rloss = 0.0; preds_all = []; labels_all = []; probs_all = []
    pbar = tqdm(loader, desc="  Train", leave=False)
    for imgs, labs in pbar:
        imgs, labs = imgs.to(device), labs.to(device)
        optimizer.zero_grad()
        if use_amp:
            with autocast():
                out = model(imgs); loss = criterion(out, labs)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            out = model(imgs); loss = criterion(out, labs)
            loss.backward(); optimizer.step()
        rloss += loss.item() * imgs.size(0)
        pr = torch.softmax(out.detach(), 1)
        preds_all.extend(pr.argmax(1).cpu().numpy())
        labels_all.extend(labs.cpu().numpy())
        probs_all.extend(pr[:, 1].cpu().numpy())
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    n = len(loader.dataset)
    try:
        auc = roc_auc_score(labels_all, probs_all)
    except:
        auc = 0.0
    return {
        'loss': rloss / n,
        'accuracy': accuracy_score(labels_all, preds_all),
        'f1': f1_score(labels_all, preds_all, average='weighted', zero_division=0),
        'auc_roc': auc
    }


def validate(model, loader, criterion, device, use_amp):
    model.eval()
    rloss = 0.0; preds_all = []; labels_all = []; probs_all = []
    with torch.no_grad():
        for imgs, labs in tqdm(loader, desc="  Val", leave=False):
            imgs, labs = imgs.to(device), labs.to(device)
            if use_amp:
                with autocast():
                    out = model(imgs); loss = criterion(out, labs)
            else:
                out = model(imgs); loss = criterion(out, labs)
            rloss += loss.item() * imgs.size(0)
            pr = torch.softmax(out, 1)
            preds_all.extend(pr.argmax(1).cpu().numpy())
            labels_all.extend(labs.cpu().numpy())
            probs_all.extend(pr[:, 1].cpu().numpy())
    n = len(loader.dataset)
    try:
        auc = roc_auc_score(labels_all, probs_all)
    except:
        auc = 0.0
    return {
        'loss': rloss / n,
        'accuracy': accuracy_score(labels_all, preds_all),
        'precision': precision_score(labels_all, preds_all, average='weighted', zero_division=0),
        'recall': recall_score(labels_all, preds_all, average='weighted', zero_division=0),
        'f1': f1_score(labels_all, preds_all, average='weighted', zero_division=0),
        'auc_roc': auc
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    args = parse_args()
    use_amp = not args.no_amp
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"\nGPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB)")
    else:
        use_amp = False
        print("\nNo GPU, using CPU")

    # Validate continuous learning args
    if args.continuous:
        if not args.old_checkpoint:
            raise ValueError(
                "--old_checkpoint is required for --continuous mode.\n"
                "Example: --old_checkpoint output/efficientnet_b4_YYYYMMDD/best_model.pth"
            )
        if not args.new_datasets:
            raise ValueError(
                "--new_datasets is required for --continuous mode.\n"
                "Example: --new_datasets updated_data_5 updated_data_6"
            )
        if not os.path.exists(args.old_checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.old_checkpoint}")
        print(f"\n{'=' * 60}")
        print(f"  CONTINUOUS LEARNING MODE (Experience Replay)")
        print(f"  Phase 1 checkpoint : {args.old_checkpoint}")
        print(f"  New datasets       : {args.new_datasets}")
        print(f"  Replay buffer size : {args.replay_buffer_size:,}")
        print(f"{'=' * 60}")

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    mode_prefix = 'continuous' if args.continuous else ''
    run = f"{args.model}{'_' + mode_prefix if mode_prefix else ''}_{ts}"
    odir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output_dir, run)
    os.makedirs(odir, exist_ok=True)
    global CONSOLE_LOG_FILE
    CONSOLE_LOG_FILE = os.path.join(odir, 'console.log')
    cfg = vars(args)
    cfg['device'] = str(device)
    cfg['run'] = run
    cfg['use_amp'] = use_amp
    cfg['mode'] = 'continuous' if args.continuous else 'standard'
    json.dump(cfg, open(os.path.join(odir, 'config.json'), 'w'), indent=2)
    log_info(f"Session started. Output dir: {odir} | Model: {args.model} | AMP: {use_amp}")

    # Select transforms based on model and enhanced_aug flag
    if args.model == 'mobilenet_v3':
        tf = mobilenet_enhanced_data_transforms if args.enhanced_aug else mobilenet_default_data_transforms
    elif args.model.startswith('efficientnet_b4'):
        tf = efficientnet_enhanced_data_transforms if args.enhanced_aug else efficientnet_default_data_transforms
    else:
        tf = efficientnet_default_data_transforms

    # ── Build datasets ─────────────────────────────────────────────────────────
    replay_dataset = None  # Only set in continuous mode
    val_new_loader = None  # Only used in continuous mode

    class_counts = None
    if args.continuous:
        print(f"\nBuilding Continuous Learning datasets...")
        train_ds, val_new_ds, val_old_ds, class_counts = build_continuous_datasets(args, tf)
        replay_dataset = train_ds  # Pass to train_one_epoch for refresh_buffer()

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers, pin_memory=True, drop_last=True)
        val_new_loader = DataLoader(val_new_ds, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_old_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)
        # val_loader = val_old (for early stopping / best model tracking)
        print(f"\nContinuous Learning loaders ready.")
    else:
        print(f"\nBuilding Phase 1 training datasets...")
        train_ds, val_ds = build_datasets(args, tf)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)

    # ── Build model ────────────────────────────────────────────────────────────
    print(f"\nBuilding {args.model}...")
    model, img_sz, *_ = model_selection(args.model, num_out_classes=2, dropout=args.dropout)
    model = model.to(device)
    tp = sum(p.numel() for p in model.parameters())
    print(f"  Params: {tp:,}")

    # Class-weighted loss to combat fake-heavy imbalance in D5/D6
    if class_counts is not None:
        n_real, n_fake = class_counts
        total = n_real + n_fake
        w_real = total / (2.0 * n_real)
        w_fake = total / (2.0 * n_fake)
        weights = torch.tensor([w_real, w_fake], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        log_info(f"  Class-weighted loss: real={w_real:.3f}, fake={w_fake:.3f}")
    else:
        criterion = nn.CrossEntropyLoss()

    # In continuous mode: use the old_checkpoint LR * 0.1 (much lower to avoid overwriting)
    if args.continuous:
        effective_lr = args.lr * 0.1
        print(f"  Continuous mode: LR = {args.lr:.2e} × 0.1 = {effective_lr:.2e} "
              f"(conservative to preserve old features)")
    elif args.fine_tune:
        effective_lr = args.lr * 0.1
        print(f"  Fine-tune mode: LR reduced to {effective_lr:.2e}")
    else:
        effective_lr = args.lr

    optimizer = optim.Adam(model.parameters(), lr=effective_lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    scaler = GradScaler() if use_amp else None

    # ── Load checkpoint ────────────────────────────────────────────────────────
    start_epoch = 0
    best_auc = 0.0

    checkpoint_path = args.old_checkpoint if args.continuous else args.resume
    if checkpoint_path:
        ck = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ck['model_state_dict'])
        if args.continuous:
            # Continuous learning: fresh optimizer, start from epoch 0
            print(f"  Loaded Phase 1 weights for continuous learning: {checkpoint_path}")
            print(f"  Phase 1 best AUC was: {ck.get('best_auc', 'N/A'):.4f}")
            start_epoch = 0
        elif args.fine_tune:
            print(f"  Loaded model weights for fine-tuning from: {checkpoint_path}")
            start_epoch = 0
        else:
            optimizer.load_state_dict(ck['optimizer_state_dict'])
            start_epoch = ck['epoch'] + 1
            best_auc = ck.get('best_auc', 0.0)

    # ── Setup logging ──────────────────────────────────────────────────────────
    log_path = os.path.join(odir, 'training_log.csv')
    if args.continuous:
        fields = ['epoch', 'lr', 'train_loss', 'train_acc', 'train_f1', 'train_auc',
                  'val_new_loss', 'val_new_acc', 'val_new_f1', 'val_new_auc',
                  'val_old_loss', 'val_old_acc', 'val_old_f1', 'val_old_auc',
                  'val_combined_auc']
    else:
        fields = ['epoch', 'lr', 'train_loss', 'train_acc', 'train_f1', 'train_auc',
                  'val_loss', 'val_acc', 'val_precision', 'val_recall', 'val_f1', 'val_auc']
    with open(log_path, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    # ── Training loop ──────────────────────────────────────────────────────────
    mode_label = 'CONTINUOUS LEARNING' if args.continuous else 'STANDARD TRAINING'
    print(f"\n{'=' * 60}\n{mode_label} - {args.model.upper()}\n{'=' * 60}")
    patience_ctr = 0
    train_start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        lr = optimizer.param_groups[0]['lr']
        log_info(f"--- Epoch {epoch + 1}/{args.epochs} | LR: {lr:.2e} ---")

        tm = train_one_epoch(model, train_loader, criterion, optimizer, scaler,
                             device, use_amp, replay_dataset=replay_dataset)

        if args.continuous:
            # Dual validation: measure both learning (new) and remembering (old)
            vm_new = validate(model, val_new_loader, criterion, device, use_amp)
            vm_old = validate(model, val_loader, criterion, device, use_amp)
            combined_auc = (vm_new['auc_roc'] + vm_old['auc_roc']) / 2.0

            scheduler.step()
            row = {
                'epoch': epoch + 1, 'lr': lr,
                'train_loss': tm['loss'], 'train_acc': tm['accuracy'],
                'train_f1': tm['f1'], 'train_auc': tm['auc_roc'],
                'val_new_loss': vm_new['loss'], 'val_new_acc': vm_new['accuracy'],
                'val_new_f1': vm_new['f1'], 'val_new_auc': vm_new['auc_roc'],
                'val_old_loss': vm_old['loss'], 'val_old_acc': vm_old['accuracy'],
                'val_old_f1': vm_old['f1'], 'val_old_auc': vm_old['auc_roc'],
                'val_combined_auc': combined_auc
            }
            with open(log_path, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)

            log_info(f"  Train        - Loss:{tm['loss']:.4f} Acc:{tm['accuracy']:.4f} F1:{tm['f1']:.4f} AUC:{tm['auc_roc']:.4f}")
            log_info(f"  Val NEW  ↑   - Loss:{vm_new['loss']:.4f} Acc:{vm_new['accuracy']:.4f} F1:{vm_new['f1']:.4f} AUC:{vm_new['auc_roc']:.4f}  ← Learning new generators")
            log_info(f"  Val OLD  ←   - Loss:{vm_old['loss']:.4f} Acc:{vm_old['accuracy']:.4f} F1:{vm_old['f1']:.4f} AUC:{vm_old['auc_roc']:.4f}  ← Remembering old generators")
            log_info(f"  Combined AUC - {combined_auc:.4f}  (avg of NEW + OLD — best model criterion)")

            # Forgetting alert
            if vm_old['auc_roc'] < 0.90:
                log_info(f"  ⚠️  WARNING: Old data AUC dropped to {vm_old['auc_roc']:.4f} — potential catastrophic forgetting!")

            best_metric = combined_auc

        else:
            vm = validate(model, val_loader, criterion, device, use_amp)
            scheduler.step()
            row = {
                'epoch': epoch + 1, 'lr': lr,
                'train_loss': tm['loss'], 'train_acc': tm['accuracy'],
                'train_f1': tm['f1'], 'train_auc': tm['auc_roc'],
                'val_loss': vm['loss'], 'val_acc': vm['accuracy'],
                'val_precision': vm['precision'], 'val_recall': vm['recall'],
                'val_f1': vm['f1'], 'val_auc': vm['auc_roc']
            }
            with open(log_path, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)
            log_info(f"  Train - Loss:{tm['loss']:.4f} Acc:{tm['accuracy']:.4f} F1:{tm['f1']:.4f} AUC:{tm['auc_roc']:.4f}")
            log_info(f"  Val   - Loss:{vm['loss']:.4f} Acc:{vm['accuracy']:.4f} P:{vm['precision']:.4f} R:{vm['recall']:.4f} F1:{vm['f1']:.4f} AUC:{vm['auc_roc']:.4f}")
            best_metric = vm['auc_roc']

        elapsed_total = time.time() - train_start_time
        hours, rem = divmod(elapsed_total, 3600)
        minutes, seconds = divmod(rem, 60)
        log_info(f"  Epoch Time: {time.time() - t0:.1f}s | Total Running Time: {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s")

        # Save best model based on combined_auc (continuous) or val_auc (standard)
        if best_metric > best_auc:
            best_auc = best_metric
            patience_ctr = 0
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_auc,
                'config': cfg
            }
            if args.continuous:
                save_dict['val_new_auc'] = vm_new['auc_roc']
                save_dict['val_old_auc'] = vm_old['auc_roc']
                save_dict['val_combined_auc'] = combined_auc
                save_dict['phase1_checkpoint'] = args.old_checkpoint
                save_dict['new_datasets'] = args.new_datasets
                save_dict['replay_buffer_size'] = args.replay_buffer_size
            else:
                save_dict['val_metrics'] = vm if not args.continuous else {}
            torch.save(save_dict, os.path.join(odir, 'best_model.pth'))
            if args.continuous:
                log_info(f"  ✅ Best model saved! Combined AUC: {best_auc:.4f} (NEW: {vm_new['auc_roc']:.4f} | OLD: {vm_old['auc_roc']:.4f})")
            else:
                log_info(f"  ✅ Best model saved! (AUC: {best_auc:.4f})")
        else:
            patience_ctr += 1
            log_info(f"  No improvement ({patience_ctr}/{args.patience})")
        if patience_ctr >= args.patience:
            log_info(f"Early stopping at epoch {epoch + 1}")
            break

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_auc': best_auc,
        'config': cfg
    }, os.path.join(odir, 'final_model.pth'))
    print(f"\nTRAINING COMPLETE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Best {'Combined ' if args.continuous else ''}AUC: {best_auc:.4f}")
    print(f"  Output: {odir}")

    if args.continuous:
        print(f"\n  ─── Continuous Learning Summary ───")
        print(f"  Phase 1 checkpoint : {args.old_checkpoint}")
        print(f"  New datasets used  : {args.new_datasets}")
        print(f"  Replay buffer size : {args.replay_buffer_size:,}")
        print(f"  Best Combined AUC  : {best_auc:.4f}")
        print(f"  Review training_log.csv to compare val_new_auc vs val_old_auc per epoch.")


if __name__ == '__main__':
    main()
