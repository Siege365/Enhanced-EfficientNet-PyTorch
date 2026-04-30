"""
Training Pipeline - EfficientNet-B4 Comparative Study

Usage:
    python train.py --model efficientnet_b4 --epochs 30 --batch_size 16
    python train.py --model efficientnet_b4_cbam --epochs 30 --batch_size 16

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
from dataset.transform import efficientnet_default_data_transforms, efficientnet_enhanced_data_transforms
from dataset.image_dataset import create_dataset


# Default data directory — points to your existing downloaded datasets
DEFAULT_DATA_DIR = r'C:\Users\natha\OneDrive\Documents\Thesis Project\FaceForensics'


def parse_args():
    p = argparse.ArgumentParser(description='Train EfficientNet-B4 for AI Image Detection')
    p.add_argument('--model', type=str, default='efficientnet_b4',
                   choices=['efficientnet_b4', 'efficientnet_b4_cbam'])
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
    p.add_argument('--oversample_d3', action='store_true', default=True,
                   help='Oversample updated_data_3 (3x) to balance modern AI images')
    return p.parse_args()


def build_datasets(args, transforms_dict):
    root = args.data_dir
    d1 = os.path.join(root, 'updated_data_1')
    d2 = os.path.join(root, 'updated_data_2')
    d3 = os.path.join(root, 'updated_data_3')
    cfgs = []
    if os.path.exists(d1):
        cfgs.append({'type': 'csv', 'path': d1, 'csv_file': 'train.csv'})
    if os.path.exists(d2):
        cfgs.append({'type': 'folder', 'path': d2})
    if os.path.exists(d3):
        cfgs.append({'type': 'folder', 'path': d3})
        if args.oversample_d3:
            cfgs.append({'type': 'folder', 'path': d3})
            cfgs.append({'type': 'folder', 'path': d3})
            print(f"  [OK] OpenFake data found (3x oversampled): {d3}")
        else:
            print(f"  [OK] OpenFake data found: {d3}")
    if not cfgs:
        raise ValueError(f"No datasets found in {root}")
    ds = create_dataset(cfgs, split='train', transform=transforms_dict['train'])
    total = len(ds)
    val_sz = int(total * args.val_split)
    train_sz = total - val_sz
    train_sub, val_sub = random_split(ds, [train_sz, val_sz],
                                       generator=torch.Generator().manual_seed(42))
    print(f"\nTrain: {train_sz} | Val: {val_sz}")
    return train_sub, val_sub


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    rloss = 0.0; preds_all = []; labels_all = []; probs_all = []
    for imgs, labs in tqdm(loader, desc="  Train", leave=False):
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

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run = f"{args.model}_{ts}"
    odir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output_dir, run)
    os.makedirs(odir, exist_ok=True)
    cfg = vars(args)
    cfg['device'] = str(device)
    cfg['run'] = run
    cfg['use_amp'] = use_amp
    json.dump(cfg, open(os.path.join(odir, 'config.json'), 'w'), indent=2)
    print(f"Output: {odir}\nModel: {args.model}\nAMP: {use_amp}")

    # Select transforms based on model
    if args.model == 'efficientnet_b4':
        tf = efficientnet_default_data_transforms
    elif args.model == 'efficientnet_b4_cbam':
        tf = efficientnet_enhanced_data_transforms
    else:
        tf = efficientnet_default_data_transforms

    train_ds, val_ds = build_datasets(args, tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    print(f"\nBuilding {args.model}...")
    model, img_sz, *_ = model_selection(args.model, num_out_classes=2, dropout=args.dropout)
    model = model.to(device)
    tp = sum(p.numel() for p in model.parameters())
    print(f"  Params: {tp:,}")

    criterion = nn.CrossEntropyLoss()
    effective_lr = args.lr * 0.1 if args.fine_tune else args.lr
    if args.fine_tune:
        print(f"  Fine-tune mode: LR reduced to {effective_lr:.2e}")
    optimizer = optim.Adam(model.parameters(), lr=effective_lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    scaler = GradScaler() if use_amp else None

    start_epoch = 0
    best_auc = 0.0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck['model_state_dict'])
        if args.fine_tune:
            print(f"  Loaded model weights for fine-tuning from: {args.resume}")
            start_epoch = 0
        else:
            optimizer.load_state_dict(ck['optimizer_state_dict'])
            start_epoch = ck['epoch'] + 1
            best_auc = ck.get('best_auc', 0.0)

    log_path = os.path.join(odir, 'training_log.csv')
    fields = ['epoch', 'lr', 'train_loss', 'train_acc', 'train_f1', 'train_auc',
              'val_loss', 'val_acc', 'val_precision', 'val_recall', 'val_f1', 'val_auc']
    with open(log_path, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    print(f"\n{'=' * 60}\nTRAINING - {args.model.upper()}\n{'=' * 60}")
    patience_ctr = 0
    train_start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch + 1}/{args.epochs} | LR: {lr:.2e}")
        tm = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
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
        print(f"  Train - Loss:{tm['loss']:.4f} Acc:{tm['accuracy']:.4f} "
              f"F1:{tm['f1']:.4f} AUC:{tm['auc_roc']:.4f}")
        print(f"  Val   - Loss:{vm['loss']:.4f} Acc:{vm['accuracy']:.4f} "
              f"P:{vm['precision']:.4f} R:{vm['recall']:.4f} "
              f"F1:{vm['f1']:.4f} AUC:{vm['auc_roc']:.4f}")

        elapsed_total = time.time() - train_start_time
        hours, rem = divmod(elapsed_total, 3600)
        minutes, seconds = divmod(rem, 60)
        curr_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"  Epoch Finished At: {curr_time}")
        print(f"  Epoch Time: {time.time() - t0:.1f}s | "
              f"Total Running Time: {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s")
        if vm['auc_roc'] > best_auc:
            best_auc = vm['auc_roc']
            patience_ctr = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_auc,
                'val_metrics': vm,
                'config': cfg
            }, os.path.join(odir, 'best_model.pth'))
            print(f"  Best model saved! (AUC: {best_auc:.4f})")
        else:
            patience_ctr += 1
            print(f"  No improvement ({patience_ctr}/{args.patience})")
        if patience_ctr >= args.patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_auc': best_auc,
        'config': cfg
    }, os.path.join(odir, 'final_model.pth'))
    print(f"\nTRAINING COMPLETE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Best AUC: {best_auc:.4f}")
    print(f"  Output: {odir}")


if __name__ == '__main__':
    main()
