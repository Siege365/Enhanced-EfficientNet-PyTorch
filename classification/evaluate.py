"""
Evaluation Pipeline - EfficientNet-B4 Comparative Study

Runs a trained model against the UNSEEN test set and produces:
  - Per-class metrics (Accuracy, Precision, Recall, F1, AUC)
  - Confusion Matrix (saved as PNG)
  - ROC Curve (saved as PNG)
  - Full metrics saved to JSON

Usage:
    python evaluate.py --model efficientnet_b4 --weights output/efficientnet_b4_XXXXXXXX_XXXXXX/best_model.pth
    python evaluate.py --model efficientnet_b4_cbam --weights output/efficientnet_b4_cbam_XXXXXXXX_XXXXXX/best_model.pth
"""
import io
import os, sys, json, argparse
from datetime import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from PIL import Image
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix, roc_curve)
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from network.models import model_selection
from dataset.transform import (
    efficientnet_default_data_transforms, 
    efficientnet_enhanced_data_transforms,
    mobilenet_default_data_transforms
)
from dataset.image_dataset import create_dataset

DEFAULT_DATA_DIR = r'E:\Thesis_Datasets\images'


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate Models on unseen test set')
    p.add_argument('--model', type=str, required=True,
                   choices=['mobilenet_v3', 'efficientnet_b4', 'efficientnet_b4_cbam', 'efficientnet_b4_spatial'])
    p.add_argument('--weights', type=str, required=True,
                   help='Path to best_model.pth checkpoint')
    p.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR)
    p.add_argument('--data_dir_external', type=str, default=None,
                   help='Path to external test dataset (e.g. updated_data_4) to completely override internal datasets')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--tta', action='store_true',
                   help='Enable Test-Time Augmentation: averages predictions across '
                        '5 augmented views (original, H-flip, JPEG-50, JPEG-75, center-crop) '
                        'for improved robustness on compressed/in-the-wild images.')
    return p.parse_args()


def build_test_datasets(args, transform):
    """
    Loads the UNSEEN test split from datasets 2, 3, and 4.

    Dataset sources (E:\\Thesis_Datasets\\images\\):
      - updated_data_1 : SKIPPED — test.csv is a Kaggle submission file with no ground-truth labels
      - updated_data_2 : DeepDetect-2025 — test/real/ + test/fake/
      - updated_data_3 : OpenFake 2025-26  — test/real/ + test/fake/
      - updated_data_4 : External test set  — test/real/ + test/fake/ + test/artificial/ + test/deepfake/
                          (artificial and deepfake both merged to label=1)
      - updated_data_5 : SuSy (if available) — test/real/ + test/fake/
      - updated_data_6 : MS COCOAI (if available) — test/real/ + test/fake/
    """
    cfgs = []
    dataset_names = []  # track names for per-dataset eval

    if args.data_dir_external:
        # Evaluate ONLY on the external dataset path provided
        ext_base = args.data_dir_external
        if os.path.exists(ext_base):
            cfgs.append({'type': 'folder', 'path': ext_base})
            dataset_names.append(os.path.basename(ext_base))
            print(f"  [EXTERNAL] Evaluating exclusively on: {ext_base}")
        else:
            raise ValueError(f"External dataset path not found: {ext_base}")
    else:
        # Default: evaluate on all available datasets
        root = args.data_dir
        all_ds = [
            ('updated_data_2', 'DeepDetect-2025'),
            ('updated_data_3', 'OpenFake 2025-26'),
            ('updated_data_4', 'External test set'),
            ('updated_data_5', 'SuSy'),
            ('updated_data_6', 'MS COCOAI (Defactify)'),
        ]

        print("  [SKIP] updated_data_1: test.csv has no ground-truth labels (Kaggle submission format)")

        for folder, label in all_ds:
            path = os.path.join(root, folder)
            if os.path.exists(path):
                cfgs.append({'type': 'folder', 'path': path})
                dataset_names.append(f"{folder} ({label})")
                print(f"  [OK] {folder} ({label}): {path}")
            else:
                print(f"  [SKIP] {folder} ({label}): not found")

    if not cfgs:
        raise ValueError("No test datasets found! Check E:\\Thesis_Datasets\\images\\ exists.")
    ds = create_dataset(cfgs, split='test', transform=transform)
    print(f"\nTest dataset: {len(ds)} total images")
    return ds, cfgs, dataset_names


def _jpeg_compress_pil(img: Image.Image, quality: int) -> Image.Image:
    """Re-compress a PIL image at the given JPEG quality and return a new PIL image."""
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def build_tta_transforms(base_size: int = 380):
    """
    Returns a list of (name, callable) pairs.  Each callable receives a PIL image
    and returns an augmented PIL image ready for the standard tensor pipeline.

    TTA views:
      1. Original   — no change
      2. H-Flip     — horizontal mirror
      3. JPEG-50    — heavy JPEG compression (quality 50)
      4. JPEG-75    — moderate JPEG compression (quality 75)
      5. CenterCrop — zoom-in 90% then resize back (simulates crop/repost)
    """
    def original(img):   return img
    def hflip(img):      return img.transpose(Image.FLIP_LEFT_RIGHT)
    def jpeg50(img):     return _jpeg_compress_pil(img, quality=50)
    def jpeg75(img):     return _jpeg_compress_pil(img, quality=75)
    def centercrop(img):
        w, h = img.size
        cw, ch = int(w * 0.9), int(h * 0.9)
        left = (w - cw) // 2
        top  = (h - ch) // 2
        return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BILINEAR)

    return [original, hflip, jpeg50, jpeg75, centercrop]


def run_inference(model, loader, device, args, base_transform):
    """
    Runs model inference and returns (preds, labels, probs).
    If args.tta is True, averages softmax probabilities across all TTA views.
    """
    if not args.tta:
        # ── Standard single-pass inference ──────────────────────────────────
        preds_all, labels_all, probs_all = [], [], []
        with torch.no_grad():
            for imgs, labs in tqdm(loader, desc="  Evaluating"):
                imgs = imgs.to(device)
                out  = model(imgs)
                pr   = torch.softmax(out, 1)
                preds_all.extend(pr.argmax(1).cpu().numpy())
                labels_all.extend(labs.numpy())
                probs_all.extend(pr[:, 1].cpu().numpy())
        return preds_all, labels_all, probs_all

    # ── TTA multi-pass inference ─────────────────────────────────────────────
    # Strategy: temporarily disable the dataset's built-in transform so we get
    # raw PIL images, apply each TTA augmentation ourselves, then run the
    # standard Resize->ToTensor->Normalize pipeline before feeding to the model.
    from torch.utils.data import ConcatDataset as _ConcatDataset

    dataset      = loader.dataset
    tta_augments = build_tta_transforms()

    to_tensor = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    # ── Disable transforms on all underlying sub-datasets ───────────────────
    saved_transforms = []
    def _save_and_disable(ds):
        if isinstance(ds, _ConcatDataset):
            for sub in ds.datasets:
                _save_and_disable(sub)
        elif hasattr(ds, 'transform'):
            saved_transforms.append((ds, ds.transform))
            ds.transform = None

    _save_and_disable(dataset)

    preds_all, labels_all, probs_all = [], [], []
    n          = len(dataset)
    batch_size = args.batch_size

    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc="  Evaluating (TTA)"):
            end     = min(start + batch_size, n)
            indices = list(range(start, end))

            # Collect labels and raw PIL images for this batch (once)
            pil_images   = []
            labels_batch = []
            for idx in indices:
                item = dataset[idx]
                lab  = item[1]
                img  = item[0]
                if not isinstance(img, Image.Image):
                    img = transforms.ToPILImage()(img.clamp(0, 1))
                pil_images.append(img)
                labels_batch.append(lab)

            labels_all.extend(labels_batch)

            # Accumulate softmax probs over all TTA views
            avg_probs = None
            for aug_fn in tta_augments:
                batch_tensors = [to_tensor(aug_fn(pil)) for pil in pil_images]
                batch  = torch.stack(batch_tensors).to(device)
                out    = model(batch)
                pr     = torch.softmax(out, 1)  # (B, 2)
                avg_probs = pr if avg_probs is None else avg_probs + pr

            avg_probs = avg_probs / len(tta_augments)
            preds_all.extend(avg_probs.argmax(1).cpu().numpy())
            probs_all.extend(avg_probs[:, 1].cpu().numpy())

    # ── Restore original transforms ──────────────────────────────────────────
    for ds, tf in saved_transforms:
        ds.transform = tf

    return preds_all, labels_all, probs_all


def evaluate_per_dataset(model, device, args, cfgs, dataset_names, transform, eval_dir):
    print("\nStarting per-dataset evaluation...")
    results = {}
    for cfg, name in zip(cfgs, dataset_names):
        print(f"  -> Evaluating {name}...")
        ds = create_dataset([cfg], split='test', transform=transform)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        
        preds_all, labels_all = [], []
        with torch.no_grad():
            for imgs, labs in loader:
                imgs = imgs.to(device)
                out = model(imgs)
                preds_all.extend(out.argmax(1).cpu().numpy())
                labels_all.extend(labs.numpy())
        
        acc = accuracy_score(labels_all, preds_all)
        results[name] = {"accuracy": float(acc), "samples": len(ds)}
        print(f"     Accuracy: {acc:.4f}")
    
    with open(os.path.join(eval_dir, 'per_dataset_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)


def plot_confusion_matrix(cm, output_path, model_name):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    classes = ['Real', 'Fake']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks); ax.set_xticklabels(classes, fontsize=12)
    ax.set_yticks(tick_marks); ax.set_yticklabels(classes, fontsize=12)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black', fontsize=14)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_title(f'Confusion Matrix\n{model_name}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_roc_curve(labels, probs, output_path, model_name, auc):
    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='#4e79a7', lw=2, label=f'AUC = {auc:.4f}')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve\n{model_name}', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    else:
        print("\nNo GPU — using CPU (evaluation will be slower)")

    # Output directory: same folder as the weights file
    weights_dir = os.path.dirname(os.path.abspath(args.weights))
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    eval_dir = os.path.join(weights_dir, f'eval_{ts}')
    os.makedirs(eval_dir, exist_ok=True)
    print(f"Evaluation output: {eval_dir}")

    # Select transforms
    if args.model == 'mobilenet_v3':
        transform = mobilenet_default_data_transforms['test']
    elif args.model in ('efficientnet_b4_cbam', 'efficientnet_b4_spatial'):
        transform = efficientnet_enhanced_data_transforms['test']
    else:
        transform = efficientnet_default_data_transforms['test']

    # Build test dataset
    print(f"\nLoading UNSEEN test data...")
    test_ds, cfgs, dataset_names = build_test_datasets(args, transform)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # Load model
    print(f"\nLoading model: {args.model}")
    model, img_sz, *_ = model_selection(args.model, num_out_classes=2, dropout=args.dropout)
    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"  Loaded weights from: {args.weights}")
    print(f"  Trained best AUC (validation): {checkpoint.get('best_auc', 'N/A'):.4f}")

    # Run inference
    tta_note = " (TTA ×5 views)" if args.tta else ""
    print(f"\nRunning inference on {len(test_ds)} test images{tta_note}...")
    if args.tta:
        print("  TTA views: original | H-flip | JPEG-50 | JPEG-75 | center-crop")
    preds_all, labels_all, probs_all = run_inference(model, test_loader, device, args, transform)

    # Compute metrics (optimized with Youden's J statistic)
    fpr, tpr, thresholds = roc_curve(labels_all, probs_all)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # Re-calculate predictions using the optimal threshold
    optimized_preds = (np.array(probs_all) >= optimal_threshold).astype(int)

    acc  = accuracy_score(labels_all, optimized_preds)
    prec = precision_score(labels_all, optimized_preds, average='weighted', zero_division=0)
    rec  = recall_score(labels_all, optimized_preds, average='weighted', zero_division=0)
    f1   = f1_score(labels_all, optimized_preds, average='weighted', zero_division=0)
    try:
        auc = roc_auc_score(labels_all, probs_all)
    except Exception:
        auc = 0.0
    cm = confusion_matrix(labels_all, optimized_preds)

    print(f"\n{'=' * 50}")
    print(f"TEST SET RESULTS — {args.model.upper()}")
    print(f"{'=' * 50}")
    print(f"  Optimal Threshold : {optimal_threshold:.4f} (via Youden's J)")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"\n  Confusion Matrix (Optimized):")
    print(f"    TP={cm[1,1]:,}  FP={cm[0,1]:,}")
    print(f"    FN={cm[1,0]:,}  TN={cm[0,0]:,}")
    print(f"{'=' * 50}\n")

    # Save metrics to JSON
    metrics = {
        'model': args.model,
        'weights': args.weights,
        'tta_enabled': args.tta,
        'tta_views': 5 if args.tta else 1,
        'external_dataset': bool(args.data_dir_external),
        'dataset_path': args.data_dir_external if args.data_dir_external else args.data_dir,
        'test_samples': len(test_ds),
        'optimal_threshold': round(float(optimal_threshold), 4),
        'accuracy': round(acc, 6),
        'precision': round(prec, 6),
        'recall': round(rec, 6),
        'f1_score': round(f1, 6),
        'auc_roc': round(auc, 6),
        'confusion_matrix': cm.tolist(),
        'evaluated_at': ts
    }
    metrics_path = os.path.join(eval_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {metrics_path}")

    # Save plots
    print("\nGenerating plots...")
    plot_confusion_matrix(cm, os.path.join(eval_dir, 'confusion_matrix.png'), args.model)
    plot_roc_curve(labels_all, probs_all, os.path.join(eval_dir, 'roc_curve.png'), args.model, auc)

    # Per-dataset breakdown (anti-forgetting evidence)
    if not args.data_dir_external and len(cfgs) > 1:
        evaluate_per_dataset(model, device, args, cfgs, dataset_names, transform, eval_dir)

    print(f"\nEvaluation complete! All results saved to:\n  {eval_dir}")


if __name__ == '__main__':
    main()
