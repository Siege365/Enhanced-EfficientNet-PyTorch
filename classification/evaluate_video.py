"""
Evaluation Pipeline - Phase 3 Video Deepfake Detection

Runs a trained video model against UNSEEN external test video sequences and produces:
  - Per-class metrics (Accuracy, Precision, Recall, F1, AUC)
  - Confusion Matrix (saved as PNG)
  - ROC Curve (saved as PNG)
  - Full metrics saved to JSON
  - Dynamically calculates the optimal threshold using Youden's J statistic

Usage:
    python evaluate_video.py --architecture baseline \
        --weights output_exp1/video_tsm_mhsa_XXXX/best_video_model.pth \
        --test_dirs /scratch1/.../dfdc_test /scratch1/.../ff_test
"""
import io
import os, sys, json, argparse
from datetime import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix, roc_curve)
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from network.video_models import video_model_selection
from dataset.video_dataset import VideoFrameSequenceDataset


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate Video Models on unseen test sets')
    p.add_argument('--architecture', type=str, required=True,
                   choices=['baseline', 'enhanced', 'actual_baseline'],
                   help='Model architecture to evaluate')
    p.add_argument('--weights', type=str, required=True,
                   help='Path to best_video_model.pth checkpoint')
    p.add_argument('--test_dirs', nargs='+', required=True,
                   help='Directories containing the test video frame sequences')
    p.add_argument('--num_frames', type=int, default=8, help='Number of frames per video sequence (T)')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--no_youden', action='store_true', help='Disable Youden J statistic and lock threshold to 0.5')
    return p.parse_args()


def get_val_transform(img_size=380):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])


def run_inference(model, loader, device):
    """
    Runs model inference and returns (preds, labels, probs).
    """
    model.eval()
    labels_all, probs_all = [], []
    with torch.no_grad():
        for imgs, labs in tqdm(loader, desc="  Evaluating"):
            imgs = imgs.to(device)
            out = model(imgs)
            pr = torch.softmax(out, 1)
            
            labels_all.extend(labs.numpy())
            probs_all.extend(pr[:, 1].cpu().numpy())
            
    return labels_all, probs_all


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

    print(f"\nLoading model architecture: {args.architecture}")
    model, img_sz, _ = video_model_selection(
        architecture=args.architecture,
        num_frames=args.num_frames,
        pretrained_checkpoint=None,  # We load the entire state dict below anyway
        dropout=args.dropout
    )
    
    print(f"  Loading weights from: {args.weights}")
    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    transform = get_val_transform(img_size=img_sz)
    
    print(f"\nLoading UNSEEN test video data from {len(args.test_dirs)} directories...")
    for d in args.test_dirs:
        print(f"  -> {d}")
        
    test_ds = VideoFrameSequenceDataset(args.test_dirs, num_frames=args.num_frames, transform=transform)
    print(f"\nTest dataset: {len(test_ds)} total video clips")
    
    use_pw = args.num_workers > 0
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True,
                             persistent_workers=use_pw, prefetch_factor=2 if use_pw else None)

    # Run inference
    print(f"\nRunning inference on {len(test_ds)} test instances...")
    labels_all, probs_all = run_inference(model, test_loader, device)

    # Compute metrics
    if args.no_youden:
        optimal_threshold = 0.5
        print("\n  [Threshold] Youden's J optimization DISABLED. Locked to 0.5000.")
    else:
        fpr, tpr, thresholds = roc_curve(labels_all, probs_all)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        print(f"\n  [Threshold] Youden's J optimization ENABLED. Found: {optimal_threshold:.4f}")

    # Re-calculate predictions using the threshold
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
    print(f"TEST SET RESULTS — {args.architecture.upper()} VIDEO MODEL")
    print(f"{'=' * 50}")
    threshold_str = " (Fixed)" if args.no_youden else " (via Youden's J)"
    print(f"  Threshold : {optimal_threshold:.4f}{threshold_str}")
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
        'architecture': args.architecture,
        'weights': args.weights,
        'test_dirs': args.test_dirs,
        'test_samples': len(test_ds),
        'youden_enabled': not args.no_youden,
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
    plot_confusion_matrix(cm, os.path.join(eval_dir, 'confusion_matrix.png'), args.architecture)
    plot_roc_curve(labels_all, probs_all, os.path.join(eval_dir, 'roc_curve.png'), args.architecture, auc)

    print(f"\nEvaluation complete! All results saved to:\n  {eval_dir}")


if __name__ == '__main__':
    main()
