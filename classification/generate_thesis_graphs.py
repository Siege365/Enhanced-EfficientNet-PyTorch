"""
Final Thesis Graph Generator — Multi-Model Comparative Study

Generates publication-ready charts comparing all trained models.
(Updated for White Background academic formatting)
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─── Configuration ───────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(BASE, 'output')
GRAPH_DIR = os.path.join(OUTPUT, 'thesis_final_graphs')
os.makedirs(GRAPH_DIR, exist_ok=True)

# Model directories and their display names
MODELS = {
    'mobilenet_v3': {
        'dir': 'mobilenet_v3_20260502_225732',
        'label': 'MobileNetV3-Small\n(2.5M params)',
        'short': 'MobileNetV3',
        'color': '#d62728',       # Academic Red
        'line_style': '--',
    },
    'efficientnet_b4': {
        'dir': 'efficientnet_b4_20260501_151231',
        'label': 'EfficientNet-B4\n(19.3M params)',
        'short': 'EfficientNet-B4',
        'color': '#2ca02c',       # Academic Green
        'line_style': '-',
    },
    'efficientnet_b4_cbam': {
        'dir': 'efficientnet_b4_cbam_20260501_235853',
        'label': 'EfficientNet-B4\n+ CBAM',
        'short': 'B4 + CBAM',
        'color': '#ff7f0e',       # Academic Orange
        'line_style': '-.',
    },
    'efficientnet_b4_spatial': {
        'dir': 'efficientnet_b4_spatial_20260502_125045',
        'label': 'EfficientNet-B4\n+ Spatial-Only',
        'short': 'B4 + Spatial',
        'color': '#9467bd',       # Academic Purple
        'line_style': ':',
    },
}

def find_latest_eval(model_dir):
    full = os.path.join(OUTPUT, model_dir)
    evals = sorted([d for d in os.listdir(full) if d.startswith('eval_')])
    if not evals:
        return None
    return os.path.join(full, evals[-1], 'metrics.json')


def load_all_metrics():
    metrics = {}
    for key, cfg in MODELS.items():
        mpath = find_latest_eval(cfg['dir'])
        if mpath and os.path.isfile(mpath):
            with open(mpath) as f:
                metrics[key] = json.load(f)
            print(f"  Loaded metrics for {cfg['short']}: AUC={metrics[key]['auc_roc']:.4f}")
        else:
            print(f"  WARNING: No eval metrics found for {cfg['short']}")
    return metrics


def load_all_logs():
    logs = {}
    for key, cfg in MODELS.items():
        lpath = os.path.join(OUTPUT, cfg['dir'], 'training_log.csv')
        if os.path.isfile(lpath):
            logs[key] = pd.read_csv(lpath)
            print(f"  Loaded training log for {cfg['short']}: {len(logs[key])} epochs")
    return logs


# ─── CHART 1: Main Performance Bar Chart ─────────────────────────────
def plot_main_comparison(metrics):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']
    metric_keys  = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']

    mob = metrics.get('mobilenet_v3', {})
    eff = metrics.get('efficientnet_b4', {})

    mob_vals = [mob.get(k, 0) for k in metric_keys]
    eff_vals = [eff.get(k, 0) for k in metric_keys]

    x = np.arange(len(metric_names))
    w = 0.35

    bars1 = ax.bar(x - w/2, mob_vals, w, label=MODELS['mobilenet_v3']['short'],
                   color=MODELS['mobilenet_v3']['color'], edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + w/2, eff_vals, w, label=MODELS['efficientnet_b4']['short'],
                   color=MODELS['efficientnet_b4']['color'], edgecolor='black', linewidth=1)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.2%}', ha='center', va='bottom',
                fontsize=10, fontweight='bold', color='black')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.2%}', ha='center', va='bottom',
                fontsize=10, fontweight='bold', color='black')

    ax.set_ylabel('Score', fontsize=12, color='black', fontweight='bold')
    ax.set_title('Model Performance Comparison (Out-of-Distribution Test)',
                 fontsize=14, color='black', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11, color='black')
    ax.set_ylim(0, 1.15)
    ax.tick_params(colors='black')
    ax.legend(fontsize=11, loc='upper left', facecolor='white', edgecolor='black', labelcolor='black')
    ax.grid(axis='y', alpha=0.3, color='gray', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for s in ax.spines.values():
        s.set_color('black')

    plt.tight_layout()
    path = os.path.join(GRAPH_DIR, 'main_performance_comparison.png')
    plt.savefig(path, dpi=300, facecolor='white')
    plt.close()


# ─── CHART 2: Enhancement Analysis Bar Chart ─────────────────────────
def plot_enhancement_analysis(metrics):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']
    metric_keys  = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']

    models_to_show = ['efficientnet_b4', 'efficientnet_b4_cbam', 'efficientnet_b4_spatial']
    colors = [MODELS['efficientnet_b4']['color'], MODELS['efficientnet_b4_cbam']['color'], MODELS['efficientnet_b4_spatial']['color']]
    labels = ['Vanilla B4', 'B4 + CBAM', 'B4 + Spatial-Only']

    x = np.arange(len(metric_names))
    w = 0.25
    offsets = [-w, 0, w]

    for i, (mkey, color, label) in enumerate(zip(models_to_show, colors, labels)):
        vals = [metrics.get(mkey, {}).get(k, 0) for k in metric_keys]
        bars = ax.bar(x + offsets[i], vals, w, label=label,
                      color=color, edgecolor='black', linewidth=1)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{bar.get_height():.2%}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold', color='black')

    ax.set_ylabel('Score', fontsize=12, color='black', fontweight='bold')
    ax.set_title('Effect of Attention Enhancement on EfficientNet-B4',
                 fontsize=14, color='black', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11, color='black')
    ax.set_ylim(0, 1.15)
    ax.tick_params(colors='black')
    ax.legend(fontsize=11, loc='upper left', facecolor='white', edgecolor='black', labelcolor='black')
    ax.grid(axis='y', alpha=0.3, color='gray', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for s in ax.spines.values():
        s.set_color('black')

    plt.tight_layout()
    path = os.path.join(GRAPH_DIR, 'enhancement_analysis.png')
    plt.savefig(path, dpi=300, facecolor='white')
    plt.close()


# ─── CHART 3: Learning Curves ────────────────────────────────────────
def plot_learning_curves(logs):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('white')

    titles = ['Training Loss', 'Validation Loss', 'Validation AUC']
    y_cols = ['train_loss', 'val_loss', 'val_auc']

    for ax, title, ycol in zip(axes, titles, y_cols):
        ax.set_facecolor('white')
        for key, cfg in MODELS.items():
            if key in logs:
                df = logs[key]
                epochs = df['epoch'].values
                vals = df[ycol].values
                ax.plot(epochs, vals, label=cfg['short'],
                        color=cfg['color'], linestyle=cfg['line_style'],
                        linewidth=2.5, marker='o', markersize=4)

        ax.set_title(title, fontsize=13, color='black', fontweight='bold', pad=10)
        ax.set_xlabel('Epoch', fontsize=11, color='black')
        ax.tick_params(colors='black')
        ax.legend(fontsize=9, facecolor='white', edgecolor='black', labelcolor='black')
        ax.grid(alpha=0.3, color='gray', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for s in ax.spines.values():
            s.set_color('black')

    plt.suptitle('Training & Validation Curves — All Architectures',
                 fontsize=15, color='black', fontweight='bold', y=1.05)
    plt.tight_layout()
    path = os.path.join(GRAPH_DIR, 'learning_curves.png')
    plt.savefig(path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()


# ─── CHART 4: Overlaid ROC Curves ────────────────────────────────────
def plot_overlaid_roc(metrics):
    import torch
    from torch.utils.data import DataLoader
    from torch.cuda.amp import autocast
    from sklearn.metrics import roc_curve, auc
    from tqdm import tqdm

    sys.path.insert(0, BASE)
    from network.models import model_selection
    from dataset.transform import (
        efficientnet_default_data_transforms,
        efficientnet_enhanced_data_transforms,
        mobilenet_default_data_transforms
    )
    from dataset.image_dataset import create_dataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = r'C:\Users\natha\OneDrive\Documents\Thesis Project\FaceForensics\updated_data_4'

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    model_configs = {
        'mobilenet_v3': {
            'transform': mobilenet_default_data_transforms['test'],
            'weights': os.path.join(OUTPUT, 'mobilenet_v3_20260502_225732', 'best_model.pth'),
        },
        'efficientnet_b4': {
            'transform': efficientnet_default_data_transforms['test'],
            'weights': os.path.join(OUTPUT, 'efficientnet_b4_20260501_151231', 'best_model.pth'),
        },
        'efficientnet_b4_cbam': {
            'transform': efficientnet_enhanced_data_transforms['test'],
            'weights': os.path.join(OUTPUT, 'efficientnet_b4_cbam_20260501_235853', 'best_model.pth'),
        },
        'efficientnet_b4_spatial': {
            'transform': efficientnet_enhanced_data_transforms['test'],
            'weights': os.path.join(OUTPUT, 'efficientnet_b4_spatial_20260502_125045', 'best_model.pth'),
        },
    }

    ds_config = [{'type': 'folder', 'path': data_path}]

    for key, mcfg in model_configs.items():
        cfg = MODELS[key]
        print(f"  Running inference for {cfg['short']}...")

        ds = create_dataset(ds_config, split='test', transform=mcfg['transform'])
        loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

        model, _, *_ = model_selection(key, num_out_classes=2, dropout=0.5)
        ckpt = torch.load(mcfg['weights'], map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(device)
        model.eval()

        all_probs = []
        all_labels = []
        with torch.no_grad():
            for images, labels in tqdm(loader, desc=f'    {cfg["short"]}', leave=False):
                images = images.to(device)
                with autocast():
                    outputs = model(images)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.numpy())

        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, label=f'{cfg["short"]} (AUC = {roc_auc:.4f})',
                color=cfg['color'], linestyle=cfg['line_style'],
                linewidth=2.5)

        del model
        torch.cuda.empty_cache()

    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.5, linewidth=1.5)

    ax.set_xlabel('False Positive Rate', fontsize=12, color='black', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, color='black', fontweight='bold')
    ax.set_title('Receiver Operating Characteristic (ROC) Curves',
                 fontsize=14, color='black', fontweight='bold', pad=15)
    ax.tick_params(colors='black')
    ax.legend(fontsize=11, loc='lower right', facecolor='white', edgecolor='black', labelcolor='black')
    ax.grid(alpha=0.3, color='gray', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for s in ax.spines.values():
        s.set_color('black')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])

    plt.tight_layout()
    path = os.path.join(GRAPH_DIR, 'roc_curves_overlay.png')
    plt.savefig(path, dpi=300, facecolor='white')
    plt.close()


if __name__ == '__main__':
    print("=" * 60)
    print("THESIS GRAPH GENERATOR — Multi-Model Comparative Study")
    print("=" * 60)

    print("\n[1/5] Loading evaluation metrics...")
    metrics = load_all_metrics()

    print("\n[2/5] Loading training logs...")
    logs = load_all_logs()

    print("\n[3/5] Generating Main Performance Comparison...")
    plot_main_comparison(metrics)

    print("\n[4/5] Generating Enhancement Analysis...")
    plot_enhancement_analysis(metrics)

    print("\n[5/5] Generating Learning Curves...")
    plot_learning_curves(logs)

    # Note: Running overlaid ROC can take ~10 minutes so we print a warning
    print("\n[6/6] Generating Overlaid ROC Curves (this may take ~10 min)...")
    try:
        plot_overlaid_roc(metrics)
    except Exception as e:
        print(f"Error generating ROC curves: {e}")

    print("\n" + "=" * 60)
    print(f"ALL GRAPHS SAVED TO: {GRAPH_DIR}")
    print("=" * 60)
