"""
Model Soups — Weight Averaging Experiment
Averages the weights of Phase 1 and Joint Training models into a single merged checkpoint.
No training required. Reference: Wortsman et al. 2022 "Model soups: averaging weights of 
multiple fine-tuned models improves accuracy without increasing inference time."

Usage (from project root):
    .\.venv\Scripts\python.exe classification/model_soup.py
"""
import os, sys, torch, json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from network.models import model_selection

WEIGHTS1 = os.path.join(os.path.dirname(__file__), "output/efficientnet_b4_20260501_151231/best_model.pth")
WEIGHTS2 = os.path.join(os.path.dirname(__file__), "output/efficientnet_b4_20260903_144444/best_model.pth")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output/efficientnet_b4_soup_20260908")

ALPHA = 0.5  # Weight for model 1. (1-ALPHA) is weight for model 2.
             # 0.5 = equal blend. Try 0.3, 0.4, 0.6, 0.7 for asymmetric blends.

def main():
    device = torch.device('cpu')  # No GPU needed -- just weight math

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    print(f"Loading checkpoint 1: {WEIGHTS1}")
    ckpt1 = torch.load(WEIGHTS1, map_location=device, weights_only=False)
    sd1 = ckpt1['model_state_dict']

    print(f"Loading checkpoint 2: {WEIGHTS2}")
    ckpt2 = torch.load(WEIGHTS2, map_location=device, weights_only=False)
    sd2 = ckpt2['model_state_dict']

    # Verify both have the same keys
    keys1 = set(sd1.keys())
    keys2 = set(sd2.keys())
    if keys1 != keys2:
        raise ValueError(f"Model state dicts have different keys!\n"
                         f"Only in ckpt1: {keys1 - keys2}\n"
                         f"Only in ckpt2: {keys2 - keys1}")
    print(f"  Both models have {len(keys1)} matching parameter keys. OK")

    # Blend weights
    print(f"\nBlending weights (alpha={ALPHA:.2f} x Model1  +  {1-ALPHA:.2f} x Model2)...")
    merged_sd = {}
    for key in sd1:
        p1 = sd1[key].float()
        p2 = sd2[key].float()
        merged_sd[key] = (ALPHA * p1 + (1.0 - ALPHA) * p2)

    # Verify the merged model loads correctly
    print("Verifying merged model loads correctly...")
    model, _, *_ = model_selection('efficientnet_b4', num_out_classes=2, dropout=0.5)
    model.load_state_dict(merged_sd)
    model.eval()
    print("  Merged model loaded successfully. OK")

    # Save
    out_path = os.path.join(OUTPUT_DIR, "best_model.pth")
    save_dict = {
        'model_state_dict': merged_sd,
        'soup_alpha': ALPHA,
        'weights1': WEIGHTS1,
        'weights2': WEIGHTS2,
        'created_at': ts,
        'method': 'model_soup_weight_average',
        'reference': 'Wortsman et al. 2022 - Model soups',
        'best_auc': (ckpt1.get('best_auc', 0) * ALPHA + ckpt2.get('best_auc', 0) * (1 - ALPHA)),
        'epoch': max(ckpt1.get('epoch', 0), ckpt2.get('epoch', 0)),
    }
    torch.save(save_dict, out_path)
    print(f"\nSaved merged model to: {out_path}")

    # Save metadata JSON
    meta = {
        'method': 'Model Soup (Weight Averaging)',
        'alpha': ALPHA,
        'model1': WEIGHTS1,
        'model2': WEIGHTS2,
        'created_at': ts,
        'note': f'alpha={ALPHA} blend: {int(ALPHA*100)}% Phase1 + {int((1-ALPHA)*100)}% JointTraining'
    }
    with open(os.path.join(OUTPUT_DIR, 'soup_config.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Model Soup Complete!")
    print(f"  Model 1 (Phase 1):          {int(ALPHA*100)}% weight")
    print(f"  Model 2 (Joint Training):   {int((1-ALPHA)*100)}% weight")
    print(f"  Output: {out_path}")
    print(f"{'='*60}")
    print(f"\nNext -- evaluate the soup on both benchmarks:")
    print(f"  D4:     evaluate.py --weights {out_path} --data_dir_external E:\\Thesis_Datasets\\images\\updated_data_4")
    print(f"  MIRAGE: evaluate.py --weights {out_path} --data_dir_external E:\\Thesis_Datasets\\images\\updated_data_7")


if __name__ == '__main__':
    main()
