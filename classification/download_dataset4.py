"""
Dataset 4 Downloader: AI-vs-Deepfake-vs-Real
=============================================
Downloads the prithivMLmods/AI-vs-Deepfake-vs-Real dataset from HuggingFace
and organizes it into a folder structure compatible with our evaluation pipeline.

Source:   https://huggingface.co/datasets/prithivMLmods/AI-vs-Deepfake-vs-Real
Classes:  0=Artificial (modern AI), 1=Deepfake (old GAN), 2=Real

Output structure:
    updated_data_4/
        test/
            real/        <- label 2 (Real photos)
            fake/        <- label 0 (Artificial) + label 1 (Deepfake) merged
        test_detailed/   <- for granular analysis in thesis
            real/
            artificial/  <- label 0: Modern AI (Midjourney, SD, etc.)
            deepfake/    <- label 1: Old GAN deepfakes

Usage:
    python download_dataset4.py
"""

import os
import sys
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = r'C:\Users\natha\OneDrive\Documents\Thesis Project\FaceForensics\updated_data_4'
DATASET_NAME = 'prithivMLmods/AI-vs-Deepfake-vs-Real'

# Label mapping from the dataset card
LABEL_MAP = {0: 'artificial', 1: 'deepfake', 2: 'real'}


def check_dependencies():
    """Check that required packages are installed."""
    missing = []
    try:
        import datasets
    except ImportError:
        missing.append('datasets')
    try:
        import huggingface_hub
    except ImportError:
        missing.append('huggingface_hub')
    try:
        from PIL import Image
    except ImportError:
        missing.append('Pillow')

    if missing:
        print(f"\n[ERROR] Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)
    print("[OK] All dependencies found.")


def setup_directories(output_dir):
    """Create the output folder structure."""
    dirs = {
        'real':       os.path.join(output_dir, 'test', 'real'),
        'fake':       os.path.join(output_dir, 'test', 'fake'),
        'det_real':   os.path.join(output_dir, 'test_detailed', 'real'),
        'det_art':    os.path.join(output_dir, 'test_detailed', 'artificial'),
        'det_deep':   os.path.join(output_dir, 'test_detailed', 'deepfake'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def main():
    print("=" * 60)
    print("DATASET 4 DOWNLOADER: AI-vs-Deepfake-vs-Real")
    print("=" * 60)

    check_dependencies()

    from datasets import load_dataset
    from huggingface_hub import login
    from PIL import Image

    # -------------------------------------------------------------------
    # HuggingFace login
    # -------------------------------------------------------------------
    print("\nYou need to be logged in to HuggingFace to access this dataset.")
    print("Enter your HuggingFace access token (from https://huggingface.co/settings/tokens):")
    token = input("Token: ").strip()
    if not token:
        print("[ERROR] No token provided. Exiting.")
        sys.exit(1)
    login(token=token)
    print("[OK] Logged in successfully.\n")

    # -------------------------------------------------------------------
    # Download dataset
    # -------------------------------------------------------------------
    print(f"Downloading dataset: {DATASET_NAME}")
    print("This may take a few minutes (1.96 GB)...")
    ds = load_dataset(DATASET_NAME, split='train', trust_remote_code=True)
    total = len(ds)
    print(f"[OK] Downloaded {total:,} images.\n")

    # -------------------------------------------------------------------
    # Create output directories
    # -------------------------------------------------------------------
    dirs = setup_directories(OUTPUT_DIR)
    print(f"Output directory: {OUTPUT_DIR}")
    print("Saving images...")

    counts = {'real': 0, 'artificial': 0, 'deepfake': 0}
    errors = 0

    for i, sample in enumerate(tqdm(ds, desc="  Extracting")):
        try:
            img = sample['image']
            label_id = sample['label']
            label_name = LABEL_MAP.get(label_id, 'unknown')

            # Convert to RGB if needed (handles RGBA/grayscale edge cases)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            filename = f"{label_name}_{i:05d}.jpg"

            # Save to binary test/real or test/fake
            if label_name == 'real':
                img.save(os.path.join(dirs['real'], filename), quality=95)
                img.save(os.path.join(dirs['det_real'], filename), quality=95)
                counts['real'] += 1
            elif label_name == 'artificial':
                img.save(os.path.join(dirs['fake'], filename), quality=95)
                img.save(os.path.join(dirs['det_art'], filename), quality=95)
                counts['artificial'] += 1
            elif label_name == 'deepfake':
                img.save(os.path.join(dirs['fake'], filename), quality=95)
                img.save(os.path.join(dirs['det_deep'], filename), quality=95)
                counts['deepfake'] += 1

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"\n  [WARN] Error on sample {i}: {e}")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    total_fake = counts['artificial'] + counts['deepfake']
    print(f"\n{'=' * 60}")
    print("DOWNLOAD COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Real images:       {counts['real']:,}")
    print(f"  Modern AI (fake):  {counts['artificial']:,}  (Midjourney, SD, DALL-E)")
    print(f"  Old Deepfake:      {counts['deepfake']:,}  (GAN-based deepfakes)")
    print(f"  Total fake:        {total_fake:,}")
    print(f"  Errors:            {errors}")
    print(f"\n  Binary test set saved to:   {os.path.join(OUTPUT_DIR, 'test')}")
    print(f"  Detailed test set saved to: {os.path.join(OUTPUT_DIR, 'test_detailed')}")
    print(f"\n  Use for evaluation with:")
    print(f"  python evaluate.py --model efficientnet_b4 \\")
    print(f"      --weights output/.../best_model.pth \\")
    print(f"      --data_dir_external \"{OUTPUT_DIR}\"")


if __name__ == '__main__':
    main()
