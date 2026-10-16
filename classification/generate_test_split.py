"""
generate_test_split.py

Generates a deterministic 15% test split from the baseline datasets
(dfdc_faces and ff_faces) by writing the held-out video folder paths
to a plain text file (baseline_test_videos.txt).

This approach requires ZERO extra disk space and does NOT move or copy
any files, making it safe to run even while Exp 4 is actively training
on the same dataset directories.

Usage (run once on COARE login node):
    /opt/modules/library/cpu/anaconda/3-2024.10-1/bin/python generate_test_split.py

Output:
    E:/AI_Generated_Media_Detection/datasets/extracted/videos-zip/baseline_test_videos.txt
"""
import os
import random
from pathlib import Path

# Fixed seed for full reproducibility — any reviewer can verify this split
SEED = 42
TEST_FRACTION = 0.15

BASE_DIR = "E:/AI_Generated_Media_Detection/datasets/extracted/videos-zip"
OUTPUT_TXT = os.path.join(BASE_DIR, "baseline_test_videos.txt")

DATASETS_TO_SPLIT = ["dfdc_faces", "ff_faces"]

VALID_EXTS = ('.jpg', '.jpeg', '.png', '.webp')


def find_video_folders(dataset_path):
    """Walk a dataset directory and return all leaf folders that contain frames."""
    video_folders = []
    for root, dirs, files in os.walk(dataset_path):
        img_files = [f for f in files if f.lower().endswith(VALID_EXTS)]
        if len(img_files) >= 2:
            video_folders.append(root)
    return video_folders


def main():
    print("=" * 60)
    print("  GENERATING BASELINE TEST SPLIT (Zero-Copy Method)")
    print(f"  Seed: {SEED} | Test Fraction: {TEST_FRACTION:.0%}")
    print("=" * 60)

    rng = random.Random(SEED)
    all_test_folders = []

    for ds_name in DATASETS_TO_SPLIT:
        ds_path = os.path.join(BASE_DIR, ds_name)
        if not os.path.exists(ds_path):
            print(f"\n[Warning] Dataset not found, skipping: {ds_path}")
            continue

        print(f"\nScanning {ds_name}...")
        video_folders = find_video_folders(ds_path)
        rng.shuffle(video_folders)

        num_test = int(len(video_folders) * TEST_FRACTION)
        test_folders = video_folders[:num_test]
        train_folders = video_folders[num_test:]

        print(f"  Total video clips  : {len(video_folders):,}")
        print(f"  Training (85%)     : {len(train_folders):,}")
        print(f"  Test (15%)         : {num_test:,} <-- held out")

        all_test_folders.extend(test_folders)

    # Write held-out paths to txt file
    with open(OUTPUT_TXT, 'w') as f:
        for folder in sorted(all_test_folders):
            f.write(folder + "\n")

    print(f"\n{'=' * 60}")
    print(f"  Done! {len(all_test_folders):,} folders written to:")
    print(f"  {OUTPUT_TXT}")
    print(f"\n  Sample of held-out test paths (verify these look correct):")
    for p in sorted(all_test_folders)[:5]:
        print(f"    {p}")
    print("=" * 60)


if __name__ == "__main__":
    main()
