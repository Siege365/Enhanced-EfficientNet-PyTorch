"""
Unified Image Dataset Loader

Supports two dataset formats:
  1. CSV-labeled datasets (updated_data_1): CSV file maps filenames -> labels
  2. Folder-structured datasets (updated_data_2, updated_data_3): real/ and fake/ subdirectories

This loader automatically detects the format and handles both seamlessly,
allowing training on combined datasets from different sources.

Author: Multi-Model Comparative Study Project
"""
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset


class CSVImageDataset(Dataset):
    """
    Dataset loader for CSV-labeled image data (updated_data_1 format).

    Expected structure:
        root_dir/
        |-- train.csv          (columns: index, file_name, label)
        |-- train_data/        (contains the images)
        |-- test.csv
        |-- test_data_v2/

    Labels: 0 = Real, 1 = Fake (AI-generated)
    """

    def __init__(self, csv_path, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.df = pd.read_csv(csv_path)

        if 'file_name' not in self.df.columns or 'label' not in self.df.columns:
            raise ValueError(
                f"CSV must contain 'file_name' and 'label' columns. "
                f"Found: {list(self.df.columns)}"
            )

        # Filter out missing files
        valid_mask = self.df['file_name'].apply(
            lambda f: os.path.exists(os.path.join(root_dir, f))
        )
        n_missing = (~valid_mask).sum()
        if n_missing > 0:
            print(f"  Warning: {n_missing} images not found, skipping them.")
        self.df = self.df[valid_mask].reset_index(drop=True)

        print(f"  Loaded {len(self.df)} images from CSV: "
              f"{(self.df['label'] == 0).sum()} real, "
              f"{(self.df['label'] == 1).sum()} fake")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row['file_name'])
        label = int(row['label'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class FolderImageDataset(Dataset):
    """
    Dataset loader for folder-structured image data (updated_data_2/3 format).

    Expected structure:
        root_dir/
        |-- real/     (real images)
        |-- fake/     (AI-generated images)

    Labels: 0 = Real, 1 = Fake (AI-generated)
    """

    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Load real images (label=0)
        real_dir = os.path.join(root_dir, 'real')
        if os.path.isdir(real_dir):
            for fname in os.listdir(real_dir):
                ext = os.path.splitext(fname)[1].lower()
                if ext in self.SUPPORTED_EXTENSIONS:
                    self.samples.append((os.path.join(real_dir, fname), 0))

        # Load fake images (label=1)
        fake_dir = os.path.join(root_dir, 'fake')
        if os.path.isdir(fake_dir):
            for fname in os.listdir(fake_dir):
                ext = os.path.splitext(fname)[1].lower()
                if ext in self.SUPPORTED_EXTENSIONS:
                    self.samples.append((os.path.join(fake_dir, fname), 1))

        n_real = sum(1 for _, l in self.samples if l == 0)
        n_fake = sum(1 for _, l in self.samples if l == 1)
        print(f"  Loaded {len(self.samples)} images from folders: "
              f"{n_real} real, {n_fake} fake")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def create_dataset(dataset_config, split='train', transform=None):
    """
    Create a unified dataset from one or more data sources.

    Args:
        dataset_config: List of dataset source dicts, each containing:
            - 'type': 'csv' or 'folder'
            - 'path': Root directory path
            - For CSV type: 'csv_file' key with CSV filename
        split: 'train', 'val', or 'test'
        transform: torchvision transforms

    Returns:
        ConcatDataset combining all sources
    """
    datasets = []

    for config in dataset_config:
        dtype = config['type']
        path = config['path']

        if dtype == 'csv':
            csv_file = config.get('csv_file', f'{split}.csv')
            csv_path = os.path.join(path, csv_file)
            if os.path.exists(csv_path):
                print(f"Loading CSV dataset: {csv_path}")
                ds = CSVImageDataset(csv_path, path, transform=transform)
                datasets.append(ds)
            else:
                print(f"  Warning: CSV not found: {csv_path}, skipping.")

        elif dtype == 'folder':
            folder_path = os.path.join(path, split)
            if os.path.isdir(folder_path):
                print(f"Loading folder dataset: {folder_path}")
                ds = FolderImageDataset(folder_path, transform=transform)
                datasets.append(ds)
            else:
                print(f"  Warning: Folder not found: {folder_path}, skipping.")

        else:
            print(f"  Warning: Unknown dataset type '{dtype}', skipping.")

    if not datasets:
        raise ValueError("No valid datasets found! Check your paths and config.")

    if len(datasets) == 1:
        return datasets[0]

    combined = ConcatDataset(datasets)
    total = len(combined)
    print(f"\nCombined dataset: {total} total images from {len(datasets)} sources")
    return combined
