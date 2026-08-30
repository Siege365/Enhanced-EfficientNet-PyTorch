"""
Unified Image Dataset Loader

Supports three dataset formats:
  1. CSV-labeled (updated_data_1)  : CSV maps filenames -> labels (Shutterstock + AI pairs)
  2. Binary folder (updated_data_2, updated_data_3) : real/ and fake/ subdirectories
  3. Multi-class folder (updated_data_4): real/, fake/, artificial/, deepfake/ subdirectories
     — artificial and deepfake are both merged into label=1 (fake)

Continuous Learning:
  4. ReplayBufferDataset: Mixes new training data with a fixed random sample of old training
     data (the "replay buffer") to prevent catastrophic forgetting during Phase 2 training.

Data root: E:\\Thesis_Datasets\\images\\

Author: Multi-Model Comparative Study Project
"""
import os
import random
import time
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

        # [CLOUD PERFORMANCE FIX]
        # We skip the os.path.exists() check because doing it for 100k+ files
        # over a Google Drive network mount takes over an hour.
        # We trust that the CSV is completely accurate.
        valid_mask = pd.Series([True] * len(self.df))
        
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
        
        try:
            image = Image.open(img_path).convert('RGB')
        except (FileNotFoundError, OSError):
            # [CLOUD SAFETY FIX] If image is corrupted or missing, randomly pick another one
            # to prevent the entire training run from crashing.
            return self.__getitem__(random.randint(0, len(self.df) - 1))
            
        if self.transform:
            image = self.transform(image)
        return image, label


class FolderImageDataset(Dataset):
    """
    Dataset loader for folder-structured image data.

    Supports two structures:

    Binary (updated_data_2, updated_data_3):
        root_dir/
        |-- real/          (label=0)
        |-- fake/          (label=1)

    Multi-class (updated_data_4) — merged into binary:
        root_dir/
        |-- real/          (label=0)
        |-- fake/          (label=1)
        |-- artificial/    (label=1, merged with fake)
        |-- deepfake/      (label=1, merged with fake)

    Labels: 0 = Real, 1 = Fake (AI-generated)
    """

    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}

    # All folder names that map to label=1 (fake)
    FAKE_FOLDER_NAMES = {'fake', 'artificial', 'deepfake'}

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        def safe_listdir(path, max_retries=3):
            # [CLOUD SAFETY FIX] Google Drive API throttles os.listdir for large folders
            # causing OSError: [Errno 5] Input/output error. We must retry a few times.
            for attempt in range(max_retries):
                try:
                    return sorted(os.listdir(path))
                except OSError as e:
                    print(f"  [Cloud I/O Error] Retry {attempt+1}/{max_retries} for {path}: {e}")
                    time.sleep(2)
            raise OSError(f"Failed to list {path} after {max_retries} retries due to Google Drive network throttling.")

        # Load real images (label=0)
        real_dir = os.path.join(root_dir, 'real')
        if os.path.isdir(real_dir):
            for fname in safe_listdir(real_dir):
                ext = os.path.splitext(fname)[1].lower()
                if ext in self.SUPPORTED_EXTENSIONS:
                    self.samples.append((os.path.join(real_dir, fname), 0))
        else:
            print(f"  Warning: No 'real/' folder found in {root_dir}")

        # Load fake images (label=1) — supports fake/, artificial/, deepfake/
        for folder_name in self.FAKE_FOLDER_NAMES:
            fake_dir = os.path.join(root_dir, folder_name)
            if os.path.isdir(fake_dir):
                count = 0
                for fname in safe_listdir(fake_dir):
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in self.SUPPORTED_EXTENSIONS:
                        self.samples.append((os.path.join(fake_dir, fname), 1))
                        count += 1
                if count > 0:
                    print(f"    [{folder_name}/] → label=1 (fake): {count} images")

        n_real = sum(1 for _, l in self.samples if l == 0)
        n_fake = sum(1 for _, l in self.samples if l == 1)
        print(f"  Loaded {len(self.samples)} images from {root_dir}: "
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


class ReplayBufferDataset(Dataset):
    """
    Experience Replay Dataset for Continuous Learning (Phase 2 training).

    Combines NEW training data with a fixed random sample (the "replay buffer")
    from OLD training data. This prevents catastrophic forgetting when the model
    is fine-tuned on new AI generators without re-training on all old data.

    Each epoch, call refresh_buffer() to re-sample a fresh random subset from old
    data. Fresh sampling each epoch prevents overfitting to specific replay samples
    and improves overall generalization.

    Usage (in train.py):
        new_ds = create_dataset(new_cfgs, split='train', transform=tf_train)
        old_ds = create_dataset(old_cfgs, split='train', transform=tf_train)
        combined = ReplayBufferDataset(new_ds, old_ds, buffer_size=15000)

        # At the start of EVERY training epoch:
        combined.refresh_buffer()

    Args:
        new_dataset  : Dataset of NEW data (D5, D6) — used in full
        old_dataset  : Dataset of OLD data (D1, D2, D3) — sampled from
        buffer_size  : Total old samples to include each epoch (default: 15000).
                       Class-balanced: buffer_size//2 real + buffer_size//2 fake.
        seed         : Random seed for reproducible initial sampling (default: 42)
    """

    def __init__(self, new_dataset, old_dataset, buffer_size=15000, seed=42):
        self.new_dataset = new_dataset
        self.old_dataset = old_dataset
        self.buffer_size = buffer_size
        self._rng = random.Random(seed)

        # Fast O(1) label lookup helper without opening image files from disk
        def _get_fast_label(ds, idx):
            if isinstance(ds, ConcatDataset):
                if idx < 0:
                    if -idx > len(ds):
                        raise ValueError("absolute value of index should not exceed dataset length")
                    idx = len(ds) + idx
                dataset_idx = bisect.bisect_right(ds.cumulative_sizes, idx)
                if dataset_idx == 0:
                    sample_idx = idx
                else:
                    sample_idx = idx - ds.cumulative_sizes[dataset_idx - 1]
                return _get_fast_label(ds.datasets[dataset_idx], sample_idx)
            if hasattr(ds, 'indices') and hasattr(ds, 'dataset'): # Subset
                return _get_fast_label(ds.dataset, ds.indices[idx])
            if hasattr(ds, 'df'): # CSVImageDataset
                return int(ds.df.iloc[idx]['label'])
            if hasattr(ds, 'samples'): # FolderImageDataset
                return int(ds.samples[idx][1])
            if hasattr(ds, 'video_clips'): # VideoFrameSequenceDataset
                return int(ds.video_clips[idx][1])
            _, label = ds[idx]
            return label

        import bisect
        print(f"\n  [Replay] Indexing old dataset for balanced buffer sampling...")
        old_real_idx = []
        old_fake_idx = []
        for i in range(len(old_dataset)):
            try:
                label = _get_fast_label(old_dataset, i)
                (old_real_idx if label == 0 else old_fake_idx).append(i)
            except Exception:
                pass

        n_real = len(old_real_idx)
        n_fake = len(old_fake_idx)
        per_class = min(buffer_size // 2, n_real, n_fake)
        self.actual_buffer_size = per_class * 2

        print(f"  [Replay] Old dataset index: {n_real:,} real | {n_fake:,} fake")
        print(f"  [Replay] Buffer size: {self.actual_buffer_size:,} "
              f"({per_class:,} real + {per_class:,} fake) "
              f"from {n_real + n_fake:,} total old images")

        self._old_real_idx = old_real_idx
        self._old_fake_idx = old_fake_idx
        self._per_class = per_class

        # Build the initial buffer
        self.buffer_indices = []
        self.refresh_buffer()

    def refresh_buffer(self):
        """
        Re-sample the replay buffer. Call at the START of each training epoch.
        Provides fresh diversity each epoch and prevents replay sample memorization.
        """
        real_sample = self._rng.sample(self._old_real_idx, self._per_class)
        fake_sample = self._rng.sample(self._old_fake_idx, self._per_class)
        self.buffer_indices = real_sample + fake_sample
        self._rng.shuffle(self.buffer_indices)

    def __len__(self):
        # Total length = all new data + replay buffer
        return len(self.new_dataset) + len(self.buffer_indices)

    def __getitem__(self, idx):
        if idx < len(self.new_dataset):
            # New sample — uses new_dataset's own transform
            return self.new_dataset[idx]
        else:
            # Replay sample from old dataset
            old_idx = self.buffer_indices[idx - len(self.new_dataset)]
            return self.old_dataset[old_idx]
