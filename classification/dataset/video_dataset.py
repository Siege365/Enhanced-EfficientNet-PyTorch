"""
Video Dataset Loader - Phase 3: TSM + MHSA Video Deepfake Detection

Implements:
  1. VideoFrameSequenceDataset: Scans directories of pre-extracted frame folders,
     determines real vs fake labels, and loads ordered sequences of T frames.
  2. VideoReplayBufferDataset: Mixes new video clip training data with a balanced
     sample of old video training data to prevent catastrophic forgetting.

Author: Multi-Model Comparative Study Project
"""
import os
import random
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class VideoFrameSequenceDataset(Dataset):
    """
    Loads sequences of pre-extracted video frames as 4D tensors (T, C, H, W).
    
    Expected folder layout:
      root_dir/
        |-- video_folder_1/
        |     |-- frame_0000.jpg
        |     |-- frame_0005.jpg
        |-- video_folder_2/ ...
    
    Labels are automatically inferred from folder/path names:
      - Real (0): 'real', 'original', '0_real', 'celeb-real', 'youtube'
      - Fake (1): 'fake', 'manipulated', 'synth', 'deepfake', 'kling', 'hailuo', 'civitai', 'celeb-synthesis'
    """
    def __init__(self, root_dirs, num_frames=8, transform=None):
        self.num_frames = num_frames
        self.transform = transform
        self.video_clips = []  # List of (folder_path, label, frame_files)
        
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]
            
        real_kw = ['real', 'original', '0_real', 'celeb-real', 'youtube']
        fake_kw = ['fake', 'manipulated', 'synth', 'deepfake', 'kling', 'hailuo', 'civitai', 'celeb-synthesis', 'dfd_fake']
        
        valid_exts = ('.jpg', '.jpeg', '.png', '.webp')
        
        for root_dir in root_dirs:
            if not os.path.exists(root_dir):
                print(f"  [Warning] Directory not found: {root_dir}")
                continue
                
            print(f"  [VideoDataset] Scanning {root_dir} for video frame sequences...")
            for root, dirs, files in os.walk(root_dir):
                img_files = sorted([f for f in files if f.lower().endswith(valid_exts)])
                if len(img_files) >= 2:
                    # Determine label from path path string
                    lower_path = root.lower().replace('\\', '/')
                    
                    is_real = any(kw in lower_path for kw in real_kw)
                    is_fake = any(kw in lower_path for kw in fake_kw)
                    
                    label = None
                    if is_fake and not is_real:
                        label = 1
                    elif is_real and not is_fake:
                        label = 0
                    elif is_fake and is_real:
                        # Tie-breaker: walk path components from leaf upward.
                        # This handles WildDeepfake whose parent dir is named
                        # 'deepfake_in_the_wild' but real clips live inside
                        # 'real_train/.../real/103' where '103' is numeric.
                        path_parts = root.replace('\\', '/').lower().split('/')
                        for part in reversed(path_parts):
                            if any(kw in part for kw in fake_kw):
                                label = 1
                                break
                            elif any(kw in part for kw in real_kw):
                                label = 0
                                break
                    
                    if label is not None:
                        full_img_paths = [os.path.join(root, f) for f in img_files]
                        self.video_clips.append((root, label, full_img_paths))

        n_real = sum(1 for _, l, _ in self.video_clips if l == 0)
        n_fake = sum(1 for _, l, _ in self.video_clips if l == 1)
        print(f"  [VideoDataset] Loaded {len(self.video_clips)} video clips ({n_real:,} Real | {n_fake:,} Fake)")

    def __len__(self):
        return len(self.video_clips)

    def _sample_indices(self, total_frames):
        T = self.num_frames
        if total_frames >= T:
            step = total_frames / T
            indices = [int(i * step) for i in range(T)]
        else:
            # Repeat/pad sequence if fewer frames than T
            indices = [i % total_frames for i in range(T)]
        return indices

    def __getitem__(self, idx):
        folder_path, label, frame_files = self.video_clips[idx]
        indices = self._sample_indices(len(frame_files))
        
        tensor_list = []
        for i in indices:
            img_path = frame_files[i]
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception:
                # Fallback blank image if corrupted
                img = Image.new('RGB', (380, 380), color='black')
                
            if self.transform:
                img_tensor = self.transform(img)
            else:
                img_tensor = transforms.ToTensor()(img)
            tensor_list.append(img_tensor)
            
        # Stack along temporal dimension T -> (T, C, H, W)
        sequence_tensor = torch.stack(tensor_list, dim=0)
        return sequence_tensor, label


class VideoReplayBufferDataset(Dataset):
    """
    Continual Learning Video Replay Buffer.
    Mixes new video clip training dataset with balanced samples of old video datasets.
    """
    def __init__(self, new_dataset, old_dataset, buffer_size=500, seed=42):
        self.new_dataset = new_dataset
        self.old_dataset = old_dataset
        self.buffer_size = buffer_size
        self._rng = random.Random(seed)

        print(f"\n  [VideoReplay] Indexing old video dataset for balanced buffer sampling...")
        old_real_idx = []
        old_fake_idx = []
        for i in range(len(old_dataset)):
            try:
                _, label, _ = old_dataset.video_clips[i]
                (old_real_idx if label == 0 else old_fake_idx).append(i)
            except Exception:
                pass

        n_real = len(old_real_idx)
        n_fake = len(old_fake_idx)
        per_class = min(buffer_size // 2, n_real, n_fake)
        self.actual_buffer_size = per_class * 2

        print(f"  [VideoReplay] Old dataset index: {n_real:,} real | {n_fake:,} fake")
        print(f"  [VideoReplay] Buffer size: {self.actual_buffer_size:,} "
              f"({per_class:,} real + {per_class:,} fake)")

        self._old_real_idx = old_real_idx
        self._old_fake_idx = old_fake_idx
        self._per_class = per_class

        self.buffer_indices = []
        self.refresh_buffer()

    def refresh_buffer(self):
        real_sample = self._rng.sample(self._old_real_idx, self._per_class) if self._per_class > 0 else []
        fake_sample = self._rng.sample(self._old_fake_idx, self._per_class) if self._per_class > 0 else []
        self.buffer_indices = real_sample + fake_sample
        self._rng.shuffle(self.buffer_indices)

    def __len__(self):
        return len(self.new_dataset) + len(self.buffer_indices)

    def __getitem__(self, idx):
        if idx < len(self.new_dataset):
            return self.new_dataset[idx]
        else:
            old_idx = self.buffer_indices[idx - len(self.new_dataset)]
            return self.old_dataset[old_idx]


if __name__ == "__main__":
    print("=" * 60)
    print("Testing VideoDataset Loader...")
    # Test scanner on scraped datasets
    test_dirs = [
        r'E:\Thesis_Datasets\videos\civitai_scraped\frames',
        r'E:\Thesis_Datasets\videos\hailuo_scraped\frames'
    ]
    default_transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor()
    ])
    ds = VideoFrameSequenceDataset(test_dirs, num_frames=8, transform=default_transform)
    if len(ds) > 0:
        seq, lbl = ds[0]
        print(f"  Sample clip 0 -> Tensor shape: {seq.shape} | Label: {lbl}")
    print("=" * 60)
    print("Test completed successfully!")
