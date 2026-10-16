import os
import shutil
import random
from pathlib import Path

# Fix the seed so the split is exactly the same if run multiple times
random.seed(42)

def split_dataset(source_base, test_base, test_fraction=0.15):
    source_path = Path(source_base)
    test_path = Path(test_base)
    
    if not source_path.exists():
        print(f"Skipping {source_base}, does not exist.")
        return
        
    print(f"Processing dataset: {source_path.name}")
    
    for class_folder in source_path.iterdir():
        if not class_folder.is_dir():
            continue
            
        video_folders = [d for d in class_folder.iterdir() if d.is_dir()]
        
        if not video_folders:
            continue
            
        random.shuffle(video_folders)
        
        num_test = int(len(video_folders) * test_fraction)
        test_folders = video_folders[:num_test]
        
        print(f"  Class '{class_folder.name}': Found {len(video_folders)} videos. Moving {num_test} to test set...")
        
        dest_class_dir = test_path / source_path.name / class_folder.name
        os.makedirs(dest_class_dir, exist_ok=True)
        
        for video_dir in test_folders:
            dest_video_dir = dest_class_dir / video_dir.name
            shutil.move(str(video_dir), str(dest_video_dir))

if __name__ == "__main__":
    print("============================================================")
    print("  CREATING STRICT VIDEO-LEVEL TEST SET (Baseline Data)      ")
    print("============================================================")
    
    base_dir = "E:/AI_Generated_Media_Detection/datasets/extracted/videos-zip"
    test_set_dir = os.path.join(base_dir, "baseline_test_set")
    
    datasets_to_split = ["dfdc_faces", "ff_faces"]
    
    for ds in datasets_to_split:
        source_ds = os.path.join(base_dir, ds)
        split_dataset(source_ds, test_set_dir, test_fraction=0.15)
        
    print("\nDataset split complete! You now have a strict, clean test set located at:")
    print(f"  {test_set_dir}")
