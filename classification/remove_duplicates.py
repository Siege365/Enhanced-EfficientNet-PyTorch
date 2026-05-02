import os
import hashlib
from tqdm import tqdm

def get_file_hash(filepath):
    """Returns the MD5 hash of a file to find exact byte-for-byte duplicates."""
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return None

def remove_duplicates(directory):
    print(f"\nScanning for exact duplicates in: {directory}")
    if not os.path.exists(directory):
        print(f"  [ERROR] Directory not found: {directory}")
        return

    seen_hashes = {}
    duplicates_removed = 0
    total_files = 0

    # Walk through all subdirectories (train/real, train/fake, etc.)
    for root, _, files in os.walk(directory):
        # Only process image files to be safe
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        if not image_files:
            continue
            
        for filename in tqdm(image_files, desc=f"Scanning {os.path.basename(root)}"):
            total_files += 1
            filepath = os.path.join(root, filename)
            
            file_hash = get_file_hash(filepath)
            if not file_hash:
                continue
                
            if file_hash in seen_hashes:
                # Duplicate found! Delete the file.
                try:
                    os.remove(filepath)
                    duplicates_removed += 1
                except Exception as e:
                    print(f"  [ERROR] Could not delete {filepath}: {e}")
            else:
                # First time seeing this image, save its hash
                seen_hashes[file_hash] = filepath

    print(f"\n--- Cleanup Summary for {os.path.basename(directory)} ---")
    print(f"  Total images scanned: {total_files}")
    print(f"  Exact duplicates deleted: {duplicates_removed}")
    print(f"  Unique images remaining: {total_files - duplicates_removed}\n")

if __name__ == '__main__':
    print("=" * 50)
    print("DATASET CLEANER: EXACT DUPLICATE REMOVER")
    print("=" * 50)
    
    # Target specifically updated_data_3 as requested
    target_dir = r"C:\Users\natha\OneDrive\Documents\Thesis Project\FaceForensics\updated_data_3"
    remove_duplicates(target_dir)
