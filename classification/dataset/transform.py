"""
Data Transforms - Model Augmentation Pipelines

Provides transform sets for:
  EfficientNet-B4 (380x380 native resolution):
    1. efficientnet_default_data_transforms:  Vanilla transforms (baseline)
    2. efficientnet_enhanced_data_transforms: Social media-hardened augmentations (p=0.5 JPEG)
    3. efficientnet_hardened_data_transforms: Mandatory JPEG + chained compression (Hardened Joint Retraining)
  MobileNetV3 (224x224 native resolution):
    4. mobilenet_default_data_transforms:     Standard MobileNetV3 transforms
    5. mobilenet_enhanced_data_transforms:    Social media-hardened augmentations

Author: Multi-Model Comparative Study Project
"""
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


# =============================================================================
# CUSTOM AUGMENTATION TRANSFORMS (Social Media Simulation)
# =============================================================================

class JPEGCompression:
    """
    Simulates JPEG compression artifacts that occur on social media uploads.
    Platforms like WhatsApp (~70), Instagram (~85), Facebook (~85) apply
    lossy compression to all uploaded images.
    """
    def __init__(self, quality_range=(30, 95)):
        self.quality_range = quality_range

    def __call__(self, img):
        quality = np.random.randint(self.quality_range[0], self.quality_range[1])
        img_array = np.array(img)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        _, enc_img = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        dec_img = cv2.imdecode(enc_img, cv2.IMREAD_COLOR)
        dec_img = cv2.cvtColor(dec_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(dec_img)


class RandomDownscaleUpscale:
    """
    Simulates the resize cycle from AI generation -> social media upload.
    """
    def __init__(self, scale_range=(0.5, 1.0)):
        self.scale_range = scale_range

    def __call__(self, img):
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        w, h = img.size
        new_w = max(int(w * scale), 32)
        new_h = max(int(h * scale), 32)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        img = img.resize((w, h), Image.BILINEAR)
        return img



class GaussianBlurPIL:
    """
    Applies Gaussian blur to simulate platform-level image processing.
    """
    def __init__(self, sigma_range=(0.1, 2.0)):
        self.sigma_range = sigma_range

    def __call__(self, img):
        sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
        img_array = np.array(img)
        ksize = int(np.ceil(sigma * 3) * 2 + 1)
        ksize = max(ksize, 3)
        if ksize % 2 == 0:
            ksize += 1
        blurred = cv2.GaussianBlur(img_array, (ksize, ksize), sigma)
        return Image.fromarray(blurred)


class ChainedJPEGCompression:
    """
    Simulates the chained compression cycle of social media sharing:
        AI-generated image -> Upload (pass 1) -> Download -> Re-share (pass 2)

    Real-world social media pipelines apply JPEG compression multiple times.
    A single JPEG pass (used in enhanced transforms) does not fully replicate
    the artifact destruction seen in actual re-shared content.

    Pass 1: Simulates first upload to platform (quality 50-90)
    Pass 2: Simulates re-download and re-upload (quality 40-80)
    """
    def __init__(self, q1_range=(50, 90), q2_range=(40, 80)):
        self.q1_range = q1_range
        self.q2_range = q2_range

    def __call__(self, img):
        img_array = np.array(img)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Pass 1: first upload
        q1 = np.random.randint(self.q1_range[0], self.q1_range[1])
        _, enc = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q1])
        img_bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)

        # Pass 2: re-share / re-upload
        q2 = np.random.randint(self.q2_range[0], self.q2_range[1])
        _, enc = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q2])
        img_bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)


# =============================================================================
# EFFICIENTNET-B4 VANILLA TRANSFORMS (380x380)
# =============================================================================

efficientnet_default_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'val': transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'test': transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
}


# =============================================================================
# EFFICIENTNET-B4 ENHANCED TRANSFORMS (Social Media-Hardened, 380x380)
# =============================================================================

efficientnet_enhanced_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((400, 400)),  # Slightly larger for random crop
        transforms.RandomCrop(380),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomApply([JPEGCompression(quality_range=(30, 95))], p=0.5),
        transforms.RandomApply([GaussianBlurPIL(sigma_range=(0.1, 2.0))], p=0.3),
        transforms.RandomApply([RandomDownscaleUpscale(scale_range=(0.5, 1.0))], p=0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ]),
    'val': transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'test': transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
}


# =============================================================================
# EFFICIENTNET-B4 HARDENED TRANSFORMS (Mandatory JPEG + Chained Compression)
# =============================================================================
#
# MOTIVATION: The enhanced transforms apply JPEG at p=0.5 (50% of samples).
# This means 50% of training images are clean/pristine, causing the model to
# rely on high-frequency pixel-level artifacts that are destroyed by social
# media compression. The hardened pipeline makes compression MANDATORY on
# every single training sample, forcing the model to learn features that
# survive multiple compression passes.
#
# Key differences from efficientnet_enhanced_data_transforms:
#   1. ChainedJPEGCompression ALWAYS applied (no RandomApply, no p=)
#   2. Stronger geometric augmentation (crop up to 15 degrees)
#   3. RandomApply kept for blur/downscale (still adds variety without always degrading)
#   4. More aggressive ColorJitter for real-world camera variance

efficientnet_hardened_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((400, 400)),           # Slightly larger for random crop
        transforms.RandomCrop(380),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        # MANDATORY: ChainedJPEGCompression on every sample (not p=0.5)
        # Simulates: generate -> share -> download -> re-share pipeline
        ChainedJPEGCompression(q1_range=(50, 90), q2_range=(40, 80)),
        transforms.RandomApply([GaussianBlurPIL(sigma_range=(0.1, 2.0))], p=0.4),
        transforms.RandomApply([RandomDownscaleUpscale(scale_range=(0.5, 1.0))], p=0.4),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ]),
    'val': transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'test': transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
}


# =============================================================================
# MOBILENETV3 VANILLA TRANSFORMS (224x224, ImageNet normalization)
# =============================================================================

# ImageNet mean/std (required for torchvision pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

mobilenet_default_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ]),
}


# =============================================================================
# MOBILENETV3 ENHANCED TRANSFORMS (Social Media-Hardened, 224x224)
# =============================================================================

mobilenet_enhanced_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((240, 240)),  # Slightly larger for random crop
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomApply([JPEGCompression(quality_range=(30, 95))], p=0.5),
        transforms.RandomApply([GaussianBlurPIL(sigma_range=(0.1, 2.0))], p=0.3),
        transforms.RandomApply([RandomDownscaleUpscale(scale_range=(0.5, 1.0))], p=0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ]),
}
