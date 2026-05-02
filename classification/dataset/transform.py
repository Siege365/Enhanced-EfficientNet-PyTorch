"""
Data Transforms - Model Augmentation Pipelines

Provides transform sets for:
  EfficientNet-B4 (380x380 native resolution):
    1. efficientnet_default_data_transforms: Vanilla transforms (baseline)
    2. efficientnet_enhanced_data_transforms: Social media-hardened augmentations
  MobileNetV3 (224x224 native resolution):
    3. mobilenet_default_data_transforms: Standard MobileNetV3 transforms
    4. mobilenet_enhanced_data_transforms: Social media-hardened augmentations

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
