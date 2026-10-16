"""
AI-Generated Media Forensic Detector — Web Dashboard
Spatiotemporal Deep Learning Framework for Detecting AI-Generated Images & Deepfake Videos.

Supports:
  - Image Models: EfficientNet-B4 (Joint Enhanced, Phase 1, Phase 2), MobileNetV3-Small, Ensemble
  - Video Models: EfficientNet-B4 + TSM + MHSA (Enhanced Model), Vanilla EfficientNet-B4 (Baseline Mean Pooling),
                  EfficientNet + Spatial ViT (Actual Baseline - Coccomini et al. 2022)

Usage:
    streamlit run app.py

Author: Multi-Model Comparative Study Project
"""
import os, sys, time, tempfile, glob, base64, zipfile
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from google import genai

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from network.models import model_selection
from network.video_models import video_model_selection
try:
    from network.vit_models import vit_model_selection
except ImportError:
    vit_model_selection = None
from dataset.transform import (
    efficientnet_default_data_transforms,
    mobilenet_default_data_transforms
)

# ─── Page Config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Forensic Media Detector — Images & Video",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

    /* Animated Background */
    .stApp {
        font-family: 'Outfit', sans-serif;
        background: radial-gradient(circle at 15% 50%, rgba(76, 29, 149, 0.18), transparent 25%),
                    radial-gradient(circle at 85% 30%, rgba(14, 165, 233, 0.18), transparent 25%),
                    radial-gradient(circle at 50% 80%, rgba(99, 102, 241, 0.12), transparent 30%);
        background-color: #0b0f19;
    }

    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] {
        background: rgba(11, 15, 25, 0.75) !important;
        backdrop-filter: blur(22px);
        -webkit-backdrop-filter: blur(22px);
        border-right: 1px solid rgba(255, 255, 255, 0.07);
    }

    /* Hero Text */
    .hero-title {
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.2rem;
        font-weight: 800;
        text-align: center;
        letter-spacing: -1px;
        margin-bottom: 0.3rem;
        animation: fadeInDown 0.8s ease-out;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        text-align: center;
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 2rem;
        animation: fadeInUp 0.8s ease-out;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 24px;
        backdrop-filter: blur(14px);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 20px;
    }
    .glass-card:hover {
        border-color: rgba(56, 189, 248, 0.35);
        box-shadow: 0 10px 40px rgba(56, 189, 248, 0.12);
    }

    /* Result Cards */
    .result-real {
        background: linear-gradient(145deg, rgba(16, 185, 129, 0.15), rgba(16, 185, 129, 0.03));
        border: 1.5px solid rgba(16, 185, 129, 0.45);
        border-radius: 20px;
        padding: 26px;
        text-align: center;
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.15);
        animation: scaleIn 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .result-fake {
        background: linear-gradient(145deg, rgba(239, 68, 68, 0.15), rgba(239, 68, 68, 0.03));
        border: 1.5px solid rgba(239, 68, 68, 0.45);
        border-radius: 20px;
        padding: 26px;
        text-align: center;
        box-shadow: 0 0 25px rgba(239, 68, 68, 0.15);
        animation: scaleIn 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .result-label {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: 1px;
    }
    .result-confidence {
        font-size: 1.05rem;
        color: #e2e8f0;
        margin-top: 5px;
    }

    /* Metric Tiles */
    .metric-row {
        display: flex;
        gap: 12px;
        margin-top: 18px;
        flex-wrap: wrap;
    }
    .metric-tile {
        flex: 1;
        min-width: 120px;
        background: rgba(30, 41, 59, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 16px 10px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-tile:hover {
        background: rgba(30, 41, 59, 0.85);
        border-color: rgba(56, 189, 248, 0.25);
        transform: translateY(-2px);
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 800;
        background: linear-gradient(to right, #f1f5f9, #cbd5e1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-name {
        font-size: 0.72rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1.3px;
        margin-top: 6px;
        font-weight: 600;
    }

    /* Frame Preview Card */
    .frame-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 6px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .frame-card:hover {
        transform: translateY(-3px);
        border-color: #38bdf8;
    }
    .frame-badge {
        font-size: 0.75rem;
        font-weight: 700;
        color: #38bdf8;
        background: rgba(56, 189, 248, 0.12);
        border-radius: 6px;
        padding: 2px 8px;
        display: inline-block;
        margin-top: 4px;
    }

    /* Category Badges */
    .badge-video {
        background: linear-gradient(135deg, rgba(168, 85, 247, 0.2), rgba(129, 140, 248, 0.2));
        border: 1px solid rgba(168, 85, 247, 0.5);
        color: #c084fc;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        display: inline-block;
    }
    .badge-image {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.2), rgba(59, 130, 246, 0.2));
        border: 1px solid rgba(56, 189, 248, 0.5);
        color: #38bdf8;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        display: inline-block;
    }

    /* XAI Card */
    .xai-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(129, 140, 248, 0.35);
        border-radius: 18px;
        padding: 22px 26px;
        margin-top: 20px;
        animation: fadeInUp 0.5s ease-out;
    }
    .xai-card h4 {
        background: linear-gradient(to right, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
        font-size: 1.15rem;
    }
    .xai-card p {
        color: #cbd5e1;
        line-height: 1.65;
        font-size: 0.95rem;
    }
    .xai-badge {
        display: inline-block;
        background: rgba(129, 140, 248, 0.15);
        border: 1px solid rgba(129, 140, 248, 0.35);
        border-radius: 8px;
        padding: 3px 10px;
        font-size: 0.72rem;
        color: #a5b4fc;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    /* Animations */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-16px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(16px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes scaleIn {
        from { opacity: 0; transform: scale(0.92); }
        to { opacity: 1; transform: scale(1); }
    }

    /* File Uploader Customization */
    [data-testid="stFileUploadDropzone"] {
        background-color: rgba(30, 41, 59, 0.35) !important;
        border: 2px dashed rgba(148, 163, 184, 0.3) !important;
        border-radius: 18px !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #38bdf8 !important;
        background-color: rgba(56, 189, 248, 0.06) !important;
    }
    
    /* Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        font-weight: 600;
        font-size: 0.95rem;
        background-color: transparent;
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom-color: #38bdf8 !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Model Discovery ─────────────────────────────────────────────────
def auto_discover_models():
    """
    Scans both classification/output/ (Image models) and
    classification/output_video/ (Video models), returning distinctly labeled
    dictionaries with full architectural metadata.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_img_dir = os.path.join(base_dir, 'output')
    output_vid_dir = os.path.join(base_dir, 'output_video')
    
    models = {}

    # 1. SCAN VIDEO MODELS (output_video/)
    # Distinct naming mapping for video experiments and checkpoints
    vid_mapping = {
        'exp4_results': {
            'label': '🎬 [VIDEO] EfficientNet-B4 + TSM + MHSA (Enhanced Model - Exp 4) ⭐',
            'architecture': 'enhanced',
            'default_thresh': 0.50,
            'desc': 'Enhanced Spatiotemporal Model (TSM + MHSA) trained on proposed 11 datasets. Peak video AUC.',
            'params': '43.3M',
            'frames': 8
        },
        'video_tsm_mhsa_20260920_190749': {
            'label': '🎬 [VIDEO] EfficientNet-B4 + TSM + MHSA (Enhanced Model - Checkpoint 20260920)',
            'architecture': 'enhanced',
            'default_thresh': 0.50,
            'desc': 'Enhanced Spatiotemporal Checkpoint (TSM + MHSA) on B4 backbone.',
            'params': '43.3M',
            'frames': 8
        },
        'exp3_results': {
            'label': '🎬 [VIDEO] Vanilla EfficientNet-B4 Mean Pooling (Baseline Model - Exp 3)',
            'architecture': 'baseline',
            'default_thresh': 0.50,
            'desc': 'Spatial-only EfficientNet-B4 with simple temporal average pooling (No TSM, No MHSA).',
            'params': '19.3M',
            'frames': 8
        },
        'video_tsm_mhsa_20260919_182904': {
            'label': '🎬 [VIDEO] Vanilla EfficientNet-B4 (Exp 1 Baseline Replication - Baseline Data)',
            'architecture': 'baseline',
            'default_thresh': 0.50,
            'desc': 'Spatial-only EfficientNet-B4 baseline trained on DFDC + FaceForensics++ (AUC: 0.9328).',
            'params': '19.3M',
            'frames': 8
        },
        'exp6_results': {
            'label': '🎬 [VIDEO] EfficientNet + Spatial ViT (Actual Baseline - Exp 6)',
            'architecture': 'actual_baseline',
            'default_thresh': 0.50,
            'desc': 'Direct replication of Coccomini et al. (2022) with EfficientNet-B0 + Spatial ViT Encoder.',
            'params': '24.9M',
            'frames': 8
        },
        'vit_spatial_20260921_095057': {
            'label': '🎬 [VIDEO] EfficientNet + Spatial ViT (Actual Baseline - Checkpoint 20260921)',
            'architecture': 'actual_baseline',
            'default_thresh': 0.50,
            'desc': 'Spatial ViT encoder replication of Coccomini et al. on video frames.',
            'params': '24.9M',
            'frames': 8
        }
    }

    if os.path.exists(output_vid_dir):
        for folder in sorted(os.listdir(output_vid_dir)):
            folder_path = os.path.join(output_vid_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            
            # Check for best_video_model.pth or best_model.pth
            cand_weights = os.path.join(folder_path, 'best_video_model.pth')
            if not os.path.isfile(cand_weights):
                cand_weights = os.path.join(folder_path, 'best_model.pth')
            if not os.path.isfile(cand_weights):
                continue

            info = vid_mapping.get(folder, None)
            if info:
                label = info['label']
                arch = info['architecture']
                thresh = info['default_thresh']
                desc = info['desc']
                params = info['params']
                frames = info['frames']
            else:
                # Dynamic inference of architecture
                arch = 'enhanced' if 'tsm' in folder.lower() or 'mhsa' in folder.lower() else 'baseline'
                label = f"🎬 [VIDEO] {folder} ({arch.upper()})"
                thresh = 0.50
                desc = f"Video checkpoint from {folder}"
                params = '43.3M' if arch == 'enhanced' else '19.3M'
                frames = 8

            models[label] = {
                'display_name': label,
                'modality': 'video',
                'model_name': 'video_model',
                'architecture': arch,
                'weights': cand_weights,
                'folder': folder,
                'default_thresh': thresh,
                'description': desc,
                'params': params,
                'frames': frames
            }

    # 2. SCAN IMAGE MODELS (output/)
    img_mapping = {
        'efficientnet_b4_20260903_144444': {
            'label': '🖼️ [IMAGE] EfficientNet-B4 (Joint Enhanced - Primary Image Model) ⭐',
            'model_name': 'efficientnet_b4',
            'default_thresh': 0.1967, # Youden's J on MIRAGE
            'desc': 'Primary image detector. Jointly trained on D1+D2+D3+D5+D6 with enhanced data augmentation.',
            'params': '19.3M'
        },
        'efficientnet_b4_20260501_151231': {
            'label': '🖼️ [IMAGE] EfficientNet-B4 (Phase 1 - D1-D3 50%)',
            'model_name': 'efficientnet_b4',
            'default_thresh': 0.50,
            'desc': 'Phase 1 baseline trained on 50% subset of D1-D3. Best raw lab-quality benchmark on D4.',
            'params': '19.3M'
        },
        'efficientnet_b4_20260723_230925': {
            'label': '🖼️ [IMAGE] EfficientNet-B4 (Phase 2 - Joint Early)',
            'model_name': 'efficientnet_b4',
            'default_thresh': 0.50,
            'desc': 'Early joint training checkpoint without enhanced augmentations.',
            'params': '19.3M'
        },
        'efficientnet_b4_20260904_204027': {
            'label': '🖼️ [IMAGE] EfficientNet-B4 (Phase 3 - 100% D1-D3 + JPEG)',
            'model_name': 'efficientnet_b4',
            'default_thresh': 0.2658,
            'desc': 'Trained on 100% D1-D3 with JPEG compression augmentation.',
            'params': '19.3M'
        },
        'efficientnet_b4_20260907_175728': {
            'label': '🖼️ [IMAGE] EfficientNet-B4 (Hardened Augmentation)',
            'model_name': 'efficientnet_b4',
            'default_thresh': 0.50,
            'desc': 'Trained with aggressive noise injection and compression hardening.',
            'params': '19.3M'
        },
        'efficientnet_b4_continuous_20260704_232952': {
            'label': '🖼️ [IMAGE] EfficientNet-B4 (Phase 2 - Continual Old)',
            'model_name': 'efficientnet_b4',
            'default_thresh': 0.50,
            'desc': 'Sequential continuous training checkpoint (historical reference).',
            'params': '19.3M'
        }
    }

    if os.path.exists(output_img_dir):
        for folder in sorted(os.listdir(output_img_dir)):
            best = os.path.join(output_img_dir, folder, 'best_model.pth')
            if os.path.isfile(best):
                info = img_mapping.get(folder, None)
                if info:
                    label = info['label']
                    m_name = info['model_name']
                    thresh = info['default_thresh']
                    desc = info['desc']
                    params = info['params']
                else:
                    m_name = 'mobilenet_v3' if 'mobilenet' in folder.lower() else 'efficientnet_b4'
                    label = f"🖼️ [IMAGE] {folder}"
                    thresh = 0.50
                    desc = f"Image checkpoint from {folder}"
                    params = '2.5M' if m_name == 'mobilenet_v3' else '19.3M'

                models[label] = {
                    'display_name': label,
                    'modality': 'image',
                    'model_name': m_name,
                    'architecture': 'image_backbone',
                    'weights': best,
                    'folder': folder,
                    'default_thresh': thresh,
                    'description': desc,
                    'params': params,
                    'frames': 1
                }

    # Add Image Ensemble if both required models exist
    p1_key = '🖼️ [IMAGE] EfficientNet-B4 (Phase 1 - D1-D3 50%)'
    p2_key = '🖼️ [IMAGE] EfficientNet-B4 (Joint Enhanced - Primary Image Model) ⭐'
    if p1_key in models and p2_key in models:
        models['🔥 [IMAGE] ENSEMBLE (Phase 1 + Joint Enhanced)'] = {
            'display_name': '🔥 [IMAGE] ENSEMBLE (Phase 1 + Joint Enhanced)',
            'modality': 'image',
            'model_name': 'ensemble',
            'architecture': 'dual_ensemble',
            'weights': None,
            'folder': 'ensemble',
            'default_thresh': 0.50,
            'description': 'Dual-model ensemble taking MAX(Fake_Prob) from Phase 1 (lab specialist) and Joint Enhanced (in-the-wild specialist).',
            'params': '38.7M (Combined)',
            'frames': 1,
            'phase1': models[p1_key],
            'phase2': models[p2_key]
        }

    return models


# ─── Model & Transform Loading ───────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_cached_model(modality, model_name, architecture, weights_path, dropout=0.5):
    """Loads a model from checkpoint and keeps it cached in memory."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if modality == 'video':
        if architecture == 'enhanced':
            model, img_sz, num_frames = video_model_selection(
                architecture='enhanced',
                num_frames=8,
                dropout=dropout
            )
            checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
            sd = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
            model.load_state_dict(sd, strict=False)
            del checkpoint
            model = model.to(device)
            model.eval()
            return model, device, img_sz, num_frames

        elif architecture == 'baseline':
            model, img_sz, num_frames = video_model_selection(
                architecture='baseline',
                num_frames=8,
                dropout=dropout
            )
            checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
            sd = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
            model.load_state_dict(sd, strict=False)
            del checkpoint
            model = model.to(device)
            model.eval()
            return model, device, img_sz, num_frames

        elif architecture in ['actual_baseline', 'efficient_vit', 'vit']:
            model, img_sz = vit_model_selection(dropout=dropout)
            checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
            sd = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
            model.load_state_dict(sd, strict=False)
            del checkpoint
            model = model.to(device)
            model.eval()
            return model, device, img_sz, 8

    # Image models
    model, img_sz, *_ = model_selection(model_name, num_out_classes=2, dropout=dropout)
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    sd = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    model.load_state_dict(sd)
    del checkpoint
    model = model.to(device)
    model.eval()
    return model, device, img_sz, 1


def get_model_transform(modality, model_name, architecture, img_size=380):
    """Returns the matching preprocessing pipeline for the given model."""
    if modality == 'video':
        if architecture in ['actual_baseline', 'efficient_vit', 'vit']:
            target_sz = 224
        else:
            target_sz = 380
        return transforms.Compose([
            transforms.Resize((target_sz, target_sz)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    else:
        if model_name == 'mobilenet_v3':
            return mobilenet_default_data_transforms['test']
        else:
            return efficientnet_default_data_transforms['test']


# ─── Video Frame Extraction ──────────────────────────────────────────
def extract_video_frames(video_input, num_frames=8):
    """
    Uniformly samples num_frames (T=8) equidistant across the video duration.
    Accepts: file path, Streamlit UploadedFile, or bytes.
    Returns:
        (frames_pil_list, metadata_dict)
    """
    temp_path = None
    try:
        if isinstance(video_input, str) and os.path.exists(video_input):
            video_path = video_input
        else:
            suffix = '.mp4'
            if hasattr(video_input, 'name') and video_input.name:
                _, ext = os.path.splitext(video_input.name)
                if ext: suffix = ext

            # Extract raw bytes for container verification
            data = None
            if hasattr(video_input, 'getvalue'):
                data = video_input.getvalue()
            elif hasattr(video_input, 'read'):
                if hasattr(video_input, 'seek'):
                    video_input.seek(0)
                data = video_input.read()
                if hasattr(video_input, 'seek'):
                    video_input.seek(0)
            elif isinstance(video_input, (bytes, bytearray)):
                data = bytes(video_input)

            if data is not None:
                head = data[:500].lower()
                if b'<html' in head or b'<!doctype' in head:
                    raise ValueError(
                        "The input data is a webpage (HTML) rather than a video file. "
                        "If you used a URL, please note that platforms like YouTube, TikTok, Reddit, "
                        "or Google Drive do not host raw MP4 files at page URLs. "
                        "Please download the video to your computer and upload it directly."
                    )
                if len(data) < 512:
                    raise ValueError("The provided file or stream is empty or too small to be a valid video container.")

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                if data is not None:
                    tmp.write(data)
                temp_path = tmp.name
                video_path = temp_path

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(
                "OpenCV could not open the video container. "
                "The video format or codec might be corrupted, unsupported, or incomplete."
            )

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or np.isnan(fps): fps = 30.0
        duration_sec = total_frames / fps if total_frames > 0 else 0.0

        if total_frames <= 0:
            # Fallback sequential read for non-indexed streams
            raw_frames = []
            while True:
                ret, frame = cap.read()
                if not ret: break
                raw_frames.append(frame)
                if len(raw_frames) >= 600: break # Safety limit
            total_frames = len(raw_frames)
            duration_sec = total_frames / fps
            if total_frames <= 0:
                cap.release()
                raise ValueError("Video contains no readable frames or is an unsupported format.")
            if total_frames >= num_frames:
                indices = [int(i * (total_frames / num_frames)) for i in range(num_frames)]
            else:
                indices = [i % total_frames for i in range(num_frames)]
            frames = [Image.fromarray(cv2.cvtColor(raw_frames[i], cv2.COLOR_BGR2RGB)) for i in indices]
        else:
            if total_frames >= num_frames:
                step = total_frames / num_frames
                indices = [min(int(i * step), total_frames - 1) for i in range(num_frames)]
            else:
                indices = [i % total_frames for i in range(num_frames)]

            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    if frames:
                        frames.append(frames[-1].copy())
                    else:
                        frames.append(Image.new('RGB', (380, 380), color='black'))
                else:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(rgb))

        cap.release()
        timestamps = [f"{(idx / fps):.2f}s" for idx in indices]

        metadata = {
            'total_frames': total_frames,
            'fps': round(fps, 2),
            'duration_sec': round(duration_sec, 2),
            'width': width,
            'height': height,
            'sampled_indices': indices,
            'timestamps': timestamps
        }
        return frames, metadata

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


# ─── Prediction Logic (Images & Video) ───────────────────────────────
def predict_image(model, device, image_pil, transform, threshold=0.5, modality='image', architecture=None):
    """Inference for a single image, with graceful handling if a video model is active."""
    if modality == 'video':
        # Replicate the single image across T=8 frames to run through the video spatiotemporal model
        frames_pil = [image_pil] * 8
        res = predict_video(model, device, frames_pil, transform, modality, architecture, threshold=threshold)
        return res

    if isinstance(model, tuple): # Image Ensemble
        m1, m2 = model
        res1 = predict_image(m1, device, image_pil, transform, threshold=threshold)
        res2 = predict_image(m2, device, image_pil, transform, threshold=threshold)
        final_fake_prob = max(res1['fake_prob'], res2['fake_prob'])
        final_real_prob = 1.0 - final_fake_prob
        return {
            'prediction': 'AI-GENERATED' if final_fake_prob >= threshold else 'REAL',
            'real_prob': final_real_prob,
            'fake_prob': final_fake_prob,
            'confidence': max(final_real_prob, final_fake_prob),
            'inference_ms': res1['inference_ms'] + res2['inference_ms']
        }

    start = time.time()
    img_tensor = transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu', enabled=(device.type == 'cuda')):
            output = model(img_tensor)
    probs = torch.softmax(output, dim=1)[0]
    elapsed = time.time() - start

    real_prob = probs[0].item()
    fake_prob = probs[1].item()
    prediction = 'AI-GENERATED' if fake_prob >= threshold else 'REAL'

    return {
        'prediction': prediction,
        'real_prob': real_prob,
        'fake_prob': fake_prob,
        'confidence': max(real_prob, fake_prob),
        'inference_ms': elapsed * 1000
    }


def predict_video(model, device, frames_pil, transform, modality, architecture, threshold=0.5):
    """
    Evaluates a video sequence represented by T=8 PIL frames.
    Works for Enhanced (TSM+MHSA), Baseline (Mean Pooling), ViT, or Image model fallback.
    """
    start = time.time()
    num_frames = len(frames_pil)
    tensor_list = [transform(f) for f in frames_pil]

    frame_probs = None
    with torch.no_grad():
        with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu', enabled=(device.type == 'cuda')):
            if modality == 'video':
                if architecture in ['enhanced', 'baseline']:
                    # Input shape: (1, T, 3, 380, 380)
                    seq_tensor = torch.stack(tensor_list, dim=0).unsqueeze(0).to(device)
                    output = model(seq_tensor)
                    probs = torch.softmax(output, dim=1)[0]
                    real_prob = probs[0].item()
                    fake_prob = probs[1].item()

                    # Check for auxiliary frame-level spatial logits if available
                    if architecture == 'enhanced' and hasattr(model, 'backbone') and hasattr(model, 'aux_classifier'):
                        try:
                            feat = model.backbone.extract_features(seq_tensor.view(num_frames, 3, 380, 380))
                            feat = model.backbone._avg_pooling(feat).flatten(start_dim=1)
                            aux_out = model.aux_classifier(feat)
                            aux_pr = torch.softmax(aux_out, dim=1)
                            frame_probs = aux_pr[:, 1].cpu().tolist()
                        except Exception:
                            frame_probs = None

                elif architecture in ['actual_baseline', 'efficient_vit', 'vit']:
                    # Input shape: (T, 3, 224, 224)
                    batch_tensor = torch.stack(tensor_list, dim=0).to(device)
                    output = model(batch_tensor)
                    pr = torch.softmax(output, dim=1)
                    frame_probs = pr[:, 1].cpu().tolist()
                    avg_pr = pr.mean(dim=0)
                    real_prob = avg_pr[0].item()
                    fake_prob = avg_pr[1].item()
                else:
                    seq_tensor = torch.stack(tensor_list, dim=0).unsqueeze(0).to(device)
                    output = model(seq_tensor)
                    probs = torch.softmax(output, dim=1)[0]
                    real_prob = probs[0].item()
                    fake_prob = probs[1].item()
            else:
                # Modality is image: Evaluate each of the 8 frames independently
                batch_tensor = torch.stack(tensor_list, dim=0).to(device)
                if isinstance(model, tuple): # Ensemble
                    m1, m2 = model
                    o1 = m1(batch_tensor)
                    o2 = m2(batch_tensor)
                    p1 = torch.softmax(o1, dim=1)[:, 1]
                    p2 = torch.softmax(o2, dim=1)[:, 1]
                    f_probs = torch.maximum(p1, p2)
                    fake_prob = f_probs.mean().item()
                    real_prob = 1.0 - fake_prob
                    frame_probs = f_probs.cpu().tolist()
                else:
                    output = model(batch_tensor)
                    pr = torch.softmax(output, dim=1)
                    frame_probs = pr[:, 1].cpu().tolist()
                    avg_pr = pr.mean(dim=0)
                    real_prob = avg_pr[0].item()
                    fake_prob = avg_pr[1].item()

    elapsed = time.time() - start
    prediction = 'AI-GENERATED' if fake_prob >= threshold else 'REAL'

    return {
        'prediction': prediction,
        'real_prob': real_prob,
        'fake_prob': fake_prob,
        'confidence': max(real_prob, fake_prob),
        'inference_ms': elapsed * 1000,
        'frame_probs': frame_probs
    }


# ─── Explainability Tools ────────────────────────────────────────────
def generate_universal_gradcam(model, device, image_pil, transform, model_name=None, pred_class_idx=1):
    """
    Universal Grad-CAM generator that works seamlessly across:
    1. Image Models: Vanilla EfficientNet-B4, MobileNetV3 (TransferModel)
    2. Video Models: EfficientNet-B4 + TSM + MHSA, Baseline Mean Pool (VideoEfficientNetB4, BaselineVideoEfficientNetB4)
    3. Image Ensembles: Runs Grad-CAM on the primary constituent model
    """
    try:
        with torch.enable_grad():
            # If ensemble tuple (m1, m2), use m1
            if isinstance(model, tuple):
                model = model[0]

            # Case 1: Image model wrapper (TransferModel -> model.model.backbone)
            if hasattr(model, 'model') and hasattr(model.model, 'backbone'):
                inner_model = model.model
                backbone = inner_model.backbone
                if model_name == 'mobilenet_v3':
                    target_layer = [backbone.features[-1]]
                else:
                    target_layer = [backbone._conv_head]

                for p in model.parameters():
                    p.requires_grad = True

                cam = GradCAM(model=model, target_layers=target_layer)
                img_tensor = transform(image_pil).unsqueeze(0).to(device)
                targets = [ClassifierOutputTarget(pred_class_idx)]
                grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0]

            # Case 2: Video model with backbone (VideoEfficientNetB4 -> model.backbone)
            elif hasattr(model, 'backbone') and hasattr(model.backbone, '_conv_head'):
                class VideoFrameSpatialWrapper(nn.Module):
                    def __init__(self, v_model):
                        super().__init__()
                        self.backbone = v_model.backbone
                        self.aux = getattr(v_model, 'aux_classifier', None)
                        if self.aux is None:
                            self.aux = getattr(v_model, 'classifier', None)
                        if self.aux is None:
                            self.aux = nn.Linear(1792, 2).to(device)

                    def forward(self, x):
                        feat = self.backbone.extract_features(x)
                        feat = self.backbone._avg_pooling(feat).flatten(start_dim=1)
                        return self.aux(feat)

                wrapper = VideoFrameSpatialWrapper(model).to(device)
                wrapper.eval()
                for p in wrapper.parameters():
                    p.requires_grad = True

                target_layer = [wrapper.backbone._conv_head]
                cam = GradCAM(model=wrapper, target_layers=target_layer)
                img_tensor = transform(image_pil).unsqueeze(0).to(device)
                targets = [ClassifierOutputTarget(pred_class_idx)]
                grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0]

            else:
                return None

            img_resized = image_pil.resize((img_tensor.shape[3], img_tensor.shape[2]))
            rgb_img = np.array(img_resized).astype(np.float32) / 255.0
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            return Image.fromarray(cam_image)
    except Exception as e:
        return None


def generate_gradcam(model, device, image_pil, transform, model_name=None, pred_class_idx=1):
    """Wrapper for backward compatibility."""
    return generate_universal_gradcam(model, device, image_pil, transform, model_name, pred_class_idx)


def generate_video_frame_gradcam(model, device, frame_pil, transform, pred_class_idx=1):
    """Wrapper for backward compatibility."""
    return generate_universal_gradcam(model, device, frame_pil, transform, None, pred_class_idx)


def get_gemini_api_key():
    """Safely retrieves GEMINI_API_KEY from environment or Streamlit secrets without crashing."""
    k = os.environ.get('GEMINI_API_KEY', '')
    if not k:
        try:
            k = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            k = ""
    return k


def generate_gemini_explanation(image_or_frames, prediction, confidence, model_label, is_video=False):
    """Generates AI Forensic Report using Gemini API."""
    api_key = get_gemini_api_key()
    if not api_key:
        return None

    try:
        client = genai.Client(api_key=api_key)

        if is_video:
            # Create a 4x2 montage grid of the 8 keyframes to send as single visual context
            frames = image_or_frames[:8]
            w, h = 320, 180
            grid = Image.new('RGB', (w * 4, h * 2))
            for i, f in enumerate(frames):
                resized = f.resize((w, h))
                x_pos = (i % 4) * w
                y_pos = (i // 4) * h
                grid.paste(resized, (x_pos, y_pos))

            buf = BytesIO()
            grid.save(buf, format='JPEG', quality=85)
            img_bytes = buf.getvalue()

            prompt = f"""You are a world-class AI video forensic analyst. A spatiotemporal deep learning model ({model_label}) analyzed this 8-frame sequence and classified the video as **{prediction}** with **{confidence:.1%} confidence**.

Provide a clear, authoritative, yet accessible forensic breakdown in 3 short sections:

**Forensic Verdict:**
In 1-2 sentences, plainly state whether this video is an authentic real recording or an AI-generated/face-swap deepfake.

**Temporal & Spatial Inconsistencies:**
In 2-3 sentences, detail specific visual cues observed across the sequence (e.g., facial boundary jitter, unnatural motion blur, blinking anomalies, temporal flickering, or synthetic texture smoothing).

**Reliability Assessment:**
In 1 sentence, explain whether the {confidence:.1%} confidence indicates a definitive or borderline classification.

Keep the total response under 150 words. Write in clean paragraphs under each header."""

        else:
            buf = BytesIO()
            image_or_frames.save(buf, format='JPEG', quality=85)
            img_bytes = buf.getvalue()

            prompt = f"""You are an expert AI forensic analyst. A deep learning model ({model_label}) analyzed this image and classified it as **{prediction}** with **{confidence:.1%} confidence**.

Structure your concise forensic breakdown:

**What does this mean?**
In 1-2 simple sentences, explain if this image is a real camera photograph or AI-generated.

**Why does the model think so?**
In 2-3 sentences, point out specific visual artifacts (e.g., unnatural skin smoothness, anomalous lighting, warped geometry, or authentic photographic grain).

**How reliable is this?**
In 1 sentence, comment on the reliability of this {confidence:.1%} confidence score.

Keep the total response under 150 words without bullet points."""

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                genai.types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg'),
                prompt
            ]
        )
        return response.text
    except Exception as e:
        return f"ERROR: Forensic explanation could not be generated ({str(e)})"


# ─── Sidebar Configuration ───────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Model Configuration")
    st.markdown("---")

    all_models = auto_discover_models()

    if not all_models:
        st.error("No model checkpoints found in classification/output/ or classification/output_video/!")
        st.stop()

    # 1. Modality Filter
    modality_filter = st.radio(
        "Filter by Media Type",
        ["🌐 All Models", "🎬 Video Models", "🖼️ Image Models"],
        horizontal=True,
        index=0
    )

    if modality_filter == "🎬 Video Models":
        filtered_models = {k: v for k, v in all_models.items() if v['modality'] == 'video'}
    elif modality_filter == "🖼️ Image Models":
        filtered_models = {k: v for k, v in all_models.items() if v['modality'] == 'image'}
    else:
        filtered_models = all_models

    if not filtered_models:
        st.warning("No models found matching this filter.")
        filtered_models = all_models

    # 2. Model Selection Dropdown
    model_keys = list(filtered_models.keys())
    # Prefer Exp 4 or Joint Enhanced as default if available
    default_idx = 0
    for idx, k in enumerate(model_keys):
        if 'Exp 4' in k or 'Joint Enhanced' in k:
            default_idx = idx
            break

    selected_label = st.selectbox(
        "Select Active Model",
        model_keys,
        index=default_idx
    )
    active_model_info = filtered_models[selected_label]

    st.markdown("---")
    
    # 3. Dynamic Threshold Slider
    st.markdown("### 🎛️ Sensitivity Threshold")
    st.caption("Adjust decision boundary for Fake / AI-Generated classification.")
    
    thresh_default = active_model_info.get('default_thresh', 0.50)
    step_val = 0.01
    val = round(float(thresh_default), 2)
    val = max(0.0, min(1.0, val))
    
    ui_threshold = st.slider(
        "Decision Boundary (Fake %)",
        min_value=0.0,
        max_value=1.0,
        value=val,
        step=step_val,
        format="%.2f",
        help="Scores at or above this threshold will be flagged as AI-Generated / Deepfake."
    )

    st.markdown("---")

    # 4. Detailed Model Information Card
    st.markdown("### 📊 Model Architecture Details")
    
    if active_model_info['modality'] == 'video':
        st.markdown(f'<span class="badge-video">🎬 Video (Spatiotemporal)</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="badge-image">🖼️ Image (Spatial)</span>', unsafe_allow_html=True)

    st.markdown(f"<p style='margin-top: 8px; font-size: 0.9rem; color: #cbd5e1;'>{active_model_info['description']}</p>", unsafe_allow_html=True)
    
    st.markdown(f"**Parameters:** `{active_model_info['params']}`")
    st.markdown(f"**Temporal Input:** `{active_model_info.get('frames', 1)} Frames`")
    st.markdown(f"**Checkpoint:** `{active_model_info['folder']}`")

    st.markdown("---")
    st.markdown("### ⚡ Hardware Accelerator")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        st.success(f"GPU: {gpu_name}")
    else:
        st.info("CPU Active (Inference running on Host CPU)")


# ─── Model Helper (On-Demand Loading) ─────────────────────────────────
modality = active_model_info['modality']
arch = active_model_info.get('architecture')
model_name = active_model_info.get('model_name')
model_img_size = 224 if (arch in ['actual_baseline', 'efficient_vit', 'vit'] or model_name == 'mobilenet_v3') else 380
active_transform = get_model_transform(modality, model_name, arch, img_size=model_img_size)

def get_loaded_model():
    with st.spinner(f"Preparing {active_model_info['display_name']}..."):
        if model_name == 'ensemble':
            m1, dev1, sz1, _ = load_cached_model('image', 'efficientnet_b4', None, active_model_info['phase1']['weights'])
            m2, dev2, sz2, _ = load_cached_model('image', 'efficientnet_b4', None, active_model_info['phase2']['weights'])
            return (m1, m2), dev1, sz1
        else:
            m, dev, sz, _ = load_cached_model(
                modality, model_name, arch, active_model_info['weights']
            )
            return m, dev, sz


# ─── Hero Banner ─────────────────────────────────────────────────────
st.markdown('<h1 class="hero-title">🛡️ AI Forensic Media Detector</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">Unified Spatiotemporal Deep Learning Framework for Detecting AI-Generated Images & Deepfake Videos</p>',
    unsafe_allow_html=True
)

# ─── Application Tabs ────────────────────────────────────────────────
tab_video, tab_image, tab_batch, tab_about = st.tabs([
    "🎬 Video Analysis",
    "🖼️ Single Image",
    "📁 Batch Image Analysis",
    "ℹ️ Thesis Findings & About"
])


# ═════════════════════════════════════════════════════════════════════
# TAB 1: VIDEO ANALYSIS
# ═════════════════════════════════════════════════════════════════════
with tab_video:
    if modality != 'video':
        st.info(
            f"💡 **Note:** You currently have an Image model selected (`{active_model_info['display_name']}`). "
            "For full spatiotemporal evaluation with Temporal Shift (TSM) and Multi-Head Self-Attention (MHSA), "
            "select a **Video Model** from the sidebar. (Running now will evaluate the 8 sampled frames with your image model)."
        )

    vid_input_mode = st.radio(
        "Video Input Source",
        ["📁 Upload Video File", "🎞️ Upload Frames Archive (ZIP)", "🔗 Direct Video URL"],
        horizontal=True,
        key='vid_mode'
    )

    video_bytes = None
    frames_extracted = None
    meta = None

    if vid_input_mode == "📁 Upload Video File":
        uploaded_video = st.file_uploader(
            "Upload a video file to evaluate",
            type=['mp4', 'mov', 'avi', 'mkv', 'webm'],
            key='video_uploader'
        )
        if uploaded_video:
            video_bytes = uploaded_video.getvalue()
            with st.spinner("Extracting 8 equidistant temporal keyframes using OpenCV..."):
                try:
                    frames_extracted, meta = extract_video_frames(uploaded_video, num_frames=8)
                except Exception as e:
                    st.error(f"Error parsing video: {e}")

    elif vid_input_mode == "🎞️ Upload Frames Archive (ZIP)":
        uploaded_zip = st.file_uploader(
            "Upload a ZIP archive containing extracted video frame images (.jpg, .png)",
            type=['zip'],
            key='zip_uploader'
        )
        if uploaded_zip:
            with st.spinner("Unpacking frame sequence from archive..."):
                try:
                    with zipfile.ZipFile(uploaded_zip, 'r') as z:
                        namelist = sorted([n for n in z.namelist() if n.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
                        if len(namelist) < 2:
                            st.error("ZIP must contain at least 2 image frames.")
                        else:
                            # Sample 8 equidistant
                            total_f = len(namelist)
                            if total_f >= 8:
                                step = total_f / 8.0
                                idxs = [int(i * step) for i in range(8)]
                            else:
                                idxs = [i % total_f for i in range(8)]
                            frames_extracted = [Image.open(BytesIO(z.read(namelist[i]))).convert('RGB') for i in idxs]
                            meta = {
                                'total_frames': total_f,
                                'fps': 30.0,
                                'duration_sec': round(total_f / 30.0, 2),
                                'width': frames_extracted[0].width,
                                'height': frames_extracted[0].height,
                                'sampled_indices': idxs,
                                'timestamps': [f"Frame {idx}" for idx in idxs]
                            }
                except Exception as e:
                    st.error(f"Could not read ZIP archive: {e}")

    else:
        video_url = st.text_input(
            "Paste video link (YouTube, Shorts, TikTok, Instagram, Facebook, or direct .mp4)",
            placeholder="https://www.youtube.com/shorts/... or https://www.tiktok.com/@... or direct .mp4",
            key='video_url_input',
            help="Supports YouTube, Shorts, TikTok, Instagram Reels, Facebook Watch, Twitter/X, Reddit, and direct .mp4/.webm links."
        )
        if video_url and video_url.strip():
            url_clean = video_url.strip()
            with st.spinner("Fetching and extracting video stream..."):
                try:
                    # Detect video platforms (YouTube, TikTok, Instagram, Facebook, etc.)
                    is_social_or_platform = any(domain in url_clean.lower() for domain in [
                        'youtube.com', 'youtu.be', 'tiktok.com', 'instagram.com', 
                        'twitter.com', 'x.com', 'reddit.com', 'vimeo.com', 
                        'facebook.com', 'fb.watch', 'threads.net', 'twitch.tv', 'dailymotion.com'
                    ]) or not any(url_clean.lower().split('?')[0].endswith(ext) for ext in ['.mp4', '.webm', '.avi', '.mov', '.mkv'])

                    if is_social_or_platform:
                        import yt_dlp
                        with tempfile.TemporaryDirectory() as tmpdir:
                            outpath = os.path.join(tmpdir, 'video.%(ext)s')
                            ydl_opts = {
                                'format': 'bestvideo[ext=mp4]/bestvideo/best[ext=mp4]/best',
                                'outtmpl': outpath,
                                'quiet': True,
                                'no_warnings': True,
                                'max_filesize': 100 * 1024 * 1024,
                            }
                            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                                ydl.download([url_clean])

                            downloaded_files = os.listdir(tmpdir)
                            if not downloaded_files:
                                raise ValueError("Could not extract streamable video from the provided link. Please ensure the link is public.")

                            fpath = os.path.join(tmpdir, downloaded_files[0])
                            with open(fpath, 'rb') as vf:
                                video_bytes = vf.read()
                            frames_extracted, meta = extract_video_frames(fpath, num_frames=8)
                    else:
                        # Direct HTTP link
                        headers = {
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                            'Accept': '*/*'
                        }
                        resp = requests.get(url_clean, headers=headers, timeout=25, stream=True)
                        resp.raise_for_status()

                        content_type = resp.headers.get('content-type', '').lower()
                        if 'text/html' in content_type:
                            import yt_dlp
                            with tempfile.TemporaryDirectory() as tmpdir:
                                outpath = os.path.join(tmpdir, 'video.mp4')
                                ydl_opts = {
                                    'format': 'bestvideo[ext=mp4]/bestvideo/best',
                                    'outtmpl': outpath,
                                    'quiet': True,
                                    'no_warnings': True,
                                }
                                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                                    ydl.download([url_clean])

                                downloaded_files = os.listdir(tmpdir)
                                if not downloaded_files:
                                    raise ValueError("The URL returned a webpage (HTML) rather than a direct raw video file.")

                                fpath = os.path.join(tmpdir, downloaded_files[0])
                                with open(fpath, 'rb') as vf:
                                    video_bytes = vf.read()
                                frames_extracted, meta = extract_video_frames(fpath, num_frames=8)
                        else:
                            video_bytes = resp.content
                            frames_extracted, meta = extract_video_frames(video_bytes, num_frames=8)
                except Exception as e:
                    st.error(f"Failed to retrieve video from URL: {e}")

    # Display video analysis if frames extracted
    if frames_extracted and meta:
        col_vid_player, col_vid_meta = st.columns([1.2, 1])

        with col_vid_player:
            st.markdown("#### 📺 Video Playback")
            if video_bytes:
                st.video(video_bytes)
            else:
                st.image(frames_extracted[0], caption="Frame 0 (Keyframe Preview)", use_container_width=True)

        with col_vid_meta:
            st.markdown("#### 📋 Video Metadata")
            st.markdown(f"""
            <div class="glass-card">
                <p style="margin: 4px 0;"><strong>Resolution:</strong> {meta['width']} × {meta['height']} px</p>
                <p style="margin: 4px 0;"><strong>Frame Rate:</strong> {meta['fps']} FPS</p>
                <p style="margin: 4px 0;"><strong>Total Frames:</strong> {meta['total_frames']:,} frames</p>
                <p style="margin: 4px 0;"><strong>Duration:</strong> {meta['duration_sec']} seconds</p>
                <p style="margin: 4px 0;"><strong>Sampling Mode:</strong> Equidistant temporal sampling (T=8)</p>
                <p style="margin: 4px 0;"><strong>Active Evaluator:</strong> {active_model_info['display_name']}</p>
            </div>
            """, unsafe_allow_html=True)

        # Run Video Inference
        active_model, active_device, _ = get_loaded_model()
        with st.spinner("Running spatiotemporal feature extraction and temporal aggregation..."):
            vid_result = predict_video(
                active_model,
                active_device,
                frames_extracted,
                active_transform,
                modality,
                arch,
                threshold=ui_threshold
            )

        # Result Banner
        if vid_result['prediction'] == 'REAL':
            st.markdown(f"""
            <div class="result-real">
                <div class="result-label" style="color: #10b981;">✅ AUTHENTIC VIDEO</div>
                <div class="result-confidence">
                    Confidence: {vid_result['confidence']:.1%} | Temporal Decision Threshold: {ui_threshold:.4f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-fake">
                <div class="result-label" style="color: #ef4444;">⚠️ AI-GENERATED / DEEPFAKE VIDEO</div>
                <div class="result-confidence">
                    Confidence: {vid_result['confidence']:.1%} | Temporal Decision Threshold: {ui_threshold:.4f}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Metrics Row
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-tile">
                <div class="metric-value">{vid_result['real_prob']:.1%}</div>
                <div class="metric-name">Authentic Prob</div>
            </div>
            <div class="metric-tile">
                <div class="metric-value">{vid_result['fake_prob']:.1%}</div>
                <div class="metric-name">AI / Deepfake Prob</div>
            </div>
            <div class="metric-tile">
                <div class="metric-value">{vid_result['inference_ms']:.0f}ms</div>
                <div class="metric-name">Latency (T=8)</div>
            </div>
            <div class="metric-tile">
                <div class="metric-value">8</div>
                <div class="metric-name">Sampled Frames</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Probability distribution bars
        st.markdown("#### Sequence Probability Distribution")
        st.progress(vid_result['real_prob'], text=f"Authentic Real Probability: {vid_result['real_prob']:.2%}")
        st.progress(vid_result['fake_prob'], text=f"AI-Generated / Deepfake Probability: {vid_result['fake_prob']:.2%}")

        # 8-Frame Temporal Gallery
        st.markdown("---")
        st.markdown("### 🎞️ Sampled Temporal Keyframes ($T=8$)")
        st.caption("Equidistant frames sampled across the video timeline and passed through the temporal model:")

        cols_frames = st.columns(8)
        for i, frame in enumerate(frames_extracted):
            with cols_frames[i]:
                st.image(frame, use_container_width=True)
                timestamp = meta['timestamps'][i] if i < len(meta['timestamps']) else f"F{i+1}"
                if vid_result['frame_probs'] and i < len(vid_result['frame_probs']):
                    f_fake = vid_result['frame_probs'][i]
                    if f_fake >= ui_threshold:
                        badge_label = f"Fake: {f_fake:.0%}"
                        badge_color = "#ef4444"
                    else:
                        badge_label = f"Real: {(1.0 - f_fake):.0%}"
                        badge_color = "#10b981"
                    st.markdown(f"""
                    <div class="frame-card">
                        <span class="frame-badge">{timestamp}</span>
                        <div style="font-size: 0.72rem; color: {badge_color}; margin-top: 3px; font-weight: 700;">
                            {badge_label}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="frame-card">
                        <span class="frame-badge">{timestamp}</span>
                    </div>
                    """, unsafe_allow_html=True)

        # ─── Video Grad-CAM Explainability Section ───────────────────
        st.markdown("---")
        st.markdown("### 🔬 Spatiotemporal Explainability: Keyframe Grad-CAM Heatmaps")
        st.caption("Inspect spatial activation heatmaps for individual video frames to see where the convolutional backbone detects manipulation artifacts:")

        # Automatically identify the most suspicious frame (highest fake probability)
        most_suspicious_idx = 0
        if vid_result.get('frame_probs'):
            most_suspicious_idx = int(np.argmax(vid_result['frame_probs']))

        frame_options = []
        for i in range(len(frames_extracted)):
            t_str = meta['timestamps'][i] if i < len(meta['timestamps']) else f"F{i+1}"
            if vid_result.get('frame_probs') and i < len(vid_result['frame_probs']):
                fp = vid_result['frame_probs'][i]
                status_str = f"Fake: {fp:.0%}" if fp >= ui_threshold else f"Real: {(1.0 - fp):.0%}"
            else:
                status_str = "Frame"
            star = " ⭐ [Most Suspicious]" if i == most_suspicious_idx and vid_result.get('frame_probs') else ""
            frame_options.append(f"Frame {i+1} ({t_str}) — {status_str}{star}")

        selected_frame_str = st.selectbox(
            "🎯 Select a Keyframe to Inspect with Grad-CAM:",
            options=frame_options,
            index=most_suspicious_idx,
            key="video_cam_frame_selector",
            help="Defaults to the most suspicious frame with highest fake risk. Select any keyframe to inspect its spatial features."
        )
        selected_idx = frame_options.index(selected_frame_str)
        selected_frame_pil = frames_extracted[selected_idx]
        selected_timestamp = meta['timestamps'][selected_idx] if selected_idx < len(meta['timestamps']) else f"Frame {selected_idx+1}"

        col_orig_frame, col_cam_frame = st.columns([1, 1])
        with col_orig_frame:
            st.image(selected_frame_pil, caption=f"Original Keyframe ({selected_timestamp})", use_container_width=True)

        with col_cam_frame:
            with st.spinner(f"Generating Grad-CAM heatmap for {selected_timestamp}..."):
                # Target class: if overall video is AI/Fake, highlight fake class (1), otherwise real class (0)
                target_class_idx = 1 if vid_result['prediction'] != 'REAL' else 0
                frame_cam = generate_video_frame_gradcam(
                    active_model,
                    active_device,
                    selected_frame_pil,
                    active_transform,
                    pred_class_idx=target_class_idx
                )
            if frame_cam:
                st.image(frame_cam, caption=f"Grad-CAM Heatmap ({selected_timestamp})", use_container_width=True)
            else:
                st.info("Grad-CAM generation is not supported for this video model architecture.")

        # Gemini Video Explanation
        st.markdown("---")
        has_gemini = bool(get_gemini_api_key())
        if has_gemini:
            if st.button("🧠 Generate Forensic Video Report (Gemini AI)", key='gemini_vid_btn'):
                with st.spinner("Analyzing temporal inconsistencies and generating forensic report..."):
                    explanation = generate_gemini_explanation(
                        frames_extracted,
                        vid_result['prediction'],
                        vid_result['confidence'],
                        active_model_info['display_name'],
                        is_video=True
                    )
                if explanation:
                    import re
                    exp_html = explanation
                    exp_html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', exp_html)
                    exp_html = exp_html.replace('\n\n', '</p><p>').replace('\n', '<br>')
                    st.markdown(f"""
                    <div class="xai-card">
                        <span class="xai-badge">🧠 Multimodal Spatiotemporal Forensic Report</span>
                        <h4>Forensic Analysis Breakdown</h4>
                        <p>{exp_html}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="xai-card" style="border-color: rgba(148, 163, 184, 0.2);">
                <span class="xai-badge" style="color: #94a3b8;">🔒 AI Video Forensic Report Locked</span>
                <p style="color: #64748b; font-size: 0.85rem;">Set your <code>GEMINI_API_KEY</code> environment variable to unlock natural language spatiotemporal forensic reporting powered by Gemini.</p>
            </div>
            """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════
# TAB 2: SINGLE IMAGE
# ═════════════════════════════════════════════════════════════════════
with tab_image:
    if modality == 'video':
        st.info(
            f"💡 **Notice:** A Video model (`{active_model_info['display_name']}`) is currently selected. "
            "Evaluating a single image will replicate it across time ($T=8$) to run inference through the spatiotemporal architecture."
        )

    img_input_mode = st.radio(
        "Image Input Mode",
        ["📁 Upload File", "🔗 Paste Image URL"],
        horizontal=True,
        key='img_mode'
    )

    image_pil = None
    if img_input_mode == "📁 Upload File":
        uploaded_img = st.file_uploader(
            "Upload an image (.jpg, .png, .webp)",
            type=['jpg', 'jpeg', 'png', 'webp', 'bmp'],
            key='single_img_upload'
        )
        if uploaded_img:
            try:
                image_pil = Image.open(uploaded_img).convert('RGB')
            except Exception:
                st.error("Invalid image format.")
    else:
        url_input = st.text_input(
            "Paste direct image URL",
            placeholder="https://example.com/image.jpg",
            key='single_img_url'
        )
        if url_input:
            try:
                resp = requests.get(url_input, timeout=6)
                resp.raise_for_status()
                image_pil = Image.open(BytesIO(resp.content)).convert('RGB')
            except Exception:
                st.error("Could not load image from the provided URL.")

    if image_pil:
        active_model, active_device, _ = get_loaded_model()
        result = predict_image(
            active_model,
            active_device,
            image_pil,
            active_transform,
            threshold=ui_threshold,
            modality=modality,
            architecture=arch
        )

        col_img, col_cam = st.columns([1, 1])

        with col_img:
            st.image(image_pil, caption="Original Input Media", use_container_width=True)

        with col_cam:
            pred_idx = 1 if result['prediction'] != 'REAL' else 0
            with st.spinner("Generating Grad-CAM attention heatmap..."):
                heatmap = generate_universal_gradcam(active_model, active_device, image_pil, active_transform, model_name, pred_idx)
            if heatmap:
                st.image(heatmap, caption="Grad-CAM Spatial Activation Heatmap", use_container_width=True)
            else:
                st.markdown(f"""
                <div class="glass-card" style="height: 100%; display: flex; flex-direction: column; justify-content: center; text-align: center;">
                    <h4>Model: {active_model_info['display_name']}</h4>
                    <p style="color: #94a3b8; font-size: 0.9rem;">
                        Spatial Grad-CAM is not supported for this model architecture.
                    </p>
                </div>
                """, unsafe_allow_html=True)

        # Result card
        if result['prediction'] == 'REAL':
            st.markdown(f"""
            <div class="result-real">
                <div class="result-label" style="color: #10b981;">✅ AUTHENTIC MEDIA</div>
                <div class="result-confidence">
                    Confidence: {result['confidence']:.1%} | Sensitivity Threshold: {ui_threshold:.4f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-fake">
                <div class="result-label" style="color: #ef4444;">⚠️ AI-GENERATED MEDIA</div>
                <div class="result-confidence">
                    Confidence: {result['confidence']:.1%} | Sensitivity Threshold: {ui_threshold:.4f}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Metric Tiles
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-tile">
                <div class="metric-value">{result['real_prob']:.1%}</div>
                <div class="metric-name">Authentic Prob</div>
            </div>
            <div class="metric-tile">
                <div class="metric-value">{result['fake_prob']:.1%}</div>
                <div class="metric-name">AI-Generated Prob</div>
            </div>
            <div class="metric-tile">
                <div class="metric-value">{result['inference_ms']:.0f}ms</div>
                <div class="metric-name">Inference Latency</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Progress Bars
        st.markdown("#### Probability Distribution")
        st.progress(result['real_prob'], text=f"Real Probability: {result['real_prob']:.2%}")
        st.progress(result['fake_prob'], text=f"AI-Generated Probability: {result['fake_prob']:.2%}")

        # Gemini Text Explanation
        has_gemini = bool(get_gemini_api_key())
        if has_gemini:
            if st.button("🧠 Generate Forensic Image Explanation", key='gemini_img_btn'):
                with st.spinner("Analyzing forensic artifacts..."):
                    explanation = generate_gemini_explanation(
                        image_pil,
                        result['prediction'],
                        result['confidence'],
                        active_model_info['display_name'],
                        is_video=False
                    )
                if explanation:
                    import re
                    exp_html = explanation
                    exp_html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', exp_html)
                    exp_html = exp_html.replace('\n\n', '</p><p>').replace('\n', '<br>')
                    st.markdown(f"""
                    <div class="xai-card">
                        <span class="xai-badge">🧠 Multimodal Forensic Analysis</span>
                        <h4>Why did the model classify this as {result['prediction']}?</h4>
                        <p>{exp_html}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="xai-card" style="border-color: rgba(148, 163, 184, 0.2);">
                <span class="xai-badge" style="color: #94a3b8;">🔒 AI Forensic Explanation Locked</span>
                <p style="color: #64748b; font-size: 0.85rem;">Set your <code>GEMINI_API_KEY</code> environment variable to enable explainable text reports.</p>
            </div>
            """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════
# TAB 3: BATCH IMAGE ANALYSIS
# ═════════════════════════════════════════════════════════════════════
with tab_batch:
    batch_mode = st.radio(
        "Batch Input Mode",
        ["📁 Upload Multiple Files", "🔗 Paste URLs List"],
        horizontal=True,
        key='batch_radio'
    )

    batch_items = []
    if batch_mode == "📁 Upload Multiple Files":
        uploaded_batch = st.file_uploader(
            "Upload multiple images",
            type=['jpg', 'jpeg', 'png', 'webp', 'bmp'],
            accept_multiple_files=True,
            key='batch_file_upload'
        )
        if uploaded_batch:
            for f in uploaded_batch:
                try:
                    img = Image.open(f).convert('RGB')
                    batch_items.append({'image': img, 'name': f.name})
                except Exception:
                    pass
    else:
        urls_text = st.text_area(
            "Paste direct image URLs (one per line)",
            placeholder="https://example.com/img1.jpg\nhttps://example.com/img2.png",
            height=120,
            key='batch_urls_textarea'
        )
        if urls_text.strip():
            urls = [u.strip() for u in urls_text.split('\n') if u.strip()]
            for u in urls:
                try:
                    r = requests.get(u, timeout=5)
                    r.raise_for_status()
                    img = Image.open(BytesIO(r.content)).convert('RGB')
                    fname = u.split('/')[-1]
                    if len(fname) > 22: fname = fname[:19] + "..."
                    batch_items.append({'image': img, 'name': fname})
                except Exception:
                    st.warning(f"Could not load: {u}")

    if batch_items:
        active_model, active_device, _ = get_loaded_model()
        st.markdown(f"**Analyzing {len(batch_items)} images with `{active_model_info['display_name']}`...**")
        prog_bar = st.progress(0)
        batch_results = []

        for i, item in enumerate(batch_items):
            res = predict_image(
                active_model,
                active_device,
                item['image'],
                active_transform,
                threshold=ui_threshold,
                modality=modality,
                architecture=arch
            )
            res['name'] = item['name']
            res['image'] = item['image']
            batch_results.append(res)
            prog_bar.progress((i + 1) / len(batch_items))

        real_count = sum(1 for r in batch_results if r['prediction'] == 'REAL')
        fake_count = len(batch_results) - real_count
        avg_ms = np.mean([r['inference_ms'] for r in batch_results])

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-tile">
                <div class="metric-value" style="color: #10b981;">{real_count}</div>
                <div class="metric-name">Authentic</div>
            </div>
            <div class="metric-tile">
                <div class="metric-value" style="color: #ef4444;">{fake_count}</div>
                <div class="metric-name">AI-Generated</div>
            </div>
            <div class="metric-tile">
                <div class="metric-value">{avg_ms:.0f}ms</div>
                <div class="metric-name">Avg Latency</div>
            </div>
            <div class="metric-tile">
                <div class="metric-value">{len(batch_results)}</div>
                <div class="metric-name">Total Evaluated</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        grid_cols = st.columns(4)
        for i, res in enumerate(batch_results):
            with grid_cols[i % 4]:
                st.image(res['image'], use_container_width=True)
                if res['prediction'] == 'REAL':
                    st.success(f"✅ Real ({res['confidence']:.0%})")
                else:
                    st.error(f"⚠️ Fake ({res['confidence']:.0%})")
                st.caption(res['name'])


# ═════════════════════════════════════════════════════════════════════
# TAB 4: THESIS FINDINGS & ABOUT
# ═════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
    <div class="glass-card">
        <h2 style="color: #818cf8; margin-bottom: 8px;">Thesis Research Project: Forensic Synthetic Media Detection</h2>
        <p style="color: #cbd5e1; line-height: 1.7;">
            This web dashboard demonstrates the comparative evaluation and deployment of deep learning architectures
            for detecting synthetic media (AI-generated images and deepfake videos) in social media contexts.
            The study rigorously benchmarks spatial-only baseline backbones against our proposed spatiotemporal framework.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #38bdf8;">🎬 Phase 3: Video Detection Framework</h3>
            <ul style="color: #cbd5e1; line-height: 1.8;">
                <li><strong>Proposed Enhanced Model:</strong> EfficientNet-B4 + Temporal Shift Module (TSM, T=8) + Multi-Head Self-Attention (MHSA, D=1792).</li>
                <li><strong>Temporal Aggregation:</strong> Zero-parameter channel shifting injects temporal awareness into MBConv blocks; MHSA captures long-range temporal dependencies.</li>
                <li><strong>Baseline Video Model:</strong> EfficientNet-B4 with naive temporal mean pooling (No TSM, No MHSA).</li>
                <li><strong>Actual Baseline:</strong> Replication of Coccomini et al. (2022) with EfficientNet-B0 + Spatial ViT Encoder.</li>
                <li><strong>Key Discovery:</strong> Enhanced TSM+MHSA achieves <strong>0.9700 Val AUC</strong>, significantly outperforming Vanilla Baseline (0.9328 Val AUC).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #a855f7;">🖼️ Phase 1 & 2: Image Detection Framework</h3>
            <ul style="color: #cbd5e1; line-height: 1.8;">
                <li><strong>Primary Image Model:</strong> Vanilla EfficientNet-B4 Joint Training (20260903).</li>
                <li><strong>Multi-Corpus Joint Training:</strong> Solved catastrophic forgetting by training across D1–D6 simultaneously.</li>
                <li><strong>In-the-Wild Benchmark (MIRAGE D7):</strong> Reached 0.7817 AUC, dramatically improving over Phase 1 baseline (0.6767 AUC).</li>
                <li><strong>Youden's J Optimization:</strong> Corrected class imbalance threshold to 0.1967, yielding +7.0% Accuracy and +6.1% F1 without retraining.</li>
                <li><strong>Attention Architecture Finding:</strong> External CBAM and Spatial attention degraded performance, proving vanilla EfficientNet compound scaling is optimal.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <h3 style="color: #fbbf24;">⚖️ Youden's J Statistic (Threshold Optimization)</h3>
        <p style="color: #cbd5e1; line-height: 1.7;">
            In social media environments, synthetic media datasets exhibit severe class imbalance (e.g., 1:8 real-to-fake in MIRAGE D7).
            A standard fixed 0.5 decision threshold causes classifiers to miss subtle generative artifacts.
            By calculating the Youden's J statistic:
        </p>
        <p style="text-align: center; font-size: 1.15rem; color: #38bdf8; font-weight: 700; margin: 12px 0;">
            J = Sensitivity + Specificity - 1
        </p>
        <p style="color: #cbd5e1; line-height: 1.7;">
            The optimal cut-off maximizes the vertical distance on the ROC curve, restoring balanced sensitivity and specificity without retraining model weights.
        </p>
    </div>
    """, unsafe_allow_html=True)
