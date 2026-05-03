"""
AI-Generated Image Detector — Professional Web Dashboard
A modern Streamlit GUI for detecting AI-generated media.

Supports: MobileNetV3-Small, EfficientNet-B4 (Vanilla)

Usage:
    streamlit run app.py

Author: Multi-Model Comparative Study Project
"""
import os, sys, time, tempfile, glob
import streamlit as st
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from PIL import Image
import numpy as np
import requests
from io import BytesIO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from network.models import model_selection
from dataset.transform import (
    efficientnet_default_data_transforms,
    mobilenet_default_data_transforms
)

# ─── Page Config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Forensic Image Detector",
    page_icon="🔍",
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
        background: radial-gradient(circle at 15% 50%, rgba(76, 29, 149, 0.15), transparent 25%),
                    radial-gradient(circle at 85% 30%, rgba(14, 165, 233, 0.15), transparent 25%);
        background-color: #0f1117; /* Slate dark */
    }

    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] {
        background: rgba(15, 17, 23, 0.6) !important;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Hero Text */
    .hero-title {
        background: linear-gradient(to right, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        letter-spacing: -1px;
        margin-bottom: 0.5rem;
        animation: fadeInDown 0.8s ease-out;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 2.5rem;
        animation: fadeInUp 0.8s ease-out;
    }

    /* Glass Cards with Hover Glow */
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 30px;
        backdrop-filter: blur(12px);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 20px;
    }
    .glass-card:hover {
        transform: translateY(-5px);
        border-color: rgba(56, 189, 248, 0.3);
        box-shadow: 0 10px 40px rgba(56, 189, 248, 0.1);
    }

    /* Result Cards */
    .result-real {
        background: linear-gradient(145deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.02));
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.1);
        animation: scaleIn 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .result-fake {
        background: linear-gradient(145deg, rgba(239, 68, 68, 0.1), rgba(239, 68, 68, 0.02));
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.1);
        animation: scaleIn 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .result-label {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: 1px;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    .result-confidence {
        font-size: 1.1rem;
        color: #cbd5e1;
        margin-top: 5px;
    }

    /* Metric Tiles */
    .metric-row {
        display: flex;
        gap: 15px;
        margin-top: 20px;
    }
    .metric-tile {
        flex: 1;
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 20px 10px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-tile:hover {
        background: rgba(30, 41, 59, 0.8);
        border-color: rgba(255, 255, 255, 0.1);
        transform: translateY(-2px);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(to right, #e2e8f0, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-name {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 8px;
        font-weight: 600;
    }

    /* Animations */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes scaleIn {
        from { opacity: 0; transform: scale(0.9); }
        to { opacity: 1; transform: scale(1); }
    }

    /* File Uploader Customization */
    [data-testid="stFileUploadDropzone"] {
        background-color: rgba(30, 41, 59, 0.3) !important;
        border: 2px dashed rgba(148, 163, 184, 0.3) !important;
        border-radius: 20px !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #38bdf8 !important;
        background-color: rgba(56, 189, 248, 0.05) !important;
    }
    
    /* Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px 8px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom-color: #38bdf8 !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Model Loading ───────────────────────────────────────────────────
@st.cache_resource
def load_model(model_name, weights_path, dropout=0.5):
    """Load a trained model from checkpoint."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, img_sz, *_ = model_selection(model_name, num_out_classes=2, dropout=dropout)
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, device


def auto_discover_models():
    """Scan the output directory for trained model checkpoints."""
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    found = {}
    for folder in sorted(os.listdir(output_dir)):
        best = os.path.join(output_dir, folder, 'best_model.pth')
        if os.path.isfile(best):
            # Parse model name from folder (e.g., "efficientnet_b4_20260501_151231")
            parts = folder.rsplit('_', 2)  # Split off timestamp
            if len(parts) >= 3:
                model_name = '_'.join(parts[:-2])
            else:
                model_name = parts[0]

            # Only show the two main models (skip CBAM/Spatial)
            if model_name in ('mobilenet_v3', 'efficientnet_b4'):
                display = {
                    'mobilenet_v3': '🟥 MobileNetV3-Small',
                    'efficientnet_b4': '🟩 EfficientNet-B4',
                }.get(model_name, model_name)
                found[display] = {
                    'model_name': model_name,
                    'weights': best,
                    'folder': folder
                }
    return found


def get_transform(model_name):
    """Get the correct preprocessing transform for a model."""
    if model_name == 'mobilenet_v3':
        return mobilenet_default_data_transforms['test']
    else:
        return efficientnet_default_data_transforms['test']


def predict_image(model, device, image_pil, transform):
    """Run inference on a single PIL image."""
    start = time.time()
    img_tensor = transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            output = model(img_tensor)
    probs = torch.softmax(output, dim=1)[0]
    elapsed = time.time() - start

    real_prob = probs[0].item()
    fake_prob = probs[1].item()
    prediction = 'REAL' if real_prob > fake_prob else 'AI-GENERATED'

    return {
        'prediction': prediction,
        'real_prob': real_prob,
        'fake_prob': fake_prob,
        'confidence': max(real_prob, fake_prob),
        'inference_ms': elapsed * 1000
    }


# ─── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔧 Configuration")
    st.markdown("---")

    models = auto_discover_models()

    if not models:
        st.error("No trained models found in the output/ directory!")
        st.stop()

    selected_display = st.selectbox(
        "Select Model",
        list(models.keys()),
        index=len(models) - 1  # Default to last (EfficientNet)
    )
    selected = models[selected_display]

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    param_counts = {
        'mobilenet_v3': '2,542,474',
        'efficientnet_b4': '19,344,370',
    }
    st.markdown(f"**Architecture:** `{selected['model_name']}`")
    st.markdown(f"**Parameters:** `{param_counts.get(selected['model_name'], 'N/A')}`")
    st.markdown(f"**Checkpoint:** `{selected['folder']}`")

    st.markdown("---")
    st.markdown("### ⚡ System")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        st.success(f"GPU: {gpu_name}")
    else:
        st.warning("CPU Mode (Slower)")


# ─── Main Content ────────────────────────────────────────────────────
st.markdown('<h1 class="hero-title"><span style="-webkit-text-fill-color: initial; -webkit-background-clip: initial; background: none;">🔍</span> AI Forensic Image Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Deep Learning Approach for Detecting AI-Generated Media on Social Media</p>',
            unsafe_allow_html=True)

# Load model
model, device = load_model(selected['model_name'], selected['weights'])
transform = get_transform(selected['model_name'])

# Tabs
tab_image, tab_batch, tab_about = st.tabs(["🖼️ Single Image", "📁 Batch Analysis", "ℹ️ About"])

# ─── Tab 1: Single Image ─────────────────────────────────────────────
with tab_image:
    uploaded = st.file_uploader(
        "Upload an image from your computer",
        type=['jpg', 'jpeg', 'png', 'webp', 'bmp'],
        key='single_upload'
    )
    
    st.markdown("<div style='text-align: center; margin: 10px 0; color: #94a3b8; font-weight: 600;'>— OR —</div>", unsafe_allow_html=True)
    
    image_url = st.text_input(
        "Paste an image URL from social media",
        placeholder="https://example.com/image.jpg",
        key='url_upload'
    )

    image = None
    if uploaded:
        try:
            image = Image.open(uploaded).convert('RGB')
        except Exception:
            st.error("Invalid image file.")
    elif image_url:
        try:
            response = requests.get(image_url, timeout=5)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
        except Exception as e:
            st.error("Could not load image from URL. Please ensure it is a direct link to an image file (.jpg, .png).")

    if image:
        result = predict_image(model, device, image, transform)

        col_img, col_result = st.columns([1, 1])

        with col_img:
            st.image(image, caption="Uploaded Image", width='stretch')

        with col_result:
            if result['prediction'] == 'REAL':
                st.markdown(f"""
                <div class="result-real">
                    <div class="result-label" style="color: #2ECC71;">✅ AUTHENTIC</div>
                    <div class="result-confidence" style="color: rgba(255,255,255,0.7);">
                        Confidence: {result['confidence']:.1%}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-fake">
                    <div class="result-label" style="color: #E74C3C;">⚠️ AI-GENERATED</div>
                    <div class="result-confidence" style="color: rgba(255,255,255,0.7);">
                        Confidence: {result['confidence']:.1%}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-tile">
                    <div class="metric-value">{result['real_prob']:.1%}</div>
                    <div class="metric-name">Real Probability</div>
                </div>
                <div class="metric-tile">
                    <div class="metric-value">{result['fake_prob']:.1%}</div>
                    <div class="metric-name">Fake Probability</div>
                </div>
                <div class="metric-tile">
                    <div class="metric-value">{result['inference_ms']:.0f}ms</div>
                    <div class="metric-name">Inference Time</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Progress bars
            st.markdown("#### Probability Distribution")
            st.progress(result['real_prob'], text=f"Real: {result['real_prob']:.2%}")
            st.progress(result['fake_prob'], text=f"AI-Generated: {result['fake_prob']:.2%}")


# ─── Tab 2: Batch Analysis ───────────────────────────────────────────
with tab_batch:
    uploaded_files = st.file_uploader(
        "Upload multiple images from your computer",
        type=['jpg', 'jpeg', 'png', 'webp', 'bmp'],
        accept_multiple_files=True,
        key='batch_upload'
    )

    st.markdown("<div style='text-align: center; margin: 10px 0; color: #94a3b8; font-weight: 600;'>— OR —</div>", unsafe_allow_html=True)

    batch_urls_text = st.text_area(
        "Paste multiple image URLs (one per line)",
        placeholder="https://example.com/image1.jpg\nhttps://example.com/image2.png",
        key='batch_url_upload',
        height=100
    )

    images_to_process = []
    
    if uploaded_files:
        for f in uploaded_files:
            try:
                img = Image.open(f).convert('RGB')
                images_to_process.append({'image': img, 'filename': f.name})
            except Exception:
                pass
                
    if batch_urls_text.strip():
        urls = [url.strip() for url in batch_urls_text.split('\\n') if url.strip()]
        for url in urls:
            try:
                response = requests.get(url, timeout=5)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content)).convert('RGB')
                filename = url.split('/')[-1]
                if len(filename) > 20: filename = filename[:17] + "..."
                images_to_process.append({'image': img, 'filename': filename})
            except Exception:
                st.warning(f"Could not load image from URL: {url}")

    if images_to_process:
        st.markdown(f"**Analyzing {len(images_to_process)} images...**")
        progress = st.progress(0)

        results = []
        for i, item in enumerate(images_to_process):
            res = predict_image(model, device, item['image'], transform)
            res['filename'] = item['filename']
            res['image'] = item['image']
            results.append(res)
            progress.progress((i + 1) / len(images_to_process))

        # Summary stats
        real_count = sum(1 for r in results if r['prediction'] == 'REAL')
        fake_count = len(results) - real_count
        avg_ms = np.mean([r['inference_ms'] for r in results])

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-tile">
                <div class="metric-value" style="color: #2ECC71;">{real_count}</div>
                <div class="metric-name">Authentic Images</div>
            </div>
            <div class="metric-tile">
                <div class="metric-value" style="color: #E74C3C;">{fake_count}</div>
                <div class="metric-name">AI-Generated Images</div>
            </div>
            <div class="metric-tile">
                <div class="metric-value">{avg_ms:.0f}ms</div>
                <div class="metric-name">Avg. Inference</div>
            </div>
            <div class="metric-tile">
                <div class="metric-value">{len(results)}</div>
                <div class="metric-name">Total Analyzed</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Grid display
        cols = st.columns(4)
        for i, res in enumerate(results):
            with cols[i % 4]:
                st.image(res['image'], width='stretch')
                if res['prediction'] == 'REAL':
                    st.success(f"✅ Real ({res['confidence']:.0%})")
                else:
                    st.error(f"⚠️ Fake ({res['confidence']:.0%})")
                st.caption(res['filename'])


# ─── Tab 3: About ────────────────────────────────────────────────────
with tab_about:
    st.markdown("""
    <div class="glass-card">
        <h2 style="color: #667eea;">About This Project</h2>
        <p style="color: rgba(255,255,255,0.7);">
            This application is the deployment interface for a comparative deep learning study
            on detecting AI-generated images in social media contexts. The research evaluates
            the architectural requirements for robust forensic detection by benchmarking a
            lightweight baseline (MobileNetV3) against a high-capacity model (EfficientNet-B4).
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #E74C3C;">🟥 MobileNetV3-Small (Baseline)</h3>
            <ul style="color: rgba(255,255,255,0.7);">
                <li><strong>Parameters:</strong> 2.5 Million</li>
                <li><strong>Input Resolution:</strong> 224×224</li>
                <li><strong>Architecture:</strong> Inverted Residuals + SE</li>
                <li><strong>AUC on Unseen Data:</strong> 0.7913</li>
                <li><strong>Verdict:</strong> Insufficient for forensic detection</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #2ECC71;">🟩 EfficientNet-B4 (Proposed)</h3>
            <ul style="color: rgba(255,255,255,0.7);">
                <li><strong>Parameters:</strong> 19.3 Million</li>
                <li><strong>Input Resolution:</strong> 380×380</li>
                <li><strong>Architecture:</strong> Compound-Scaled MBConv + SE</li>
                <li><strong>AUC on Unseen Data:</strong> 0.9808</li>
                <li><strong>Verdict:</strong> Superior forensic detection capability</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <h3 style="color: #E67E22;">📌 Key Research Finding</h3>
        <p style="color: rgba(255,255,255,0.7);">
            This study scientifically proves that <strong>model capacity matters</strong> for
            AI-generated media detection. While both architectures share the same MBConv foundation
            and Squeeze-and-Excitation attention, only the high-capacity EfficientNet-B4 (with its
            Compound Scaling of depth, width, and resolution) has sufficient feature extraction power
            to detect the subtle spatial artifacts left by modern generative AI models (Midjourney,
            Stable Diffusion, DALL-E).
        </p>
        <p style="color: rgba(255,255,255,0.7);">
            Additionally, experiments with external attention mechanisms (CBAM, Spatial-Only) on
            EfficientNet-B4 demonstrated <strong>catastrophic interference</strong> with the
            NAS-optimized architecture, reducing AUC from 0.98 to ~0.88. This finding cautions
            against arbitrary architectural modifications to compound-scaled models.
        </p>
    </div>
    """, unsafe_allow_html=True)
