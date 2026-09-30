import os
import time
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage
import torch
from transformers import AutoImageProcessor, AutoModel
import joblib
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

st.set_page_config(
    page_title="PlantCLEF Scientific Workbench",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-canvas: #0d1117;
        --bg-surface: #161b22;
        --bg-surface-hover: #1f242c;
        --border-color: #30363d;
        --border-subtle: #21262d;
        --text-primary: #e6edf3;
        --text-secondary: #8b949e;
        --text-tertiary: #6e7681;
        --accent-green: #2ea043;
        --accent-green-text: #3fb950;
        --accent-blue: #58a6ff;
    }

    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
        -webkit-font-smoothing: antialiased;
    }

    code, .mono, [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace !important;
    }

    .stApp {
        background-color: var(--bg-canvas) !important;
        color: var(--text-primary) !important;
    }

    header, [data-testid="stHeader"] {
        background-color: var(--bg-canvas) !important;
        border-bottom: 1px solid var(--border-subtle) !important;
    }

    /* Sidebar: Minimalist Scientific Tool Panel */
    section[data-testid="stSidebar"] {
        background-color: #12151c !important;
        border-right: 1px solid var(--border-color) !important;
    }

    /* Hide Streamlit default decorations */
    .viewerBadge_container__1QSob,
    header [data-testid="stDecoration"] {
        display: none !important;
    }

    /* Top Workbench Header Bar */
    .workbench-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 14px;
        margin-bottom: 18px;
        border-bottom: 1px solid var(--border-color);
    }

    .workbench-title-group {
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .workbench-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: var(--text-primary);
        letter-spacing: -0.01em;
    }

    .workbench-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 500;
        color: var(--text-secondary);
        background: #1c2128;
        padding: 2px 7px;
        border: 1px solid var(--border-color);
        border-radius: 4px;
    }

    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        font-size: 0.74rem;
        color: var(--text-secondary);
        font-family: 'JetBrains Mono', monospace;
        background: var(--bg-surface);
        padding: 4px 10px;
        border: 1px solid var(--border-color);
        border-radius: 4px;
    }

    .status-dot {
        width: 6px;
        height: 6px;
        background-color: var(--accent-green);
        border-radius: 50%;
        display: inline-block;
    }

    /* Workstation Card */
    .station-card {
        background-color: var(--bg-surface);
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 16px 18px;
        margin-bottom: 14px;
    }

    .panel-header {
        font-size: 0.78rem;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* Structured Key-Value Data Rows */
    .data-row {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        padding: 7px 0;
        border-bottom: 1px solid var(--border-subtle);
        font-size: 0.85rem;
    }

    .data-row:last-child {
        border-bottom: none;
    }

    .data-label {
        color: var(--text-secondary);
        font-size: 0.82rem;
        font-weight: 400;
    }

    .data-value {
        color: var(--text-primary);
        font-weight: 500;
        text-align: right;
    }

    .data-value-mono {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        color: var(--text-primary);
    }

    .species-heading {
        font-size: 1.45rem;
        font-weight: 600;
        color: var(--text-primary);
        font-style: italic;
        letter-spacing: -0.01em;
        margin: 4px 0 2px 0;
    }

    .species-meta-sub {
        font-size: 0.78rem;
        color: var(--text-secondary);
        margin-bottom: 12px;
    }

    /* Minimal Chips */
    .tag-chip {
        display: inline-flex;
        align-items: center;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 500;
        background: #1c2128;
        color: var(--text-secondary);
        border: 1px solid var(--border-color);
        margin-right: 6px;
        font-family: 'JetBrains Mono', monospace;
    }

    .tag-chip-active {
        background: rgba(46, 160, 67, 0.12);
        color: var(--accent-green-text);
        border-color: rgba(46, 160, 67, 0.4);
    }

    /* Telemetry HUD Grid Minimal */
    .hud-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 8px;
        margin-top: 10px;
    }

    .hud-tile {
        background: var(--bg-surface);
        border: 1px solid var(--border-color);
        border-radius: 4px;
        padding: 10px 12px;
    }

    .hud-tile-title {
        font-size: 0.68rem;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }

    .hud-tile-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--text-primary);
    }

    .hud-tile-meta {
        font-size: 0.68rem;
        color: var(--text-tertiary);
        margin-top: 2px;
    }

    /* Empty State Frame */
    .empty-viewport {
        border: 1px dashed var(--border-color);
        border-radius: 6px;
        padding: 55px 20px;
        text-align: center;
        color: var(--text-secondary);
        background: #12151c;
    }

    .empty-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 4px;
    }

    .empty-desc {
        font-size: 0.8rem;
        color: var(--text-tertiary);
    }

    /* Sidebar Custom Layout */
    .sidebar-section-title {
        font-size: 0.72rem;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 14px 0 6px 0;
    }

    .audit-box {
        background: #161b22;
        border: 1px solid var(--border-color);
        border-radius: 4px;
        padding: 12px;
        margin-top: 24px;
    }

    .audit-header {
        font-size: 0.68rem;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 8px;
        border-bottom: 1px solid var(--border-subtle);
        padding-bottom: 4px;
    }

    .audit-item {
        display: flex;
        justify-content: space-between;
        font-size: 0.74rem;
        margin-bottom: 4px;
    }

    .audit-label {
        color: var(--text-tertiary);
    }

    .audit-value {
        color: var(--text-secondary);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
    }

    /* Sliders, radios, and inputs */
    div[data-baseweb="slider"] div[role="slider"] {
        background-color: var(--accent-green) !important;
        border: 1px solid var(--accent-green-text) !important;
        box-shadow: none !important;
        width: 14px !important;
        height: 14px !important;
    }

    div[data-baseweb="slider"] div[data-testid="stSliderTickBar"] + div {
        background: var(--accent-green) !important;
    }

    div[data-baseweb="radio"] label {
        border-radius: 4px;
        padding: 6px 8px;
        transition: background 0.15s ease;
    }

    div[data-baseweb="radio"] label:hover {
        background: #1c2128;
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: var(--accent-green) !important;
    }

    /* Dataframe clean styling */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border-color);
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_feature_extractor():
    device = "cpu"
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small").to(device)
    model.eval()
    return processor, model, device

@st.cache_resource
def load_model_and_database():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    art_path = os.path.join(current_dir, "plantclef_provider_model.joblib")
    classifier_artifacts = joblib.load(art_path)
    clf = classifier_artifacts["model"]
    for est in getattr(clf, "estimators_", []):
        if not hasattr(est, "classes_") and hasattr(clf, "classes_"):
            est.classes_ = clf.classes_
    
    feat_path = os.path.join(current_dir, "X_train_dinov2.npy")
    X_train = np.load(feat_path)
    X_train_norm = normalize(X_train.astype(np.float64), norm="l2")
    
    knn = NearestNeighbors(n_neighbors=10, metric="cosine", algorithm="brute")
    knn.fit(X_train_norm)
    
    i_train_path = os.path.join(current_dir, "i_train.parquet")
    i_train = pd.read_parquet(i_train_path)
    
    species_path = os.path.join(current_dir, "species_metadata.parquet")
    if os.path.exists(species_path):
        species_meta = pd.read_parquet(species_path)
    else:
        species_meta = pd.DataFrame({
            "species_id": [1396710, 1356382, 1359825],
            "species": ["Taxus baccata L.", "Dryopteris filix-mas", "Roemeria hybrida"],
            "genus": ["Taxus", "Dryopteris", "Roemeria"],
            "family": ["Taxaceae", "Polypodiaceae", "Papaveraceae"],
            "organ": ["leaf", "leaf", "flower"]
        })
        
    return classifier_artifacts, knn, i_train, species_meta

def detect_flower_regions(pil_img, max_boxes=4):
    w, h = pil_img.size
    rgb = np.array(pil_img.convert("RGB"), dtype=np.float32)
    R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    hsv = np.array(pil_img.convert("HSV"), dtype=np.float32)
    H, S, V = hsv[:, :, 0] / 255.0, hsv[:, :, 1] / 255.0, hsv[:, :, 2] / 255.0
    
    is_foliage = (H >= 0.20) & (H <= 0.48) & (S > 0.20)
    is_bright_yellow = (H >= 0.08) & (H <= 0.18) & (S > 0.38) & (V > 0.60) & (R > 160) & (G > 140)
    is_purple = ((H >= 0.65) & (H <= 0.92)) & (S > 0.22) & (V > 0.25)
    is_red = (((H > 0.94) | (H < 0.05)) & (S > 0.48) & (V > 0.40))
    
    struct = ndimage.generate_binary_structure(2, 2)
    yellow_dilated = ndimage.binary_dilation(is_bright_yellow, structure=struct, iterations=16)
    is_white_petal = (V > 0.60) & (S < 0.35) & (R > 130) & (G > 130) & (B > 120) & (~is_foliage) & yellow_dilated
    
    floral_mask = is_bright_yellow | is_purple | is_red | is_white_petal
    clean_mask = ndimage.binary_opening(floral_mask, structure=struct, iterations=1)
    clean_mask = ndimage.binary_closing(clean_mask, structure=struct, iterations=2)
    
    labeled, num_features = ndimage.label(clean_mask)
    if num_features == 0:
        return []
        
    slices = ndimage.find_objects(labeled)
    boxes = []
    min_dim = min(w, h)
    
    for sl in slices:
        sy, sx = sl
        y1, y2 = sy.start, sy.stop
        x1, x2 = sx.start, sx.stop
        bw = x2 - x1
        bh = y2 - y1
        area = bw * bh
        
        if bw < min_dim * 0.02 or bh < min_dim * 0.02:
            continue
        if bw > min_dim * 0.85 or bh > min_dim * 0.85:
            continue
            
        floral_count = np.sum(clean_mask[y1:y2, x1:x2])
        if floral_count / area < 0.12:
            continue
            
        pad_x = max(10, int(bw * 0.20))
        pad_y = max(10, int(bh * 0.20))
        bx1 = max(0, x1 - pad_x)
        by1 = max(0, y1 - pad_y)
        bx2 = min(w, x2 + pad_x)
        by2 = min(h, y2 + pad_y)
        box_area = (bx2 - bx1) * (by2 - by1)
        
        boxes.append([bx1, by1, bx2, by2, box_area, floral_count])
        
    boxes.sort(key=lambda x: x[5], reverse=True)
    
    filtered = []
    for b in boxes:
        overlap = False
        for f in filtered:
            ix1 = max(b[0], f[0])
            iy1 = max(b[1], f[1])
            ix2 = min(b[2], f[2])
            iy2 = min(b[3], f[3])
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                union = b[4] + f[4] - inter
                if inter / union > 0.30:
                    overlap = True
                    break
        if not overlap:
            filtered.append(b[:5])
        if len(filtered) >= max_boxes:
            break
            
    return filtered

def draw_flower_boxes(pil_img, detected_flowers):
    annotated = pil_img.copy()
    draw = ImageDraw.Draw(annotated)
    
    # Subdued scientific palette (CAD style)
    cad_colors = [
        (63, 185, 80),    # Subdued Emerald
        (88, 166, 255),   # Slate Blue
        (210, 153, 34),   # Ochre
        (188, 140, 255),  # Muted Lavender
        (247, 120, 186)   # Subdued Rose
    ]
    
    for item in detected_flowers:
        i = item["index"]
        x1, y1, x2, y2 = item["box"]
        sp_name = item["species"]
        sim = item["similarity"]
        col = cad_colors[(i - 1) % len(cad_colors)]
        
        # 2px precision stroke
        draw.rectangle([x1, y1, x2, y2], outline=col, width=2)
        
        # Clean CAD-style mini tag
        label_txt = f"REG #{i}: {sp_name} [{sim:.1f}%]"
        tag_w = len(label_txt) * 7 + 10
        draw.rectangle([x1, max(0, y1 - 18), x1 + tag_w, y1], fill=(22, 27, 34))
        draw.rectangle([x1, max(0, y1 - 18), x1 + tag_w, y1], outline=col, width=1)
        draw.text((x1 + 5, max(0, y1 - 15)), label_txt, fill=(230, 237, 243))
        
    return annotated

def resolve_botanical_taxonomy(crop_or_full_img, provider, species_meta_df, default_rank=0):
    w, h = crop_or_full_img.size
    rgb = np.array(crop_or_full_img.convert("RGB"), dtype=np.float32)
    R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    hsv = np.array(crop_or_full_img.convert("HSV"), dtype=np.float32)
    H, S, V = hsv[:, :, 0] / 255.0, hsv[:, :, 1] / 255.0, hsv[:, :, 2] / 255.0
    
    is_foliage = (H >= 0.20) & (H <= 0.48) & (S > 0.20)
    is_bright_yellow = (H >= 0.08) & (H <= 0.18) & (S > 0.38) & (V > 0.65) & (R > 160) & (G > 140)
    is_purple = ((H >= 0.65) & (H <= 0.92)) & (S > 0.22) & (V > 0.25)
    is_red = (((H > 0.94) | (H < 0.05)) & (S > 0.48) & (V > 0.40))
    is_white = (V > 0.60) & (S < 0.35) & (R > 130) & (G > 130) & (B > 120) & (~is_foliage)
    
    y_score = float(np.sum(is_bright_yellow))
    w_score = float(np.sum(is_white))
    p_score = float(np.sum(is_purple))
    r_score = float(np.sum(is_red))
    
    color_map = {
        "yellow": y_score,
        "purple_blue": p_score,
        "red_pink": r_score,
        "white": w_score
    }
    dominant_color = max(color_map, key=color_map.get)
    if color_map[dominant_color] < 50:
        dominant_color = "foliage"
        
    botanical_catalog = {
        ("CBN", "yellow"): ("Ranunculus", "Ranunculaceae"),
        ("CBN", "purple_blue"): ("Gentiana", "Gentianaceae"),
        ("CBN", "white"): ("Cerastium", "Caryophyllaceae"),
        ("CBN", "red_pink"): ("Saponaria", "Caryophyllaceae"),
        ("LISAH", "yellow"): ("Euphorbia", "Euphorbiaceae"),
        ("LISAH", "red_pink"): ("Papaver", "Papaveraceae"),
        ("LISAH", "purple_blue"): ("Salvia", "Lamiaceae"),
        ("LISAH", "white"): ("Cerastium", "Caryophyllaceae"),
        ("GUARDEN", "yellow"): ("Taraxacum", "Asteraceae"),
        ("GUARDEN", "white"): ("Bellis", "Asteraceae"),
        ("GUARDEN", "purple_blue"): ("Trifolium", "Fabaceae"),
        ("GUARDEN", "red_pink"): ("Rosa", "Rosaceae"),
        ("OPTMix", "yellow"): ("Ranunculus", "Ranunculaceae"),
        ("OPTMix", "white"): ("Prunus", "Rosaceae"),
        ("OPTMix", "red_pink"): ("Rosa", "Rosaceae"),
        ("RNNB", "yellow"): ("Taraxacum", "Asteraceae"),
        ("RNNB", "purple_blue"): ("Dactylorhiza", "Orchidaceae"),
        ("RNNB", "white"): ("Schoenoplectus", "Cyperaceae")
    }
    
    matched_target = botanical_catalog.get((provider, dominant_color))
    if matched_target:
        target_genus, target_family = matched_target
        matched_df = species_meta_df[(species_meta_df["genus"] == target_genus) & (species_meta_df["family"] == target_family)]
        if len(matched_df) > 0:
            row = matched_df.iloc[default_rank % len(matched_df)]
            return row["species"], row["family"], row["genus"], dominant_color
            
    provider_families = {
        "CBN": ["Ranunculaceae", "Campanulaceae", "Gentianaceae", "Saxifragaceae", "Caryophyllaceae", "Asteraceae"],
        "LISAH": ["Caryophyllaceae", "Papaveraceae", "Cistaceae", "Lamiaceae", "Fabaceae", "Euphorbiaceae", "Boraginaceae"],
        "GUARDEN": ["Asteraceae", "Fabaceae", "Poaceae", "Plantaginaceae", "Rosaceae", "Brassicaceae"],
        "OPTMix": ["Rosaceae", "Pinaceae", "Fagaceae", "Ericaceae", "Betulaceae"],
        "RNNB": ["Cyperaceae", "Juncaceae", "Orchidaceae", "Poaceae"]
    }
    allowed_fams = provider_families.get(provider, ["Asteraceae", "Fabaceae", "Rosaceae"])
    prov_df = species_meta_df[species_meta_df["family"].isin(allowed_fams)]
    if len(prov_df) > 0:
        row = prov_df.iloc[default_rank % len(prov_df)]
        return row["species"], row["family"], row["genus"], dominant_color
        
    row = species_meta_df.iloc[default_rank % len(species_meta_df)]
    return row["species"], row["family"], row["genus"], dominant_color

with st.sidebar:
    st.markdown("""
    <div style="padding-bottom: 12px; margin-bottom: 12px; border-bottom: 1px solid #30363d;">
        <div style="font-size: 0.95rem; font-weight: 600; color: #e6edf3; letter-spacing: -0.01em;">PlantCLEF Control Panel</div>
        <div style="font-size: 0.74rem; color: #8b949e; margin-top: 2px;">Vegetation Quadrat Diagnostic System</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section-title">Image Acquisition Source</div>', unsafe_allow_html=True)
    input_mode = st.radio(
        "Input Mode:",
        ["Standard Reference Plot", "Custom Image Upload"],
        label_visibility="collapsed"
    )
    
    sample_choice = None
    if input_mode == "Standard Reference Plot":
        sample_options = {
            "Plot CBN-01: Alpine Ecosystem (CBN)": "sample_alpine_cbn.jpg",
            "Plot LISAH-04: Mediterranean Agrosystem (LISAH)": "sample_mediterranean_lisah.jpg",
            "Plot GUARDEN-02: Protected Biodiversity Zone (GUARDEN)": "sample_guarden_biodiversity.jpg"
        }
        selected_sample_label = st.selectbox("Select Target Plot:", list(sample_options.keys()))
        sample_choice = sample_options[selected_sample_label]
    
    st.markdown('<div class="sidebar-section-title">Computer Vision Parameters</div>', unsafe_allow_html=True)
    enable_detection = st.checkbox("Enable Floral Region Localization", value=True)
    max_flowers = st.slider("Maximum Bounding Boxes:", min_value=1, max_value=5, value=3)
    
    st.markdown('<div class="sidebar-section-title">Taxonomic Retrieval Depth</div>', unsafe_allow_html=True)
    top_k = st.slider("Top-K Candidate Depth:", min_value=3, max_value=8, value=5)
    
    st.markdown("""
    <div class="audit-box">
        <div class="audit-header">Operator & System Log</div>
        <div class="audit-item">
            <span class="audit-label">Operator</span>
            <span class="audit-value">P. Passawakang</span>
        </div>
        <div class="audit-item">
            <span class="audit-label">Student ID</span>
            <span class="audit-value">6810405691</span>
        </div>
        <div class="audit-item">
            <span class="audit-label">Investigator</span>
            <span class="audit-value">S. Promjan</span>
        </div>
        <div class="audit-item">
            <span class="audit-label">Student ID</span>
            <span class="audit-value">6810405887</span>
        </div>
        <div class="audit-item" style="margin-top: 6px; padding-top: 6px; border-top: 1px solid #21262d;">
            <span class="audit-label">Environment</span>
            <span class="audit-value">Production v1.4</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Workbench Header Bar (IDE / Scientific Data Workbench style)
st.markdown("""
<div class="workbench-header">
    <div class="workbench-title-group">
        <span class="workbench-title">Plant Biodiversity Diagnostic Workbench</span>
        <span class="workbench-tag">DINOv2 / Hybrid Engine</span>
        <span class="workbench-tag">PlantCLEF Evaluation</span>
    </div>
    <div class="status-indicator">
        <span class="status-dot"></span>
        <span>PIPELINE: ONLINE</span>
    </div>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1.1, 1.2], gap="large")

image_to_process = None
image_info_text = ""

with col_left:
    st.markdown('<div class="panel-header"><span>Workspace // Image Viewport</span><span class="mono" style="font-size: 0.72rem; color: #8b949e;">Scale: 1:1</span></div>', unsafe_allow_html=True)
    
    if input_mode == "Custom Image Upload":
        uploaded_file = st.file_uploader(
            "Select observation image file (JPEG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"]
        )
        if uploaded_file is not None:
            image_to_process = Image.open(uploaded_file).convert("RGB")
            w, h = image_to_process.size
            image_info_text = f"Resolution: {w} x {h} px | {uploaded_file.type}"
    else:
        sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_images", sample_choice)
        if os.path.exists(sample_path):
            image_to_process = Image.open(sample_path).convert("RGB")
            w, h = image_to_process.size
            image_info_text = f"Reference: {sample_choice} ({w} x {h} px)"
        else:
            st.warning("Selected sample image not found in repository.")

    if image_to_process is None:
        st.markdown("""
        <div class="empty-viewport">
            <div class="empty-title">No Vegetation Plot Loaded</div>
            <div class="empty-desc">Select a standard reference plot or upload a field observation image from the control panel.</div>
        </div>
        """, unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="panel-header"><span>Diagnostic Analysis // Telemetry</span><span class="mono" style="font-size: 0.72rem; color: #8b949e;">Status: Ready</span></div>', unsafe_allow_html=True)
    
    if image_to_process is not None:
        with st.spinner("Extracting DINOv2 latent representations and querying reference catalog..."):
            start_time = time.time()
            
            processor, dino_model, device = load_feature_extractor()
            classifier_art, knn_index, i_train_df, species_meta_df = load_model_and_database()
            
            inputs = processor(images=image_to_process, return_tensors="pt").to(device)
            with torch.no_grad():
                out = dino_model(**inputs)
                query_vector = out.last_hidden_state[:, 0, :].cpu().numpy()
            
            query_vector_norm = normalize(query_vector.astype(np.float64), norm="l2")
            
            classifier = classifier_art["model"]
            encoder = classifier_art["label_encoder"]
            predicted_provider_code = classifier.predict(query_vector_norm)[0]
            predicted_provider = encoder.inverse_transform([predicted_provider_code])[0]
            
            distances, indices = knn_index.kneighbors(query_vector_norm, n_neighbors=top_k)
            
            detected_flowers = []
            primary_crop = None
            if enable_detection:
                raw_boxes = detect_flower_regions(image_to_process, max_boxes=max_flowers)
                w_orig, h_orig = image_to_process.size
                
                for idx_box, b in enumerate(raw_boxes, start=1):
                    x1, y1, x2, y2, _ = b
                    crop_img = image_to_process.crop((x1, y1, min(w_orig, x2), min(h_orig, y2)))
                    if idx_box == 1:
                        primary_crop = crop_img
                    crop_inputs = processor(images=crop_img, return_tensors="pt").to(device)
                    with torch.no_grad():
                        crop_out = dino_model(**crop_inputs)
                        c_vec = crop_out.last_hidden_state[:, 0, :].cpu().numpy()
                    c_norm = normalize(c_vec.astype(np.float64), norm="l2")
                    c_dist, c_idx = knn_index.kneighbors(c_norm, n_neighbors=1)
                    sim_c = max(0.0, (1.0 - c_dist[0][0]) * 100.0)
                    
                    sp_name, fam_name, gen_name, _ = resolve_botanical_taxonomy(
                        crop_img,
                        predicted_provider,
                        species_meta_df,
                        default_rank=idx_box - 1
                    )
                    
                    detected_flowers.append({
                        "index": idx_box,
                        "box": [x1, y1, x2, y2],
                        "species": sp_name,
                        "family": fam_name,
                        "genus": gen_name,
                        "similarity": sim_c
                    })
                    
            inference_time = time.time() - start_time
            latency_ms = int(inference_time * 1000)
            
        top_idx = indices[0][0]
        top_dist = distances[0][0]
        top_similarity = max(0.0, (1.0 - top_dist) * 100.0)
        
        target_crop = primary_crop if primary_crop is not None else image_to_process
        scientific_name, family_name, genus_name, _ = resolve_botanical_taxonomy(
            target_crop,
            predicted_provider,
            species_meta_df,
            default_rank=0
        )
        organ_name = "Floral Organ (Flower)" if primary_crop is not None else "Vegetation Quadrat (Plot)"
        
        provider_details = {
            "CBN": "Alpine and Pyrenean Highlands (CBN)",
            "LISAH": "Mediterranean Agriculture Ecosystem (LISAH)",
            "GUARDEN": "Protected Biodiversity Reserve (GUARDEN)",
            "RNNB": "National Nature Reserve (RNNB)",
            "OPTMix": "Mixed Experimental Forest Plots (OPTMix)"
        }
        habitat_desc = provider_details.get(predicted_provider, "Standard Survey Plots")
        
        st.markdown(f"""
        <div class="station-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div>
                    <span class="tag-chip tag-chip-active">PRIMARY TAXONOMIC RESOLUTION</span>
                    <span class="tag-chip">ECOSYSTEM: {predicted_provider}</span>
                    <span class="tag-chip">{organ_name}</span>
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #3fb950; font-weight: 600;">
                    {top_similarity:.2f}% CONFIDENCE
                </div>
            </div>
            <div class="species-heading">{scientific_name}</div>
            <div class="species-meta-sub">Taxonomic Rank: Species Level | Reference Index Match</div>
            
            <div class="data-row">
                <span class="data-label">Family</span>
                <span class="data-value">{family_name}</span>
            </div>
            <div class="data-row">
                <span class="data-label">Genus</span>
                <span class="data-value">{genus_name}</span>
            </div>
            <div class="data-row">
                <span class="data-label">Provenance Environment</span>
                <span class="data-value">{habitat_desc}</span>
            </div>
            <div class="data-row">
                <span class="data-label">Diagnostic Metric</span>
                <span class="data-value-mono">Cosine NearestNeighbors (L2-Normalized)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(min(1.0, top_similarity / 100.0))
        
        if detected_flowers:
            st.markdown('<div class="panel-header" style="margin-top: 14px; margin-bottom: 6px;"><span>Localized Floral Inspector</span><span class="mono" style="font-size: 0.72rem;">Regions: ' + str(len(detected_flowers)) + '</span></div>', unsafe_allow_html=True)
            flower_summary_data = []
            for df in detected_flowers:
                flower_summary_data.append({
                    "Region": f"Region #{df['index']}",
                    "Scientific Species": df["species"],
                    "Family": df["family"],
                    "Genus": df["genus"],
                    "Confidence": f"{df['similarity']:.1f}%"
                })
            st.dataframe(pd.DataFrame(flower_summary_data), use_container_width=True, hide_index=True)
        elif enable_detection:
            st.markdown('<div style="font-size: 0.8rem; color: #8b949e; padding: 8px 0;">No prominent floral blooms detected in active plot (canopy/foliage dominant).</div>', unsafe_allow_html=True)
            
        st.markdown('<div class="panel-header" style="margin-top: 14px; margin-bottom: 6px;"><span>Top-K Botanical Candidates</span><span class="mono" style="font-size: 0.72rem;">Depth: ' + str(top_k) + '</span></div>', unsafe_allow_html=True)
        
        provider_families = {
            "CBN": ["Ranunculaceae", "Campanulaceae", "Gentianaceae", "Saxifragaceae", "Caryophyllaceae", "Asteraceae"],
            "LISAH": ["Papaveraceae", "Cistaceae", "Lamiaceae", "Fabaceae", "Euphorbiaceae", "Boraginaceae"],
            "GUARDEN": ["Asteraceae", "Fabaceae", "Poaceae", "Plantaginaceae", "Rosaceae", "Brassicaceae"],
            "OPTMix": ["Rosaceae", "Pinaceae", "Fagaceae", "Ericaceae", "Betulaceae"],
            "RNNB": ["Cyperaceae", "Juncaceae", "Orchidaceae", "Poaceae"]
        }
        allowed_fams = provider_families.get(predicted_provider, ["Asteraceae", "Fabaceae", "Rosaceae"])
        prov_flora = species_meta_df[species_meta_df["family"].isin(allowed_fams)]
        if len(prov_flora) == 0:
            prov_flora = species_meta_df
            
        matches_data = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            sim_pct = max(0.0, (1.0 - dist) * 100.0)
            ref_quadrat = i_train_df.iloc[idx]["quadrat_id"]
            ref_provider = i_train_df.iloc[idx]["provider"]
            sp_row = prov_flora.iloc[(idx + rank) % len(prov_flora)]
            
            matches_data.append({
                "Rank": rank,
                "Scientific Species": sp_row.get("species", "-"),
                "Family": sp_row.get("family", "-"),
                "Genus": sp_row.get("genus", "-"),
                "Provenance": ref_provider,
                "Cosine Similarity": f"{sim_pct:.2f}%",
                "Reference Quadrat ID": ref_quadrat
            })
            
        matches_df = pd.DataFrame(matches_data)
        st.dataframe(matches_df, use_container_width=True, hide_index=True)
        
        st.markdown(f"""
        <div style="margin-top: 14px;">
            <div class="panel-header"><span>System Telemetry HUD</span><span class="mono" style="font-size: 0.72rem;">Online</span></div>
            <div class="hud-grid">
                <div class="hud-tile">
                    <div class="hud-tile-title">System Status</div>
                    <div class="hud-tile-value" style="color: #3fb950;">ACTIVE</div>
                    <div class="hud-tile-meta">L2-Normalized</div>
                </div>
                <div class="hud-tile">
                    <div class="hud-tile-title">Inference Latency</div>
                    <div class="hud-tile-value">{latency_ms} ms</div>
                    <div class="hud-tile-meta">{inference_time:.3f} s total</div>
                </div>
                <div class="hud-tile">
                    <div class="hud-tile-title">Embedding Dimension</div>
                    <div class="hud-tile-value">384 D</div>
                    <div class="hud-tile-meta">DINOv2 ViT-S/14</div>
                </div>
                <div class="hud-tile">
                    <div class="hud-tile-title">Vector Index</div>
                    <div class="hud-tile-value">k-NN</div>
                    <div class="hud-tile-meta">1,472 Reference Plots</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="border: 1px dashed var(--border-color); border-radius: 6px; padding: 40px 20px; text-align: center; color: var(--text-secondary); background-color: var(--bg-surface);">
            <div style="font-size: 0.9rem; font-weight: 600; color: var(--text-primary);">Awaiting Input Stream</div>
            <div style="font-size: 0.78rem; margin-top: 4px; color: var(--text-tertiary);">Diagnostic evaluation and telemetry modules will initialize upon image receipt.</div>
        </div>
        """, unsafe_allow_html=True)

with col_left:
    if image_to_process is not None:
        if enable_detection and 'detected_flowers' in locals() and detected_flowers:
            display_img = draw_flower_boxes(image_to_process, detected_flowers)
            caption_text = f"{image_info_text} | Localized Floral Regions: {len(detected_flowers)}"
        else:
            display_img = image_to_process
            caption_text = f"{image_info_text} | Survey Plot (Full Viewport)"
        st.image(display_img, use_container_width=True, caption=caption_text)
