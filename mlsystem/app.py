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
    page_title="PlantCLEF AI - Scientific Biodiversity Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: rgb(11, 19, 32);
        color: rgb(248, 250, 252);
    }
    header, [data-testid="stHeader"] {
        background-color: rgb(11, 19, 32) !important;
    }
    section[data-testid="stSidebar"] {
        background-color: rgb(15, 25, 42) !important;
        border-right: 1px solid rgb(30, 48, 74);
    }
    .sci-header {
        font-size: 2.0rem;
        font-weight: 800;
        color: rgb(248, 250, 252);
        letter-spacing: -0.5px;
        margin-bottom: 2px;
    }
    .sci-sub {
        font-size: 0.95rem;
        color: rgb(148, 163, 184);
        margin-bottom: 16px;
    }
    .sci-card {
        background-color: rgb(21, 34, 56);
        border: 1px solid rgb(34, 55, 85);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .species-name {
        font-size: 1.55rem;
        font-weight: 700;
        color: rgb(52, 211, 153);
        font-style: italic;
        margin-top: 4px;
        margin-bottom: 4px;
    }
    .badge-sci {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .badge-emerald {
        background-color: rgba(16, 185, 129, 0.18);
        color: rgb(52, 211, 153);
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .badge-cyan {
        background-color: rgba(6, 182, 212, 0.18);
        color: rgb(34, 211, 238);
        border: 1px solid rgba(6, 182, 212, 0.4);
    }
    .badge-purple {
        background-color: rgba(168, 85, 247, 0.18);
        color: rgb(192, 132, 252);
        border: 1px solid rgba(168, 85, 247, 0.4);
    }
    .telemetry-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
        margin-top: 10px;
    }
    .telemetry-card {
        background-color: rgb(15, 25, 42);
        border: 1px solid rgb(30, 48, 74);
        border-radius: 8px;
        padding: 10px 12px;
    }
    .telemetry-title {
        font-size: 0.72rem;
        color: rgb(148, 163, 184);
        text-transform: uppercase;
        font-weight: 600;
    }
    .telemetry-value {
        font-size: 1.15rem;
        font-weight: 700;
        color: rgb(52, 211, 153);
        margin-top: 2px;
    }
    .telemetry-sub {
        font-size: 0.7rem;
        color: rgb(100, 116, 139);
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
    hsv = np.array(pil_img.convert("HSV"), dtype=np.float32)
    H = hsv[:, :, 0] / 255.0
    S = hsv[:, :, 1] / 255.0
    V = hsv[:, :, 2] / 255.0
    
    is_foliage = (H >= 0.18) & (H <= 0.45) & (S > 0.25) & (V < 0.80)
    is_yellow = (H >= 0.08) & (H < 0.18) & (S > 0.40) & (V > 0.55)
    is_purple_red = ((H > 0.60) | (H < 0.06)) & (S > 0.20) & (V > 0.35)
    is_white = (V > 0.85) & (S < 0.25) & (~is_foliage)
    
    flower_mask = is_yellow | is_purple_red | is_white
    
    struct = ndimage.generate_binary_structure(2, 2)
    clean_mask = ndimage.binary_dilation(flower_mask, structure=struct, iterations=3)
    clean_mask = ndimage.binary_closing(clean_mask, structure=struct, iterations=3)
    
    labeled, num_features = ndimage.label(clean_mask)
    slices = ndimage.find_objects(labeled)
    
    boxes = []
    min_area = (w * h) * 0.008
    max_area = (w * h) * 0.35
    
    for s in slices:
        y1, y2 = s[0].start, s[0].stop
        x1, x2 = s[1].start, s[1].stop
        bw, bh = x2 - x1, y2 - y1
        area = bw * bh
        if min_area < area < max_area and bw > 25 and bh > 25:
            aspect = max(bw / max(1, bh), bh / max(1, bw))
            if aspect < 3.5:
                pad = 12
                boxes.append([max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad), area])
                
    if not boxes:
        boxes.append([int(w * 0.15), int(h * 0.15), int(w * 0.85), int(h * 0.85), int(w * h * 0.5)])
        
    boxes.sort(key=lambda x: x[4], reverse=True)
    
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
                if inter / union > 0.25:
                    overlap = True
                    break
        if not overlap:
            filtered.append(b)
        if len(filtered) >= max_boxes:
            break
            
    return filtered

def draw_flower_boxes(pil_img, detected_flowers):
    annotated = pil_img.copy()
    draw = ImageDraw.Draw(annotated)
    box_colors = [
        (16, 185, 129),
        (6, 182, 212),
        (245, 158, 11),
        (168, 85, 247),
        (236, 72, 153)
    ]
    
    for item in detected_flowers:
        i = item["index"]
        x1, y1, x2, y2 = item["box"]
        sp_name = item["species"]
        sim = item["similarity"]
        col = box_colors[(i - 1) % len(box_colors)]
        
        draw.rectangle([x1, y1, x2, y2], outline=col, width=4)
        label_txt = f"[{i}] {sp_name} ({sim:.1f}%)"
        tag_w = len(label_txt) * 8 + 14
        draw.rectangle([x1, max(0, y1 - 24), x1 + tag_w, y1], fill=col)
        draw.text((x1 + 6, max(0, y1 - 20)), label_txt, fill=(11, 19, 32))
        
    return annotated

with st.sidebar:
    st.subheader("Control Center")
    st.caption("ระบบวิเคราะห์และจำแนกแปลงพืชธรรมชาติ")
    st.divider()
    
    st.write("แหล่งที่มาของภาพสำรวจ:")
    input_mode = st.radio(
        "โหมดการนำเข้า:",
        ["เลือกภาพตัวอย่างจากระบบ", "อัปโหลดภาพของคุณเอง"],
        label_visibility="collapsed"
    )
    
    sample_choice = None
    if input_mode == "เลือกภาพตัวอย่างจากระบบ":
        sample_options = {
            "แปลงพืชเทือกเขาแอลป์ (Alpine Plot - CBN)": "sample_alpine_cbn.jpg",
            "แปลงพืชทุ่งหญ้าเมดิเตอร์เรเนียน (LISAH)": "sample_mediterranean_lisah.jpg",
            "แปลงคุ้มครองความหลากหลายทางชีวภาพ (GUARDEN)": "sample_guarden_biodiversity.jpg"
        }
        selected_sample_label = st.selectbox("เลือกแปลงสำรวจ:", list(sample_options.keys()))
        sample_choice = sample_options[selected_sample_label]
    
    st.divider()
    st.write("การตรวจจับดอกไม้ (Visual Localization):")
    enable_detection = st.checkbox("ตีกรอบระบุตำแหน่งดอกไม้ในแปลง", value=True)
    max_flowers = st.slider("จำนวนดอกไม้สูงสุดที่ตรวจจับ:", min_value=1, max_value=5, value=3)
    
    st.divider()
    top_k = st.slider("จำนวนชนิดพืชในตาราง Top-K Inspector:", min_value=3, max_value=8, value=5)
    
    st.divider()
    st.write("คณะผู้พัฒนา:")
    st.write("- ปิยังกูร ปัสสาวะกัง (6810405691)")
    st.write("- ศิวภูมิ พรหมจรรย์ (6810405887)")

st.markdown('<div class="sci-header">Plant Biodiversity Identification Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sci-sub">ระบบจำแนกชนิดพืชในแปลงสำรวจธรรมชาติด้วยสถาปัตยกรรม Hybrid (DINOv2 + Vector Search)</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([1.1, 1.2], gap="large")

image_to_process = None
image_info_text = ""

with col_left:
    st.subheader("Vegetation Quadrat Workspace")
    
    if input_mode == "อัปโหลดภาพของคุณเอง":
        uploaded_file = st.file_uploader(
            "ลากไฟล์ภาพแปลงพืชมาวางที่นี่ หรือกด Browse",
            type=["jpg", "jpeg", "png", "webp"]
        )
        if uploaded_file is not None:
            image_to_process = Image.open(uploaded_file).convert("RGB")
            w, h = image_to_process.size
            image_info_text = f"ความละเอียด: {w} x {h} px | {uploaded_file.type}"
    else:
        sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_images", sample_choice)
        if os.path.exists(sample_path):
            image_to_process = Image.open(sample_path).convert("RGB")
            w, h = image_to_process.size
            image_info_text = f"ภาพแปลงมาตรฐาน: {sample_choice} ({w} x {h} px)"
        else:
            st.warning("ไม่พบไฟล์ภาพตัวอย่างในระบบ")

    if image_to_process is None:
        st.markdown("""
        <div style="border: 2px dashed rgb(34, 55, 85); border-radius: 12px; padding: 60px 20px; text-align: center; color: rgb(148, 163, 184); background-color: rgb(15, 25, 42);">
            <div style="font-size: 1.1rem; font-weight: 600;">ยังไม่มีข้อมูลภาพแปลงสำรวจ</div>
            <div style="font-size: 0.85rem; margin-top: 6px;">โปรดอัปโหลดภาพ หรือเลือกภาพตัวอย่างจากเมนูด้านซ้ายเพื่อเริ่มการวิเคราะห์</div>
        </div>
        """, unsafe_allow_html=True)

with col_right:
    st.subheader("Diagnostic & Telemetry Panel")
    
    if image_to_process is not None:
        with st.spinner("กำลังประมวลผลเวกเตอร์ลักษณะภาพ DINOv2 และสืบค้นคลังพฤกษศาสตร์..."):
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
                    sp_row_c = species_meta_df.iloc[c_idx[0][0] % len(species_meta_df)]
                    sim_c = max(0.0, (1.0 - c_dist[0][0]) * 100.0)
                    
                    detected_flowers.append({
                        "index": idx_box,
                        "box": [x1, y1, x2, y2],
                        "species": sp_row_c.get("species", "Taxus baccata L."),
                        "family": sp_row_c.get("family", "Taxaceae"),
                        "genus": sp_row_c.get("genus", "Taxus"),
                        "similarity": sim_c
                    })
                    
            inference_time = time.time() - start_time
            latency_ms = int(inference_time * 1000)
            
        top_idx = indices[0][0]
        top_dist = distances[0][0]
        top_similarity = max(0.0, (1.0 - top_dist) * 100.0)
        
        top_species_row = species_meta_df.iloc[top_idx % len(species_meta_df)]
        scientific_name = top_species_row.get("species", "Taxus baccata L.")
        genus_name = top_species_row.get("genus", "Taxus")
        family_name = top_species_row.get("family", "Taxaceae")
        organ_name = top_species_row.get("organ", "ใบ (Leaf)")
        
        provider_details = {
            "CBN": "เทือกเขาแอลป์และพีเรนีส (Conservatoire Botanique National)",
            "LISAH": "ทุ่งหญ้าและพื้นที่เกษตรเมดิเตอร์เรเนียน (LISAH Agrosystem)",
            "GUARDEN": "เขตคุ้มครองความหลากหลายทางชีวภาพ (GUARDEN Biodiversity)",
            "RNNB": "เขตอนุรักษ์ธรรมชาติแห่งชาติ (National Nature Reserve)",
            "OPTMix": "แปลงทดลองป่าไม้ผสมผสาน (OPTMix Mixed Forest)"
        }
        habitat_desc = provider_details.get(predicted_provider, "แปลงสำรวจธรรมชาติทั่วไป")
        
        st.markdown(f"""
        <div class="sci-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <span class="badge-sci badge-emerald">PRIMARY IDENTIFICATION</span>
                    <span class="badge-sci badge-cyan">{predicted_provider}</span>
                    <span class="badge-sci badge-purple">{organ_name}</span>
                    <div class="species-name">{scientific_name}</div>
                    <div style="color: rgb(148, 163, 184); font-size: 0.9rem; margin-top: 4px;">
                        วงศ์ (Family): <b style="color: rgb(248, 250, 252);">{family_name}</b> | 
                        สกุล (Genus): <b style="color: rgb(248, 250, 252);">{genus_name}</b>
                    </div>
                    <div style="color: rgb(100, 116, 139); font-size: 0.8rem; margin-top: 2px;">
                        สภาพแวดล้อม: {habitat_desc}
                    </div>
                </div>
            </div>
            <div style="margin-top: 14px;">
                <div style="display: flex; justify-content: space-between; font-size: 0.82rem; color: rgb(148, 163, 184); margin-bottom: 4px;">
                    <span>Visual Similarity Metric</span>
                    <span style="color: rgb(52, 211, 153); font-weight: 700;">{top_similarity:.2f}%</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(min(1.0, top_similarity / 100.0))
        
        if detected_flowers:
            st.markdown('<div style="font-weight: 700; font-size: 1.05rem; color: rgb(248, 250, 252); margin-top: 14px; margin-bottom: 6px;">Localized Species Inspector (ตรวจจับแยกช่อดอก)</div>', unsafe_allow_html=True)
            flower_summary_data = []
            for df in detected_flowers:
                flower_summary_data.append({
                    "กรอบดอกไม้": f"ดอกไม้ที่ {df['index']}",
                    "ชื่อวิทยาศาสตร์ (Species)": df["species"],
                    "วงศ์ (Family)": df["family"],
                    "สกุล (Genus)": df["genus"],
                    "ความมั่นใจ": f"{df['similarity']:.1f}%"
                })
            st.dataframe(pd.DataFrame(flower_summary_data), use_container_width=True, hide_index=True)
            
        st.markdown('<div style="font-weight: 700; font-size: 1.05rem; color: rgb(248, 250, 252); margin-top: 14px; margin-bottom: 6px;">Top-K Nearest Botanical Candidates (คลังพืชใกล้เคียง)</div>', unsafe_allow_html=True)
        
        matches_data = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            sim_pct = max(0.0, (1.0 - dist) * 100.0)
            ref_quadrat = i_train_df.iloc[idx]["quadrat_id"]
            ref_provider = i_train_df.iloc[idx]["provider"]
            sp_row = species_meta_df.iloc[idx % len(species_meta_df)]
            
            matches_data.append({
                "อันดับ": rank,
                "ชื่อวิทยาศาสตร์": sp_row.get("species", "-"),
                "วงศ์ (Family)": sp_row.get("family", "-"),
                "สกุล (Genus)": sp_row.get("genus", "-"),
                "ถิ่นกำเนิด": ref_provider,
                "ความคล้ายคลึง": f"{sim_pct:.2f}%",
                "แปลงอ้างอิง": ref_quadrat
            })
            
        matches_df = pd.DataFrame(matches_data)
        st.dataframe(matches_df, use_container_width=True, hide_index=True)
        
        st.markdown(f"""
        <div class="sci-card" style="margin-top: 16px;">
            <div style="font-size: 0.85rem; font-weight: 700; color: rgb(148, 163, 184); text-transform: uppercase; letter-spacing: 0.5px;">System Telemetry HUD</div>
            <div class="telemetry-grid">
                <div class="telemetry-card">
                    <div class="telemetry-title">Model Status</div>
                    <div class="telemetry-value" style="color: rgb(52, 211, 153);">Active</div>
                    <div class="telemetry-sub">Ready for query</div>
                </div>
                <div class="telemetry-card">
                    <div class="telemetry-title">Latency</div>
                    <div class="telemetry-value">{latency_ms} ms</div>
                    <div class="telemetry-sub">{inference_time:.3f} seconds</div>
                </div>
                <div class="telemetry-card">
                    <div class="telemetry-title">Embedding</div>
                    <div class="telemetry-value" style="color: rgb(34, 211, 238);">384 D</div>
                    <div class="telemetry-sub">DINOv2 ViT-S/14</div>
                </div>
                <div class="telemetry-card">
                    <div class="telemetry-title">Architecture</div>
                    <div class="telemetry-value" style="font-size: 0.95rem; color: rgb(192, 132, 252); margin-top: 6px;">Hybrid</div>
                    <div class="telemetry-sub">k-NN + Ensemble</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="border: 1px solid rgb(30, 48, 74); border-radius: 12px; padding: 40px 20px; text-align: center; color: rgb(100, 116, 139); background-color: rgb(15, 25, 42);">
            <div style="font-size: 1rem; font-weight: 600;">รอการนำเข้าข้อมูลภาพแปลงสำรวจ</div>
            <div style="font-size: 0.82rem; margin-top: 4px;">แผงวิเคราะห์เชิงลึกและ Telemetry จะทำงานทันทีเมื่อได้รับภาพ</div>
        </div>
        """, unsafe_allow_html=True)

with col_left:
    if image_to_process is not None:
        if enable_detection and 'detected_flowers' in locals() and detected_flowers:
            display_img = draw_flower_boxes(image_to_process, detected_flowers)
            caption_text = f"{image_info_text} | ตรวจพบตำแหน่งดอกไม้ {len(detected_flowers)} ตำแหน่ง"
        else:
            display_img = image_to_process
            caption_text = image_info_text
        st.image(display_img, use_container_width=True, caption=caption_text)
