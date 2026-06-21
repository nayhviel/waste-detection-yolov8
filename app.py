pip install ultralytics
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import pandas as pd

st.set_page_config(
    page_title="Recyclable Waste Detection",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS styling - matching the design
st.markdown("""
<style>
    body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%);
        min-height: 100vh;
    }
    
    [data-testid="stMainBlockContainer"] {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    .header-container {
        text-align: center;
        padding: 2rem 1rem;
        margin-bottom: 2rem;
    }
    
    .header-icon {
        font-size: 3.5rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .header-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2e7d32;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        font-size: 0.95rem;
        color: #558b2f;
        font-weight: 500;
    }
    
    .section-label {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        color: #558b2f;
        text-transform: uppercase;
        margin-bottom: 1.2rem;
        display: block;
    }
    
    .section-description {
        font-size: 0.85rem;
        color: #558b2f;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }
    
    .upload-area {
        border: 2px dashed #81c784;
        border-radius: 12px;
        padding: 3rem 2rem;
        text-align: center;
        background-color: #f1f8e9;
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .upload-area:hover {
        border-color: #4caf50;
        background-color: #e8f5e9;
    }
    
    .upload-icon {
        font-size: 3rem;
        margin-bottom: 0.75rem;
        display: block;
        color: #4caf50;
    }
    
    .upload-text {
        font-size: 0.95rem;
        color: #2e7d32;
        font-weight: 600;
        margin-bottom: 0.3rem;
        display: block;
    }
    
    .upload-subtext {
        font-size: 0.8rem;
        color: #81c784;
        display: block;
    }
    
    .result-container {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    
    .result-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2e7d32;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #e8f5e9;
    }
    
    .section-divider {
        margin: 2rem 0;
        border-top: 1px solid #e8f5e9;
        padding-top: 2rem;
    }
    
    .detection-item {
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        background-color: #f1f8e9;
        border-left: 4px solid #4caf50;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 0.95rem;
        color: #1b5e20;
    }
    
    .detection-item-check {
        color: #4caf50;
        margin-right: 0.75rem;
        font-weight: bold;
        font-size: 1.1rem;
        flex-shrink: 0;
    }
    
    .detection-item-text {
        flex-grow: 1;
        margin: 0 0.5rem;
    }
    
    .detection-item-conf {
        color: #558b2f;
        font-weight: 600;
        font-family: 'Courier New', monospace;
        text-align: right;
        flex-shrink: 0;
    }
    
    .total-count {
        font-size: 1rem;
        font-weight: 600;
        color: #2e7d32;
        margin-top: 1.5rem;
        padding-top: 1.5rem;
        border-top: 2px solid #e8f5e9;
        text-align: center;
    }
    
    .footer {
        text-align: center;
        padding: 2rem 1rem 1rem;
        font-size: 0.85rem;
        color: #81c784;
    }
    
    .image-label {
        font-size: 0.9rem;
        font-weight: 600;
        color: #2e7d32;
        margin-bottom: 0.75rem;
    }
    
    /* Style file uploader */
    [data-testid="stFileUploader"] {
        background-color: transparent !important;
    }
    
    [data-testid="stFileUploader"] > section {
        border: 2px dashed #81c784 !important;
        border-radius: 12px !important;
        padding: 3rem 2rem !important;
        background-color: #f1f8e9 !important;
    }
    
    [data-testid="stFileUploader"] > section:hover {
        border-color: #4caf50 !important;
        background-color: #e8f5e9 !important;
    }
    
    [data-testid="stFileUploaderDropzone"] {
        text-align: center;
    }
    
    [data-testid="stFileUploaderDropzone"] > div {
        color: #2e7d32 !important;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
<div class="header-container">
    <span class="header-icon">♻️</span>
    <h1 class="header-title">Recyclable Waste Detection</h1>
    <p class="header-subtitle">Deteksi limbah daur ulang menggunakan YOLOv8s + Lighting Augmentation</p>
</div>
""", unsafe_allow_html=True)

# Model Selection Section
st.markdown('<span class="section-label">Pilih Model Inferensi</span>', unsafe_allow_html=True)

model_option = st.selectbox(
    "Pilih model untuk deteksi:",
    ["YOLOv8s Baseline", "YOLOv8s + Lighting Augmentation"],
    label_visibility="collapsed",
    key="model_select"
)

if model_option == "YOLOv8s Baseline":
    st.markdown('<p class="section-description">Model dengan feature extractor standar YOLOv8s dan classifier dasar.</p>', unsafe_allow_html=True)
    model_path = "baseline_best.pt"
else:
    st.markdown('<p class="section-description">Model dengan augmentasi pencahayaan untuk performa lebih baik pada kondisi cahaya berbeda.</p>', unsafe_allow_html=True)
    model_path = "augmented_best.pt"

st.divider()

# Load model
@st.cache_resource
def load_model(path):
    return YOLO(path)

model = load_model(model_path)

# Upload Section
st.markdown('<span class="section-label">Unggah Gambar</span>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📤 Seret & letakkan gambar limbah di sini atau klik untuk memilih file",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
    key="file_uploader"
)

st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)

# Detection Button
st.markdown("<div style='text-align: center; margin: 2rem 0;'>", unsafe_allow_html=True)
detect_button = st.button("🔍 Deteksi Limbah", use_container_width=False, key="detect_btn")
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None and detect_button:
    
    image = Image.open(uploaded_file)
    
    # Convert RGBA to RGB if necessary (PNG files have transparency)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        results = model(tmp.name)
    
    result_img = results[0].plot()
    
    # Start single large container for all results
    st.markdown("""
    <div class="result-container">
        <div class="result-title">📊 Hasil Deteksi</div>
    """, unsafe_allow_html=True)
    
    # Display images side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="image-label">Gambar Asli</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
    
    with col2:
        st.markdown('<div class="image-label">Hasil Deteksi</div>', unsafe_allow_html=True)
        st.image(result_img, use_container_width=True)
    
    # Statistics, Details & Objects Section - All in one container
    boxes = results[0].boxes
    names = model.names
    
    if len(boxes) > 0:
        # Calculate statistics
        confidences = [float(box.conf[0]) for box in boxes]
        avg_confidence = sum(confidences) / len(confidences)
        max_confidence = max(confidences)
        total_detections = len(boxes)
        
        # Build table data
        details_data = []
        for idx, box in enumerate(boxes, 1):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            width = x2 - x1
            height = y2 - y1
            
            details_data.append({
                "ID": idx,
                "Class": names[cls_id],
                "Confidence": f"{conf*100:.2f}%",
                "X": x1,
                "Y": y1,
                "Width": width,
                "Height": height
            })
        
        # Build complete HTML in ONE string
        complete_html = f"""
        <div class="section-divider">
            <div class="result-title">📊 Detection Statistics</div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem;">
                <div style="background: #f1f8e9; padding: 1.5rem; border-radius: 8px; text-align: center;">
                    <div style="font-size: 2rem; font-weight: bold; color: #2e7d32;">{total_detections}</div>
                    <div style="font-size: 0.9rem; color: #558b2f; margin-top: 0.5rem;">Total Detections</div>
                </div>
                <div style="background: #f1f8e9; padding: 1.5rem; border-radius: 8px; text-align: center;">
                    <div style="font-size: 2rem; font-weight: bold; color: #2e7d32;">{avg_confidence*100:.2f}%</div>
                    <div style="font-size: 0.9rem; color: #558b2f; margin-top: 0.5rem;">Avg Confidence</div>
                </div>
                <div style="background: #f1f8e9; padding: 1.5rem; border-radius: 8px; text-align: center;">
                    <div style="font-size: 2rem; font-weight: bold; color: #2e7d32;">{max_confidence*100:.2f}%</div>
                    <div style="font-size: 0.9rem; color: #558b2f; margin-top: 0.5rem;">Max Confidence</div>
                </div>
                <div style="background: #f1f8e9; padding: 1.5rem; border-radius: 8px; text-align: center;">
                    <div style="font-size: 2rem; font-weight: bold; color: #2e7d32;">Uploaded</div>
                    <div style="font-size: 0.9rem; color: #558b2f; margin-top: 0.5rem;">Image Source</div>
                </div>
            </div>
        </div>
        
        <div class="section-divider">
            <div class="result-title">📋 Detection Details</div>
            <table style="width: 100%; border-collapse: collapse; margin-bottom: 1.5rem;">
                <thead>
                    <tr style="background-color: #f1f8e9; border-bottom: 2px solid #e8f5e9;">
                        <th style="padding: 0.75rem; text-align: left; color: #2e7d32; font-weight: 600; font-size: 0.9rem;">ID</th>
                        <th style="padding: 0.75rem; text-align: left; color: #2e7d32; font-weight: 600; font-size: 0.9rem;">Class</th>
                        <th style="padding: 0.75rem; text-align: left; color: #2e7d32; font-weight: 600; font-size: 0.9rem;">Confidence</th>
                        <th style="padding: 0.75rem; text-align: left; color: #2e7d32; font-weight: 600; font-size: 0.9rem;">X</th>
                        <th style="padding: 0.75rem; text-align: left; color: #2e7d32; font-weight: 600; font-size: 0.9rem;">Y</th>
                        <th style="padding: 0.75rem; text-align: left; color: #2e7d32; font-weight: 600; font-size: 0.9rem;">Width</th>
                        <th style="padding: 0.75rem; text-align: left; color: #2e7d32; font-weight: 600; font-size: 0.9rem;">Height</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for data in details_data:
            complete_html += '<tr style="border-bottom: 1px solid #e8f5e9;">'
            for col in ["ID", "Class", "Confidence", "X", "Y", "Width", "Height"]:
                value = data[col]
                complete_html += f'<td style="padding: 0.75rem; color: #333; font-size: 0.9rem;">{value}</td>'
            complete_html += '</tr>'
        
        complete_html += """
                </tbody>
            </table>
        </div>
        
        <div class="section-divider">
            <div class="result-title">✨ Objek Terdeteksi</div>
        """
        
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            complete_html += f'<div class="detection-item"><span class="detection-item-check">✓</span><span class="detection-item-text">{names[cls_id]}</span><span class="detection-item-conf">{conf:.2f}</span></div>'
        
        complete_html += f'<div class="total-count">📈 Total Objek: <strong>{len(boxes)}</strong></div>'
        complete_html += '</div>'
        
        st.markdown(complete_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('</div>', unsafe_allow_html=True)
        st.warning("⚠️ Tidak ada objek terdeteksi.")
    
    # Container already closed above

# Footer
st.markdown("""
<div class="footer">
    © 2026 Sistem Deteksi Limbah Daur Ulang
</div>
""", unsafe_allow_html=True)
