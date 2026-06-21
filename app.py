import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import pandas as pd
import numpy as np
from collections import Counter

st.set_page_config(
    page_title="Recyclable Waste Detection",
    page_icon="♻️",
    layout="wide", # Diubah ke wide agar gambar dan tabel lebih lega
    initial_sidebar_state="collapsed"
)

# Custom CSS styling - Dipertahankan untuk background dan header
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
        background: white;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    .header-icon {
        font-size: 3.5rem;
        margin-bottom: 0.5rem;
        display: block;
    }
    
    .header-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2e7d32;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        font-size: 1rem;
        color: #558b2f;
        font-weight: 500;
    }
    
    .section-label {
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        color: #558b2f;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
        display: block;
    }
    
    /* Style file uploader */
    [data-testid="stFileUploader"] > section {
        border: 2px dashed #81c784 !important;
        border-radius: 12px !important;
        padding: 2rem !important;
        background-color: #f1f8e9 !important;
    }
    
    [data-testid="stFileUploader"] > section:hover {
        border-color: #4caf50 !important;
        background-color: #e8f5e9 !important;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
<div class="header-container">
    <span class="header-icon">♻️</span>
    <h1 class="header-title">Sistem Deteksi Sampah Daur Ulang</h1>
    <p class="header-subtitle">Optimasi Kinerja YOLOv8 melalui Augmentasi Pencahayaan</p>
</div>
""", unsafe_allow_html=True)

# Pilih Model dan Upload di dalam kolom
col_opt, col_up = st.columns([1, 2])

with col_opt:
    st.markdown('<span class="section-label">⚙️ Pengaturan Model</span>', unsafe_allow_html=True)
    model_option = st.selectbox(
        "Pilih model inferensi:",
        ["YOLOv8s Baseline", "YOLOv8s + Lighting Augmentation"],
        key="model_select"
    )
    
    if model_option == "YOLOv8s Baseline":
        st.caption("Model standar YOLOv8s tanpa optimasi cahaya tambahan.")
        model_path = "best.pt" # Sesuaikan nama file asli modelmu
    else:
        st.caption("Model yang telah dioptimasi dengan variasi nilai Hue, Saturation, dan Value.")
        model_path = "best.pt" # Sesuaikan nama file asli modelmu

with col_up:
    st.markdown('<span class="section-label">📂 Unggah Gambar Uji</span>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Pilih gambar dari perangkatmu",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

st.markdown("---")

# Load model
@st.cache_resource
def load_model(path):
    return YOLO(path)

# Mencegah error kalau model belum ada saat tes UI
try:
    model = load_model(model_path)
    model_loaded = True
except Exception as e:
    st.warning(f"⚠️ Menunggu file model '{model_path}' dimasukkan ke folder yang sama dengan app.py")
    model_loaded = False

if uploaded_file is not None and model_loaded:
    image = Image.open(uploaded_file)
    
    # Convert RGBA to RGB if necessary (PNG files have transparency)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    with st.spinner('🔍 AI sedang memproses gambar...'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            results = model(tmp.name)
        
        # PERBAIKAN WARNA GAMBAR: YOLO menghasilkan BGR, Streamlit butuh RGB
        result_img_bgr = results[0].plot()
        result_img_rgb = result_img_bgr[..., ::-1] # Konversi BGR ke RGB
        
        boxes = results[0].boxes
        names = model.names

    # === BAGIAN HASIL (MENGGUNAKAN TABS AGAR RAPI) ===
    tab1, tab2 = st.tabs(["🖼️ VISUALISASI HASIL", "📊 ANALISIS & STATISTIK DATA"])
    
    with tab1:
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.markdown('**📸 Gambar Input Asli**')
            st.image(image, use_container_width=True)
            
        with img_col2:
            st.markdown('**🎯 Hasil Deteksi Bounding Box**')
            st.image(result_img_rgb, use_container_width=True)
            
    with tab2:
        if len(boxes) > 0:
            # 1. Metrik Singkat (Atas)
            confidences = [float(box.conf[0]) for box in boxes]
            class_ids = [int(box.cls[0]) for box in boxes]
            class_names_detected = [names[cls_id] for cls_id in class_ids]
            
            st.subheader("Ringkasan Performa")
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Sampah Terdeteksi", f"{len(boxes)} Objek")
            m2.metric("Rata-rata Tingkat Keyakinan", f"{(sum(confidences) / len(confidences)) * 100:.1f}%")
            
            # Hitung objek mayoritas
            most_common_class = Counter(class_names_detected).most_common(1)[0][0]
            m3.metric("Sampah Terbanyak", most_common_class.capitalize())
            
            st.markdown("---")
            
            # 2. Grafik dan Tabel (Bawah)
            chart_col, table_col = st.columns([1, 1.5])
            
            with chart_col:
                st.markdown("**Distribusi Jenis Sampah**")
                # Buat DataFrame khusus untuk grafik
                df_counts = pd.DataFrame(Counter(class_names_detected).items(), columns=['Jenis Sampah', 'Jumlah'])
                df_counts = df_counts.set_index('Jenis Sampah')
                st.bar_chart(df_counts, color="#4caf50")
                
            with table_col:
                st.markdown("**Tabel Detail Probabilitas (Confidence)**")
                # Data untuk tabel Streamlit yang interaktif
                details_data = []
                for idx, box in enumerate(boxes, 1):
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    details_data.append({
                        "No": idx,
                        "Jenis Sampah": names[cls_id].capitalize(),
                        "Akurasi (Confidence)": f"{conf*100:.1f} %",
                    })
                df_details = pd.DataFrame(details_data)
                # Tampilkan sebagai dataframe (bisa di scroll dan sort otomatis)
                st.dataframe(df_details, use_container_width=True, hide_index=True)
                
            # Info Bounding Box Mentah (Disembunyikan di expander agar tidak semak)
            with st.expander("Lihat Koordinat Piksel Bounding Box"):
                coord_data = []
                for idx, box in enumerate(boxes, 1):
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                    coord_data.append({"No": idx, "X1": x1, "Y1": y1, "X2": x2, "Y2": y2, "Width": x2-x1, "Height": y2-y1})
                st.dataframe(pd.DataFrame(coord_data), hide_index=True, use_container_width=True)
                
        else:
            st.info("⚠️ Tidak ada objek sampah daur ulang yang berhasil dideteksi pada gambar ini.")

# Footer
st.markdown("""
<div style='text-align: center; margin-top: 50px; color: #81c784; font-size: 0.85rem;'>
    © 2026 Skripsi Mar'atu Nadhifah Nayla | Universitas Negeri Semarang
</div>
""", unsafe_allow_html=True)