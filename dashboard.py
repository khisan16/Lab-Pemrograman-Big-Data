import streamlit as st
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

# ============================
# 🔧 Konfigurasi Halaman
# ============================
st.set_page_config(
    page_title="Object Detection Dashboard (YOLOv8)",
    page_icon="🪼",
    layout="wide",
)

# ============================
# 🎨 CSS Tema Gelap
# ============================
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #0b0c10;
        background-image: linear-gradient(160deg, #0b0c10 0%, #1f2833 100%);
        color: #c5c6c7;
    }
    [data-testid="stSidebar"] {
        background-color: #1f2833;
    }
    h1, h2, h3, h4 {
        color: #66fcf1;
        font-weight: 700;
    }
    .stButton button {
        background: linear-gradient(90deg, #45a29e, #66fcf1);
        color: #0b0c10;
        font-weight: bold;
        border-radius: 8px;
        border: none;
    }
    .stFileUploader label {
        color: #c5c6c7;
        font-size: 1rem;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# ============================
# 🪼 Header
# ============================
st.title("🪼 Object Detection Dashboard (YOLOv8)")
st.markdown("**Deteksi Objek Otomatis pada Gambar Menggunakan Model YOLOv8**")

# ============================
# 📦 Load Model
# ============================
try:
    model = YOLO("model/best.pt")
except Exception as e:
    st.error(f"Gagal memuat model YOLOv8: {e}")
    st.stop()

# ============================
# 📁 Upload Gambar
# ============================
# Upload file
uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])

# Muat model YOLOv8
try:
    model = YOLO("best.pt")
except Exception as e:
    st.error(f"Gagal memuat model YOLOv8: {e}")
    st.stop()

if uploaded_file is not None:
    # Simpan sementara
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Prediksi YOLO
    try:
        results = model.predict(source=file_path, conf=0.25)
        result_img = results[0].plot()

        # Tampilkan dalam dua kolom
        col1, col2 = st.columns(2)
        with col1:
            st.image(Image.open(file_path), caption="Gambar Asli", use_container_width=True)
        with col2:
            st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
