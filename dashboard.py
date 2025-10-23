import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# Judul aplikasi
st.title("🧠 Object Detection Dashboard (YOLOv8)")

# Load model
MODEL_PATH = "model/best.pt"

@st.cache_resource
def load_model():
    try:
        model = YOLO(MODEL_PATH)
        st.success
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

model = load_model()

# Pilihan upload
st.sidebar.header("Upload File")
upload_type = st.sidebar.radio("Pilih jenis file:", ["Gambar"])

if upload_type == "Gambar":
    uploaded_file = st.file_uploader("Upload gambar...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None and model:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar Asli", use_container_width=True)

        # Simpan sementara untuk inferensi
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            image.save(temp.name)
            results = model(temp.name)
            res_plotted = results[0].plot()

        st.image(res_plotted, caption="Hasil Deteksi", use_container_width=True)

st.sidebar.info("Model YOLOv8 - Deteksi berbasis file best.pt")
