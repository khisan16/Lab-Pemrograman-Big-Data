import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

# Judul aplikasi
st.title("🧠 Object Detection Dashboard (YOLOv8)")

# Path model
MODEL_PATH = "model/best.pt"

# Load model
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# Upload gambar
uploaded_file = st.file_uploader("Upload gambar untuk deteksi...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar Asli", use_container_width=True)

    # Simpan file sementara untuk deteksi
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        image.save(temp.name)
        results = model(temp.name)
        res_plotted = results[0].plot()

    # Tampilkan hasil deteksi
    st.image(res_plotted, caption="Hasil Deteksi", use_container_width=True)

st.sidebar.info("Model: YOLOv8 - Deteksi Objek dari file best.pt")
