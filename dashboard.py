import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageDraw

# ==========================
# Konfigurasi Halaman
# ==========================
st.set_page_config(
    page_title="Dashboard YOLO & Klasifikasi Sensitif",
    page_icon="🧠",
    layout="wide",
)

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    try:
        yolo_model = YOLO("model/best.pt")  # YOLO object detection
    except Exception as e:
        st.error(f"❌ Gagal memuat YOLO model: {e}")
        yolo_model = None

    try:
        classifier = tf.keras.models.load_model("model/hisan_model.h5")  # Classification
    except Exception as e:
        st.error(f"❌ Gagal memuat classifier model: {e}")
        classifier = None

    return yolo_model, classifier

yolo_model, classifier = load_models()
model_loaded = yolo_model is not None or classifier is not None

# ==========================
# UI Utama
# ==========================
st.title("🧠 Image Classification & Object Detection (Sensitif)")

menu = st.sidebar.selectbox(
    "Pilih Mode:",
    ["Deteksi Objek (YOLO Sensitif)", "Klasifikasi Gambar"],
)

uploaded_file = st.file_uploader("📸 Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# Fungsi resize untuk YOLO
# ==========================
def prepare_image(img, size=(640, 640)):
    return img.resize(size)

# ==========================
# Deteksi Objek Sensitif
# ==========================
def detect_objects(img):
    img_resized = prepare_image(img)
    # Sangat peka: conf rendah, iou rendah
    results = yolo_model(img_resized, conf=0.01, iou=0.05, verbose=False)
    return results

# ==========================
# Main Logic
# ==========================
if uploaded_file is not None and model_loaded:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang diunggah", use_container_width=True)

    if menu == "Deteksi Objek (YOLO Sensitif)":
        if yolo_model is None:
            st.error("YOLO model belum dimuat.")
        else:
            with st.spinner("🔍 Sedang mendeteksi objek..."):
                results = detect_objects(img)
                result_img = results[0].plot()  # YOLO default plot
                st.image(result_img, caption="Hasil Deteksi (Sangat Sensitif)", use_container_width=True)

                boxes = results[0].boxes
                if len(boxes) > 0:
                    st.subheader("📦 Detil Objek Terdeteksi (Sangat Sensitif):")
                    for i, box in enumerate(boxes):
                        cls_name = results[0].names[int(box.cls)]
                        conf = float(box.conf)
                        st.write(f"**Objek {i+1}:** {cls_name} ({conf:.2%})")
                else:
                    st.info("Tidak ada objek terdeteksi sama sekali (coba unggah gambar lain).")

    elif menu == "Klasifikasi Gambar":
        if classifier is None:
            st.error("Classifier model belum dimuat.")
        else:
            with st.spinner("🧠 Sedang melakukan klasifikasi..."):
                try:
                    img_rgb = img.convert("RGB")
                    target_size = classifier.input_shape[1:3]
                    img_resized = img_rgb.resize(target_size)

                    img_array = image.img_to_array(img_resized)
                    img_array = np.expand_dims(img_array, axis=0) / 255.0

                    prediction = classifier.predict(img_array)
                    class_index = int(np.argmax(prediction))
                    confidence = float(np.max(prediction))

                    st.success(f"### 🏷️ Kelas Prediksi: {class_index}")
                    st.progress(confidence)
                    st.caption(f"Probabilitas: {confidence:.2%}")

                    if prediction.shape[1] > 1:
                        st.subheader("📊 Confidence per Kelas")
                        for i, conf in enumerate(prediction[0]):
                            st.write(f"**Kelas {i}**: {conf:.2%}")

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat klasifikasi: {e}")
else:
    st.info("⬆️ Silakan unggah gambar terlebih dahulu untuk memulai analisis.")
