import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ==========================
# Konfigurasi Halaman
# ==========================
st.set_page_config(
    page_title="AI Vision Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/best.pt")  # Model deteksi objek
    classifier = tf.keras.models.load_model("model/hisan_model.h5")  # Model klasifikasi
    return yolo_model, classifier

yolo_model, classifier = load_models()

# Daftar label kelas (ubah sesuai model CNN kamu)
CLASS_NAMES = ["Kelas 1", "Kelas 2", "Kelas 3"]

# ==========================
# UI
# ==========================
st.title("🧠 AI Vision Dashboard")
st.markdown(
    """
    Selamat datang di **Dashboard Deteksi & Klasifikasi Gambar**.  
    Pilih mode di sidebar untuk menggunakan model **YOLOv8 Object Detection** atau **Image Classification CNN**.
    """
)

menu = st.sidebar.radio("🧩 Pilih Mode:", ["📦 Deteksi Objek (YOLO)", "🖼️ Klasifikasi Gambar"])

uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# LOGIKA UTAMA
# ==========================
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(img, caption="📸 Gambar Diupload", use_container_width=True)

    with col2:
        # ==========================
        # MODE YOLO DETECTION
        # ==========================
        if "YOLO" in menu:
            with st.spinner("🔍 Sedang mendeteksi objek..."):
                # Tambahkan parameter agar YOLO lebih sensitif
                results = yolo_model.predict(
                    source=img,
                    conf=0.25,  # tingkat kepercayaan minimal
                    iou=0.45,   # intersection over union threshold
                    imgsz=640   # ukuran gambar
                )

                # Gambar hasil deteksi
                result_img = results[0].plot()
                st.image(result_img, caption="Hasil Deteksi YOLO", use_container_width=True)

                # Debug: tampilkan hasil deteksi mentah
                st.write("### 🔍 Debug Data Deteksi:")
                st.write(results[0].boxes.data)

                # Tampilkan nama label model
                st.write("📋 Label Model:", yolo_model.names)

                # Analisis hasil deteksi
                detections = results[0].boxes.data
                if len(detections) > 0:
                    st.success(f"✅ {len(detections)} objek terdeteksi!")
                    for i, det in enumerate(detections, 1):
                        cls_id = int(det[5])
                        conf = float(det[4])
                        label = yolo_model.names.get(cls_id, f"Class {cls_id}")
                        st.write(f"**{i}. {label} ({conf*100:.2f}% confidence)**")
                else:
                    st.warning("⚠️ Tidak ada objek terdeteksi. Coba turunkan nilai `conf` di kode YOLO.")
        
        # ==========================
        # MODE CNN CLASSIFICATION
        # ==========================
        elif "Klasifikasi" in menu:
            with st.spinner("🧠 Sedang melakukan klasifikasi..."):
                # Preprocessing gambar untuk model CNN
                img_resized = img.resize((96, 96))  # ubah sesuai input model CNN kamu
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0) / 255.0

                # Prediksi
                prediction = classifier.predict(img_array)
                class_index = int(np.argmax(prediction))
                confidence = float(np.max(prediction))

                # Hasil klasifikasi
                class_name = CLASS_NAMES[class_index] if class_index < len(CLASS_NAMES) else f"Kelas {class_index}"
                st.success("✅ Klasifikasi Berhasil!")
                st.write(f"### 🔎 Hasil Prediksi: **{class_name}**")
                st.progress(confidence)
                st.write(f"Probabilitas: **{confidence*100:.2f}%**")

                # Tampilkan probabilitas semua kelas
                st.subheader("📊 Probabilitas Semua Kelas:")
                for i, prob in enumerate(prediction[0]):
                    label = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Kelas {i}"
                    st.write(f"{label}: {prob*100:.2f}%")
                    st.progress(float(prob))

else:
    st.info("👆 Silakan unggah gambar terlebih dahulu untuk memulai.")

# ==========================
# Footer
# ==========================
st.markdown(
    """
    ---
    🧑‍💻 **Dikembangkan oleh:** Tim AI Vision  
    ⚙️ Model: YOLOv8 + TensorFlow CNN  
    """
)
