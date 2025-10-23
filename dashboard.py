import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import os

# ==========================
# CEK & IMPORT YOLOv8
# ==========================
try:
    from ultralytics import YOLO
    yolov8_available = True
except ImportError:
    yolov8_available = False
    st.warning("❌ Modul 'ultralytics' belum terinstal. Jalankan: pip install ultralytics")

# ==========================
# CEK & IMPORT TensorFlow
# ==========================
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image
    tf_available = True
except ImportError:
    tf_available = False
    st.warning("❌ Modul 'tensorflow' belum terinstal. Jalankan: pip install tensorflow")

# ==========================
# KONFIGURASI STREAMLIT
# ==========================
st.set_page_config(page_title="Dashboard YOLO & Klasifikasi", page_icon="🧠", layout="wide")
st.title("🧠 Image Classification & Object Detection (Sensitif)")

# ==========================
# LOAD MODEL DENGAN CACHE
# ==========================
@st.cache_resource
def load_yolo_model(path="model/best.pt"):
    if not yolov8_available:
        return None
    if not os.path.exists(path):
        st.error(f"❌ File model YOLO tidak ditemukan di: {path}")
        return None
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat YOLOv8: {e}")
        return None

@st.cache_resource
def load_classifier_model(path="model/hisan_model.h5"):
    if not tf_available:
        return None
    if not os.path.exists(path):
        st.error(f"❌ File model klasifikasi tidak ditemukan di: {path}")
        return None
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model klasifikasi: {e}")
        return None

# ==========================
# LOAD MODEL
# ==========================
yolo_model = load_yolo_model()
classifier_model = load_classifier_model()

# ==========================
# LOAD LABEL KELAS TENSORFLOW (opsional)
# ==========================
classes_file = "model/classes.txt"
if os.path.exists(classes_file):
    with open(classes_file, "r") as f:
        tf_class_labels = [line.strip() for line in f.readlines()]
else:
    tf_class_labels = None

# ==========================
# SIDEBAR
# ==========================
menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
uploaded_file = st.file_uploader("📸 Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# FUNGSI BANTUAN
# ==========================
def prepare_image(img, size=(640, 640)):
    return img.resize(size)

def preprocess_for_classifier(img, target_shape):
    img_rgb = img.convert("RGB")
    img_resized = img_rgb.resize(target_shape)
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# ==========================
# DETEKSI OBJEK DENGAN YOLO
# ==========================
def detect_objects(img):
    img_resized = prepare_image(img)
    results = yolo_model(img_resized, conf=0.25, iou=0.45, verbose=False)
    return results

# ==========================
# TAMPILKAN KELAS YOLO
# ==========================
if yolo_model is not None:
    st.sidebar.subheader("📌 Kelas YOLO")
    df_yolo = pd.DataFrame(list(yolo_model.names.items()), columns=["Index", "Nama Kelas"])
    st.sidebar.table(df_yolo)

# ==========================
# BAGIAN UTAMA
# ==========================
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang diunggah", use_container_width=True)

    # MODE YOLO
    if menu == "Deteksi Objek (YOLO)":
        if yolo_model is None:
            st.warning("YOLOv8 tidak tersedia. Silakan cek requirements dan model best.pt.")
        else:
            with st.spinner("🔍 Sedang mendeteksi objek..."):
                results = detect_objects(img)
                result_img = results[0].plot()
                st.image(result_img, caption="Hasil Deteksi YOLO", use_container_width=True)

                boxes = results[0].boxes
                if len(boxes) > 0:
                    st.subheader("📦 Detil Objek Terdeteksi:")
                    for i, box in enumerate(boxes):
                        cls_name = results[0].names[int(box.cls)]
                        conf = float(box.conf)
                        st.write(f"**Objek {i+1}:** {cls_name} ({conf:.2%})")
                else:
                    st.info("Tidak ada objek terdeteksi.")

    # MODE KLASIFIKASI
    elif menu == "Klasifikasi Gambar":
        if classifier_model is None:
            st.warning("Model klasifikasi tidak tersedia.")
        else:
            with st.spinner("🧠 Sedang mengklasifikasikan..."):
                target_size = classifier_model.input_shape[1:3]
                img_array = preprocess_for_classifier(img, target_size)
                prediction = classifier_model.predict(img_array)
                class_index = int(np.argmax(prediction))
                confidence = float(np.max(prediction))

                label = tf_class_labels[class_index] if tf_class_labels else class_index
                st.success(f"### 🏷️ Prediksi: {label}")
                st.progress(confidence)
                st.caption(f"Probabilitas: {confidence:.2%}")

else:
    st.info("⬆️ Silakan unggah gambar terlebih dahulu.")
