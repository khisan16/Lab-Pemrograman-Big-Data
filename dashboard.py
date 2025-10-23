import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import os

# ==========================
# Import YOLO & TF
# ==========================
try:
    from ultralytics import YOLO
    yolov8_available = True
except ImportError:
    yolov8_available = False

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image
    tf_available = True
except ImportError:
    tf_available = False

# ==========================
# Halaman Streamlit
# ==========================
st.set_page_config(
    page_title="Dashboard YOLO & Klasifikasi Sensitif",
    page_icon="🧠",
    layout="wide",
)
st.title("🧠 Image Classification & Object Detection (Sensitif)")

# ==========================
# Cek kelas YOLO sebelum load cached
# ==========================
if yolov8_available:
    try:
        model_check = YOLO("model/best.pt")
        yolo_classes = model_check.names  # dict {0: 'class_name', ...}
        st.sidebar.subheader("📌 Kelas YOLO (cek awal)")
        df_yolo_check = pd.DataFrame(list(yolo_classes.items()), columns=["Index", "Nama Kelas"])
        st.sidebar.table(df_yolo_check)
    except Exception as e:
        st.sidebar.warning(f"Gagal load YOLO untuk cek kelas: {e}")
        yolo_classes = None
else:
    yolo_classes = None
    st.sidebar.info("YOLOv8 tidak tersedia")

# ==========================
# Load Models dengan caching
# ==========================
@st.cache_resource
def load_yolo_model(path="model/best.pt"):
    try:
        yolo_model = YOLO(path)
        return yolo_model
    except Exception as e:
        st.warning(f"❌ Gagal memuat YOLO: {e}")
        return None

@st.cache_resource
def load_classifier_model(path="model/hisan_model.h5"):
    try:
        classifier_model = tf.keras.models.load_model(path)
        return classifier_model
    except Exception as e:
        st.warning(f"❌ Gagal memuat classifier: {e}")
        return None

yolo_model = load_yolo_model() if yolov8_available else None
classifier_model = load_classifier_model() if tf_available else None

# ==========================
# Load kelas TensorFlow (opsional)
# ==========================
classes_file = "model/classes.txt"
if os.path.exists(classes_file):
    with open(classes_file, "r") as f:
        tf_class_labels = [line.strip() for line in f.readlines()]
else:
    tf_class_labels = None

# ==========================
# Sidebar Mode
# ==========================
menu = st.sidebar.selectbox(
    "Pilih Mode:",
    ["Deteksi Objek (YOLO Sensitif)", "Klasifikasi Gambar"]
)

uploaded_file = st.file_uploader("📸 Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# Fungsi Resize & Preprocess
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
# Deteksi Objek Sensitif
# ==========================
def detect_objects(img):
    img_resized = prepare_image(img)
    results = yolo_model(img_resized, conf=0.01, iou=0.05, verbose=False)
    return results

# ==========================
# Tampilkan daftar kelas YOLO di sidebar setelah load
# ==========================
if yolo_model is not None:
    st.sidebar.subheader("📌 Kelas YOLO (loaded)")
    df_yolo_classes = pd.DataFrame(list(yolo_model.names.items()), columns=["Index", "Nama Kelas"])
    st.sidebar.table(df_yolo_classes)

# ==========================
# Main Logic
# ==========================
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang diunggah", use_container_width=True)

    # ======================
    # Deteksi Objek YOLO
    # ======================
    if menu == "Deteksi Objek (YOLO Sensitif)":
        if yolo_model is None:
            st.warning("YOLOv8 tidak tersedia. Silakan cek requirements dan model best.pt.")
        else:
            with st.spinner("🔍 Sedang mendeteksi objek..."):
                results = detect_objects(img)
                result_img = results[0].plot()
                st.image(result_img, caption="Hasil Deteksi (Sangat Sensitif)", use_container_width=True)

                boxes = results[0].boxes
                if len(boxes) > 0:
                    st.subheader("📦 Detil Objek Terdeteksi:")
                    for i, box in enumerate(boxes):
                        cls_name = results[0].names[int(box.cls)]
                        conf = float(box.conf)
                        st.write(f"**Objek {i+1}:** {cls_name} ({conf:.2%})")
                else:
                    st.info("Tidak ada objek terdeteksi. Coba unggah gambar lain atau pastikan objek terlihat jelas.")

    # ======================
    # Klasifikasi Gambar
    # ======================
    elif menu == "Klasifikasi Gambar":
        if classifier_model is None:
            st.warning("Model klasifikasi tidak tersedia. Silakan cek path hisan_model.h5.")
        else:
            # Tampilkan jumlah kelas & nama kelas TensorFlow
            num_classes = classifier_model.output_shape[1]
            st.info(f"Jumlah kelas TensorFlow: {num_classes}")
            if tf_class_labels:
                st.info(f"Nama kelas TensorFlow: {tf_class_labels}")
            else:
                st.info(f"Kelas (index): {list(range(num_classes))}")

            with st.spinner("🧠 Sedang melakukan klasifikasi..."):
                try:
                    target_size = classifier_model.input_shape[1:3]
                    img_array = preprocess_for_classifier(img, target_size)
                    prediction = classifier_model.predict(img_array)
                    class_index = int(np.argmax(prediction))
                    confidence = float(np.max(prediction))

                    label = tf_class_labels[class_index] if tf_class_labels else class_index
                    st.success(f"### 🏷️ Kelas Prediksi: {label}")
                    st.progress(confidence)
                    st.caption(f"Probabilitas: {confidence:.2%}")

                    if prediction.shape[1] > 1:
                        st.subheader("📊 Confidence per Kelas")
                        for i, conf in enumerate(prediction[0]):
                            lbl = tf_class_labels[i] if tf_class_labels else i
                            st.write(f"**{lbl}**: {conf:.2%}")

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat klasifikasi: {e}")

else:
    st.info("⬆️ Silakan unggah gambar terlebih dahulu untuk memulai analisis.")
