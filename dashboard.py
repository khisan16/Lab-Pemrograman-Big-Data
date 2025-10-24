import streamlit as st
from PIL import Image
from ultralytics import YOLO
import tempfile, os, shutil, time
import pandas as pd

# ============================
# 🔧 Konfigurasi Halaman
# ============================
st.set_page_config(page_title="YOLOv8 Object Detection Dashboard", page_icon="🪼", layout="wide")

# ============================
# 🎨 CSS Kustom
# ============================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0b0c10;
    background-image: linear-gradient(160deg, #0b0c10 0%, #1f2833 100%);
    color: #c5c6c7;
}
h1, h2, h3 { color: #66fcf1; font-weight: 700; }
.sidebar .sidebar-content { background-color: #1f2833; }
.stButton button {
    background: linear-gradient(90deg, #45a29e, #66fcf1);
    color: #0b0c10; font-weight: bold; border-radius: 8px;
}
.result-card {
    background-color: #1f2833;
    padding: 15px; border-radius: 12px;
    box-shadow: 0px 0px 10px rgba(102,252,241,0.2);
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# ============================
# 🪼 Header
# ============================
st.title("🪼 Object Detection Dashboard (YOLOv8)")
st.markdown("**Deteksi Objek Otomatis dengan Dashboard Interaktif**")

# ============================
# ⚙️ Sidebar Pengaturan
# ============================
st.sidebar.header("⚙️ Pengaturan Deteksi")
conf_thres = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
show_labels = st.sidebar.checkbox("Tampilkan Label di Gambar", True)
show_stats = st.sidebar.checkbox("Tampilkan Statistik Hasil", True)
st.sidebar.divider()
st.sidebar.caption("Dikembangkan oleh **HISAN** ✨")

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
uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])

if uploaded_file:
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if not os.path.exists(file_path):
        st.error("❌ File tidak ditemukan setelah upload. Coba unggah ulang.")
    else:
        img = Image.open(file_path)
        st.image(img, caption="📸 Gambar Asli", use_container_width=True)

        with st.spinner("🧠 Sedang mendeteksi objek..."):
            start_time = time.time()
            results = model.predict(source=file_path, conf=conf_thres, verbose=False)
            elapsed = time.time() - start_time

        result_image = results[0].plot()
        boxes = results[0].boxes

        col1, col2 = st.columns([1, 1])
        with col2:
            st.image(result_image, caption="🔍 Hasil Deteksi YOLOv8", use_container_width=True)

        with col1:
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.subheader("📊 Statistik Deteksi")
            st.write(f"**Waktu Proses:** {elapsed:.2f} detik")
            st.write(f"**Total Objek:** {len(boxes)}")
            st.markdown("</div>", unsafe_allow_html=True)

    shutil.rmtree(temp_dir, ignore_errors=True)

    # ============================
    # 🎭 Hasil Deteksi
    # ============================
    col1, col2 = st.columns([1, 1])
    with col2:
        st.image(result_image, caption="🔍 Hasil Deteksi YOLOv8", use_container_width=True)

    if show_stats:
        with col1:
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.subheader("📊 Statistik Deteksi")
            st.write(f"**Waktu Proses:** {elapsed:.2f} detik")
            st.write(f"**Total Objek:** {len(boxes)}")

            if len(boxes) > 0:
                data = []
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[cls_id]
                    data.append({"Label": label, "Confidence": round(conf, 2)})
                df = pd.DataFrame(data)
                st.table(df)
                st.bar_chart(df.set_index("Label")["Confidence"])
            st.markdown("</div>", unsafe_allow_html=True)

    shutil.rmtree(temp_dir, ignore_errors=True)
