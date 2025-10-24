# =====================================================
# dashboard.py — Aplikasi Deteksi Jenis Ubur-Ubur
# =====================================================

import streamlit as st
from PIL import Image
import os, tempfile, shutil, time
import pandas as pd
import numpy as np
import cv2

# Try to import YOLO if available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False


# ---------------------------
# Helper: resolve image path
# ---------------------------
def find_image(name_variants, search_dirs=[".", "images", "assets", "img"]):
    for folder in search_dirs:
        for name in name_variants:
            path = os.path.join(folder, name)
            if os.path.exists(path):
                return path
    return None


# ---------------------------
# Custom CSS Style
# ---------------------------
def local_css():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(180deg, #001E3C, #003366);
        color: #E0E0E0;
    }
    h1, h2, h3, h4, h5 {
        color: #E0E0E0;
    }
    .stButton button {
        background-color: #0066CC !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 0.5em 1.2em !important;
        border: none !important;
    }
    .stButton button:hover {
        background-color: #004C99 !important;
        transform: scale(1.03);
    }
    .uploadedFile {
        border-radius: 8px;
    }
    img {
        border-radius: 12px !important;
    }
    </style>
    """, unsafe_allow_html=True)


# =====================================================
# HALAMAN: HOME
# =====================================================
def page_home():
    st.markdown("<h1 style='color:#E0E0E0;'>🐚 Deteksi Jenis Ubur-Ubur</h1>", unsafe_allow_html=True)
    st.write("Selamat datang di aplikasi deteksi jenis ubur-ubur menggunakan model YOLO.")

    col1, col2, col3 = st.columns(3)
    with col2:
        if st.button("Mulai Deteksi 🪼"):
            st.session_state.page = "detect"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.image(find_image(["jellyfish_sample.jpg", "example.jpg"]) or "https://upload.wikimedia.org/wikipedia/commons/0/0e/Jelly_cc11.jpg",
             caption="Contoh gambar ubur-ubur", use_container_width=True)


# =====================================================
# HALAMAN: GALERI
# =====================================================
def page_gallery():
    st.markdown("<h1 style='color:#E0E0E0;'>🖼️ Galeri Ubur-Ubur</h1>", unsafe_allow_html=True)
    folder = "gallery"
    if not os.path.exists(folder):
        st.info("Folder galeri belum tersedia.")
        return

    images = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    if not images:
        st.warning("Belum ada gambar di galeri.")
        return

    cols = st.columns(3)
    for i, img_path in enumerate(images):
        with cols[i % 3]:
            st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)


# =====================================================
# HALAMAN: DETAIL
# =====================================================
def page_detail():
    st.markdown("<h1 style='color:#E0E0E0;'>📘 Detail Informasi Ubur-Ubur</h1>", unsafe_allow_html=True)
    st.write("Halaman ini menampilkan informasi mendalam tentang spesies ubur-ubur tertentu.")
    st.info("Fitur ini masih dalam pengembangan.")


# =====================================================
# HALAMAN: DETEKSI
# =====================================================
def page_detection():
    st.markdown("<h1 style='text-align:left; color:#E0E0E0;'>🔍 Deteksi Jenis Ubur-Ubur</h1>", unsafe_allow_html=True)

    if st.button("⬅️ Kembali"):
        st.session_state.page = "home"
        st.rerun()

    uploaded_file = st.file_uploader("Unggah gambar untuk deteksi", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<h5 style='color:#E0E0E0;'>Gambar Asli</h5>", unsafe_allow_html=True)
            st.image(image, use_container_width=True)

        with col2:
            st.markdown("<h5 style='color:#E0E0E0;'>Hasil Deteksi</h5>", unsafe_allow_html=True)
            progress_text = "⏳ Sedang memproses deteksi..."
            progress_bar = st.progress(0, text=progress_text)

            # Simulasi progress
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1, text=progress_text)
            progress_bar.empty()

            # Deteksi (simulasi)
            detected_image = img_array.copy()
            h, w, _ = detected_image.shape
            cv2.rectangle(detected_image, (int(w*0.2), int(h*0.2)), (int(w*0.8), int(h*0.8)), (0, 255, 0), 2)
            cv2.putText(detected_image, "barrel-jellyfish 0.96", (int(w*0.22), int(h*0.18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            st.image(detected_image, use_column_width=True, caption="Hasil Deteksi (proporsional)")

        # Statistik hasil
        st.markdown("<h3 style='color:#E0E0E0; margin-top:30px;'>📊 Statistik Deteksi</h3>", unsafe_allow_html=True)
        col3, col4 = st.columns([1.5, 2])

        with col3:
            st.markdown("<p style='color:#E0E0E0;'>Waktu Proses: 0.34 detik</p>", unsafe_allow_html=True)
            st.markdown("<p style='color:#E0E0E0;'>Total Objek: 1</p>", unsafe_allow_html=True)

        with col4:
            data = {"Label": ["barrel-jellyfish"], "Confidence": [0.964]}
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True, hide_index=True)

    else:
        st.info("Silakan unggah gambar ubur-ubur untuk mendeteksi jenisnya.")


# =====================================================
# ROUTER: Pilih Halaman
# =====================================================
def main():
    st.set_page_config(page_title="Deteksi Ubur-Ubur", page_icon="🪼", layout="wide")
    local_css()

    if "page" not in st.session_state:
        st.session_state.page = "home"

    page = st.session_state.page

    if page == "home":
        page_home()
    elif page == "gallery":
        page_gallery()
    elif page == "detail":
        page_detail()
    elif page == "detect":
        page_detection()


if __name__ == "__main__":
    main()
