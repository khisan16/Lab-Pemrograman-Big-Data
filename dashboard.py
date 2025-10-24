# dashboard.py
import streamlit as st
from PIL import Image
import os, tempfile, shutil, time
import pandas as pd

# Try to import YOLO if available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

# ---------------------------
# Helper: resolve image path
# ---------------------------
def find_image(name_variants, search_dirs=[".","images","assets","img"]):
    for d in search_dirs:
        for v in name_variants:
            p = os.path.join(d, v)
            if os.path.exists(p):
                return p
    return None

# ---------------------------
# Data: species + possible filenames + description
# ---------------------------
SPECIES = [
    {
        "key": "moon-jellyfish",
        "labels": ["moon-jellyfish.jpg", "moon-jellyfish.jpeg", "moon_jellyfish.jpg"],
        "title": "Moon Jellyfish",
        "desc": """• Bentuk dan warna: Tubuh transparan berbentuk lonceng pipih. 
• Organ reproduksi: Gonad berbentuk empat cincin yang terlihat. 
• Tentakel: Pendek dan halus, tidak berbahaya untuk manusia umumnya.
• Ukuran: Diameter bisa mencapai ~30 cm."""
    },
    {
        "key": "blue-jellyfish",
        "labels": ["blue-jellyfish.jpg", "blue_jellyfish.jpg"],
        "title": "Blue Jellyfish",
        "desc": "• Bentuk: Lonjong dengan rona kebiruan.\n• Tentakel: Lebih panjang dari moon jellyfish.\n• Ciri khas: Warna biru yang dominan."
    },
    {
        "key": "mauve-stinger-jellyfish",
        "labels": ["mauve-stinger-jellyfish.jpg","mauve-stinger-jellyfish.jpg"],
        "title": "Mauve Stinger Jellyfish",
        "desc": "• Bentuk: Memiliki tentakel panjang menyerupai string.\n• Ciri khas: Warna ungu/mauve yang kontras.\n• Ukuran: Tentakel yang panjang dapat mencapai jarak yang signifikan."
    },
    {
        "key": "lions-mane-jellyfish",
        "labels": ["lions-mane-jellyfish.jpg", "lions-mane-jellyfish.jpg","lion's-mane-jellyfish.jpg","lion’s-mane-jellyfish.jpg"],
        "title": "Lion's Mane Jellyfish",
        "desc": "• Bentuk: Lonceng besar dengan tentakel sangat panjang.\n• Ciri khas: Mirip surai singa (banyak tentakel).\n• Potensi bahaya: Bisa menyebabkan sengatan menyakitkan."
    },
    {
        "key": "compass-jellyfish",
        "labels": ["compass-jellyfish.jpg", "compass_jellyfish.jpg"],
        "title": "Compass Jellyfish",
        "desc": "• Ciri khas: Pola seperti kompas di loncengnya.\n• Tentakel: Rata-rata panjang, bentuk khas pada permukaan."
    },
    {
        "key": "barrel-jellyfish",
        "labels": ["barrel-jellyfish.jpg", "barrel_jellyfish.jpg"],
        "title": "Barrel Jellyfish",
        "desc": "• Bentuk: Lonceng besar, menyerupai barel/ember.\n• Ukuran: Dapat menjadi relatif besar dibanding jenis lain."
    }
]

# Pre-resolve image paths (best-effort)
for s in SPECIES:
    s["img_path"] = find_image(s["labels"])  # can be None if not found

# ---------------------------
# Page state & nav helpers
# ---------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"
if "selected" not in st.session_state:
    st.session_state.selected = None
if "conf_thres" not in st.session_state:
    st.session_state.conf_thres = 0.25

def nav_to(p):
    st.session_state.page = p

def select_species(key):
    st.session_state.selected = key
    st.session_state.page = "detail"

# ---------------------------
# CSS / Styling
# ---------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');

html, body, [class*="css"]  {
  font-family: 'Poppins', sans-serif !important;
}

/* Gaya umum latar belakang laut */
body {
  background: linear-gradient(180deg, #0d1b4a 0%, #102a7a 40%, #1b8fbf 100%);
  color: #f5f6fa;
  background-attachment: fixed;
}

/* Hapus warna putih default kontainer Streamlit */
section.main, .stApp {
  background: transparent !important;
}

.stImage img {
        height: 220px !important;
        object-fit: cover;
        border-radius: 15px;
    }

.hero-box {
  background: linear-gradient(135deg, #7a32ff 0%, #44e0ff 100%);
  border-radius: 24px;
  box-shadow: 0 8px 25px rgba(0,0,0,0.4);
  padding: 40px 20px;
  text-align: center;
  margin-bottom: 40px;
}

.hero-title {
  font-size: 36px;
  font-weight: 800;
  margin-bottom: 8px;
  color: #ffffff;
}

.hero-sub {
  font-size: 22px;
  font-weight: 600;
  color: #eaf9ff;
}

.hero-title {
  font-size: 36px;
  font-weight: 800;
  margin-bottom: 8px;
  color: #ffffff;
}

.hero-sub {
  font-size: 22px;
  font-weight: 600;
  margin-bottom: 30px;
  color: #eaf9ff;
}

.hero-btn {
  display: inline-block;
  margin: 8px;
  padding: 14px 26px;
  border: none;
  border-radius: 30px;
  background: rgba(255,255,255,0.2);
  color: white;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  transition: all 0.3s ease;
  cursor: pointer;
}
.hero-btn:hover {
  background: white;
  color: #4e00c2;
  transform: translateY(-2px);
}

.jelly-icon {
  position: absolute;
  top: 20px;
  right: 20px;
  width: 100px;
  height: auto;
}

/* Tambahan efek halus */
.stButton>button {
  border-radius: 20px;
  background: rgba(255,255,255,0.1);
  color: white;
  border: 1px solid rgba(255,255,255,0.3);
  transition: 0.3s ease;
}
.stButton>button:hover {
  background: rgba(255,255,255,0.3);
  color: #1b003f;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Load model (once)
# ---------------------------
model = None
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

if YOLO_AVAILABLE and not st.session_state.model_loaded:
    try:
        # Attempt to load model from model/best.pt (common location in repo)
        model_path_candidates = ["model/best.pt", "best.pt", "models/best.pt"]
        found = None
        for p in model_path_candidates:
            if os.path.exists(p):
                found = p
                break
        if found:
            model = YOLO(found)
            st.session_state.model_loaded = True
            st.session_state.model_path = found
        else:
            st.session_state.model_loaded = False
            st.session_state.model_path = None
    except Exception as e:
        st.session_state.model_loaded = False
        st.session_state.model_error = str(e)

# If already loaded earlier in session, reuse
if st.session_state.model_loaded:
    try:
        if model is None:
            model = YOLO(st.session_state.model_path)
    except Exception:
        pass

# ---------------------------
# ---------- PAGES ----------
# ---------------------------

# ---------- HOME ----------
def page_home():
    st.markdown("<div class='hero-container'>", unsafe_allow_html=True)

    # Ubur-ubur lucu online (ikon transparan)
    st.markdown("<img src='https://cdn-icons-png.flaticon.com/512/616/616408.png' class='jelly-icon'>", unsafe_allow_html=True)

    # Judul di dalam kotak gradasi
    st.markdown("""
    <div class='hero-box'>
        <div class='hero-title'>HALO! SELAMAT DATANG</div>
        <div class='hero-sub'>MAU NGAPAIN NIH?</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("MENGENAL JENIS-JENIS UBUR-UBUR", key="btn_gallery", use_container_width=True):
            nav_to("gallery")
        if st.button("DETEKSI JENIS UBUR-UBUR", key="btn_detect", use_container_width=True):
            nav_to("detect")

    st.markdown("</div>", unsafe_allow_html=True)
    
# ---------- GALERI ----------
def page_gallery():
    st.markdown(
        "<h2 style='text-align:center; color:#ffffff; margin-bottom:20px;'>MARI MENGENAL JENIS-JENIS UBUR-UBUR</h2>",
        unsafe_allow_html=True,
    )

    # Tombol Back
    back_col, _ = st.columns([1, 6])
    with back_col:
        st.button("⬅️ Back", key="back_to_home", on_click=nav_to, args=("home",))

    # Grid 3 kolom
    cols = st.columns(3)
    for idx, s in enumerate(SPECIES):
        col = cols[idx % 3]
        with col:
            st.markdown("<div style='text-align:center; padding:12px;'>", unsafe_allow_html=True)

            img_src = s.get("img_path") or "https://via.placeholder.com/300x300.png?text=No+Image"
            st.image(img_src, use_container_width=False, width=220)

            # 🔹 Nama gambar — sedikit lebih kecil dari sebelumnya
            st.markdown(
                f"""
                <h4 style="
                    color:#eaf9ff;
                    margin-top:10px;
                    font-family:Poppins, sans-serif;
                    font-size:17px;
                    font-weight:600;
                ">
                    {s['title']}
                </h4>
                """,
                unsafe_allow_html=True,
            )

            # Tombol lihat detail (tetap di tengah)
            st.button(
                "Lihat Detail",
                key=f"btn_{s['key']}",
                on_click=select_species,
                args=(s['key'],)
            )

            st.markdown("</div>", unsafe_allow_html=True)

# ---------- DETAIL ----------
def page_detail():
    # find selected species dict
    key = st.session_state.selected
    s = next((x for x in SPECIES if x['key'] == key), None)
    if s is None:
        st.error("Spesies tidak ditemukan. Kembali ke Galeri.")
        st.button("⬅️ Kembali ke Galeri", key="back_to_gallery_missing", on_click=nav_to, args=("gallery",))
        return

    # Tombol kembali (kiri atas) — on_click lebih andal
    st.button("⬅️ Back", key="back_from_detail", on_click=nav_to, args=("gallery",))

    # layout dua kolom: gambar di kiri, teks di kanan
    left, right = st.columns([1, 2])
    with left:
        img_src = s.get("img_path") or "https://via.placeholder.com/380x380.png?text=No+Image"
        st.image(img_src, use_container_width=False, width=300)

    with right:
        # Judul cerah
        st.markdown(f"<h2 style='color:#ffffff; margin-bottom:6px;'>{s['title']}</h2>", unsafe_allow_html=True)

        # Ubah deskripsi menjadi list poin ke bawah
        raw_desc = s.get('desc', '')
        # split by bullet marker; dukung format "• " atau newline
        if "•" in raw_desc:
            parts = [p.strip() for p in raw_desc.split("•") if p.strip()]
        else:
            parts = [p.strip() for p in raw_desc.split("\n") if p.strip()]

        if parts:
            list_items = "".join([f"<li style='margin-bottom:8px; color:#eaf9ff;'>{p}</li>" for p in parts])
            st.markdown(f"<ul style='padding-left:18px;'>{list_items}</ul>", unsafe_allow_html=True)
        else:
            st.write(s.get('desc', 'Deskripsi belum tersedia.'))

def page_detection():
    import cv2
    import os
    import time
    import pandas as pd
    from PIL import Image
    import streamlit as st

    st.markdown(
        "<h2 style='text-align:center; color:#eaf9ff; margin-bottom:20px;'>Deteksi Jenis Ubur-Ubur</h2>",
        unsafe_allow_html=True,
    )

    # Tombol Back (berfungsi kembali ke home)
    back_col, _ = st.columns([1, 5])
    with back_col:
        st.button("⬅️ Back", key="back_from_detection", on_click=nav_to, args=("home",))

    uploaded_file = st.file_uploader(
        "Unggah gambar untuk deteksi",
        type=["jpg", "jpeg", "png"],
        help="Limit 200MB per file • JPG, JPEG, PNG"
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        temp_path = os.path.join("temp", uploaded_file.name)
        os.makedirs("temp", exist_ok=True)
        image.save(temp_path)

        # Proses deteksi YOLO
        start_time = time.time()
        results = model.predict(temp_path, conf=0.5, imgsz=640, verbose=False)
        elapsed_time = time.time() - start_time

        # Ambil hasil
        annotated_frame = results[0].plot(line_width=2, font_size=16)
        detected_pil = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        detected_pil = detected_pil.resize(image.size)  # samakan ukuran hasil dengan input

        labels = results[0].boxes.cls
        confs = results[0].boxes.conf

        # Tampilkan hasil
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h4 style='color:#f2faff;'>Gambar Asli</h4>", unsafe_allow_html=True)
            st.image(image, width="stretch", clamp=True)
        with col2:
            st.markdown("<h4 style='color:#f2faff;'>Hasil Deteksi</h4>", unsafe_allow_html=True)
            st.image(detected_pil, width="stretch", clamp=True)

        # Statistik & Tabel sejajar
        st.markdown(
            "<h3 style='color:#d7f3ff; margin-top:25px;'>📊 Statistik Deteksi</h3>",
            unsafe_allow_html=True
        )

        stat_col, table_col = st.columns([1.2, 1.8])
        with stat_col:
            st.markdown(f"<p style='color:#ffffff;'>Waktu Proses: {elapsed_time:.2f} detik</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:#ffffff;'>Total Objek: {len(labels)}</p>", unsafe_allow_html=True)

        with table_col:
            st.markdown("<p style='color:#ffffff; margin-bottom:5px;'>Tabel Deteksi</p>", unsafe_allow_html=True)
            if len(labels) > 0:
                df = pd.DataFrame({
                    "Label": [model.names[int(l)] for l in labels],
                    "Confidence": [f"{float(c):.2f}" for c in confs]
                })
            else:
                df = pd.DataFrame({"Label": ["Tidak ada deteksi"], "Confidence": ["-"]})
            st.dataframe(df, width="stretch")
            
# ---------------------------
# Router (manual via st.session_state and query param fallback)
# ---------------------------
# support simple GET param navigation (back buttons using href hack)
query_params = st.query_params
if "nav" in query_params:
    q = query_params["nav"][0]
    if q == "home":
        st.session_state.page = "home"
    elif q == "gallery":
        st.session_state.page = "gallery"
    elif q == "detect":
        st.session_state.page = "detect"

# Render header nav bar
nav1, nav2, nav3 = st.columns([1,1,1])
with nav1:
    if st.button("🏠 Home"):
        nav_to("home")
with nav2:
    if st.button("📚 Mengenal Jenis"):
        nav_to("gallery")
with nav3:
    if st.button("🔍 Deteksi"):
        nav_to("detect")

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# Page switch
if st.session_state.page == "home":
    page_home()
elif st.session_state.page == "gallery":
    page_gallery()
elif st.session_state.page == "detail":
    page_detail()
elif st.session_state.page == "detect":
    page_detection()
else:
    page_home()

# ---------------------------
# Footer credit
# ---------------------------
st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; color:#bfc9d3'>Aplikasi edukasi & deteksi jenis ubur-ubur — dibuat oleh HISAN</div>", unsafe_allow_html=True) 
