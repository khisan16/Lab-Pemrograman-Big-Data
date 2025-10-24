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
# Data: species + description
# ---------------------------
SPECIES = [
    {
        "key": "moon-jellyfish",
        "labels": ["moon-jellyfish.jpg", "moon_jellyfish.jpg"],
        "title": "Moon Jellyfish",
        "desc": """• Bentuk & warna: Tubuh transparan berbentuk lonceng pipih.
• Organ reproduksi: Gonad berbentuk empat cincin terlihat jelas.
• Tentakel: Pendek & halus, tidak berbahaya untuk manusia.
• Ukuran: Diameter mencapai ±30 cm.""",
        "fallback": "https://upload.wikimedia.org/wikipedia/commons/0/0b/Aurelia_aurita_%28Moon_jellyfish%29.jpg"
    },
    {
        "key": "blue-jellyfish",
        "labels": ["blue-jellyfish.jpg", "blue_jellyfish.jpg"],
        "title": "Blue Blubber Jellyfish",
        "desc": "• Bentuk: Lonjong dengan rona kebiruan.\n• Tentakel: Lebih panjang dari Moon Jellyfish.\n• Ciri khas: Warna biru dominan.",
        "fallback": "https://upload.wikimedia.org/wikipedia/commons/b/b0/Catostylus_mosaicus_-_Blue_blubber_jellyfish.jpg"
    },
    {
        "key": "mauve-stringer-jellyfish",
        "labels": ["mauve-stringer-jellyfish.jpg","mauve_stringer_jellyfish.jpg"],
        "title": "Mauve Stinger Jellyfish",
        "desc": "• Tentakel panjang menyerupai tali.\n• Warna ungu muda yang indah.\n• Mampu memancarkan cahaya di malam hari.",
        "fallback": "https://upload.wikimedia.org/wikipedia/commons/2/20/Pelagia_noctiluca_%28Mauve_Stinger%29.jpg"
    },
    {
        "key": "lions-mane-jellyfish",
        "labels": ["lions-mane-jellyfish.jpg","lion's-mane-jellyfish.jpg"],
        "title": "Lion’s Mane Jellyfish",
        "desc": "• Bentuk: Lonceng besar dengan tentakel panjang.\n• Ciri khas: Mirip surai singa.\n• Bahaya: Dapat menyebabkan sengatan menyakitkan.",
        "fallback": "https://upload.wikimedia.org/wikipedia/commons/e/e8/Cyanea_capillata_%28Lion%27s_mane_jellyfish%29.jpg"
    },
    {
        "key": "compass-jellyfish",
        "labels": ["compass-jellyfish.jpg", "compass_jellyfish.jpg"],
        "title": "Compass Jellyfish",
        "desc": "• Ciri khas: Pola seperti kompas di loncengnya.\n• Tentakel: Panjang dengan ujung khas.",
        "fallback": "https://upload.wikimedia.org/wikipedia/commons/5/5f/Chrysaora_hysoscella_%28Compass_jellyfish%29.jpg"
    },
    {
        "key": "barrel-jellyfish",
        "labels": ["barrel-jellyfish.jpg", "barrel_jellyfish.jpg"],
        "title": "Barrel Jellyfish",
        "desc": "• Bentuk: Lonceng besar menyerupai barel.\n• Ukuran: Salah satu ubur-ubur terbesar di perairan Eropa.",
        "fallback": "https://upload.wikimedia.org/wikipedia/commons/a/a5/Rhizostoma_pulmo_%28Barrel_jellyfish%29.jpg"
    }
]

# Pre-resolve image paths (best-effort)
for s in SPECIES:
    s["img_path"] = find_image(s["labels"]) or s["fallback"]

# ---------------------------
# Page state
# ---------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"
if "selected" not in st.session_state:
    st.session_state.selected = None
if "conf_thres" not in st.session_state:
    st.session_state.conf_thres = 0.25

def nav_to(p): st.session_state.page = p
def select_species(k): st.session_state.selected = k; st.session_state.page = "detail"

# ---------------------------
# CSS Styling
# ---------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif !important; }
.stApp { background: linear-gradient(180deg,#0d1b4a,#107dac); color:white; }
.stImage img { height:220px !important; width:100% !important; object-fit:cover; border-radius:16px; box-shadow:0 4px 10px rgba(0,0,0,0.4); }
.hero-box { background:linear-gradient(135deg,#7a32ff 0%,#44e0ff 100%); border-radius:20px; padding:35px 20px; text-align:center; margin-bottom:30px; }
.hero-title { font-size:34px; font-weight:800; margin:0; }
.hero-sub { font-size:20px; font-weight:600; margin-top:10px; color:#eaf9ff; }
.stButton>button { border-radius:12px; border:none; background:rgba(255,255,255,0.2); color:white; padding:12px; font-weight:600; transition:0.3s; }
.stButton>button:hover { background:white; color:#4e00c2; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# PAGE: HOME
# ---------------------------
def page_home():
    st.markdown("<img src='https://cdn-icons-png.flaticon.com/512/616/616408.png' width='90' style='position:absolute; right:40px; top:40px;'>", unsafe_allow_html=True)
    st.markdown("""
    <div class='hero-box'>
        <div class='hero-title'>HALO! SELAMAT DATANG</div>
        <div class='hero-sub'>MAU NGAPAIN NIH?</div>
    </div>
    """, unsafe_allow_html=True)
    col = st.columns(1)[0]
    with col:
        if st.button("📚 Mengenal Jenis Ubur-Ubur", use_container_width=True):
            nav_to("gallery")
        if st.button("🔍 Deteksi Jenis Ubur-Ubur", use_container_width=True):
            nav_to("detect")

# ---------------------------
# PAGE: GALLERY
# ---------------------------
def page_gallery():
    st.markdown("## 🪼 MARI MENGENAL JENIS-JENIS UBUR-UBUR 🪼")
    cols = st.columns(3)
    for i, s in enumerate(SPECIES):
        with cols[i % 3]:
            st.image(s["img_path"], use_container_width=True)
            st.markdown(f"### {s['title']}")
            st.write(s["desc"])
            if st.button("Lihat Detail", key=s["key"]):
                select_species(s["key"])
    if st.button("⬅️ Kembali ke Beranda"):
        nav_to("home")

# ---------------------------
# PAGE: DETAIL
# ---------------------------
def page_detail():
    s = next((x for x in SPECIES if x["key"] == st.session_state.selected), None)
    if not s:
        st.error("Spesies tidak ditemukan.")
        if st.button("Kembali"):
            nav_to("gallery")
        return
    st.button("⬅️ Kembali", on_click=lambda: nav_to("gallery"))
    col1, col2 = st.columns([1,2])
    with col1:
        st.image(s["img_path"], use_container_width=True)
    with col2:
        st.markdown(f"## {s['title']}")
        st.write(s["desc"])

# ---------------------------
# PAGE: DETECTION (same as before)
# ---------------------------
def page_detect():
    st.markdown("## 🔍 Deteksi Jenis Ubur-Ubur")
    st.sidebar.header("Pengaturan Deteksi")
    conf = st.sidebar.slider("Confidence Threshold", 0.05, 1.0, st.session_state.conf_thres, 0.05)
    st.session_state.conf_thres = conf

    uploaded = st.file_uploader("Unggah gambar ubur-ubur", type=["jpg","jpeg","png"])
    if uploaded:
        temp = tempfile.mkdtemp()
