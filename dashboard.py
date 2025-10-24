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
        "key": "mauve-stringer-jellyfish",
        "labels": ["mauve-stringer-jellyfish.jpg","mauve-stringer.jpg"],
        "title": "Mauve Stringer Jellyfish",
        "desc": "• Bentuk: Memiliki tentakel panjang menyerupai string.\n• Ciri khas: Warna ungu/mauve yang kontras.\n• Ukuran: Tentakel yang panjang dapat mencapai jarak yang signifikan."
    },
    {
        "key": "lions-mane-jellyfish",
        "labels": ["lions-mane-jellyfish.jpg", "lion-mane-jellyfish.jpg","lion's-mane-jellyfish.jpg","lion’s-mane-jellyfish.jpg"],
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

/* Hero card tetap punya gradasi ubur-ubur */
.hero-container {
  position: relative;
  margin: 80px auto;
  width: 80%;
  max-width: 700px;
  background: linear-gradient(135deg, #7a32ff 0%, #44e0ff 100%);
  border-radius: 24px;
  box-shadow: 0 8px 25px rgba(0,0,0,0.4);
  padding: 60px 40px;
  text-align: center;
  overflow: hidden;
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
    
# ---------- GALLERY ----------
def page_gallery():
    st.markdown("<div style='display:flex; justify-content:space-between; align-items:center;'>"
                f"<h2 class='grid-title'>MARI MENGENAL JENIS-JENIS UBUR-UBUR</h2>"
                f"<button class='back-btn' onclick=\"window.location.href='?nav=home'\">Back</button>"
                "</div>", unsafe_allow_html=True)

    # grid 3 columns x 2 rows
    cols = st.columns(3)
    idx = 0
    for r in range(2):
        for c in range(3):
            if idx >= len(SPECIES):
                break
            s = SPECIES[idx]
            with cols[c]:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                if s.get("img_path"):
                    st.image(s["img_path"], caption=None, use_column_width=False, width=160)
                else:
                    st.image("https://via.placeholder.com/160.png?text=No+Image", width=160)
                st.markdown(f"**{s['title']}**")
                # clickable button to open detail
                if st.button("Lihat Detail", key=f"open_{s['key']}"):
                    select_species(s['key'])
                st.markdown("</div>", unsafe_allow_html=True)
            idx += 1

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

# ---------- DETAIL ----------
def page_detail():
    # find selected species dict
    key = st.session_state.selected
    s = next((x for x in SPECIES if x['key'] == key), None)
    if s is None:
        st.error("Spesies tidak ditemukan. Kembali ke Galeri.")
        if st.button("Back to Gallery"):
            nav_to("gallery")
        return

    top_col, _ = st.columns([1,4])
    with top_col:
        if st.button("⬅️ Back", key="back_from_detail"):
            nav_to("gallery")

    left, right = st.columns([1,2])
    with left:
        if s.get("img_path"):
            st.image(s["img_path"], use_column_width=True)
        else:
            st.image("https://via.placeholder.com/380x380.png?text=No+Image", use_column_width=True)
    with right:
        st.markdown(f"### {s['title']}")
        st.markdown(f"<div class='small-muted'>{s.get('desc','Deskripsi belum tersedia.')}</div>", unsafe_allow_html=True)

# ---------- DETECT ----------
def page_detect():
    st.markdown("<div style='display:flex; justify-content:space-between; align-items:center;'>"
                f"<h2 class='grid-title'>Deteksi Jenis Ubur-Ubur</h2>"
                f"<button class='back-btn' onclick=\"window.location.href='?nav=home'\">Back</button>"
                "</div>", unsafe_allow_html=True)

    st.sidebar.header("Pengaturan Deteksi")
    conf = st.sidebar.slider("Confidence Threshold", 0.05, 1.0, st.session_state.conf_thres, 0.05)
    st.session_state.conf_thres = conf

    uploaded_file = st.file_uploader("Unggah gambar untuk deteksi", type=["jpg","jpeg","png"])

    if uploaded_file:
        temp_dir = tempfile.mkdtemp()
        fpath = os.path.join(temp_dir, uploaded_file.name)
        with open(fpath, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if not os.path.exists(fpath):
            st.error("Terjadi kesalahan saat menyimpan file. Coba unggah ulang.")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return

        # Show original image left, detection result right
        col_left, col_right = st.columns([1,1])
        with col_left:
            st.markdown("**Gambar Asli**")
            st.image(fpath, use_column_width=True)

        # If model available, run predict
        if st.session_state.model_loaded:
            try:
                with st.spinner("Menjalankan deteksi..."):
                    t0 = time.time()
                    results = model.predict(source=fpath, conf=conf, verbose=False)
                    elapsed = time.time() - t0
                # plot result (returns np array or PIL)
                result_img = results[0].plot()
                boxes = results[0].boxes
                # prepare stats & table
                data = []
                for b in boxes:
                    # b.cls and b.conf indexing
                    try:
                        cls_id = int(b.cls[0])
                        confv = float(b.conf[0])
                        label = model.names[cls_id] if hasattr(model, "names") else str(cls_id)
                    except Exception:
                        label = "unknown"
                        confv = float(b.conf[0]) if hasattr(b, "conf") else 0.0
                    data.append({"Label": label, "Confidence": round(confv,3)})
                df = pd.DataFrame(data)
                with col_right:
                    st.markdown("**Hasil Deteksi**")
                    st.image(result_img, use_column_width=True)
                # Below: stats and table
                st.markdown("---")
                st.subheader("📊 Statistik Deteksi")
                lefts, rights = st.columns([1,1])
                with lefts:
                    st.write(f"**Waktu Proses:** {elapsed:.2f} detik")
                    st.write(f"**Total Objek:** {len(boxes)}")
                with rights:
                    st.markdown("<div class='table-card'><b>Tabel Deteksi</b></div>", unsafe_allow_html=True)
                    if not df.empty:
                        st.table(df)
                    else:
                        st.info("Tidak ada objek terdeteksi.")
            except Exception as e:
                st.error(f"Gagal saat proses deteksi: {e}")
        else:
            with col_right:
                st.warning("Model YOLO tidak ditemukan / tidak bisa dimuat. Menampilkan placeholder hasil.")
                st.image("https://via.placeholder.com/640x480.png?text=No+model+loaded", use_column_width=True)
                st.info("Letakkan model di model/best.pt untuk mengaktifkan deteksi.")

        shutil.rmtree(temp_dir, ignore_errors=True)
    else:
        st.info("Unggah gambar (.jpg/.png) untuk melakukan deteksi. Jika model belum tersedia, aplikasi tetap menampilkan UI.")

# ---------------------------
# Router (manual via st.session_state and query param fallback)
# ---------------------------
# support simple GET param navigation (back buttons using href hack)
query_params = st.experimental_get_query_params()
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
    page_detect()
else:
    page_home()

# ---------------------------
# Footer credit
# ---------------------------
st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; color:#bfc9d3'>Aplikasi edukasi & demo deteksi ubur-ubur — dibuat oleh HISAN</div>", unsafe_allow_html=True)
