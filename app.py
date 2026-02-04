# app.py
import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from datetime import datetime
import re
import numpy as np
import json
import glob

st.set_page_config(
    page_title="Status Tera Ulang Pasar – Kab. Tangerang",
    page_icon="🏪",
    layout="wide"
)

st.title("🏪 Status Tera Ulang Timbangan Pasar – Kabupaten Tangerang")
with st.sidebar:
    st.markdown("## 📌 Pilih Dashboard")
    page = st.radio(
        "Menu",
        ["🏪 Pasar (Tera Ulang)", "⛽ SPBU"],
        index=0,
        label_visibility="collapsed"
    )
    st.markdown("---")

st.caption("Dinas Perindustrian dan Perdagangan • Bidang Kemetrologian")

# =========================
# FILE UTAMA (SEFOLDER)
# =========================
FILE_EXCEL = "DATA_DASHBOARD_PASAR.xlsx"          # <-- sesuai nama Dayu
FILE_GEOJSON = "batas_kecamatan_tangerang.geojson"  # <-- sesuaikan kalau beda

# =========================
# UTIL
# =========================
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())

def parse_coord(val):
    """Parse koordinat dari berbagai format"""
    try:
        if pd.isna(val) or val == "":
            return np.nan, np.nan

        s = str(val).strip()
        # Handle format: "-6.26435, 106.42592"
        if ',' in s:
            parts = [p.strip() for p in s.split(',')]
            if len(parts) >= 2:
                lat = float(parts[0])
                lon = float(parts[1])
                # Auto-swap jika format terbalik (lon, lat)
                if abs(lat) > 90 and abs(lon) <= 90:
                    lat, lon = lon, lat
                return lat, lon

        # fallback: ambil 2 angka pertama
        nums = re.findall(r"-?\d+(?:\.\d+)?", s)
        if len(nums) >= 2:
            lat = float(nums[0]); lon = float(nums[1])
            if abs(lat) > 90 and abs(lon) <= 90:
                lat, lon = lon, lat
            return lat, lon

    except Exception as e:
        st.warning(f"Gagal parse koordinat: {val}, error: {e}")
    return np.nan, np.nan

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    # strip nama kolom biar aman dari spasi
    df.columns = [str(c).strip() for c in df.columns]

    rename_mapping = {
        'Nama Pasar': 'nama_pasar',
        'Alamat': 'alamat',
        'Kecamatan': 'kecamatan',
        'Koordinat': 'koordinat',
        'Tahun Tera Ulang': 'tera_ulang_tahun',
        'Total UTTP': 'jumlah_timbangan_tera_ulang',
        'Total Pedagang': 'total_pedagang'
    }

    existing_rename = {k: v for k, v in rename_mapping.items() if k in df.columns}
    df = df.rename(columns=existing_rename)

    if 'koordinat' in df.columns:
        coords = df['koordinat'].apply(parse_coord)
        df['lat'] = coords.apply(lambda x: x[0])
        df['lon'] = coords.apply(lambda x: x[1])

    # tambahkan Dacin juga (data Dayu ada)
    timbangan_cols = [
        'Timb. Pegas', 'Timb. Meja', 'Timb. Elektronik',
        'Timb. Sentisimal', 'Timb. Bobot Ingsut', 'Neraca', 'Dacin'
    ]
    available_timbangan_cols = [col for col in timbangan_cols if col in df.columns]

    if available_timbangan_cols:
        def summarize_timbangan(row):
            parts = []
            for col in available_timbangan_cols:
                try:
                    val = pd.to_numeric(row[col], errors='coerce')
                    if pd.notna(val) and val > 0:
                        label = col.replace('Timb. ', '').replace('Timb.', '').strip()
                        parts.append(f"{label}: {int(val)}")
                except Exception:
                    continue
            return "; ".join(parts) if parts else "Tidak ada data"

        df['jenis_timbangan'] = df.apply(summarize_timbangan, axis=1)

    # norm untuk join ke geojson
    if 'kecamatan' in df.columns:
        df['kec_norm'] = df['kecamatan'].apply(_norm)
    if 'nama_pasar' in df.columns:
        df['pasar_norm'] = df['nama_pasar'].apply(_norm)

    return df

def create_sample_data():
    """Buat data sample jika file asli bermasalah"""
    st.warning("Membuat data sample karena file asli bermasalah")

    sample_data = {
        'nama_pasar': ['Cisoka', 'Curug', 'Mauk', 'Cikupa', 'Pasar Kemis'],
        'kecamatan': ['Cisoka', 'Curug', 'Mauk', 'Cikupa', 'Pasar Kemis'],
        'alamat': [
            'Jl. Ps. Cisoka No.44, Cisoka, Kec. Cisoka, Kabupaten Tangerang, Banten 15730',
            'Jl. Raya Curug, Curug Wetan, Kec. Curug, Kabupaten Tangerang, Banten 15810',
            'East Mauk, Mauk, Tangerang Regency, Banten 15530',
            'Jl. Raya Serang, Cikupa, Kec. Cikupa, Kabupaten Tangerang, Banten 15710',
            'RGPJ+FJX, Jalan Raya, Sukaasih, Pasar Kemis, Tangerang Regency, Banten 15560'
        ],
        'lat': [-6.26435, -6.26100, -6.06044, -6.22907, -6.16365],
        'lon': [106.42592, 106.55858, 106.51129, 106.51981, 106.53155],
        'tera_ulang_tahun': [2025, 2025, 2025, 2025, 2025],
        'jumlah_timbangan_tera_ulang': [195, 251, 161, 257, 174],
        'jenis_timbangan': [
            'Pegas: 77; Meja: 30; Elektronik: 87',
            'Pegas: 60; Meja: 76; Elektronik: 107',
            'Pegas: 80; Meja: 10; Elektronik: 71',
            'Pegas: 36; Meja: 88; Elektronik: 130',
            'Pegas: 54; Meja: 48; Elektronik: 72'
        ]
    }
    df = pd.DataFrame(sample_data)
    df['kec_norm'] = df['kecamatan'].apply(_norm)
    df['pasar_norm'] = df['nama_pasar'].apply(_norm)
    return df

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Pastikan tipe data konsisten"""
    if 'tera_ulang_tahun' in df.columns:
        df['tera_ulang_tahun'] = pd.to_numeric(df['tera_ulang_tahun'], errors='coerce').fillna(0).astype(int)

    if 'jumlah_timbangan_tera_ulang' in df.columns:
        df['jumlah_timbangan_tera_ulang'] = pd.to_numeric(df['jumlah_timbangan_tera_ulang'], errors='coerce').fillna(0).astype(int)

    if 'total_pedagang' in df.columns:
        df['total_pedagang'] = pd.to_numeric(df['total_pedagang'], errors='coerce').fillna(0).astype(int)

    for col in ['nama_pasar', 'alamat', 'kecamatan', 'jenis_timbangan']:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    for c in ['lat', 'lon']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    return df

def clean_str_series(series: pd.Series) -> pd.Series:
    """Bersihkan series string"""
    if series is None:
        return pd.Series([], dtype=str)
    s = series.astype(str).str.strip()
    s = s.str.title()
    mask_bad = s.str.lower().isin(["", "nan", "none", "null", "na", "n/a", "-", "--"])
    return s[~mask_bad]

def uniq_clean(series: pd.Series) -> list:
    """Ambil nilai unik yang sudah dibersihkan"""
    return sorted(clean_str_series(series).unique().tolist())

def marker_color(year: int, selected_year: int):
    """Warna marker relatif terhadap tahun yang dipilih (selected_year)"""
    if year is None or year == 0:
        return "gray"
    if year == selected_year:
        return "green"
    elif year == selected_year - 1:
        return "orange"
    else:
        return "red"


@st.cache_data
def load_excel(path_like: str):
    """
    Load Excel sefolder. Jika user kasih tanpa ekstensi, auto-detect.
    """
    try:
        # auto-detect kalau user menulis "DATA_DASHBOARD_PASAR" tanpa ekstensi
        if "." not in path_like:
            matches = glob.glob(path_like + ".*")
            if not matches:
                raise FileNotFoundError(f"File {path_like}.* tidak ditemukan sefolder app.py")
            path = matches[0]
        else:
            path = path_like

        df = pd.read_excel(path, sheet_name=0, engine="openpyxl")
        df = standardize_columns(df)
        return df, None

    except Exception as e:
        st.error(f"❌ Error loading Excel: {e}")
        return create_sample_data(), "Menggunakan data sample"

@st.cache_data
def load_geojson(path: str):
    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    # Tambahkan kec_norm dari field wadmkc (sesuai geojson Dayu)
    for ft in gj.get("features", []):
        props = ft.get("properties", {})
        wadmkc = props.get("wadmkc", "")  # INI YANG DIPAKAI
        props["kec_norm"] = _norm(wadmkc)
        props["kec_label"] = wadmkc
        ft["properties"] = props

    return gj
def render_dashboard_pasar():
    # >>> PASTE SELURUH KODE DASHBOARD PASAR DI SINI <<<
    # tips: semua key session_state gunakan prefix "pasar_"
    pass


def render_dashboard_spbu():
    # >>> PASTE SELURUH KODE DASHBOARD SPBU DI SINI <<<
    # tips: semua key session_state gunakan prefix "spbu_"
    pass
# 1) IMPORT
import streamlit as st
import pandas as pd
...

st.set_page_config(...)

# 2) MENU (di sidebar, taruh dekat atas)
with st.sidebar:
    st.markdown("## 📌 Pilih Dashboard")
    page = st.radio("Menu", ["🏪 Pasar (Tera Ulang)", "⛽ SPBU"], index=0, label_visibility="collapsed")
    st.markdown("---")

# 3) HELPER FUNCTIONS
def parse_coord(...):
    ...

def uniq_clean(...):
    ...

# 4) DASHBOARD FUNCTIONS
def render_dashboard_pasar():
    # seluruh kode dashboard pasar kamu dipindah ke sini
    ...

def render_dashboard_spbu():
    # seluruh kode dashboard spbu kamu dipindah ke sini
    ...

# 5) ✅ POIN 3 ADA DI SINI (PALING BAWAH FILE)
if page == "🏪 Pasar (Tera Ulang)":
    render_dashboard_pasar()
else:
    render_dashboard_spbu()


# =========================
# MAIN APP
# =========================
# Baca Excel utama
df, err = load_excel("DATA_DASHBOARD_PASAR.xlsx")
if err:
    st.warning(f"Peringatan: {err}")

df = coerce_types(df)


# SIDEBAR FILTERS (TANPA RADIO MODE, TETAP FLEKSIBEL)
# =========================
with st.sidebar:
    st.header("Filter")

    # --- Pilih tahun (dropdown) default = terbaru ---
    if 'tera_ulang_tahun' in df.columns and df['tera_ulang_tahun'].notna().any():
        years = sorted(df['tera_ulang_tahun'].dropna().astype(int).unique().tolist())
        year_options = years[::-1]  # terbaru di atas
        year_pick = st.selectbox("Tahun Tera Ulang", options=year_options, index=0)
    else:
        year_pick = datetime.now().year
        st.info("Kolom tahun tidak ditemukan, default tahun berjalan.")

        # --- Data khusus tahun terpilih (opsi kec/pasar mengikuti tahun) ---
    df_y = df.copy()
    if 'tera_ulang_tahun' in df_y.columns:
        df_y = df_y[df_y['tera_ulang_tahun'] == int(year_pick)]

    all_kec = uniq_clean(df_y['kecamatan']) if 'kecamatan' in df_y.columns else []
    all_pasar = uniq_clean(df_y['nama_pasar']) if 'nama_pasar' in df_y.columns else []

    # init state
    st.session_state.setdefault("kec_sel", "(Semua)")
    st.session_state.setdefault("pasar_sel", "(Semua)")
    st.session_state.setdefault("year_prev", int(year_pick))

    # kalau tahun berubah -> reset pilihan agar tidak nyangkut ke opsi yang hilang
    if int(year_pick) != int(st.session_state["year_prev"]):
        st.session_state["kec_sel"] = "(Semua)"
        st.session_state["pasar_sel"] = "(Semua)"
        st.session_state["year_prev"] = int(year_pick)

    # callback: saat kec berubah
    def on_kec_change():
        k = st.session_state["kec_sel"]
        if k != "(Semua)" and {'kecamatan','nama_pasar'}.issubset(df_y.columns):
            valid_pasar = ["(Semua)"] + uniq_clean(df_y.loc[df_y['kecamatan'] == k, 'nama_pasar'])
            if st.session_state["pasar_sel"] not in valid_pasar:
                st.session_state["pasar_sel"] = "(Semua)"

    # callback: saat pasar berubah
    def on_pasar_change():
        p = st.session_state["pasar_sel"]
        # jika user pilih pasar dulu, kec otomatis mengikuti (berdasarkan tahun terpilih)
        if p != "(Semua)" and {'nama_pasar','kecamatan'}.issubset(df_y.columns):
            kec_auto = df_y.loc[df_y['nama_pasar'] == p, 'kecamatan'].dropna()
            if not kec_auto.empty:
                st.session_state["kec_sel"] = kec_auto.iloc[0]

    # validasi state sebelum render widget (hindari error "value not in options")
    kec_opsi = ["(Semua)"] + all_kec
    if st.session_state["kec_sel"] not in kec_opsi:
        st.session_state["kec_sel"] = "(Semua)"

    if st.session_state["kec_sel"] != "(Semua)" and {'kecamatan','nama_pasar'}.issubset(df_y.columns):
        pasar_opsi = ["(Semua)"] + uniq_clean(df_y.loc[df_y['kecamatan'] == st.session_state["kec_sel"], 'nama_pasar'])
    else:
        pasar_opsi = ["(Semua)"] + all_pasar

    if st.session_state["pasar_sel"] not in pasar_opsi:
        st.session_state["pasar_sel"] = "(Semua)"

    st.selectbox("Kecamatan", kec_opsi, key="kec_sel", on_change=on_kec_change)
    st.selectbox("Nama Pasar", pasar_opsi, key="pasar_sel", on_change=on_pasar_change)

    # variabel final untuk dipakai di bawah
    kec = st.session_state["kec_sel"]
    nama_pasar = st.session_state["pasar_sel"]

    # --- Kartu info pasar terpilih (ungu elegan) ---
    if ('nama_pasar' in df.columns) and (nama_pasar != "(Semua)"):
        info = df_y.loc[df_y['nama_pasar'] == nama_pasar].head(1)
        if not info.empty:
            nama = info['nama_pasar'].iat[0]
            alamat = info['alamat'].iat[0] if 'alamat' in info.columns else "Alamat tidak tersedia"
            kecamatan = info['kecamatan'].iat[0] if 'kecamatan' in info.columns else "–"

            st.markdown("---")
            st.markdown(
                f"""
                <div style="
                    background-color:#f3e8ff;
                    padding:14px 16px;
                    border-radius:12px;
                    border-left:5px solid #8000FF;
                    box-shadow:0px 1px 4px rgba(0,0,0,0.15);
                    margin-top:10px;
                    ">
                    <h4 style="margin-bottom:6px; color:#4B0082; font-size:16px;">
                        🏪 {nama}
                    </h4>
                    <p style="margin:2px 0; font-size:13px; color:#222;">
                        <b>Kecamatan:</b> {kecamatan}<br>
                        <b>Alamat:</b> {alamat}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

    # === Ringkasan Timbangan di Sidebar ===
    st.markdown("---")
    st.subheader("⚖️ Total Timbangan Tera Ulang")

    timb_map = {
        "Pegas":        ["Timb. Pegas", "Timb Pegas", "Pegas"],
        "Meja":         ["Timb. Meja", "Timb Meja", "Meja"],
        "Elektronik":   ["Timb. Elektronik", "Timb Elektronik", "Elektronik"],
        "Sentisimal":   ["Timb. Sentisimal", "Timb Sentisimal", "Sentisimal"],
        "Bobot Ingsut": ["Timb. Bobot Ingsut", "Timb Bobot Ingsut", "Bobot Ingsut"],
        "Neraca":       ["Neraca"],
        "Dacin":        ["Dacin"]
    }

    def sum_first_existing(df_src: pd.DataFrame, candidates: list) -> int:
        for c in candidates:
            if c in df_src.columns:
                return int(pd.to_numeric(df_src[c], errors="coerce").fillna(0).sum())
        return 0

    # Gunakan df terfilter untuk sidebar totals
    # Gunakan df yang sudah terfilter agar sesuai pilihan tahun/kecamatan/pasar
    fdf_sidebar = df.copy()

    if 'tera_ulang_tahun' in fdf_sidebar.columns:
        fdf_sidebar = fdf_sidebar[fdf_sidebar['tera_ulang_tahun'] == int(year_pick)]

    if 'kecamatan' in fdf_sidebar.columns and kec != "(Semua)":
        fdf_sidebar = fdf_sidebar[fdf_sidebar['kecamatan'] == kec]

    if 'nama_pasar' in fdf_sidebar.columns and nama_pasar != "(Semua)":
        fdf_sidebar = fdf_sidebar[fdf_sidebar['nama_pasar'] == nama_pasar]

    totals = {label: sum_first_existing(fdf_sidebar, cands) for label, cands in timb_map.items()}

    if 'jumlah_timbangan_tera_ulang' in fdf_sidebar.columns:
        total_uttp = int(pd.to_numeric(fdf_sidebar['jumlah_timbangan_tera_ulang'],
                                       errors='coerce').fillna(0).sum())
    else:
        total_uttp = sum(totals.values())

    for label, val in totals.items():
        st.markdown(f"**{label}**: {val:,}")

    st.markdown(f"**Total UTTP (semua jenis):** {total_uttp:,}")
    st.markdown("---")

    # (Opsional) Toggle choropleth
    #show_choro = st.checkbox("Tampilkan choropleth per kecamatan", value=False)
    #choro_metric = st.selectbox("Indikator choropleth", ["jumlah_pasar", "total_uttp", "total_pedagang"], index=1)

# =========================
# FILTER DATA (UTAMA)
# =========================
fdf = df.copy()

if 'tera_ulang_tahun' in fdf.columns:
    fdf = fdf[fdf['tera_ulang_tahun'] == int(year_pick)]

if 'kecamatan' in fdf.columns and kec != "(Semua)":
    fdf = fdf[fdf['kecamatan'] == kec]

if 'nama_pasar' in fdf.columns and nama_pasar != "(Semua)":
    fdf = fdf[fdf['nama_pasar'] == nama_pasar]

# =========================
# =========================
# KPIs (DINAMIS)
# =========================
if kec == "(Semua)" and nama_pasar == "(Semua)":
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        total_kec = clean_str_series(fdf['kecamatan']).nunique() if 'kecamatan' in fdf.columns else 0
        st.metric("Total Kecamatan", total_kec)
    with c2:
        total_pasar = clean_str_series(fdf['nama_pasar']).nunique() if 'nama_pasar' in fdf.columns else 0
        st.metric("Total Seluruh Pasar", total_pasar)
    with c3:
        st.metric("Tahun", int(year_pick))
    with c4:
        total_timb = int(pd.to_numeric(fdf.get('jumlah_timbangan_tera_ulang', 0), errors='coerce').fillna(0).sum())
        st.metric("Total Timbangan", total_timb)

# ======= KECAMATAN SAJA =======
elif kec != "(Semua)" and nama_pasar == "(Semua)":
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Kecamatan", kec)
    with c2:
        total_pasar_kec = clean_str_series(fdf['nama_pasar']).nunique() if 'nama_pasar' in fdf.columns else 0
        st.metric("Total Pasar", total_pasar_kec)
    with c3:
        st.metric("Tahun", int(year_pick))
    with c4:
        total_timb = int(pd.to_numeric(fdf.get('jumlah_timbangan_tera_ulang', 0), errors='coerce').fillna(0).sum())
        st.metric("Total Timbangan", total_timb)

# ======= PASAR DIPILIH =======
else:
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Nama Pasar", nama_pasar if nama_pasar != "(Semua)" else "—")

    with c2:
        if kec != "(Semua)":
            st.metric("Kecamatan", kec)
        else:
            kec_auto = fdf['kecamatan'].dropna().iloc[0] if ('kecamatan' in fdf.columns and not fdf.empty) else "—"
            st.metric("Kecamatan", kec_auto)

    with c3:
        st.metric("Tahun", int(year_pick))

    with c4:
        total_timb = int(pd.to_numeric(fdf.get('jumlah_timbangan_tera_ulang', 0), errors='coerce').fillna(0).sum())
        st.metric("Total Timbangan", total_timb)

# =========================
# MAP
# =========================
st.subheader("🗺️ Peta Lokasi Pasar")

default_center = [-6.2, 106.55]
default_zoom = 10

has_coords = {'lat', 'lon'}.issubset(fdf.columns)
coords = None
if has_coords:
    try:
        coords = fdf[['lat', 'lon']].astype(float).dropna()
    except Exception as e:
        st.warning(f"Error processing coordinates: {e}")
        coords = None

center_loc = default_center
zoom_start = default_zoom

if 'nama_pasar' in fdf.columns and nama_pasar != "(Semua)" and coords is not None and not coords.empty:
    row_sel = fdf[fdf['nama_pasar'] == nama_pasar].head(1)
    if not row_sel.empty:
        try:
            lat0 = float(row_sel['lat'].iloc[0])
            lon0 = float(row_sel['lon'].iloc[0])
            if pd.notna(lat0) and pd.notna(lon0):
                center_loc = [lat0, lon0]
                zoom_start = 16
        except Exception:
            pass
elif coords is not None and len(coords) == 1:
    center_loc = [coords.iloc[0]['lat'], coords.iloc[0]['lon']]
    zoom_start = 14

m = folium.Map(location=center_loc, zoom_start=zoom_start, control_scale=True, tiles="OpenStreetMap")

# Pane agar batas bisa di atas marker
from folium.map import CustomPane
m.add_child(CustomPane("batas-top", z_index=650))

# Load geojson (kalau ada)
geo = None
try:
    geo = load_geojson(FILE_GEOJSON)
except Exception as e:
    st.warning(f"⚠️ GeoJSON tidak terbaca: {e}")

# gambar batas kecamatan (ungu tebal, tanpa fill)
if geo is not None:
    tooltip = folium.GeoJsonTooltip(fields=["kec_label"], aliases=["Kecamatan:"])
    style_fn = lambda x: {
        "color": "#8000FF",
        "weight": 2,
        "opacity": 1.0,
        "fill": False,
        "fillOpacity": 0
    }

    # supaya tidak error kalau folium versi tertentu tidak menerima argumen pane
    try:
        folium.GeoJson(
            geo,
            name="Batas Kecamatan",
            pane="batas-top",
            style_function=style_fn,
            tooltip=tooltip
        ).add_to(m)
    except TypeError:
        gj_layer = folium.GeoJson(
            geo,
            name="Batas Kecamatan",
            style_function=style_fn,
            tooltip=tooltip
        )
        # fallback: set pane via options jika tersedia
        try:
            gj_layer.options["pane"] = "batas-top"
        except Exception:
            pass
        gj_layer.add_to(m)

# choropleth opsional (tetap selaras tampilan)
#if (geo is not None) and show_choro and ('kec_norm' in fdf.columns):
#    agg = (
#        fdf.groupby("kec_norm")
#           .agg(
#               jumlah_pasar=("nama_pasar", "nunique"),
#               total_uttp=("jumlah_timbangan_tera_ulang", "sum") if "jumlah_timbangan_tera_ulang" in fdf.columns else ("nama_pasar", "size"),
#               total_pedagang=("total_pedagang", "sum") if "total_pedagang" in fdf.columns else ("nama_pasar", "size"),
#           )
#           .reset_index()
#    )

#    try:
#        folium.Choropleth(
#            geo_data=geo,
#            data=agg,
#            columns=["kec_norm", choro_metric],
#            key_on="feature.properties.kec_norm",
#            fill_opacity=0.55,
#            line_opacity=0.35,
#            legend_name=f"{choro_metric} (hasil filter)",
#            nan_fill_opacity=0.08
#        ).add_to(m)
#    except Exception as e:
#        st.warning(f"Choropleth gagal dibuat: {e}")

# marker pasar
if has_coords and coords is not None and not coords.empty:
    cluster = MarkerCluster(name="Pasar", show=True).add_to(m)

    for _, r in fdf.iterrows():
        try:
            lat = float(r.get('lat', float('nan')))
            lon = float(r.get('lon', float('nan')))
        except Exception:
            lat, lon = float("nan"), float("nan")

        if pd.isna(lat) or pd.isna(lon):
            continue

        nama = str(r.get('nama_pasar', 'Unknown'))
        alamat = str(r.get('alamat', 'Tidak ada alamat'))
        tahun = r.get('tera_ulang_tahun', None)
        jumlah = r.get('jumlah_timbangan_tera_ulang', None)
        jenis = str(r.get('jenis_timbangan', 'Tidak ada data'))
        pedagang = r.get('total_pedagang', None)

        html = f"""
        <div style='width: 280px; font-family: Arial, sans-serif;'>
            <h4 style='margin:8px 0; color: #2E86AB;'>{nama}</h4>
            <div style='font-size: 12px; color:#666; margin-bottom:8px;'>{alamat}</div>
            <hr style='margin:6px 0'/>
            <table style='font-size: 12px; width: 100%;'>
                <tr><td><b>Tera Ulang</b></td><td style='padding-left:8px'>: {tahun if pd.notna(tahun) else 'Tidak ada data'}</td></tr>
                <tr><td><b>Total UTTP</b></td><td style='padding-left:8px'>: {jumlah if pd.notna(jumlah) else 'Tidak ada data'}</td></tr>
                <tr><td><b>Total Pedagang</b></td><td style='padding-left:8px'>: {int(pedagang) if pd.notna(pedagang) else 'Tidak ada data'}</td></tr>
                <tr><td><b>Jenis Timbangan</b></td><td style='padding-left:8px'>: {jenis}</td></tr>
            </table>
        </div>
        """

        tooltip_text = f"{nama} - {tahun if pd.notna(tahun) else 'Tahun tidak diketahui'}"
        popup = folium.Popup(html, max_width=320)
        tooltip2 = folium.Tooltip(tooltip_text)

        try:
            y = int(tahun) if pd.notna(tahun) else None
        except Exception:
            y = None

        col = marker_color(y, int(year_pick))


        folium.CircleMarker(
            location=[lat, lon],
            radius=10,
            color=col,
            fill=True,
            fill_color=col,
            fill_opacity=0.7,
            weight=2,
            tooltip=tooltip2,
            popup=popup
        ).add_to(cluster)

    # Auto-fit bounds
    if not ('nama_pasar' in fdf.columns and nama_pasar != "(Semua)") and len(coords) > 1:
        try:
            sw = [coords['lat'].min(), coords['lon'].min()]
            ne = [coords['lat'].max(), coords['lon'].max()]
            m.fit_bounds([sw, ne], padding=(30, 30))
        except Exception as e:
            st.warning(f"Tidak bisa auto-fit peta: {e}")
else:
    st.warning("⚠️ Tidak ada data koordinat yang valid untuk ditampilkan di peta")

folium.LayerControl(collapsed=False).add_to(m)
st_folium(m, height=500, use_container_width=True)

# (opsional) tabel bawah biar enak cek
#with st.expander("📋 Lihat data (hasil filter)"):
 #   show_cols = [c for c in [
  #      "nama_pasar", "kecamatan", "tera_ulang_tahun",
  #      "jumlah_timbangan_tera_ulang", "total_pedagang", "jenis_timbangan",
   #     "alamat", "koordinat"
    #] if c in fdf.columns]
    #st.dataframe(fdf[show_cols], use_container_width=True)
# =========================
# Tampilkan peta
map_state = st_folium(m, height=500, width="stretch", key="map")

# =========================
# HANDLE KLIK MARKER -> pilih pasar + kecamatan otomatis
# =========================
def _pick_pasar_from_click(map_state: dict, df_context: pd.DataFrame) -> bool:
    """
    Return True kalau berhasil set pilihan pasar+kecamatan dari klik map.
    df_context = dataframe yang dipakai untuk menggambar marker (mis. fdf pada tahun terpilih)
    """
    if not map_state:
        return False

    clicked = map_state.get("last_object_clicked")
    if not clicked:
        return False

    latc = clicked.get("lat")
    lonc = clicked.get("lng")
    if latc is None or lonc is None:
        return False

    if df_context is None or df_context.empty or not {'lat','lon','nama_pasar','kecamatan'}.issubset(df_context.columns):
        return False

    # cari baris terdekat (klik marker -> koordinat sama/terdekat)
    tmp = df_context[['lat','lon','nama_pasar','kecamatan']].dropna().copy()
    if tmp.empty:
        return False

    # jarak kuadrat (lebih cepat)
    d2 = (tmp['lat'].astype(float) - float(latc))**2 + (tmp['lon'].astype(float) - float(lonc))**2
    idx = d2.idxmin()

    # threshold biar tidak salah pilih (0.0001 derajat ~ 11 m)
    if float(d2.loc[idx]) > 1e-8:
        return False

    pasar_clicked = str(df_context.loc[idx, 'nama_pasar'])
    kec_clicked   = str(df_context.loc[idx, 'kecamatan'])

    # set state dropdown
    st.session_state["last_changed"] = "pasar"
    st.session_state["pasar_sel"] = pasar_clicked
    st.session_state["kec_sel"] = kec_clicked

    return True

# df untuk konteks klik = dataset marker yang sedang ditampilkan (umumnya fdf pada tahun terpilih)
# Kalau map kamu pakai fdf (yang sudah filter year_pick + kec/pasar), pakai fdf.
changed = _pick_pasar_from_click(map_state, fdf)

if changed:
    st.rerun()

import altair as alt

def line_chart_tight_y(df_plot: pd.DataFrame, x_col: str, y_col: str, title: str, pad_ratio: float = 0.08):
    d = df_plot[[x_col, y_col]].dropna().copy()
    if d.empty:
        st.info(f"Tidak ada data untuk {title}.")
        return

    y_min = float(d[y_col].min())
    y_max = float(d[y_col].max())

    if y_min == y_max:
        pad = max(1.0, abs(y_min) * pad_ratio)
        y0, y1 = y_min - pad, y_max + pad
    else:
        span = y_max - y_min
        pad = span * pad_ratio
        y0, y1 = y_min - pad, y_max + pad

    chart = (
        alt.Chart(d)
        .mark_line(point=True)
        .encode(
            x=alt.X(f"{x_col}:O", title="Tahun"),
            y=alt.Y(f"{y_col}:Q", title=title, scale=alt.Scale(domain=[y0, y1])),
            tooltip=[alt.Tooltip(f"{x_col}:O", title="Tahun"), alt.Tooltip(f"{y_col}:Q", title=title)]
        )
        .properties(height=280)
    )
    st.altair_chart(chart, use_container_width=True)


# GRAFIK (di bawah MAP)
# =========================
st.subheader("📈 Grafik Tren (Tahun ke Tahun)")

# Grafik menyesuaikan pilihan user (kecamatan/pasar),
# tapi untuk tren tidak dibatasi year_pick (biar terlihat 2021–2025).
gdf = df.copy()

# scope sesuai pilihan user
if 'kecamatan' in gdf.columns and kec != "(Semua)":
    gdf = gdf[gdf['kecamatan'] == kec]
if 'nama_pasar' in gdf.columns and nama_pasar != "(Semua)":
    gdf = gdf[gdf['nama_pasar'] == nama_pasar]

# pastikan tahun numerik
if 'tera_ulang_tahun' in gdf.columns:
    gdf['tera_ulang_tahun'] = pd.to_numeric(gdf['tera_ulang_tahun'], errors='coerce')
    gdf = gdf.dropna(subset=['tera_ulang_tahun'])
    gdf['tera_ulang_tahun'] = gdf['tera_ulang_tahun'].astype(int)

# agregasi per tahun
agg = gdf.groupby('tera_ulang_tahun', as_index=False).agg(
    jumlah_kecamatan=('kecamatan', 'nunique') if 'kecamatan' in gdf.columns else ('tera_ulang_tahun', 'size'),
    jumlah_pasar=('nama_pasar', 'nunique') if 'nama_pasar' in gdf.columns else ('tera_ulang_tahun', 'size'),
    total_uttp=('jumlah_timbangan_tera_ulang', 'sum') if 'jumlah_timbangan_tera_ulang' in gdf.columns else ('tera_ulang_tahun', 'size'),
    total_pedagang=('total_pedagang', 'sum') if 'total_pedagang' in gdf.columns else ('tera_ulang_tahun', 'size'),
).sort_values('tera_ulang_tahun')

if agg.empty:
    st.info("Tidak ada data untuk ditampilkan pada grafik (cek pilihan filter).")
    
else:
    agg_plot = agg.copy()
    agg_plot["Tahun"] = agg_plot["tera_ulang_tahun"].astype(int).astype(str)
    agg_plot = agg_plot.set_index("Tahun")
    agg_plot.index.name = "Tahun"
    
    st.line_chart(agg_plot[['jumlah_kecamatan']])
    # ====== Level 1: Semua Kecamatan & Semua Pasar ======
if kec == "(Semua)" and nama_pasar == "(Semua)":
    st.caption("Menampilkan tren seluruh kabupaten per tahun.")

    # agg sudah ada (hasil groupby per tahun)
    agg_show = agg.copy()
    agg_show["Tahun"] = agg_show["tera_ulang_tahun"].astype(int).astype(str)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Tren Jumlah Pasar**")
        line_chart_tight_y(agg_show, "Tahun", "jumlah_pasar", "Jumlah Pasar", pad_ratio=0.05)

    with c2:
        st.markdown("**Tren Total UTTP**")
        line_chart_tight_y(agg_show, "Tahun", "total_uttp", "Total UTTP", pad_ratio=0.05)

    with c3:
        st.markdown("**Tren Total Pedagang**")
        line_chart_tight_y(agg_show, "Tahun", "total_pedagang", "Total Pedagang", pad_ratio=0.05)


    # ====== Level 2: Kecamatan dipilih (Pasar semua) ======
elif kec != "(Semua)" and nama_pasar == "(Semua)":
    st.caption(f"Menampilkan tren Kecamatan **{kec}** per tahun.")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Tren Jumlah Pasar (di kecamatan terpilih)**")
        st.line_chart(agg_plot[['jumlah_pasar']])

    with c2:
        st.markdown("**Tren Total UTTP (di kecamatan terpilih)**")
        st.line_chart(agg_plot[['total_uttp']])

    # Top pasar pada tahun terpilih (year_pick)
    if {'nama_pasar', 'tera_ulang_tahun', 'jumlah_timbangan_tera_ulang', 'kecamatan'}.issubset(df.columns):
        top_df = df.copy()
        top_df['tera_ulang_tahun'] = pd.to_numeric(top_df['tera_ulang_tahun'], errors='coerce')
        top_df = top_df.dropna(subset=['tera_ulang_tahun'])
        top_df['tera_ulang_tahun'] = top_df['tera_ulang_tahun'].astype(int)

        top_df = top_df[(top_df['tera_ulang_tahun'] == int(year_pick)) & (top_df['kecamatan'] == kec)]

        if not top_df.empty:
            top_pasar = (
                top_df.groupby('nama_pasar', as_index=False)['jumlah_timbangan_tera_ulang']
                .sum()
                .sort_values('jumlah_timbangan_tera_ulang', ascending=False)
                .head(10)
            )

            st.markdown(f"**Top 10 Pasar (UTTP) di tahun {int(year_pick)} – Kecamatan {kec}**")
            top_plot = top_pasar.set_index('nama_pasar')[['jumlah_timbangan_tera_ulang']]
            top_plot.index.name = "Nama Pasar"
            st.bar_chart(top_plot)
        else:
            st.info(f"Tidak ada data Top Pasar untuk tahun {int(year_pick)} di Kecamatan {kec}.")

# ====== Level 3: Pasar dipilih ======
else:
    st.caption(f"Menampilkan tren Pasar **{nama_pasar}** per tahun.")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Tren Total UTTP (pasar terpilih)**")
        st.line_chart(agg_plot[['total_uttp']])

    with c2:
        st.markdown("**Tren Total Pedagang (pasar terpilih)**")
        st.line_chart(agg_plot[['total_pedagang']])

    # Komposisi jenis timbangan per tahun (stacked)
    timb_cols = [c for c in [
        'Timb. Pegas','Timb. Meja','Timb. Elektronik','Timb. Sentisimal',
        'Timb. Bobot Ingsut','Neraca','Dacin'
    ] if c in gdf.columns]

    if timb_cols:
        try:
            import altair as alt

            g_year = gdf.groupby('tera_ulang_tahun', as_index=False)[timb_cols].sum()
            m2 = g_year.melt('tera_ulang_tahun', var_name='Jenis', value_name='Jumlah')

            st.markdown("**Komposisi Jenis Timbangan per Tahun (pasar terpilih)**")
            chart = (
                alt.Chart(m2)
                .mark_bar()
                .encode(
                    x=alt.X('tera_ulang_tahun:O', title='Tahun'),
                    y=alt.Y('Jumlah:Q', title='Jumlah'),
                    color='Jenis:N',
                    tooltip=['tera_ulang_tahun:O', 'Jenis:N', 'Jumlah:Q']
                )
                .properties(height=320)
            )
            st.altair_chart(chart, use_container_width=True)

        except Exception as e:
            st.warning(f"Grafik komposisi (Altair) tidak bisa dibuat: {e}")
            st.dataframe(gdf.groupby('tera_ulang_tahun')[timb_cols].sum())
