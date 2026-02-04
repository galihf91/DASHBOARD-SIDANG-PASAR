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


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Dashboard Kemetrologian – Kab. Tangerang",
    page_icon="📊",
    layout="wide"
)

# =========================
# FILE UTAMA (SEFOLDER)
# =========================
FILE_EXCEL   = "DATA_DASHBOARD_PASAR.xlsx"
FILE_GEOJSON = "batas_kecamatan_tangerang.geojson"
FILE_SPBU    = "Data SPBU Kab. Tangerang.csv"


# =========================
# UTIL
# =========================
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())

def parse_coord(val):
    """Parse koordinat dari berbagai format ke (lat, lon)"""
    try:
        if pd.isna(val) or str(val).strip() == "":
            return np.nan, np.nan

        s = str(val).strip()

        # format: "-6.26435, 106.42592"
        if "," in s:
            parts = [p.strip() for p in s.split(",")]
            if len(parts) >= 2:
                lat = float(parts[0])
                lon = float(parts[1])
                # swap jika terbalik (lon,lat)
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

    except Exception:
        pass

    return np.nan, np.nan

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

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

    # Tambahkan ringkasan jenis timbangan
    timbangan_cols = [
        'Timb. Pegas', 'Timb. Meja', 'Timb. Elektronik',
        'Timb. Sentisimal', 'Timb. Bobot Ingsut', 'Neraca', 'Dacin'
    ]
    available = [c for c in timbangan_cols if c in df.columns]

    if available:
        def summarize(row):
            parts = []
            for col in available:
                val = pd.to_numeric(row[col], errors="coerce")
                if pd.notna(val) and val > 0:
                    label = col.replace("Timb.", "").replace("Timb", "").replace(".", "").strip()
                    parts.append(f"{label}: {int(val)}")
            return "; ".join(parts) if parts else "Tidak ada data"
        df["jenis_timbangan"] = df.apply(summarize, axis=1)

    if "kecamatan" in df.columns:
        df["kec_norm"] = df["kecamatan"].apply(_norm)
    if "nama_pasar" in df.columns:
        df["pasar_norm"] = df["nama_pasar"].apply(_norm)

    return df

def create_sample_data():
    sample_data = {
        'nama_pasar': ['Cisoka', 'Curug', 'Mauk', 'Cikupa', 'Pasar Kemis'],
        'kecamatan': ['Cisoka', 'Curug', 'Mauk', 'Cikupa', 'Pasar Kemis'],
        'alamat': [
            'Jl. Ps. Cisoka No.44, Cisoka, Kec. Cisoka, Kabupaten Tangerang',
            'Jl. Raya Curug, Curug Wetan, Kec. Curug, Kabupaten Tangerang',
            'East Mauk, Mauk, Tangerang Regency',
            'Jl. Raya Serang, Cikupa, Kec. Cikupa, Kabupaten Tangerang',
            'Jalan Raya, Sukaasih, Pasar Kemis, Tangerang'
        ],
        'lat': [-6.26435, -6.26100, -6.06044, -6.22907, -6.16365],
        'lon': [106.42592, 106.55858, 106.51129, 106.51981, 106.53155],
        'tera_ulang_tahun': [2025, 2025, 2025, 2025, 2025],
        'jumlah_timbangan_tera_ulang': [195, 251, 161, 257, 174],
        'total_pedagang': [100, 120, 90, 150, 110],
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
    if series is None:
        return pd.Series([], dtype=str)
    s = series.astype(str).str.strip()
    s = s.str.title()
    bad = s.str.lower().isin(["", "nan", "none", "null", "na", "n/a", "-", "--"])
    return s[~bad]

def uniq_clean(series: pd.Series) -> list:
    return sorted(clean_str_series(series).unique().tolist())

def marker_color(year: int, selected_year: int):
    if year is None or year == 0:
        return "gray"
    if year == selected_year:
        return "green"
    elif year == selected_year - 1:
        return "orange"
    else:
        return "red"


# =========================
# LOADERS
# =========================
@st.cache_data
def load_excel(path_like: str):
    try:
        if "." not in path_like:
            matches = glob.glob(path_like + ".*")
            if not matches:
                raise FileNotFoundError(f"File {path_like}.* tidak ditemukan sefolder app.py")
            path = matches[0]
        else:
            path = path_like

        df = pd.read_excel(path, sheet_name=0, engine="openpyxl")
        df = standardize_columns(df)
        df = coerce_types(df)
        return df, None
    except Exception as e:
        st.error(f"❌ Error loading Excel: {e}")
        return create_sample_data(), "Menggunakan data sample"

@st.cache_data
def load_geojson(path: str):
    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    for ft in gj.get("features", []):
        props = ft.get("properties", {})
        wadmkc = props.get("wadmkc", "")
        props["kec_norm"] = _norm(wadmkc)
        props["kec_label"] = wadmkc
        ft["properties"] = props
    return gj

@st.cache_data
def load_spbu_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df = df.rename(columns={
        "No. SPBU": "nama_spbu",
        "Alamat": "alamat",
        "Kecamatan": "kecamatan",
        "Koordinat": "koordinat",
        "Media BBM": "media_bbm",
    })

    if "koordinat" in df.columns:
        coords = df["koordinat"].apply(parse_coord)
        df["lat"] = coords.apply(lambda x: x[0])
        df["lon"] = coords.apply(lambda x: x[1])

    def _split_media(x):
        if pd.isna(x) or str(x).strip() == "":
            return []
        return [m.strip() for m in str(x).split(",") if m.strip()]

    df["media_list"] = df["media_bbm"].apply(_split_media) if "media_bbm" in df.columns else [[]]*len(df)

    for c in ["nama_spbu", "alamat", "kecamatan", "media_bbm"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)

    for c in ["lat", "lon"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# =========================
# CHART HELPERS (ALTAIR)
# =========================
import altair as alt

def _show_altair(chart):
    """Aman untuk streamlit baru/lama."""
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:
        st.altair_chart(chart, use_container_width=True)

def line_chart_alt(df_plot: pd.DataFrame, x_col: str, y_col: str, title: str, pad_ratio: float = 0.08):
    """Line chart: X jelas + domain Y diperkecil supaya tren terlihat."""
    d = df_plot[[x_col, y_col]].dropna().copy()
    if d.empty:
        st.info("Tidak ada data untuk grafik.")
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
            x=alt.X(f"{x_col}:O", title="Tahun", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{y_col}:Q", title=title, scale=alt.Scale(domain=[y0, y1])),
            tooltip=[alt.Tooltip(f"{x_col}:O", title="Tahun"),
                     alt.Tooltip(f"{y_col}:Q", title=title, format=",.0f")]
        )
        .properties(height=260)
    )
    _show_altair(chart)

def bar_chart_alt(df_plot: pd.DataFrame, x_col: str, y_col: str, title: str, horizontal: bool = False):
    """Bar chart: default vertical, opsi horizontal untuk label panjang."""
    d = df_plot[[x_col, y_col]].dropna().copy()
    if d.empty:
        st.info("Tidak ada data untuk grafik.")
        return

    if horizontal:
        chart = (
            alt.Chart(d)
            .mark_bar()
            .encode(
                y=alt.Y(f"{x_col}:N", sort="-x", title=None),
                x=alt.X(f"{y_col}:Q", title=title),
                tooltip=[alt.Tooltip(f"{x_col}:N", title=x_col),
                         alt.Tooltip(f"{y_col}:Q", title=title, format=",.0f")]
            )
            .properties(height=360)
        )
    else:
        chart = (
            alt.Chart(d)
            .mark_bar()
            .encode(
                x=alt.X(f"{x_col}:N", title=x_col, sort="-y", axis=alt.Axis(labelAngle=-35)),
                y=alt.Y(f"{y_col}:Q", title=title),
                tooltip=[alt.Tooltip(f"{x_col}:N", title=x_col),
                         alt.Tooltip(f"{y_col}:Q", title=title, format=",.0f")]
            )
            .properties(height=320)
        )

    _show_altair(chart)

# =========================
# MAP CLICK -> PICK HELPER
# =========================
def pick_from_click(map_state: dict, df_context: pd.DataFrame, lat_col="lat", lon_col="lon",
                    name_col="nama_pasar", kec_col="kecamatan",
                    state_prefix="pasar"):
    """Jika user klik marker, set session_state pilihan (nama + kec) lalu return True."""
    if not map_state:
        return False

    clicked = map_state.get("last_object_clicked")
    if not clicked:
        return False

    latc = clicked.get("lat")
    lonc = clicked.get("lng")
    if latc is None or lonc is None:
        return False

    if df_context is None or df_context.empty:
        return False
    need_cols = {lat_col, lon_col, name_col, kec_col}
    if not need_cols.issubset(df_context.columns):
        return False

    tmp = df_context[[lat_col, lon_col, name_col, kec_col]].dropna().copy()
    if tmp.empty:
        return False

    d2 = (tmp[lat_col].astype(float) - float(latc))**2 + (tmp[lon_col].astype(float) - float(lonc))**2
    idx = d2.idxmin()

    # threshold klik agar tidak salah pilih (1e-8 ~ sangat dekat)
    if float(d2.loc[idx]) > 1e-8:
        return False

    nm = str(df_context.loc[idx, name_col])
    kc = str(df_context.loc[idx, kec_col])

    st.session_state[f"{state_prefix}_last_changed"] = "name"
    st.session_state[f"{state_prefix}_kec_sel"] = kc
    st.session_state[f"{state_prefix}_name_sel"] = nm
    return True


# =========================
# DASHBOARD PASAR
# =========================
def render_dashboard_pasar():
    st.title("🏪 Status Tera Ulang Timbangan Pasar – Kabupaten Tangerang")
    st.caption("Dinas Perindustrian dan Perdagangan • Bidang Kemetrologian")

    df, err = load_excel(FILE_EXCEL)
    if err:
        st.warning(f"Peringatan: {err}")

    # geojson batas
    geo = None
    try:
        geo = load_geojson(FILE_GEOJSON)
    except Exception as e:
        st.warning(f"⚠️ GeoJSON tidak terbaca: {e}")

    # ===== SIDEBAR FILTER (PASAR) =====
    with st.sidebar:
        st.header("Filter Pasar")

        # pilih tahun top-down, default terbaru
        if "tera_ulang_tahun" in df.columns and df["tera_ulang_tahun"].notna().any():
            years = sorted(df["tera_ulang_tahun"].dropna().astype(int).unique().tolist())
            year_options = years[::-1]
        else:
            year_options = [datetime.now().year]

        year_pick = st.selectbox("Tahun Tera Ulang", options=year_options, index=0, key="pasar_year_pick")

        # data tahun terpilih -> supaya opsi kec/pasar menyesuaikan tahun tsb
        df_year = df[df["tera_ulang_tahun"] == int(year_pick)].copy()

        # state (prefix pasar_)
        if "pasar_last_changed" not in st.session_state:
            st.session_state["pasar_last_changed"] = "kec"
        if "pasar_kec_sel" not in st.session_state:
            st.session_state["pasar_kec_sel"] = "(Semua)"
        if "pasar_name_sel" not in st.session_state:
            st.session_state["pasar_name_sel"] = "(Semua)"

        def _set_last(which: str):
            st.session_state["pasar_last_changed"] = which

        all_kec = uniq_clean(df_year["kecamatan"]) if "kecamatan" in df_year.columns else []
        all_pasar = uniq_clean(df_year["nama_pasar"]) if "nama_pasar" in df_year.columns else []

        # kalau state lama tidak ada di opsi tahun itu -> reset
        if st.session_state["pasar_kec_sel"] != "(Semua)" and st.session_state["pasar_kec_sel"] not in (["(Semua)"] + all_kec):
            st.session_state["pasar_kec_sel"] = "(Semua)"
        if st.session_state["pasar_name_sel"] != "(Semua)" and st.session_state["pasar_name_sel"] not in (["(Semua)"] + all_pasar):
            st.session_state["pasar_name_sel"] = "(Semua)"

        # dropdown kecamatan
        kec_opsi = ["(Semua)"] + all_kec
        kec = st.selectbox("Kecamatan", kec_opsi, key="pasar_kec_sel", on_change=_set_last, args=("kec",))

        # dropdown pasar (menyesuaikan kec jika terakhir ubah kec)
        if st.session_state["pasar_last_changed"] == "kec" and st.session_state["pasar_kec_sel"] != "(Semua)":
            pasar_opsi = ["(Semua)"] + uniq_clean(df_year.loc[df_year["kecamatan"] == st.session_state["pasar_kec_sel"], "nama_pasar"])
        else:
            pasar_opsi = ["(Semua)"] + all_pasar

        nama_pasar = st.selectbox("Nama Pasar", pasar_opsi, key="pasar_name_sel", on_change=_set_last, args=("name",))

        # kalau user pilih pasar dulu -> kecamatan ikut
        if st.session_state["pasar_last_changed"] == "name" and nama_pasar != "(Semua)" and {"nama_pasar","kecamatan"}.issubset(df_year.columns):
            kec_auto = df_year.loc[df_year["nama_pasar"] == nama_pasar, "kecamatan"].dropna()
            if not kec_auto.empty:
                st.session_state["pasar_kec_sel"] = kec_auto.iloc[0]

        # jika user pilih kec dulu tapi pasar tidak cocok -> reset pasar
        if st.session_state["pasar_last_changed"] == "kec" and kec != "(Semua)" and nama_pasar != "(Semua)":
            cek = df_year[(df_year["kecamatan"] == kec) & (df_year["nama_pasar"] == nama_pasar)]
            if cek.empty:
                st.session_state["pasar_name_sel"] = "(Semua)"

        # final
        kec = st.session_state["pasar_kec_sel"]
        nama_pasar = st.session_state["pasar_name_sel"]

        # kartu info pasar
        if nama_pasar != "(Semua)":
            info = df_year.loc[df_year["nama_pasar"] == nama_pasar].head(1)
            if not info.empty:
                nama = info["nama_pasar"].iat[0]
                alamat = info["alamat"].iat[0] if "alamat" in info.columns else "Alamat tidak tersedia"
                kecamatan = info["kecamatan"].iat[0] if "kecamatan" in info.columns else "–"
                st.markdown("---")
                st.markdown(
                    f"""
                    <div style="
                        background-color:#f3e8ff;
                        padding:14px 16px;
                        border-radius:12px;
                        border-left:5px solid #8000FF;
                        box-shadow:0px 1px 4px rgba(0,0,0,0.15);
                        margin-top:10px;">
                        <h4 style="margin-bottom:6px; color:#4B0082; font-size:16px;">🏪 {nama}</h4>
                        <p style="margin:2px 0; font-size:13px; color:#222;">
                            <b>Kecamatan:</b> {kecamatan}<br>
                            <b>Alamat:</b> {alamat}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # ringkasan timbangan (mengikuti filter tahun/kec/pasar)
        st.markdown("---")
        st.subheader("⚖️ Total Timbangan Tera Ulang")

        timb_map = {
            "Pegas":        ["Timb. Pegas"],
            "Meja":         ["Timb. Meja"],
            "Elektronik":   ["Timb. Elektronik"],
            "Sentisimal":   ["Timb. Sentisimal"],
            "Bobot Ingsut": ["Timb. Bobot Ingsut"],
            "Neraca":       ["Neraca"],
            "Dacin":        ["Dacin"],
        }

        fdf_sidebar = df_year.copy()
        if kec != "(Semua)":
            fdf_sidebar = fdf_sidebar[fdf_sidebar["kecamatan"] == kec]
        if nama_pasar != "(Semua)":
            fdf_sidebar = fdf_sidebar[fdf_sidebar["nama_pasar"] == nama_pasar]

        def sum_first_existing(df_src: pd.DataFrame, candidates: list) -> int:
            for c in candidates:
                if c in df_src.columns:
                    return int(pd.to_numeric(df_src[c], errors="coerce").fillna(0).sum())
            return 0

        totals = {label: sum_first_existing(fdf_sidebar, cands) for label, cands in timb_map.items()}
        total_uttp = int(pd.to_numeric(fdf_sidebar.get("jumlah_timbangan_tera_ulang", 0), errors="coerce").fillna(0).sum())

        for label, val in totals.items():
            st.markdown(f"**{label}**: {val:,}")
        st.markdown(f"**Total UTTP (semua jenis):** {total_uttp:,}")


    # FILTER DATA (UTAMA)
    # =========================
    fdf = df.copy()
    
    if "tera_ulang_tahun" in fdf.columns:
        fdf = fdf[fdf["tera_ulang_tahun"] == int(year_pick)]
    
    if "kecamatan" in fdf.columns and kec != "(Semua)":
        fdf = fdf[fdf["kecamatan"] == kec]
    
    if "nama_pasar" in fdf.columns and nama_pasar != "(Semua)":
        fdf = fdf[fdf["nama_pasar"] == nama_pasar]


# =========================
# KPIs (DINAMIS)
# =========================
    if kec == "(Semua)" and nama_pasar == "(Semua)":
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            total_kec = clean_str_series(fdf["kecamatan"]).nunique() if "kecamatan" in fdf.columns else 0
            st.metric("Total Kecamatan", total_kec)
        with c2:
            total_pasar = clean_str_series(fdf["nama_pasar"]).nunique() if "nama_pasar" in fdf.columns else 0
            st.metric("Total Seluruh Pasar", total_pasar)
        with c3:
            st.metric("Tahun", int(year_pick))
        with c4:
            total_timb = int(pd.to_numeric(fdf.get("jumlah_timbangan_tera_ulang", 0), errors="coerce").fillna(0).sum())
            st.metric("Total Timbangan", total_timb)

elif kec != "(Semua)" and nama_pasar == "(Semua)":
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Kecamatan", kec)
    with c2:
        total_pasar_kec = clean_str_series(fdf["nama_pasar"]).nunique() if "nama_pasar" in fdf.columns else 0
        st.metric("Total Pasar", total_pasar_kec)
    with c3:
        st.metric("Tahun", int(year_pick))
    with c4:
        total_timb = int(pd.to_numeric(fdf.get("jumlah_timbangan_tera_ulang", 0), errors="coerce").fillna(0).sum())
        st.metric("Total Timbangan", total_timb)

else:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Nama Pasar", nama_pasar if nama_pasar != "(Semua)" else "—")
    with c2:
        kec_auto = (
            fdf["kecamatan"].dropna().iloc[0]
            if ("kecamatan" in fdf.columns and not fdf.empty)
            else (kec if kec != "(Semua)" else "—")
        )
        st.metric("Kecamatan", kec_auto)
    with c3:
        st.metric("Tahun", int(year_pick))
    with c4:
        total_timb = int(pd.to_numeric(fdf.get("jumlah_timbangan_tera_ulang", 0), errors="coerce").fillna(0).sum())
        st.metric("Total Timbangan", total_timb)


# =========================
# MAP (PANGGIL SEKALI SAJA)
# =========================
st.subheader("🗺️ Peta Lokasi Pasar")

default_center = [-6.2, 106.55]
default_zoom = 10

has_coords = {"lat", "lon"}.issubset(fdf.columns)
coords = None
if has_coords:
    try:
        coords = fdf[["lat", "lon"]].astype(float).dropna()
    except Exception as e:
        st.warning(f"Error processing coordinates: {e}")
        coords = None

center_loc = default_center
zoom_start = default_zoom

if "nama_pasar" in fdf.columns and nama_pasar != "(Semua)" and coords is not None and not coords.empty:
    row_sel = fdf[fdf["nama_pasar"] == nama_pasar].head(1)
    if not row_sel.empty:
        try:
            lat0 = float(row_sel["lat"].iloc[0])
            lon0 = float(row_sel["lon"].iloc[0])
            if pd.notna(lat0) and pd.notna(lon0):
                center_loc = [lat0, lon0]
                zoom_start = 16
        except Exception:
            pass
elif coords is not None and len(coords) == 1:
    center_loc = [coords.iloc[0]["lat"], coords.iloc[0]["lon"]]
    zoom_start = 14

# tiles disembunyikan dari LayerControl (tetap tampil)
m = folium.Map(location=center_loc, zoom_start=zoom_start, control_scale=True, tiles=None)
folium.TileLayer("OpenStreetMap", control=False).add_to(m)

# Pane agar batas bisa di atas marker
from folium.map import CustomPane
m.add_child(CustomPane("batas-top", z_index=650))

# Load geojson (kalau ada)
geo = None
try:
    geo = load_geojson(FILE_GEOJSON)
except Exception as e:
    st.warning(f"⚠️ GeoJSON tidak terbaca: {e}")

if geo is not None:
    tooltip = folium.GeoJsonTooltip(fields=["kec_label"], aliases=["Kecamatan:"])
    style_fn = lambda x: {"color": "#8000FF", "weight": 2, "opacity": 1.0, "fill": False, "fillOpacity": 0}
    try:
        folium.GeoJson(
            geo,
            name="Batas Kecamatan",
            pane="batas-top",
            style_function=style_fn,
            tooltip=tooltip,
        ).add_to(m)
    except TypeError:
        gj_layer = folium.GeoJson(geo, name="Batas Kecamatan", style_function=style_fn, tooltip=tooltip)
        try:
            gj_layer.options["pane"] = "batas-top"
        except Exception:
            pass
        gj_layer.add_to(m)

# Marker pasar
if has_coords and coords is not None and not coords.empty:
    cluster = MarkerCluster(name="Pasar", show=True).add_to(m)

    for _, r in fdf.iterrows():
        try:
            lat = float(r.get("lat", float("nan")))
            lon = float(r.get("lon", float("nan")))
        except Exception:
            lat, lon = float("nan"), float("nan")

        if pd.isna(lat) or pd.isna(lon):
            continue

        nama = str(r.get("nama_pasar", "Unknown"))
        alamat = str(r.get("alamat", "Tidak ada alamat"))
        tahun = r.get("tera_ulang_tahun", None)
        jumlah = r.get("jumlah_timbangan_tera_ulang", None)
        jenis = str(r.get("jenis_timbangan", "Tidak ada data"))
        pedagang = r.get("total_pedagang", None)

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
            tooltip=folium.Tooltip(tooltip_text),
            popup=popup,
        ).add_to(cluster)

    if not ("nama_pasar" in fdf.columns and nama_pasar != "(Semua)") and len(coords) > 1:
        try:
            sw = [coords["lat"].min(), coords["lon"].min()]
            ne = [coords["lat"].max(), coords["lon"].max()]
            m.fit_bounds([sw, ne], padding=(30, 30))
        except Exception as e:
            st.warning(f"Tidak bisa auto-fit peta: {e}")
else:
    st.warning("⚠️ Tidak ada data koordinat yang valid untuk ditampilkan di peta")

folium.LayerControl(collapsed=False).add_to(m)

# >>> PANGGIL st_folium SEKALI, sekalian ambil map_state
map_state = st_folium(m, height=500, use_container_width=True, key="pasar_map")


# =========================
# HANDLE KLIK MARKER -> dropdown ikut pasar yang diklik
# =========================
def _pick_pasar_from_click(map_state: dict, df_context: pd.DataFrame) -> bool:
    if not map_state:
        return False

    clicked = map_state.get("last_object_clicked")
    if not clicked:
        return False

    latc = clicked.get("lat")
    lonc = clicked.get("lng")
    if latc is None or lonc is None:
        return False

    if df_context is None or df_context.empty:
        return False
    if not {"lat", "lon", "nama_pasar", "kecamatan"}.issubset(df_context.columns):
        return False

    tmp = df_context[["lat", "lon", "nama_pasar", "kecamatan"]].dropna().copy()
    if tmp.empty:
        return False

    d2 = (tmp["lat"].astype(float) - float(latc)) ** 2 + (tmp["lon"].astype(float) - float(lonc)) ** 2
    idx = d2.idxmin()

    # threshold agar tidak salah pilih (klik jauh dari marker)
    if float(d2.loc[idx]) > 1e-8:
        return False

    pasar_clicked = str(df_context.loc[idx, "nama_pasar"])
    kec_clicked = str(df_context.loc[idx, "kecamatan"])

    st.session_state["pasar_sel"] = pasar_clicked
    st.session_state["kec_sel"] = kec_clicked
    return True

if _pick_pasar_from_click(map_state, fdf):
    st.rerun()


# =========================
# GRAFIK (DI BAWAH MAP)
# =========================
st.subheader("📈 Grafik (Tahun ke Tahun)")

gdf = df.copy()
if "kecamatan" in gdf.columns and kec != "(Semua)":
    gdf = gdf[gdf["kecamatan"] == kec]
if "nama_pasar" in gdf.columns and nama_pasar != "(Semua)":
    gdf = gdf[gdf["nama_pasar"] == nama_pasar]

if "tera_ulang_tahun" in gdf.columns:
    gdf["tera_ulang_tahun"] = pd.to_numeric(gdf["tera_ulang_tahun"], errors="coerce")
    gdf = gdf.dropna(subset=["tera_ulang_tahun"])
    gdf["tera_ulang_tahun"] = gdf["tera_ulang_tahun"].astype(int)

agg = gdf.groupby("tera_ulang_tahun", as_index=False).agg(
    jumlah_pasar=("nama_pasar", "nunique") if "nama_pasar" in gdf.columns else ("tera_ulang_tahun", "size"),
    total_uttp=("jumlah_timbangan_tera_ulang", "sum") if "jumlah_timbangan_tera_ulang" in gdf.columns else ("tera_ulang_tahun", "size"),
    total_pedagang=("total_pedagang", "sum") if "total_pedagang" in gdf.columns else ("tera_ulang_tahun", "size"),
).sort_values("tera_ulang_tahun")

if agg.empty:
    st.info("Tidak ada data untuk grafik (cek filter).")
else:
    agg_show = agg.copy()
    agg_show["Tahun"] = agg_show["tera_ulang_tahun"].astype(int).astype(str)

    # LEVEL 1
    if kec == "(Semua)" and nama_pasar == "(Semua)":
        st.caption("Menampilkan tren seluruh kabupaten per tahun.")
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

    # LEVEL 2
    elif kec != "(Semua)" and nama_pasar == "(Semua)":
        st.caption(f"Menampilkan tren Kecamatan **{kec}** per tahun.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Tren Jumlah Pasar**")
            line_chart_tight_y(agg_show, "Tahun", "jumlah_pasar", "Jumlah Pasar", pad_ratio=0.05)
        with c2:
            st.markdown("**Tren Total UTTP**")
            line_chart_tight_y(agg_show, "Tahun", "total_uttp", "Total UTTP", pad_ratio=0.05)

        # Top 10 pasar pada tahun terpilih
        if {"nama_pasar", "tera_ulang_tahun", "jumlah_timbangan_tera_ulang", "kecamatan"}.issubset(df.columns):
            top_df = df.copy()
            top_df["tera_ulang_tahun"] = pd.to_numeric(top_df["tera_ulang_tahun"], errors="coerce")
            top_df = top_df.dropna(subset=["tera_ulang_tahun"])
            top_df["tera_ulang_tahun"] = top_df["tera_ulang_tahun"].astype(int)
            top_df = top_df[(top_df["tera_ulang_tahun"] == int(year_pick)) & (top_df["kecamatan"] == kec)]

            if not top_df.empty:
                top_pasar = (
                    top_df.groupby("nama_pasar", as_index=False)["jumlah_timbangan_tera_ulang"]
                    .sum()
                    .sort_values("jumlah_timbangan_tera_ulang", ascending=False)
                    .head(10)
                )
                st.markdown(f"**Top 10 Pasar (UTTP) di tahun {int(year_pick)} – Kecamatan {kec}**")
                st.bar_chart(top_pasar.set_index("nama_pasar")[["jumlah_timbangan_tera_ulang"]])

    # LEVEL 3 (pasar dipilih)
    else:
        st.caption(f"Menampilkan tren Pasar **{nama_pasar}** per tahun.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Tren Total UTTP**")
            line_chart_tight_y(agg_show, "Tahun", "total_uttp", "Total UTTP", pad_ratio=0.05)
        with c2:
            st.markdown("**Tren Total Pedagang**")
            line_chart_tight_y(agg_show, "Tahun", "total_pedagang", "Total Pedagang", pad_ratio=0.05)

        # Komposisi jenis timbangan per tahun (stacked)
        timb_cols = [c for c in [
            "Timb. Pegas","Timb. Meja","Timb. Elektronik","Timb. Sentisimal",
            "Timb. Bobot Ingsut","Neraca","Dacin"
        ] if c in gdf.columns]

        if timb_cols:
            import altair as alt
            g_year = gdf.groupby("tera_ulang_tahun", as_index=False)[timb_cols].sum()
            m2 = g_year.melt("tera_ulang_tahun", var_name="Jenis", value_name="Jumlah")

            st.markdown("**Komposisi Jenis Timbangan per Tahun**")
            chart = (
                alt.Chart(m2)
                .mark_bar()
                .encode(
                    x=alt.X("tera_ulang_tahun:O", title="Tahun"),
                    y=alt.Y("Jumlah:Q", title="Jumlah"),
                    color="Jenis:N",
                    tooltip=["tera_ulang_tahun:O", "Jenis:N", "Jumlah:Q"],
                )
                .properties(height=320)
            )
            st.altair_chart(chart, use_container_width=True)

# =========================
# DASHBOARD SPBU
# =========================
def render_dashboard_spbu():
    st.title("⛽ Dashboard SPBU – Kabupaten Tangerang")
    st.caption("Dinas Perindustrian dan Perdagangan • Bidang Kemetrologian")

    df_spbu = load_spbu_csv(FILE_SPBU)

    geo = None
    try:
        geo = load_geojson(FILE_GEOJSON)
    except Exception as e:
        st.warning(f"⚠️ GeoJSON tidak terbaca: {e}")

    # ===== SIDEBAR FILTER SPBU =====
    with st.sidebar:
        st.header("Filter SPBU")

        # media filter
        all_media = sorted({m for lst in df_spbu["media_list"] for m in lst})
        media_pick = st.multiselect("Media BBM", options=all_media, default=[], key="spbu_media_pick")

        base = df_spbu.copy()
        if media_pick:
            media_set = set(media_pick)
            base = base[base["media_list"].apply(lambda L: any(m in media_set for m in L))]

        # state prefix spbu_
        if "spbu_last_changed" not in st.session_state:
            st.session_state["spbu_last_changed"] = "kec"
        if "spbu_kec_sel" not in st.session_state:
            st.session_state["spbu_kec_sel"] = "(Semua)"
        if "spbu_name_sel" not in st.session_state:
            st.session_state["spbu_name_sel"] = "(Semua)"

        def _set_last(which: str):
            st.session_state["spbu_last_changed"] = which

        all_kec = uniq_clean(base["kecamatan"]) if "kecamatan" in base.columns else []
        all_spbu = uniq_clean(base["nama_spbu"]) if "nama_spbu" in base.columns else []

        # reset kalau state lama tidak ada (akibat filter media)
        if st.session_state["spbu_kec_sel"] != "(Semua)" and st.session_state["spbu_kec_sel"] not in (["(Semua)"] + all_kec):
            st.session_state["spbu_kec_sel"] = "(Semua)"
        if st.session_state["spbu_name_sel"] != "(Semua)" and st.session_state["spbu_name_sel"] not in (["(Semua)"] + all_spbu):
            st.session_state["spbu_name_sel"] = "(Semua)"

        kec_opsi = ["(Semua)"] + all_kec
        kec = st.selectbox("Kecamatan", kec_opsi, key="spbu_kec_sel", on_change=_set_last, args=("kec",))

        if st.session_state["spbu_last_changed"] == "kec" and st.session_state["spbu_kec_sel"] != "(Semua)":
            spbu_opsi = ["(Semua)"] + uniq_clean(base.loc[base["kecamatan"] == st.session_state["spbu_kec_sel"], "nama_spbu"])
        else:
            spbu_opsi = ["(Semua)"] + all_spbu

        nama_spbu = st.selectbox("Nama SPBU", spbu_opsi, key="spbu_name_sel", on_change=_set_last, args=("name",))

        # kalau pilih spbu dulu -> kec ikut
        if st.session_state["spbu_last_changed"] == "name" and nama_spbu != "(Semua)" and {"nama_spbu","kecamatan"}.issubset(base.columns):
            kec_auto = base.loc[base["nama_spbu"] == nama_spbu, "kecamatan"].dropna()
            if not kec_auto.empty:
                st.session_state["spbu_kec_sel"] = kec_auto.iloc[0]

        # kalau pilih kec dulu tapi spbu tidak cocok -> reset
        if st.session_state["spbu_last_changed"] == "kec" and kec != "(Semua)" and nama_spbu != "(Semua)":
            cek = base[(base["kecamatan"] == kec) & (base["nama_spbu"] == nama_spbu)]
            if cek.empty:
                st.session_state["spbu_name_sel"] = "(Semua)"

        kec = st.session_state["spbu_kec_sel"]
        nama_spbu = st.session_state["spbu_name_sel"]

        # kartu info spbu
        if nama_spbu != "(Semua)":
            info = base.loc[base["nama_spbu"] == nama_spbu].head(1)
            if not info.empty:
                almt = info["alamat"].iat[0]
                media_txt = info["media_bbm"].iat[0] if "media_bbm" in info.columns else "-"
                st.markdown("---")
                st.markdown(
                    f"""
                    <div style="
                        background-color:#f3e8ff;
                        padding:14px 16px;
                        border-radius:12px;
                        border-left:5px solid #8000FF;
                        box-shadow:0px 1px 4px rgba(0,0,0,0.15);
                        margin-top:10px;">
                        <h4 style="margin-bottom:6px; color:#4B0082; font-size:16px;">⛽ {nama_spbu}</h4>
                        <p style="margin:2px 0; font-size:13px; color:#222;">
                            <b>Kecamatan:</b> {kec}<br>
                            <b>Alamat:</b> {almt}<br>
                            <b>Media BBM:</b> {media_txt}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # ===== FILTER UTAMA SPBU =====
    fdf = base.copy()
    if kec != "(Semua)":
        fdf = fdf[fdf["kecamatan"] == kec]
    if nama_spbu != "(Semua)":
        fdf = fdf[fdf["nama_spbu"] == nama_spbu]

    # ===== KPI SPBU =====
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Kecamatan", clean_str_series(fdf["kecamatan"]).nunique() if "kecamatan" in fdf.columns else 0)
    with c2:
        st.metric("Total SPBU", clean_str_series(fdf["nama_spbu"]).nunique() if "nama_spbu" in fdf.columns else 0)
    with c3:
        media_unik = sorted({m for lst in fdf["media_list"] for m in lst})
        st.metric("Varian Media BBM", len(media_unik))

    # ===== MAP SPBU =====
    st.subheader("🗺️ Peta Lokasi SPBU")

    default_center = [-6.2, 106.55]
    default_zoom = 10

    coords = None
    if {"lat","lon"}.issubset(fdf.columns):
        coords = fdf[["lat","lon"]].dropna()

    center_loc = default_center
    zoom_start = default_zoom
    if nama_spbu != "(Semua)" and coords is not None and not coords.empty:
        center_loc = [float(coords.iloc[0]["lat"]), float(coords.iloc[0]["lon"])]
        zoom_start = 16

    m = folium.Map(location=center_loc, zoom_start=zoom_start, control_scale=True, tiles="OpenStreetMap")

    if geo is not None:
        tooltip = folium.GeoJsonTooltip(fields=["kec_label"], aliases=["Kecamatan:"])
        style_fn = lambda x: {"color":"#8000FF","weight":2,"opacity":1.0,"fill":False,"fillOpacity":0}
        try:
            folium.GeoJson(geo, name="Batas Kecamatan", style_function=style_fn, tooltip=tooltip).add_to(m)
        except Exception:
            pass

    if coords is not None and not coords.empty:
        cluster = MarkerCluster(name="SPBU", show=True).add_to(m)
        for _, r in fdf.iterrows():
            lat = r.get("lat"); lon = r.get("lon")
            if pd.isna(lat) or pd.isna(lon):
                continue
            nm = r.get("nama_spbu", "-")
            almt = r.get("alamat", "-")
            media_txt = r.get("media_bbm", "-")

            html = f"""
            <div style='width: 280px; font-family: Arial, sans-serif;'>
                <h4 style='margin:8px 0; color: #2E86AB;'>{nm}</h4>
                <div style='font-size: 12px; color:#666; margin-bottom:8px;'>{almt}</div>
                <hr style='margin:6px 0'/>
                <table style='font-size: 12px; width: 100%;'>
                    <tr><td><b>Kecamatan</b></td><td style='padding-left:8px'>: {r.get("kecamatan","-")}</td></tr>
                    <tr><td><b>Media BBM</b></td><td style='padding-left:8px'>: {media_txt}</td></tr>
                </table>
            </div>
            """

            folium.CircleMarker(
                location=[float(lat), float(lon)],
                radius=9,
                color="#8000FF",
                fill=True,
                fill_color="#8000FF",
                fill_opacity=0.65,
                weight=2,
                tooltip=folium.Tooltip(str(nm)),
                popup=folium.Popup(html, max_width=320),
            ).add_to(cluster)

        if nama_spbu == "(Semua)" and len(coords) > 1:
            sw = [coords["lat"].min(), coords["lon"].min()]
            ne = [coords["lat"].max(), coords["lon"].max()]
            m.fit_bounds([sw, ne], padding=(30, 30))
    else:
        st.warning("⚠️ Tidak ada koordinat valid untuk ditampilkan.")

    folium.LayerControl(collapsed=False).add_to(m)
    map_state = st_folium(m, height=520, width="stretch", key="spbu_map")

    # klik marker -> pilih spbu + kecamatan otomatis (pakai base supaya match filter media)
    if pick_from_click(
        map_state,
        df_context=base,
        name_col="nama_spbu",
        kec_col="kecamatan",
        state_prefix="spbu"
    ):
        st.rerun()

    # ===== GRAFIK SPBU =====
    st.subheader("📊 Analisis SPBU")

    if nama_spbu != "(Semua)" and not fdf.empty:
        media = fdf.iloc[0]["media_list"]
        if media:
            dd = pd.DataFrame({"Media BBM": media, "Jumlah": [1]*len(media)})
            st.markdown("**Media BBM pada SPBU terpilih**")
            bar_chart_alt(dd, "Media BBM", "Jumlah", "Jumlah")
        else:
            st.info("SPBU ini belum punya data Media BBM.")
    else:
        c1, c2 = st.columns(2)

        with c1:
            if not fdf.empty:
                by_kec = fdf.groupby("kecamatan").size().sort_values(ascending=False).head(15).reset_index()
                by_kec.columns = ["kecamatan", "jumlah_spbu"]
                st.markdown("**Top Kecamatan berdasarkan jumlah SPBU (sesuai filter)**")
                bar_chart_alt(by_kec, "kecamatan", "jumlah_spbu", "Jumlah SPBU")

        with c2:
            if not fdf.empty:
                rows = []
                for _, r in fdf.iterrows():
                    for m1 in r["media_list"]:
                        rows.append(m1)
                if rows:
                    by_media = pd.Series(rows).value_counts().head(10).reset_index()
                    by_media.columns = ["media", "jumlah_spbu"]
                    st.markdown("**Top Media BBM (jumlah SPBU yang menyediakan)**")
                    bar_chart_alt(by_media, "media", "jumlah_spbu", "Jumlah SPBU")


# =========================
# MENU (SIDEBAR) + ROUTING
# =========================
with st.sidebar:
    st.markdown("## 📌 Pilih Dashboard")
    page = st.radio(
        "Menu",
        ["🏪 Pasar (Tera Ulang)", "⛽ SPBU"],
        index=0,
        label_visibility="collapsed",
        key="page_menu"
    )
    st.markdown("---")

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
        year_pick = st.selectbox("Tahun Tera Ulang", options=year_options, index=0, key="year_pick")
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

    # kalau tahun berubah -> reset pilihan agar tidak nyangkut
    if int(year_pick) != int(st.session_state["year_prev"]):
        st.session_state["kec_sel"] = "(Semua)"
        st.session_state["pasar_sel"] = "(Semua)"
        st.session_state["year_prev"] = int(year_pick)

    # callback: saat kec berubah
    def on_kec_change():
        k = st.session_state["kec_sel"]
        if k != "(Semua)" and {'kecamatan', 'nama_pasar'}.issubset(df_y.columns):
            valid_pasar = ["(Semua)"] + uniq_clean(df_y.loc[df_y['kecamatan'] == k, 'nama_pasar'])
            if st.session_state["pasar_sel"] not in valid_pasar:
                st.session_state["pasar_sel"] = "(Semua)"

    # callback: saat pasar berubah
    def on_pasar_change():
        p = st.session_state["pasar_sel"]
        if p != "(Semua)" and {'nama_pasar', 'kecamatan'}.issubset(df_y.columns):
            kec_auto = df_y.loc[df_y['nama_pasar'] == p, 'kecamatan'].dropna()
            if not kec_auto.empty:
                st.session_state["kec_sel"] = kec_auto.iloc[0]

    # validasi state sebelum render widget (hindari "value not in options")
    kec_opsi = ["(Semua)"] + all_kec
    if st.session_state["kec_sel"] not in kec_opsi:
        st.session_state["kec_sel"] = "(Semua)"

    if st.session_state["kec_sel"] != "(Semua)" and {'kecamatan', 'nama_pasar'}.issubset(df_y.columns):
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
    if ('nama_pasar' in df_y.columns) and (nama_pasar != "(Semua)"):
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
                    margin-top:10px;">
                    <h4 style="margin-bottom:6px; color:#4B0082; font-size:16px;">🏪 {nama}</h4>
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

    # pakai df_y dulu (sudah tahun terpilih), lalu filter kec/pasar
    fdf_sidebar = df_y.copy()
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

# =========================
# FILTER DATA (UTAMA)
# =========================
# NOTE: pakai df_y (sudah terfilter tahun di sidebar)
fdf = df_y.copy()

if 'kecamatan' in fdf.columns and kec != "(Semua)":
    fdf = fdf[fdf['kecamatan'] == kec]

if 'nama_pasar' in fdf.columns and nama_pasar != "(Semua)":
    fdf = fdf[fdf['nama_pasar'] == nama_pasar]


# =========================
# KPIs (DINAMIS)
# =========================
def _safe_nunique(df_src: pd.DataFrame, col: str) -> int:
    if df_src is None or df_src.empty or col not in df_src.columns:
        return 0
    return int(clean_str_series(df_src[col]).nunique())

def _safe_sum(df_src: pd.DataFrame, col: str) -> int:
    if df_src is None or df_src.empty or col not in df_src.columns:
        return 0
    return int(pd.to_numeric(df_src[col], errors="coerce").fillna(0).sum())

year_show = int(year_pick)

# ===== Level 1: Semua Kecamatan & Semua Pasar =====
if kec == "(Semua)" and nama_pasar == "(Semua)":
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Kecamatan", _safe_nunique(fdf, "kecamatan"))
    with c2:
        st.metric("Total Seluruh Pasar", _safe_nunique(fdf, "nama_pasar"))
    with c3:
        st.metric("Tahun", year_show)
    with c4:
        st.metric("Total Timbangan", _safe_sum(fdf, "jumlah_timbangan_tera_ulang"))

# ===== Level 2: Kecamatan dipilih (Pasar semua) =====
elif kec != "(Semua)" and nama_pasar == "(Semua)":
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Kecamatan", kec)
    with c2:
        st.metric("Total Pasar", _safe_nunique(fdf, "nama_pasar"))
    with c3:
        st.metric("Tahun", year_show)
    with c4:
        st.metric("Total Timbangan", _safe_sum(fdf, "jumlah_timbangan_tera_ulang"))

# ===== Level 3: Pasar dipilih =====
else:
    # amankan kecamatan kalau user pilih pasar dulu dan kec belum kebaca
    kec_show = kec
    if kec_show == "(Semua)":
        if (fdf is not None) and (not fdf.empty) and ("kecamatan" in fdf.columns):
            vals = clean_str_series(fdf["kecamatan"])
            kec_show = vals.iloc[0] if len(vals) else "—"
        else:
            kec_show = "—"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Nama Pasar", nama_pasar if nama_pasar != "(Semua)" else "—")
    with c2:
        st.metric("Kecamatan", kec_show)
    with c3:
        st.metric("Tahun", year_show)
    with c4:
        st.metric("Total Timbangan", _safe_sum(fdf, "jumlah_timbangan_tera_ulang"))
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

# tentukan center & zoom
center_loc = default_center
zoom_start = default_zoom

if (nama_pasar != "(Semua)") and (coords is not None) and (not coords.empty):
    row_sel = fdf.head(1)  # karena fdf sudah terfilter pasar jika pasar dipilih
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

# --- bikin map (nama layer OSM dikosongkan biar tulisan "OpenStreetMap" tidak muncul) ---
m = folium.Map(location=center_loc, zoom_start=zoom_start, control_scale=True, tiles=None)
folium.TileLayer("OpenStreetMap", name=" ").add_to(m)

# Pane agar batas bisa di atas marker
from folium.map import CustomPane
m.add_child(CustomPane("batas-top", z_index=650))

# Load geojson
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
        try:
            gj_layer.options["pane"] = "batas-top"
        except Exception:
            pass
        gj_layer.add_to(m)

# marker pasar
if has_coords and coords is not None and not coords.empty:
    cluster = MarkerCluster(name="Pasar", show=True).add_to(m)

    # opsional: hindari marker dobel kalau ada duplikasi baris
    fdf_mark = fdf.dropna(subset=["lat", "lon"]).copy()
    if {"nama_pasar", "kecamatan", "lat", "lon"}.issubset(fdf_mark.columns):
        fdf_mark = fdf_mark.drop_duplicates(subset=["nama_pasar", "kecamatan", "lat", "lon"])

    for _, r in fdf_mark.iterrows():
        lat = pd.to_numeric(r.get("lat"), errors="coerce")
        lon = pd.to_numeric(r.get("lon"), errors="coerce")
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
                <tr><td><b>Tera Ulang</b></td><td style='padding-left:8px'>: {tahun if pd.notna(tahun) else '-'}</td></tr>
                <tr><td><b>Total UTTP</b></td><td style='padding-left:8px'>: {jumlah if pd.notna(jumlah) else '-'}</td></tr>
                <tr><td><b>Total Pedagang</b></td><td style='padding-left:8px'>: {int(pedagang) if pd.notna(pedagang) else '-'}</td></tr>
                <tr><td><b>Jenis Timbangan</b></td><td style='padding-left:8px'>: {jenis}</td></tr>
            </table>
        </div>
        """

        tooltip_text = f"{nama} - {tahun if pd.notna(tahun) else 'Tahun tidak diketahui'}"
        popup = folium.Popup(html, max_width=320)

        try:
            y = int(tahun) if pd.notna(tahun) else None
        except Exception:
            y = None

        col = marker_color(y, int(year_pick))

        folium.CircleMarker(
            location=[float(lat), float(lon)],
            radius=10,
            color=col,
            fill=True,
            fill_color=col,
            fill_opacity=0.7,
            weight=2,
            tooltip=folium.Tooltip(tooltip_text),
            popup=popup
        ).add_to(cluster)

    # Auto-fit bounds (kalau bukan mode pasar spesifik)
    if (nama_pasar == "(Semua)") and (coords is not None) and (len(coords) > 1):
        try:
            sw = [coords['lat'].min(), coords['lon'].min()]
            ne = [coords['lat'].max(), coords['lon'].max()]
            m.fit_bounds([sw, ne], padding=(30, 30))
        except Exception as e:
            st.warning(f"Tidak bisa auto-fit peta: {e}")
else:
    st.warning("⚠️ Tidak ada data koordinat yang valid untuk ditampilkan di peta")

folium.LayerControl(collapsed=False).add_to(m)

# =========================
# TAMPILKAN PETA (HANYA SEKALI)
# =========================
try:
    map_state = st_folium(m, height=500, width="stretch", key="pasar_map")
except TypeError:
    # fallback kalau versi streamlit_folium lama
    map_state = st_folium(m, height=500, use_container_width=True, key="pasar_map")


# =========================
# HANDLE KLIK MARKER -> pilih pasar + kecamatan otomatis (anti-loop rerun)
# =========================
def _pick_pasar_from_click(map_state: dict, df_context: pd.DataFrame) -> bool:
    if not map_state:
        return False

    clicked = map_state.get("last_object_clicked")
    if not clicked:
        return False

    latc = clicked.get("lat")
    lonc = clicked.get("lng")
    if latc is None or lonc is None:
        return False

    # anti loop: kalau klik yang sama, jangan rerun lagi
    click_key = (round(float(latc), 6), round(float(lonc), 6))
    prev = st.session_state.get("pasar_last_click")
    if prev == click_key:
        return False
    st.session_state["pasar_last_click"] = click_key

    if df_context is None or df_context.empty or not {'lat','lon','nama_pasar','kecamatan'}.issubset(df_context.columns):
        return False

    tmp = df_context[['lat','lon','nama_pasar','kecamatan']].dropna().copy()
    if tmp.empty:
        return False

    d2 = (tmp['lat'].astype(float) - float(latc))**2 + (tmp['lon'].astype(float) - float(lonc))**2
    idx = d2.idxmin()

    # threshold biar tidak salah pilih (sqrt(1e-8)=0.0001 deg ~ 11 m)
    if float(d2.loc[idx]) > 1e-8:
        return False

    pasar_clicked = str(df_context.loc[idx, 'nama_pasar'])
    kec_clicked   = str(df_context.loc[idx, 'kecamatan'])

    # sinkronkan dropdown sidebar (sesuai key sidebar terbaru kamu)
    st.session_state["pasar_sel"] = pasar_clicked
    st.session_state["kec_sel"] = kec_clicked

    return True

changed = _pick_pasar_from_click(map_state, fdf)

if changed:
    st.rerun()
# =========================
# GRAFIK (di bawah MAP)
# =========================
st.subheader("📈 Grafik Tren (Tahun ke Tahun)")

import altair as alt

def _show_altair(chart):
    """Aman untuk streamlit versi baru/lama (width vs use_container_width)."""
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:
        st.altair_chart(chart, use_container_width=True)

def line_chart_tight_y(df_plot: pd.DataFrame, x_col: str, y_col: str, y_title: str, pad_ratio: float = 0.06):
    d = df_plot[[x_col, y_col]].dropna().copy()
    if d.empty:
        st.info(f"Tidak ada data untuk {y_title}.")
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
            x=alt.X(f"{x_col}:O", title="Tahun", sort=list(d[x_col].astype(str).unique())),
            y=alt.Y(f"{y_col}:Q", title=y_title, scale=alt.Scale(domain=[y0, y1])),
            tooltip=[alt.Tooltip(f"{x_col}:O", title="Tahun"),
                     alt.Tooltip(f"{y_col}:Q", title=y_title)]
        )
        .properties(height=280)
    )
    _show_altair(chart)

# Grafik tren menyesuaikan pilihan user (kecamatan/pasar),
# tapi untuk tren tidak dibatasi year_pick (biar terlihat 2021–2025).
gdf = df.copy()

if 'kecamatan' in gdf.columns and kec != "(Semua)":
    gdf = gdf[gdf['kecamatan'] == kec]
if 'nama_pasar' in gdf.columns and nama_pasar != "(Semua)":
    gdf = gdf[gdf['nama_pasar'] == nama_pasar]

# pastikan tahun numerik
if 'tera_ulang_tahun' in gdf.columns:
    gdf['tera_ulang_tahun'] = pd.to_numeric(gdf['tera_ulang_tahun'], errors='coerce')
    gdf = gdf.dropna(subset=['tera_ulang_tahun'])
    gdf['tera_ulang_tahun'] = gdf['tera_ulang_tahun'].astype(int)

if gdf.empty or 'tera_ulang_tahun' not in gdf.columns:
    st.info("Tidak ada data untuk ditampilkan pada grafik (cek pilihan filter).")
else:
    # agregasi per tahun (TANPA jumlah_kecamatan)
    agg = (
        gdf.groupby('tera_ulang_tahun', as_index=False)
           .agg(
               jumlah_pasar=('nama_pasar', 'nunique') if 'nama_pasar' in gdf.columns else ('tera_ulang_tahun', 'size'),
               total_uttp=('jumlah_timbangan_tera_ulang', 'sum') if 'jumlah_timbangan_tera_ulang' in gdf.columns else ('tera_ulang_tahun', 'size'),
               total_pedagang=('total_pedagang', 'sum') if 'total_pedagang' in gdf.columns else ('tera_ulang_tahun', 'size'),
           )
           .sort_values('tera_ulang_tahun')
    )

    if agg.empty:
        st.info("Tidak ada data tren untuk pilihan ini.")
    else:
        agg_show = agg.copy()
        agg_show["Tahun"] = agg_show["tera_ulang_tahun"].astype(int).astype(str)

        # ====== Level 1: Semua Kecamatan & Semua Pasar ======
        if kec == "(Semua)" and nama_pasar == "(Semua)":
            st.caption("Menampilkan tren seluruh kabupaten per tahun.")
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
                line_chart_tight_y(agg_show, "Tahun", "jumlah_pasar", "Jumlah Pasar", pad_ratio=0.06)

            with c2:
                st.markdown("**Tren Total UTTP (di kecamatan terpilih)**")
                line_chart_tight_y(agg_show, "Tahun", "total_uttp", "Total UTTP", pad_ratio=0.06)

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
                line_chart_tight_y(agg_show, "Tahun", "total_uttp", "Total UTTP", pad_ratio=0.06)

            with c2:
                st.markdown("**Tren Total Pedagang (pasar terpilih)**")
                line_chart_tight_y(agg_show, "Tahun", "total_pedagang", "Total Pedagang", pad_ratio=0.06)

            # Komposisi jenis timbangan per tahun (stacked)
            timb_cols = [c for c in [
                'Timb. Pegas','Timb. Meja','Timb. Elektronik','Timb. Sentisimal',
                'Timb. Bobot Ingsut','Neraca','Dacin'
            ] if c in gdf.columns]

            if timb_cols:
                try:
                    g_year = gdf.groupby('tera_ulang_tahun', as_index=False)[timb_cols].sum()
                    g_year["Tahun"] = g_year["tera_ulang_tahun"].astype(int).astype(str)

                    m2 = g_year.melt('Tahun', value_vars=timb_cols, var_name='Jenis', value_name='Jumlah')

                    st.markdown("**Komposisi Jenis Timbangan per Tahun (pasar terpilih)**")
                    chart = (
                        alt.Chart(m2)
                        .mark_bar()
                        .encode(
                            x=alt.X('Tahun:O', title='Tahun'),
                            y=alt.Y('Jumlah:Q', title='Jumlah'),
                            color=alt.Color('Jenis:N', title='Jenis'),
                            tooltip=['Tahun:O', 'Jenis:N', 'Jumlah:Q']
                        )
                        .properties(height=320)
                    )
                    _show_altair(chart)

                except Exception as e:
                    st.warning(f"Grafik komposisi (Altair) tidak bisa dibuat: {e}")
                    st.dataframe(gdf.groupby('tera_ulang_tahun')[timb_cols].sum())
