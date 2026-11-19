import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap

# ================= KONFIGURASI =================
st.set_page_config(page_title="Dashboard Analisis Titanic", layout="wide")

GRAD = ["#7F5AF0", "#C026D3", "#FF6AC1", "#FF9E7A", "#00D4FF"]  # ungu -> magenta -> pink -> peach -> cyan
CARD, TEXT, GRID = "#1A1033", "#F8F5FF", "#3B2A66"
C_SELAMAT, C_TEWAS = "#00D4FF", "#FF6AC1"
CMAP = LinearSegmentedColormap.from_list("ungu_cyan", GRAD)

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg, #0F0A1F 0%, #1A0B33 50%, #08162E 100%); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2A1257 0%, #1A1033 55%, #0F0A1F 100%);
        border-right: 1px solid #7F5AF0;
    }
    [class*="st-key-bg_"] {
        background:
            linear-gradient(135deg, #1E1240 0%, #2B1658 55%, #1A1B4B 100%) padding-box,
            linear-gradient(135deg, #7F5AF0, #FF6AC1, #00D4FF) border-box;
        border: 2px solid transparent;
        border-radius: 16px;
        padding: 16px;
        box-shadow: 0 6px 24px rgba(127, 90, 240, 0.25);
    }
    .block-container { padding-top: 4rem; }
    div.judul {
        font-size: 3.8rem !important; font-weight: 900 !important; margin: 0.2rem 0 0.6rem 0;
        text-align: center; letter-spacing: 1px; line-height: 1.15;
        background: linear-gradient(90deg, #7F5AF0, #FF6AC1, #FF9E7A, #00D4FF);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    div.subjudul {
        font-size: 1.4rem !important; text-align: center; color: #CFC3FF;
        margin-bottom: 1.5rem; font-weight: 500 !important;
    }
    @media (max-width: 768px) {
        div.judul { font-size: 2.2rem !important; }
        div.subjudul { font-size: 1rem !important; }
    }
    [data-testid="stMetricValue"] {
        background: linear-gradient(90deg, #FF6AC1, #00D4FF);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    [data-testid="stMetricLabel"] { color: #CFC3FF; }
    </style>
    """,
    unsafe_allow_html=True,
)
sns.set_theme(style="darkgrid")

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
    d = pd.read_csv(url)
    d["status"] = d["survived"].map({1: "Selamat", 0: "Tidak Selamat"})
    return d

df = load_data()

# ================= FILTER =================
st.sidebar.header("🎨 Filter Dashboard")
st.sidebar.caption("Atur pilihan data sesuai kebutuhan analisis.")

pilih_kelas = st.sidebar.selectbox("Kelas Tiket (Pclass)", ["All"] + sorted(df["pclass"].unique().tolist()))
pilih_sex = st.sidebar.selectbox("Jenis Kelamin", ["All"] + sorted(df["sex"].unique().tolist()))
pilih_port = st.sidebar.selectbox("Pelabuhan Keberangkatan", ["All"] + sorted(df["embark_town"].dropna().unique().tolist()))
pilih_status = st.sidebar.selectbox("Status Keselamatan", ["All", "Selamat", "Tidak Selamat"])

umur_min, umur_max = int(df["age"].min()), int(df["age"].max())
pilih_umur = st.sidebar.slider("Rentang Usia", umur_min, umur_max, (umur_min, umur_max))
termasuk_kosong = st.sidebar.checkbox("Sertakan usia yang kosong (NaN)", value=True)

cond_umur = df["age"].between(*pilih_umur)
if termasuk_kosong:
    cond_umur |= df["age"].isna()

mask = cond_umur
if pilih_kelas != "All":
    mask &= df["pclass"] == pilih_kelas
if pilih_sex != "All":
    mask &= df["sex"] == pilih_sex
if pilih_port != "All":
    mask &= df["embark_town"] == pilih_port
if pilih_status != "All":
    mask &= df["status"] == pilih_status

dff = df[mask]

if dff.empty:
    st.warning("Tidak ada data yang sesuai dengan filter. Silakan ubah pilihan filter atau pilih All.")
    st.stop()

# ================= JUDUL & KPI =================
st.markdown('<div class="judul">DASHBOARD ANALISIS PENUMPANG TITANIC</div>', unsafe_allow_html=True)
st.markdown('<div class="subjudul">Tugas Evaluasi Praktikum 9 Sains Data — Dataset Titanic | Arda Niansa (F1A224005)</div>', unsafe_allow_html=True)

with st.container(border=True, key="bg_kpi"):
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Jumlah Penumpang", f"{len(dff):,}")
    k2.metric("Tingkat Keselamatan", f"{dff['survived'].mean() * 100:.1f}%")
    k3.metric("Jumlah Selamat", f"{int(dff['survived'].sum()):,}")
    k4.metric("Rata-rata Usia", f"{dff['age'].mean():.1f} th")
    k5.metric("Rata-rata Tarif", f"${dff['fare'].mean():,.2f}")

# ================= HELPER STYLE =================
def gaya(fig, ax, xl="", yl=""):
    fig.patch.set_facecolor(CARD)
    ax.set_facecolor(CARD)
    ax.set_xlabel(xl, color=TEXT, fontweight="bold")
    ax.set_ylabel(yl, color=TEXT, fontweight="bold")
    ax.tick_params(colors=TEXT)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)

def gaya_legend(ax, judul):
    leg = ax.get_legend()
    if leg:
        leg.set_title(judul)
        leg.get_title().set_color(TEXT)
        leg.get_frame().set_facecolor(CARD)
        leg.get_frame().set_edgecolor(GRID)
        for t in leg.get_texts():
            t.set_color(TEXT)

COL_SEX = {"female": "#FF6AC1", "male": "#7F5AF0"}
COL_KELAS = {1: "#7F5AF0", 2: "#C026D3", 3: "#FF9E7A"}
FLIER = dict(markerfacecolor="#00D4FF", markeredgecolor="#00D4FF", markersize=3)

def donut(ax, fig, hitung, warna):
    fig.patch.set_facecolor(CARD)
    ax.pie(hitung.values, labels=hitung.index, colors=warna, autopct="%1.1f%%",
           startangle=90, pctdistance=0.78,
           wedgeprops=dict(edgecolor=CARD, linewidth=2, width=0.55),
           textprops=dict(color=TEXT, fontsize=10))

def heat(ax, fig, data, fmt):
    fig.patch.set_facecolor(CARD)
    ax.set_facecolor(CARD)
    sns.heatmap(data, annot=True, fmt=fmt, cmap=CMAP, linecolor=CARD, linewidths=1.5, ax=ax,
                annot_kws=dict(color="white", fontweight="bold"))
    ax.tick_params(colors=TEXT)
    ax.set_xlabel(ax.get_xlabel(), color=TEXT, fontweight="bold")
    ax.set_ylabel(ax.get_ylabel(), color=TEXT, fontweight="bold")
    ax.collections[0].colorbar.ax.tick_params(colors=TEXT)

# ================= BOXPLOT (2 grafik) =================
st.markdown("#### Distribusi Usia dan Tarif Berdasarkan Kelas Tiket")
with st.container(border=True, key="bg_box"):
    b1, b2 = st.columns(2)
    with b1:
        st.markdown("**Boxplot Usia per Kelas Tiket dan Gender**")
        fig, ax = plt.subplots()
        sns.boxplot(dff, x="pclass", y="age", hue="sex", palette=COL_SEX,
                    linecolor="#F8F5FF", linewidth=1.2, flierprops=FLIER, ax=ax)
        gaya(fig, ax, "Kelas Tiket", "Usia (tahun)")
        gaya_legend(ax, "Gender")
        st.pyplot(fig); plt.close(fig)
    with b2:
        st.markdown("**Boxplot Tarif per Kelas Tiket dan Gender**")
        fig, ax = plt.subplots()
        sns.boxplot(dff, x="pclass", y="fare", hue="sex", palette=COL_SEX,
                    linecolor="#F8F5FF", linewidth=1.2, flierprops=FLIER, ax=ax)
        gaya(fig, ax, "Kelas Tiket", "Tarif ($)")
        gaya_legend(ax, "Gender")
        st.pyplot(fig); plt.close(fig)

# ================= BARPLOT (2 grafik) =================
st.markdown("#### Keselamatan dan Jumlah Penumpang")
with st.container(border=True, key="bg_bar"):
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("**Tingkat Keselamatan per Kelas dan Gender**")
        tabel = dff.groupby(["pclass", "sex"])["survived"].mean().mul(100).reset_index()
        fig, ax = plt.subplots()
        sns.barplot(tabel, x="pclass", y="survived", hue="sex", palette=COL_SEX, ax=ax)
        gaya(fig, ax, "Kelas Tiket", "Selamat (%)")
        for cont in ax.containers:
            ax.bar_label(cont, fmt="%.1f%%", color=TEXT, fontsize=9, padding=3)
        gaya_legend(ax, "Gender")
        st.pyplot(fig); plt.close(fig)
    with r2:
        st.markdown("**Jumlah Penumpang per Kelas Tiket**")
        jml = dff["pclass"].value_counts().sort_index().reset_index()
        jml.columns = ["pclass", "jumlah"]
        fig, ax = plt.subplots()
        sns.barplot(jml, x="pclass", y="jumlah", hue="pclass", palette=COL_KELAS,
                    legend=False, ax=ax)
        gaya(fig, ax, "Kelas Tiket", "Jumlah Penumpang")
        for cont in ax.containers:
            ax.bar_label(cont, fmt="%d", color=TEXT, fontsize=9, padding=3)
        st.pyplot(fig); plt.close(fig)

# ================= PIE CHART (2 grafik) =================
st.markdown("#### Proporsi Penumpang")
with st.container(border=True, key="bg_pie"):
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("**Proporsi Penumpang per Pelabuhan**")
        hitung = dff["embark_town"].value_counts()
        fig, ax = plt.subplots()
        donut(ax, fig, hitung, ["#7F5AF0", "#FF6AC1", "#00D4FF"][:len(hitung)])
        st.pyplot(fig); plt.close(fig)
    with p2:
        st.markdown("**Proporsi Status Keselamatan**")
        hitung = dff["status"].value_counts()
        fig, ax = plt.subplots()
        donut(ax, fig, hitung, [C_SELAMAT if k == "Selamat" else C_TEWAS for k in hitung.index])
        st.pyplot(fig); plt.close(fig)

# ================= HISTOGRAM (2 grafik) =================
st.markdown("#### Distribusi Usia dan Tarif")
with st.container(border=True, key="bg_hist"):
    h1, h2 = st.columns(2)
    with h1:
        st.markdown("**Distribusi Usia Penumpang**")
        fig, ax = plt.subplots()
        sns.histplot(dff.dropna(subset=["age"]), x="age", hue="status", bins=25,
                     multiple="stack", palette={"Selamat": C_SELAMAT, "Tidak Selamat": C_TEWAS},
                     edgecolor=CARD, ax=ax)
        gaya(fig, ax, "Usia (tahun)", "Jumlah Penumpang")
        gaya_legend(ax, "Status")
        st.pyplot(fig); plt.close(fig)
    with h2:
        st.markdown("**Distribusi Tarif Tiket**")
        fig, ax = plt.subplots()
        sns.histplot(dff["fare"], bins=30, kde=True, color="#C026D3", edgecolor=CARD, ax=ax,
                     line_kws=dict(color="#00D4FF", linewidth=2))
        gaya(fig, ax, "Tarif ($)", "Jumlah Penumpang")
        st.pyplot(fig); plt.close(fig)

# ================= HEATMAP (2 grafik) =================
st.markdown("#### Korelasi dan Sebaran Penumpang")
with st.container(border=True, key="bg_heat"):
    m1, m2 = st.columns(2)
    with m1:
        st.markdown("**Heatmap Korelasi Variabel Numerik**")
        kolom = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
        fig, ax = plt.subplots()
        heat(ax, fig, dff[kolom].corr(), ".2f")
        st.pyplot(fig); plt.close(fig)
    with m2:
        st.markdown("**Heatmap Jumlah Penumpang: Kelas Tiket dan Pelabuhan**")
        silang = pd.crosstab(dff["pclass"], dff["embark_town"])
        fig, ax = plt.subplots()
        heat(ax, fig, silang, "d")
        ax.set_xlabel("Pelabuhan", color=TEXT, fontweight="bold")
        ax.set_ylabel("Kelas Tiket", color=TEXT, fontweight="bold")
        st.pyplot(fig); plt.close(fig)

# ================= VIEW DATASET + STATISTIK DESKRIPTIF =================
with st.expander("**Dataset Overview**"):
    data_asli = dff.drop(columns="status")
    st.markdown("### Data Titanic")
    st.write(f"Jumlah baris: **{len(data_asli):,}** | Jumlah kolom: **{data_asli.shape[1]}**")
    st.dataframe(data_asli, width="stretch")

    st.markdown("### 📊 Statistik Deskriptif")
    tab_num, tab_kat = st.tabs(["Variabel Numerik", "Variabel Kategorik"])

    with tab_num:
        num = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
        stat = dff[num].describe().T
        stat["median"] = dff[num].median()
        stat["skewness"] = dff[num].skew()
        stat["kosong"] = dff[num].isna().sum()
        stat = stat[["count", "kosong", "mean", "median", "std", "min", "25%", "75%", "max", "skewness"]]
        stat.columns = ["Jumlah Data", "Data Kosong", "Rata-rata", "Median", "Std Dev",
                        "Min", "Q1 (25%)", "Q3 (75%)", "Maks", "Skewness"]
        st.dataframe(stat.round(3), width="stretch")
        st.caption("Skewness > 0 berarti distribusi menceng ke kanan (ekor panjang di nilai besar). "
                   "Nilai rata-rata `survived` adalah tingkat keselamatan.")

    with tab_kat:
        baris = []
        for kol in ["sex", "pclass", "embark_town", "who", "status"]:
            frek = dff[kol].value_counts()
            baris.append({
                "Variabel": kol,
                "Jumlah Kategori": dff[kol].nunique(),
                "Modus": frek.index[0],
                "Frekuensi Modus": int(frek.iloc[0]),
                "Persentase Modus (%)": round(frek.iloc[0] / len(dff) * 100, 2),
            })
        st.dataframe(pd.DataFrame(baris).set_index("Variabel").astype({"Modus": str}),
                     width="stretch")
