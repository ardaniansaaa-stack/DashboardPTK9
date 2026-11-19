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
    .judul {
        font-size: 2.4rem; font-weight: 800; margin-bottom: 0;
        background: linear-gradient(90deg, #7F5AF0, #FF6AC1, #FF9E7A, #00D4FF);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
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

kelas_opsi = sorted(df["pclass"].unique())
pilih_kelas = st.sidebar.multiselect("Kelas Tiket (Pclass)", kelas_opsi, default=kelas_opsi)

sex_opsi = sorted(df["sex"].unique())
pilih_sex = st.sidebar.multiselect("Jenis Kelamin", sex_opsi, default=sex_opsi)

port_opsi = sorted(df["embark_town"].dropna().unique())
pilih_port = st.sidebar.multiselect("Pelabuhan Keberangkatan", port_opsi, default=port_opsi)

status_opsi = ["Selamat", "Tidak Selamat"]
pilih_status = st.sidebar.multiselect("Status Keselamatan", status_opsi, default=status_opsi)

umur_min, umur_max = int(df["age"].min()), int(df["age"].max())
pilih_umur = st.sidebar.slider("Rentang Usia", umur_min, umur_max, (umur_min, umur_max))
termasuk_kosong = st.sidebar.checkbox("Sertakan usia yang kosong (NaN)", value=True)

cond_umur = df["age"].between(*pilih_umur)
if termasuk_kosong:
    cond_umur |= df["age"].isna()

dff = df[
    df["pclass"].isin(pilih_kelas)
    & df["sex"].isin(pilih_sex)
    & (df["embark_town"].isin(pilih_port) | df["embark_town"].isna())
    & df["status"].isin(pilih_status)
    & cond_umur
]

if dff.empty:
    st.warning("Tidak ada data yang sesuai dengan filter. Silakan pilih minimal satu opsi pada setiap filter.")
    st.stop()

# ================= JUDUL & KPI =================
st.markdown('<p class="judul">DASHBOARD ANALISIS PENUMPANG TITANIC</p>', unsafe_allow_html=True)
st.caption("Tugas Evaluasi Praktikum 9 Sains Data — Dataset Titanic (NIM Ganjil)")

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

# ================= BARIS 1: BOXPLOT, BARPLOT, PIE =================
with st.container(border=True, key="bg_row1"):
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Boxplot Usia per Kelas Tiket dan Gender**")
        fig, ax = plt.subplots()
        sns.boxplot(dff, x="pclass", y="age", hue="sex",
                    palette={"female": "#FF6AC1", "male": "#7F5AF0"},
                    linecolor="#F8F5FF", linewidth=1.2, ax=ax,
                    flierprops=dict(markerfacecolor="#00D4FF", markeredgecolor="#00D4FF"))
        gaya(fig, ax, "Kelas Tiket", "Usia (tahun)")
        gaya_legend(ax, "Gender")
        st.pyplot(fig); plt.close(fig)

    with c2:
        st.markdown("**Tingkat Keselamatan per Kelas dan Gender**")
        tabel = dff.groupby(["pclass", "sex"])["survived"].mean().mul(100).reset_index()
        fig, ax = plt.subplots()
        sns.barplot(tabel, x="pclass", y="survived", hue="sex",
                    palette={"female": "#FF6AC1", "male": "#7F5AF0"}, ax=ax)
        gaya(fig, ax, "Kelas Tiket", "Selamat (%)")
        for cont in ax.containers:
            ax.bar_label(cont, fmt="%.1f%%", color=TEXT, fontsize=9, padding=3)
        gaya_legend(ax, "Gender")
        st.pyplot(fig); plt.close(fig)

    with c3:
        st.markdown("**Proporsi Penumpang per Pelabuhan**")
        hitung = dff["embark_town"].value_counts()
        fig, ax = plt.subplots()
        fig.patch.set_facecolor(CARD)
        ax.pie(hitung.values, labels=hitung.index, colors=["#7F5AF0", "#FF6AC1", "#00D4FF"][:len(hitung)],
               autopct="%1.1f%%", startangle=90, pctdistance=0.78,
               wedgeprops=dict(edgecolor=CARD, linewidth=2, width=0.55),
               textprops=dict(color=TEXT, fontsize=10))
        st.pyplot(fig); plt.close(fig)

# ================= BARIS 2: HISTOGRAM & HEATMAP =================
c_hist, c_heat = st.columns(2)

with c_hist:
    with st.container(border=True, key="bg_hist"):
        st.markdown("**Distribusi Usia Penumpang**")
        fig, ax = plt.subplots()
        sns.histplot(dff.dropna(subset=["age"]), x="age", hue="status", bins=25,
                     multiple="stack", palette={"Selamat": C_SELAMAT, "Tidak Selamat": C_TEWAS},
                     edgecolor=CARD, ax=ax)
        gaya(fig, ax, "Usia (tahun)", "Jumlah Penumpang")
        gaya_legend(ax, "Status")
        st.pyplot(fig); plt.close(fig)

with c_heat:
    with st.container(border=True, key="bg_heat"):
        st.markdown("**Heatmap Korelasi Variabel Numerik**")
        kolom = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
        korelasi = dff[kolom].corr()
        fig, ax = plt.subplots()
        fig.patch.set_facecolor(CARD)
        ax.set_facecolor(CARD)
        sns.heatmap(korelasi, annot=True, fmt=".2f", cmap=CMAP, vmin=-1, vmax=1,
                    linecolor=CARD, linewidths=1.5, ax=ax,
                    annot_kws=dict(color="white", fontweight="bold"))
        ax.tick_params(colors=TEXT)
        st.pyplot(fig); plt.close(fig)

# ================= VIEW DATASET =================
with st.expander("**Dataset Overview**"):
    st.dataframe(dff, use_container_width=True)
