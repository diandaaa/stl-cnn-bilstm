"""
Dashboard Analisis Time Series – CNN-BiLSTM + STL Decomposition
Backend  : PyTorch (kompatibel Python 3.14 / Streamlit Cloud)
Run      : streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Batasi jumlah thread PyTorch
import torch
torch.set_num_threads(2)

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SST · CNN-BiLSTM + STL",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
h1,h2,h3{font-family:'Space Mono',monospace;}

.sec-title {
    font-family:'Space Mono',monospace;
    font-size:1rem;
    font-weight:700;
    color:#e2e8f0;
    background:linear-gradient(90deg,#1e3a5f,transparent);
    border-left:4px solid #38bdf8;
    padding:.45rem .8rem;
    margin:1.2rem 0 .7rem;
    border-radius:0 6px 6px 0;
    letter-spacing:.04em;
}
.metric-card{
    background:linear-gradient(135deg,#1a2236,#151c30);
    border:1px solid #2d3a56;border-radius:12px;
    padding:1rem 1.2rem;text-align:center;margin-bottom:.5rem;
}
.metric-card .label{font-size:.7rem;color:#94a3b8;letter-spacing:.12em;text-transform:uppercase;}
.metric-card .value{font-size:1.45rem;font-family:'Space Mono',monospace;color:#38bdf8;font-weight:700;}
.metric-card .sub{font-size:.72rem;color:#64748b;margin-top:2px;}
.badge-ok {background:#064e3b;color:#34d399;padding:3px 10px;border-radius:999px;font-size:.78rem;}
.badge-err{background:#450a0a;color:#f87171;padding:3px 10px;border-radius:999px;font-size:.78rem;}
.narasi{
    background:#111827;border:1px solid #1e3a5f;border-radius:8px;
    padding:.8rem 1rem;color:#cbd5e1;font-size:.88rem;line-height:1.6;
    margin:.6rem 0 1rem;
}
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor":"#0e1525","axes.facecolor":"#0e1525",
    "axes.edgecolor":"#2d3a56","axes.labelcolor":"#94a3b8",
    "xtick.color":"#64748b","ytick.color":"#64748b",
    "text.color":"#e2e8f0","grid.color":"#1e2d45","grid.alpha":.6,
    "legend.facecolor":"#111827","legend.edgecolor":"#2d3a56",
    "legend.fontsize":8,"font.family":"monospace",
    "axes.titlecolor":"#e2e8f0","axes.titlesize":10,"axes.titleweight":"bold",
})
PAL = dict(actual="#38bdf8",train="#34d399",val="#fbbf24",
           test="#f87171",trend="#a78bfa",season="#fb923c",resid="#94a3b8",
           future="#f0abfc")

def sec(txt):
    st.markdown(f'<div class="sec-title">{txt}</div>', unsafe_allow_html=True)

def narasi(txt):
    st.markdown(f'<div class="narasi">💡 {txt}</div>', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PYTORCH MODEL dengan Batch Normalization (opsional)
# ═════════════════════════════════════════════════════════════════════════════
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class CNNBiLSTM(nn.Module):
    def __init__(self, lookback, conv_filters, kernel_size,
                 lstm_units, dense_units, dropout=0.0, use_bn=False):
        super().__init__()
        self.use_bn = use_bn
        self.pad    = nn.ConstantPad1d((kernel_size-1, 0), 0)
        self.conv   = nn.Conv1d(1, conv_filters, kernel_size)
        self.bn_conv = nn.BatchNorm1d(conv_filters) if use_bn else nn.Identity()
        self.relu   = nn.ReLU()
        self.bilstm = nn.LSTM(conv_filters, lstm_units,
                              batch_first=True, bidirectional=True)
        self.drop   = nn.Dropout(dropout)
        self.fc1    = nn.Linear(lstm_units*2, dense_units)
        self.bn_fc  = nn.BatchNorm1d(dense_units) if use_bn else nn.Identity()
        self.fc2    = nn.Linear(dense_units, 1)

    def forward(self, x):
        x = x.permute(0,2,1)               # (B,1,L)
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn_conv(x)
        x = self.relu(x)
        x = x.permute(0,2,1)               # (B,L,conv_filters)
        out,_ = self.bilstm(x)
        x = self.drop(out[:,-1,:])          # last timestep
        x = self.relu(self.fc1(x))
        x = self.bn_fc(x)
        return self.fc2(x).squeeze(-1)


def train_model(model, X_tr, y_tr, X_val, y_val,
                epochs, batch_size, lr, patience=20, use_bn=False,
                weight_decay=1e-5):
    loss_fn = nn.MSELoss()  # selalu MSE
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)

    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    Xv = torch.tensor(X_val, dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.float32)

    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=False)

    best_val, best_state, wait = float("inf"), None, 0
    hist_tr, hist_val = [], []

    for _ in range(epochs):
        model.train()
        ep = []
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb.unsqueeze(-1)), yb)
            loss.backward()
            opt.step()
            ep.append(loss.item())
        tr = float(np.mean(ep))

        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xv.unsqueeze(-1)), yv).item()

        hist_tr.append(tr)
        hist_val.append(vl)
        sched.step(vl)

        if vl < best_val - 1e-6:
            best_val = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return hist_tr, hist_val


def predict_model(model, X):
    model.eval()
    with torch.no_grad():
        t = torch.tensor(X, dtype=torch.float32)
        return model(t.unsqueeze(-1)).numpy().flatten()


def recursive_forecast(model, last_window, steps):
    model.eval()
    w = last_window.copy().tolist()
    out = []
    lb = len(last_window)
    with torch.no_grad():
        for _ in range(steps):
            xin = torch.tensor(w[-lb:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            p = model(xin).item()
            out.append(p)
            w.append(p)
    return np.array(out)

# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def mcard(col, label, value, sub=""):
    col.markdown(f"""<div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        <div class="sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

def mape(yt, yp):
    yt, yp = np.array(yt), np.array(yp)
    m = yt != 0
    return np.mean(np.abs((yt[m] - yp[m]) / yt[m])) * 100

def build_dataset(arr, lb):
    X, y = [], []
    for i in range(lb, len(arr)):
        X.append(arr[i-lb:i])
        y.append(arr[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def mae_fn(yt, yp):
    from sklearn.metrics import mean_absolute_error
    return mean_absolute_error(yt, yp)

def fungsi_spektral(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    k = round((n - 1) / 2)
    t = np.arange(1, n + 1)
    pg = np.zeros(k)
    for i in range(1, k + 1):
        w = (2 * np.pi * i) / n
        a = (2 / n) * np.sum(x * np.cos(w * t))
        b = (2 / n) * np.sum(x * np.sin(w * t))
        pg[i-1] = a**2 + b**2
    km = np.argmax(pg) + 1
    per = int(round((2 * np.pi) / ((2 * np.pi * km) / n)))
    Th = np.max(pg) / np.sum(pg)
    Tt = 0.13135
    return per, Th, Tt, Th > Tt, pg

def fdGPH(x, bw=0.5):
    import statsmodels.api as sm
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    m = int(np.floor(n**bw))
    j = np.arange(1, m + 1)
    lam = 2 * np.pi * j / n
    fv = np.fft.fft(x)
    I = (1 / (2 * np.pi * n)) * np.abs(fv[j])**2
    Y = np.log(I)
    Xm = sm.add_constant(np.log(4 * (np.sin(lam / 2)**2)))
    mo = sm.OLS(Y, Xm).fit()
    return -mo.params[1], mo

def generate_random_sst(n=4071, seed=42):
    """Sintetis mirip SST tropis: range ~28-31°C, seasonal lemah, noise rendah."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    trend = 29.3 + 0.00008 * t
    seasonal = 0.35 * np.sin(2 * np.pi * t / 365 - 0.3)
    noise = rng.normal(0, 0.08, n)
    for i in range(1, n):
        noise[i] += 0.6 * noise[i-1]
    sst = trend + seasonal + noise
    dates = pd.date_range("2015-01-01", periods=n, freq="D")
    return pd.DataFrame({"tgl": dates.strftime("%-m/%-d/%Y"), "sst": np.round(sst, 5)})

def tight_ylim(ax, data_list, pad=0.15):
    """Set y-axis limits with tight padding around actual data range."""
    all_vals = np.concatenate([np.asarray(d).flatten() for d in data_list if len(d) > 0])
    lo, hi = np.nanmin(all_vals), np.nanmax(all_vals)
    r = (hi - lo) * pad
    ax.set_ylim(lo - r, hi + r)

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌊 SST Forecast")
    st.markdown("**CNN-BiLSTM + STL**")
    st.caption("Backend: PyTorch")
    st.divider()

    st.markdown("### 📂 Data Source")
    data_source = st.radio("Pilih sumber data:",
                           ["📁 Upload CSV", "🎲 Generate Data Contoh"])
    uploaded = None
    if data_source == "📁 Upload CSV":
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        st.caption("Minimal: kolom tanggal + kolom SST numerik")
    else:
        gen_n = st.slider("Jumlah hari", 365, 5000, 4071, 365)
        gen_seed = st.number_input("Seed data", value=42)

    date_col = st.text_input("Nama kolom tanggal", value="tgl")
    sst_col = st.text_input("Nama kolom target", value="sst")
    st.divider()

    st.markdown("### 🔧 Data Split")
    train_r = st.slider("Train ratio", 0.50, 0.95, 0.90, 0.01)
    val_r = st.slider("Val ratio", 0.01, 0.20, 0.05, 0.01)
    test_r = round(1 - train_r - val_r, 4)
    st.markdown(f"**Test ratio (auto):** `{max(test_r, 0):.2f}`")

    st.markdown("### 📅 STL")
    auto_period = st.checkbox("Auto-detect period (spektral)", value=True)
    manual_period = 180
    if not auto_period:
        manual_period = st.number_input("Period manual", 2, 730, 180)
    stl_robust = st.checkbox("STL robust", value=True)

    st.markdown("### 🔧 Lookback")
    lookback = st.slider("Lookback (umum)", 30, 365, 90, 10)
    separate_lookback = st.checkbox("Gunakan lookback berbeda untuk seasonal", value=True)
    if separate_lookback:
        lookback_seasonal = st.slider("Lookback Seasonal", 30, 365, 180, 10)
    else:
        lookback_seasonal = lookback

    st.markdown("### 🧠 Trend Model")
    t_conv_f = st.slider("Conv1D filters", 16, 128, 32, 16)
    t_kern = st.slider("Kernel size", 2, 15, 5, 1)
    t_lstm = st.slider("BiLSTM units", 16, 128, 64, 16)
    t_drop = st.slider("Dropout", 0.0, 0.5, 0.2, 0.05)
    t_dense = st.slider("Dense units", 16, 128, 32, 8)
    t_lr = st.number_input("LR trend", value=0.0007, format="%.4f")

    st.markdown("### 🧠 Seasonal Model")
    s_conv_f = st.slider("Conv1D filters (S)", 32, 256, 128, 16)
    s_kern = st.slider("Kernel size (S)", 2, 15, 5, 1)
    s_lstm = st.slider("BiLSTM units (S)", 32, 256, 128, 16)
    s_drop = st.slider("Dropout Seasonal", 0.0, 0.5, 0.2, 0.05)
    s_dense = st.slider("Dense units (S)", 32, 128, 64, 8)
    s_lr = st.number_input("LR seasonal", value=0.0003, format="%.4f")

    st.markdown("### ⚙️ Training")
    epochs = st.slider("Max epochs", 10, 200, 100, 10)
    batch_size = st.selectbox("Batch size", [16, 32, 64, 128], index=2)
    patience = st.slider("Early stopping patience", 10, 50, 20, 5)
    use_bn = st.checkbox("Gunakan Batch Normalization", value=False)
    scaler_type = st.selectbox("Scaler", ["MinMax", "Standard"], index=0)
    weight_decay = st.number_input("Weight decay", value=1e-5, format="%.6f", step=1e-5)
    seed = st.number_input("Random seed", value=42)

    st.divider()
    run_btn = st.button("▶  Run Analysis", use_container_width=True, type="primary")

# ═════════════════════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("# 🌊 SST Time Series Analysis Dashboard")
st.markdown("*Hybrid CNN-BiLSTM + STL Decomposition · Sea Surface Temperature Forecasting*")

if not run_btn:
    st.info("👈 Pilih sumber data & atur parameter di sidebar, lalu klik **▶ Run Analysis**.")
    with st.expander("📋 Format CSV yang dibutuhkan"):
        s = generate_random_sst(10, 42)
        st.dataframe(s, use_container_width=True)
        st.caption("Kolom minimal: tanggal (`tgl`) + nilai SST (`sst`). Nama bisa diubah di sidebar.")
        st.download_button("⬇ Download contoh CSV",
                           s.to_csv(index=False).encode(), "contoh_data.csv", "text/csv")
    st.stop()

# ── set seed & imports ────────────────────────────────────────────────────────
torch.manual_seed(int(seed))
np.random.seed(int(seed))
random.seed(int(seed))

try:
    from statsmodels.tsa.seasonal import STL
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# ── load data ─────────────────────────────────────────────────────────────────
if data_source == "🎲 Generate Data Contoh":
    df_raw = generate_random_sst(int(gen_n), int(gen_seed))
    st.success(f"✅ Data sintetis: {len(df_raw):,} baris")
else:
    if uploaded is None:
        st.error("⚠️ Upload file CSV terlebih dahulu.")
        st.stop()
    df_raw = pd.read_csv(uploaded)

if date_col not in df_raw.columns or sst_col not in df_raw.columns:
    st.error(f"Kolom `{date_col}` atau `{sst_col}` tidak ada.\n"
             f"Kolom tersedia: `{list(df_raw.columns)}`")
    st.stop()

df = df_raw.copy()
df[date_col] = pd.to_datetime(df[date_col], dayfirst=False, infer_datetime_format=True)
df = df.sort_values(date_col).set_index(date_col)
y_full = df[sst_col].values.astype(float)
dates = df.index
n = len(y_full)

n_train = int(n * train_r)
n_val = int(n * val_r)
n_test = n - n_train - n_val
if n_test <= 0:
    st.error("Test set kosong. Kurangi train/val ratio.")
    st.stop()

y_trainval = y_full[:n_train + n_val]

# ═════════════════════════════════════════════════════════════════════════════
# STL (dijalankan di awal, dipakai banyak tab)
# ═════════════════════════════════════════════════════════════════════════════
with st.spinner("Running STL decomposition..."):
    periode = fungsi_spektral(y_trainval)[0] if auto_period else manual_period
    stl_res = STL(y_trainval, period=periode, robust=stl_robust).fit()

trend_tv = stl_res.trend
season_tv = stl_res.seasonal

# ═════════════════════════════════════════════════════════════════════════════
# SCALING & DATASET (dipakai tab training & forecast)
# ═════════════════════════════════════════════════════════════════════════════
t_arr = trend_tv[:n_train]
t_val_arr = trend_tv[n_train:n_train + n_val]
s_arr = season_tv[:n_train]
s_val_arr = season_tv[n_train:n_train + n_val]

if scaler_type == "MinMax":
    sc_t = MinMaxScaler().fit(t_arr.reshape(-1, 1))
    sc_s = MinMaxScaler().fit(s_arr.reshape(-1, 1))
else:
    sc_t = StandardScaler().fit(t_arr.reshape(-1, 1))
    sc_s = StandardScaler().fit(s_arr.reshape(-1, 1))

t_tr_s = sc_t.transform(t_arr.reshape(-1, 1)).flatten().astype(np.float32)
t_vl_s = sc_t.transform(t_val_arr.reshape(-1, 1)).flatten().astype(np.float32)
s_tr_s = sc_s.transform(s_arr.reshape(-1, 1)).flatten().astype(np.float32)
s_vl_s = sc_s.transform(s_val_arr.reshape(-1, 1)).flatten().astype(np.float32)

Xtt, ytt = build_dataset(t_tr_s, lookback)
Xvt, yvt = build_dataset(np.concatenate([t_tr_s[-lookback:], t_vl_s]), lookback)

Xts, yts = build_dataset(s_tr_s, lookback_seasonal)
Xvs, yvs = build_dataset(np.concatenate([s_tr_s[-lookback_seasonal:], s_vl_s]), lookback_seasonal)

# ═════════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════════
t0, t1, t2, t3, t4, t5 = st.tabs([
    "📊 Data Overview",
    "🔬 STL & Karakteristik",
    "🤖 Model Training",
    "🎯 Forecast Results",
    "📋 Metrics",
    "🔮 Future Forecast",
])

# ─── TAB 0 · DATA OVERVIEW ───────────────────────────────────────────────────
with t0:
    c1, c2, c3, c4 = st.columns(4)
    mcard(c1, "Total Points", f"{n:,}")
    mcard(c2, "Train", f"{n_train:,}", f"{train_r * 100:.0f}%")
    mcard(c3, "Val", f"{n_val:,}", f"{val_r * 100:.0f}%")
    mcard(c4, "Test", f"{n_test:,}", f"{(n_test / n) * 100:.1f}%")

    sec("📈 Visualisasi Data & Split")
    fig, ax = plt.subplots(figsize=(14, 3.8))
    for sl, col, lbl in [
        (slice(None, n_train), PAL["train"], f"Train ({n_train})"),
        (slice(n_train, n_train + n_val), PAL["val"], f"Val ({n_val})"),
        (slice(n_train + n_val, None), PAL["test"], f"Test ({n_test})"),
    ]:
        ax.plot(dates[sl], y_full[sl], color=col, lw=1.1, label=lbl)
        ax.fill_between(dates[sl], y_full[sl], alpha=.08, color=col)
    ax.axvline(dates[n_train], color=PAL["val"], lw=1.5, ls="--", alpha=.8)
    ax.axvline(dates[n_train + n_val], color=PAL["test"], lw=1.5, ls="--", alpha=.8)
    ax.set_title("Data Split Visualization", color="#e2e8f0")
    ax.set_ylabel("SST (°C)")
    tight_ylim(ax, [y_full])
    ax.legend()
    ax.grid(True, lw=.4)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    sec("📋 Tabel Data")
    col_tbl, col_norm = st.columns(2)
    with col_tbl:
        st.markdown("**Data Asli (10 baris pertama)**")
        raw_df = df[[sst_col]].reset_index().head(10).copy()
        raw_df.columns = ["Tanggal", "SST (°C)"]
        st.dataframe(raw_df.style.format({"SST (°C)": "{:.5f}"}), use_container_width=True)
    with col_norm:
        sc_full = MinMaxScaler().fit(y_full.reshape(-1, 1))
        y_norm = sc_full.transform(y_full.reshape(-1, 1)).flatten()
        norm_df = pd.DataFrame({
            "Tanggal": dates[:10],
            "SST (°C)": y_full[:10],
            "SST Normalisasi": y_norm[:10],
        })
        st.markdown("**Sebelum & Sesudah Normalisasi (10 baris)**")
        st.dataframe(norm_df.style.format({"SST (°C)": "{:.5f}", "SST Normalisasi": "{:.5f}"}),
                     use_container_width=True)

    sec("📊 Statistik Deskriptif")
    st.dataframe(df[[sst_col]].describe().T.round(4), use_container_width=True)
    st.download_button("⬇ Download data",
                       df[[sst_col]].reset_index().to_csv(index=False).encode(),
                       "data_used.csv", "text/csv")

# ─── TAB 1 · STL & KARAKTERISTIK ────────────────────────────────────────────
with t1:
    st.success(f"STL selesai · Period = **{periode}** hari")

    sec("🔬 STL Decomposition")
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    for ax, (name, val, col) in zip(axes, [
        ("Observed", stl_res.observed, PAL["actual"]),
        ("Trend", stl_res.trend, PAL["trend"]),
        ("Seasonal", stl_res.seasonal, PAL["season"]),
        ("Residual", stl_res.resid, PAL["resid"]),
    ]):
        ax.plot(val, color=col, lw=1.2)
        ax.fill_between(range(len(val)), val, alpha=.12, color=col)
        ax.set_ylabel(name, fontsize=9, color="#e2e8f0")
        ax.grid(True, lw=.4)
        tight_ylim(ax, [val])
    axes[-1].set_xlabel("Index")
    fig.suptitle(f"STL Decomposition  (period={periode})", fontsize=11, color="#e2e8f0")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    vo = np.var(stl_res.observed)
    ve_t = (1 - np.var(stl_res.observed - stl_res.trend) / vo) * 100
    ve_s = (1 - np.var(stl_res.observed - stl_res.seasonal) / vo) * 100
    ve_r = np.var(stl_res.resid) / vo * 100
    c1, c2, c3 = st.columns(3)
    mcard(c1, "Variance – Trend", f"{ve_t:.1f}%")
    mcard(c2, "Variance – Seasonal", f"{ve_s:.1f}%")
    mcard(c3, "Variance – Residual", f"{ve_r:.1f}%")

    # ── KARAKTERISTIK TREND ─────────────────────────────────────────────────
    sec("📐 Karakteristik Komponen Trend (GPH)")
    with st.spinner("Menghitung GPH..."):
        d_gph, _ = fdGPH(trend_tv, bw=0.5)

    if d_gph < 0:
        mc, ms = "Anti-persistent", "d < 0: mean-reverting kuat"
    elif d_gph < 0.5:
        mc, ms = "Long Memory – Stasioner", "0 < d < 0.5"
    elif d_gph < 1.0:
        mc, ms = "Long Memory – Non-Stasioner", "0.5 ≤ d < 1"
    else:
        mc, ms = "Non-Stasioner Kuat", "d ≥ 1"

    c1, c2 = st.columns(2)
    mcard(c1, "GPH d estimate", f"{d_gph:.4f}")
    mcard(c2, "Memory Class", mc, ms)

    trend_slope = (trend_tv[-1] - trend_tv[0]) / len(trend_tv)
    arah = "meningkat" if trend_slope > 0 else "menurun"
    trend_range = trend_tv.max() - trend_tv.min()

    if d_gph < 0:
        mem_narasi = (f"Nilai d = {d_gph:.4f} (d < 0) menunjukkan pola **anti-persistent**: "
                      "guncangan pada tren cenderung langsung berbalik arah, sehingga tren "
                      "bergerak sangat mulus tanpa efek memori jangka panjang.")
    elif d_gph < 0.5:
        mem_narasi = (f"Nilai d = {d_gph:.4f} (0 < d < 0.5) mengindikasikan **long memory stasioner**: "
                      "tren memiliki ketergantungan jangka panjang namun tetap stasioner. "
                      "Fluktuasi masa lalu masih berpengaruh signifikan terhadap nilai masa depan.")
    elif d_gph < 1.0:
        mem_narasi = (f"Nilai d = {d_gph:.4f} (0.5 ≤ d < 1) menunjukkan **long memory non-stasioner**: "
                      "tren memiliki persistensi sangat kuat dan tidak stasioner. "
                      "Diferensiasi fraksional disarankan sebelum pemodelan klasik.")
    else:
        mem_narasi = (f"Nilai d = {d_gph:.4f} (d ≥ 1) menunjukkan tren **non-stasioner kuat**, "
                      "serupa dengan proses random walk atau lebih. Integrasi orde tinggi terindikasi.")

    narasi(
        f"Komponen tren memperlihatkan arah yang secara umum **{arah}** sepanjang periode observasi, "
        f"dengan rentang nilai sebesar **{trend_range:.4f}°C** (dari {trend_tv.min():.4f} hingga {trend_tv.max():.4f}°C). "
        f"{mem_narasi}"
    )

    # ── KARAKTERISTIK SEASONAL ──────────────────────────────────────────────
    sec("📈 Karakteristik Komponen Musiman (Spektral)")
    per_sp, Th, Tt, mus, pg = fungsi_spektral(y_trainval)

    c1, c2, c3 = st.columns(3)
    mcard(c1, "Dominant Period", f"{per_sp} hari")
    mcard(c2, "T-hitung", f"{Th:.5f}")
    mcard(c3, "T-tabel", f"{Tt:.5f}")

    badge = ('<span class="badge-ok">✓ Pola Musiman Terdeteksi</span>' if mus
             else '<span class="badge-err">✗ Tidak Musiman</span>')
    st.markdown(f"**Kesimpulan Uji:** {badge}", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(14, 3.2))
    ax.plot(pg, color=PAL["season"], lw=1.2)
    ax.fill_between(range(len(pg)), pg, alpha=.15, color=PAL["season"])
    ix = np.argmax(pg)
    ax.axvline(ix, color="#f87171", lw=1.5, ls="--", label=f"Peak @ idx={ix}")
    ax.set_title("Periodogram – Komponen Musiman", color="#e2e8f0")
    ax.set_xlabel("Frequency Index")
    ax.set_ylabel("Power")
    ax.legend()
    ax.grid(True, lw=.4)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    seas_amp = season_tv.max() - season_tv.min()
    seas_std = season_tv.std()

    if mus:
        mus_narasi = (f"Uji spektral mendeteksi **pola musiman yang signifikan** (T-hitung = {Th:.5f} > "
                      f"T-tabel = {Tt:.5f}). Periode dominan teridentifikasi pada **{per_sp} hari**, "
                      f"yang konsisten dengan siklus tahunan (~365 hari) atau sub-siklus musiman SST. ")
    else:
        mus_narasi = (f"Uji spektral **tidak mendeteksi** pola musiman yang signifikan (T-hitung = {Th:.5f} < "
                      f"T-tabel = {Tt:.5f}). Komponen seasonal pada dekomposisi STL mungkin bersifat residual noise.")

    narasi(
        f"{mus_narasi}"
        f"Amplitudo komponen musiman mencapai **{seas_amp:.4f}°C** dengan standar deviasi **{seas_std:.4f}°C**, "
        f"menunjukkan intensitas fluktuasi musiman yang {'cukup signifikan' if seas_amp > 0.5 else 'relatif kecil'} "
        f"terhadap nilai SST keseluruhan."
    )

# ─── TAB 2 · MODEL TRAINING ──────────────────────────────────────────────────
with t2:
    sec("🤖 Training CNN-BiLSTM")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🔵 Trend Model")
        prog = st.progress(0, text="Training…")
        TM = CNNBiLSTM(lookback, t_conv_f, t_kern, t_lstm, t_dense, t_drop, use_bn=use_bn)
        ht_tr, ht_val = train_model(
            TM, Xtt, ytt, Xvt, yvt,
            epochs, batch_size, t_lr,
            patience=patience, use_bn=use_bn,
            weight_decay=weight_decay
        )
        prog.progress(100, text=f"Done · {len(ht_tr)} epochs")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(ht_tr, color=PAL["train"], lw=1.5, label="Train Loss")
        ax.plot(ht_val, color=PAL["val"], lw=1.5, ls="--", label="Val Loss")
        ax.set_title("Trend Model – Loss Curve", color="#e2e8f0")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend()
        ax.grid(True, lw=.4)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col2:
        st.markdown("#### 🟠 Seasonal Model")
        prog2 = st.progress(0, text="Training…")
        SM = CNNBiLSTM(lookback_seasonal, s_conv_f, s_kern, s_lstm, s_dense, s_drop, use_bn=use_bn)
        hs_tr, hs_val = train_model(
            SM, Xts, yts, Xvs, yvs,
            epochs, batch_size, s_lr,
            patience=patience, use_bn=use_bn,
            weight_decay=weight_decay
        )
        prog2.progress(100, text=f"Done · {len(hs_tr)} epochs")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(hs_tr, color=PAL["season"], lw=1.5, label="Train Loss")
        ax.plot(hs_val, color=PAL["val"], lw=1.5, ls="--", label="Val Loss")
        ax.set_title("Seasonal Model – Loss Curve", color="#e2e8f0")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend()
        ax.grid(True, lw=.4)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.session_state.update(dict(
        trained=True, TM=TM, SM=SM, sc_t=sc_t, sc_s=sc_s,
        t_tr_s=t_tr_s, t_vl_s=t_vl_s, s_tr_s=s_tr_s, s_vl_s=s_vl_s,
    ))

# ─── TAB 3 · FORECAST RESULTS ────────────────────────────────────────────────
with t3:
    if "trained" not in st.session_state:
        st.info("Jalankan training terlebih dahulu (tab 🤖).")
        st.stop()

    TM = st.session_state["TM"]
    SM = st.session_state["SM"]
    sc_t = st.session_state["sc_t"]
    sc_s = st.session_state["sc_s"]
    t_tr_s = st.session_state["t_tr_s"]
    t_vl_s = st.session_state["t_vl_s"]
    s_tr_s = st.session_state["s_tr_s"]
    s_vl_s = st.session_state["s_vl_s"]

    tp_tr = sc_t.inverse_transform(predict_model(TM, Xtt).reshape(-1, 1)).flatten()
    tp_vl = sc_t.inverse_transform(predict_model(TM, Xvt).reshape(-1, 1)).flatten()
    sp_tr = sc_s.inverse_transform(predict_model(SM, Xts).reshape(-1, 1)).flatten()
    sp_vl = sc_s.inverse_transform(predict_model(SM, Xvs).reshape(-1, 1)).flatten()

    tl = np.concatenate([t_tr_s, t_vl_s])[-lookback:]
    sl = np.concatenate([s_tr_s, s_vl_s])[-lookback_seasonal:]

    with st.spinner("Recursive forecast pada test set…"):
        tp_te = sc_t.inverse_transform(recursive_forecast(TM, tl, n_test).reshape(-1, 1)).flatten()
        sp_te = sc_s.inverse_transform(recursive_forecast(SM, sl, n_test).reshape(-1, 1)).flatten()

    h_tr = tp_tr + sp_tr
    h_vl = tp_vl + sp_vl
    h_te = tp_te + sp_te

    # ── Diagnostic: Plot prediksi seasonal pada validation set
    sec("📊 Diagnostic: Seasonal Validation")
    fig, ax = plt.subplots(figsize=(14, 3.5))
    val_idx = slice(n_train, n_train + n_val)
    ax.plot(dates[val_idx], season_tv[val_idx], color=PAL["actual"], lw=1.5, label="Actual Seasonal Val")
    ax.plot(dates[val_idx], sp_vl, color=PAL["val"], lw=1.5, ls="--", label="Predicted Seasonal Val")
    ax.set_title("Seasonal Component – Validation Set", color="#e2e8f0")
    tight_ylim(ax, [season_tv[val_idx], sp_vl])
    ax.legend()
    ax.grid(True, lw=.4)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── Trend plot ──────────────────────────────────────────────────────────
    sec("📉 Trend – Actual vs Predicted")
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.plot(dates[:n_train + n_val], trend_tv, color=PAL["actual"], lw=1.2, label="Actual Trend")
    ax.plot(dates[lookback:n_train], tp_tr, color=PAL["train"], lw=1.1, ls="--", label="Train")
    ax.plot(dates[n_train:n_train + n_val], tp_vl, color=PAL["val"], lw=1.1, ls="--", label="Val")
    ax.plot(dates[n_train + n_val:], tp_te, color=PAL["test"], lw=1.1, ls="--", label="Test")
    ax.set_title("Trend – Actual vs Predicted", color="#e2e8f0")
    tight_ylim(ax, [trend_tv, tp_tr, tp_vl, tp_te])
    ax.legend()
    ax.grid(True, lw=.4)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── Seasonal plot ───────────────────────────────────────────────────────
    sec("🌊 Seasonal – Actual vs Predicted")
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.plot(dates[:n_train + n_val], season_tv, color=PAL["actual"], lw=1.2, label="Actual Seasonal")
    ax.plot(dates[lookback_seasonal:n_train], sp_tr, color=PAL["train"], lw=1.1, ls="--", label="Train")
    ax.plot(dates[n_train:n_train + n_val], sp_vl, color=PAL["val"], lw=1.1, ls="--", label="Val")
    ax.plot(dates[n_train + n_val:], sp_te, color=PAL["test"], lw=1.1, ls="--", label="Test")
    ax.set_title("Seasonal – Actual vs Predicted", color="#e2e8f0")
    tight_ylim(ax, [season_tv, sp_tr, sp_vl, sp_te])
    ax.legend()
    ax.grid(True, lw=.4)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── Hybrid full ─────────────────────────────────────────────────────────
    sec("🔀 Hybrid Reconstruction – Full Series")
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(dates, y_full, color=PAL["actual"], lw=1.3, label="Actual SST")
    ax.plot(dates[lookback:n_train], h_tr, color=PAL["train"], lw=1.1, ls="--", label="Hybrid Train")
    ax.plot(dates[n_train:n_train + n_val], h_vl, color=PAL["val"], lw=1.1, ls="--", label="Hybrid Val")
    ax.plot(dates[n_train + n_val:], h_te, color=PAL["test"], lw=1.3, ls="--", label="Hybrid Test")
    ax.set_title("Hybrid Reconstruction – Full Series", color="#e2e8f0")
    tight_ylim(ax, [y_full, h_tr, h_vl, h_te])
    ax.legend()
    ax.grid(True, lw=.4)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── Test only ────────────────────────────────────────────────────────────
    sec("🎯 Test Set – Actual vs Predicted")
    fig, ax = plt.subplots(figsize=(12, 3.8))
    actual_test = y_full[n_train + n_val:]
    ax.plot(dates[n_train + n_val:], actual_test, color=PAL["actual"], lw=1.5, label="Actual")
    ax.plot(dates[n_train + n_val:], h_te, color=PAL["test"], lw=1.5, ls="--", label="Predicted")
    ax.fill_between(dates[n_train + n_val:], actual_test, h_te, alpha=.1, color=PAL["test"])
    ax.set_title("TEST SET – Actual vs Predicted", color="#e2e8f0")
    tight_ylim(ax, [actual_test, h_te])
    ax.legend()
    ax.grid(True, lw=.4)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── Tabel 10 data terakhir testing ───────────────────────────────────────
    sec("📋 Tabel Perbandingan – 10 Data Terakhir Test")
    last10_dates = dates[n_train + n_val:][-10:]
    last10_actual = actual_test[-10:]
    last10_pred = h_te[-10:]
    last10_err = last10_actual - last10_pred
    last10_ape = np.abs(last10_err / last10_actual) * 100

    df_last10 = pd.DataFrame({
        "Tanggal": last10_dates,
        "Aktual (°C)": last10_actual,
        "Prediksi (°C)": last10_pred,
        "Error": last10_err,
        "APE (%)": last10_ape,
    })
    st.dataframe(df_last10.style
                 .format({"Aktual (°C)": "{:.4f}", "Prediksi (°C)": "{:.4f}",
                          "Error": "{:.4f}", "APE (%)": "{:.2f}%"})
                 .background_gradient(subset=["APE (%)"], cmap="RdYlGn_r"),
                 use_container_width=True)

    df_pred_full = pd.DataFrame({
        "date": dates[n_train + n_val:], "actual": actual_test,
        "predicted": h_te, "trend_pred": tp_te, "seasonal_pred": sp_te,
    })
    st.download_button("⬇ Download Hasil Prediksi Test",
                       df_pred_full.to_csv(index=False).encode(),
                       "test_predictions.csv", "text/csv")

    st.session_state.update(dict(
        h_tr=h_tr, h_vl=h_vl, h_te=h_te,
        tp_te=tp_te, sp_te=sp_te,
        tl_last=tl, sl_last=sl,
    ))

# ─── TAB 4 · METRICS ─────────────────────────────────────────────────────────
with t4:
    if "h_te" not in st.session_state:
        st.info("Jalankan forecast terlebih dahulu.")
        st.stop()

    h_tr = st.session_state["h_tr"]
    h_vl = st.session_state["h_vl"]
    h_te = st.session_state["h_te"]

    sec("📊 Evaluasi Model Hybrid – MAPE & MAE")

    sets = {
        "Training": (y_full[lookback:n_train], h_tr),
        "Validation": (y_full[n_train:n_train + n_val], h_vl),
        "Testing": (y_full[n_train + n_val:], h_te),
    }
    cols_m = st.columns(3)
    results = {}
    for (label, (yt, yp)), col in zip(sets.items(), cols_m):
        mp = mape(yt, yp)
        ma = mae_fn(yt, yp)
        results[label] = {"MAPE (%)": mp, "MAE": ma}
        col.markdown(f"**{label}**")
        mcard(col, "MAPE", f"{mp:.2f}%")
        mcard(col, "MAE", f"{ma:.4f}")

    sec("📋 Summary Table")
    sdf = pd.DataFrame(results).T.round(4)
    st.dataframe(
        sdf.style.background_gradient(subset=["MAPE (%)", "MAE"], cmap="RdYlGn_r"),
        use_container_width=True,
    )
    st.download_button("⬇ Download Metrics CSV",
                       sdf.to_csv().encode(), "metrics_summary.csv", "text/csv")

    te_mape = results["Testing"]["MAPE (%)"]
    te_mae = results["Testing"]["MAE"]
    if te_mape < 1:
        kual = "sangat baik (< 1%)"
    elif te_mape < 5:
        kual = "baik (1–5%)"
    elif te_mape < 10:
        kual = "cukup (5–10%)"
    else:
        kual = "perlu perbaikan (> 10%)"

    narasi(
        f"Model hybrid CNN-BiLSTM+STL menghasilkan MAPE testing sebesar **{te_mape:.2f}%** "
        f"dan MAE sebesar **{te_mae:.4f}°C**, yang tergolong **{kual}** untuk prediksi SST. "
        f"Perbedaan antara performa training ({results['Training']['MAPE (%)']:.2f}%) dan testing "
        f"({'tinggi' if te_mape - results['Training']['MAPE (%)'] > 3 else 'wajar'}) mengindikasikan "
        f"{'potensi overfitting, pertimbangkan regularisasi lebih kuat' if te_mape - results['Training']['MAPE (%)'] > 5 else 'generalisasi model yang memadai'}."
    )

# ─── TAB 5 · FUTURE FORECAST ─────────────────────────────────────────────────
with t5:
    if "tl_last" not in st.session_state:
        st.info("Jalankan forecast terlebih dahulu di tab 🎯.")
        st.stop()

    TM = st.session_state["TM"]
    SM = st.session_state["SM"]
    sc_t = st.session_state["sc_t"]
    sc_s = st.session_state["sc_s"]
    tl = st.session_state["tl_last"]
    sl = st.session_state["sl_last"]

    tp_te = st.session_state["tp_te"]
    sp_te = st.session_state["sp_te"]

    t_all_s = np.concatenate([
        st.session_state["t_tr_s"],
        st.session_state["t_vl_s"],
        sc_t.transform(tp_te.reshape(-1, 1)).flatten().astype(np.float32),
    ])
    s_all_s = np.concatenate([
        st.session_state["s_tr_s"],
        st.session_state["s_vl_s"],
        sc_s.transform(sp_te.reshape(-1, 1)).flatten().astype(np.float32),
    ])
    tl_fut = t_all_s[-lookback:]
    sl_fut = s_all_s[-lookback_seasonal:]

    FUTURE_STEPS = 10
    last_date = dates[-1]
    freq_guess = pd.infer_freq(dates[:50])
    if freq_guess is None:
        freq_guess = "D"
    future_dates = pd.date_range(last_date, periods=FUTURE_STEPS + 1, freq=freq_guess)[1:]

    with st.spinner("Menghitung future forecast…"):
        tf_fut_s = recursive_forecast(TM, tl_fut, FUTURE_STEPS)
        sf_fut_s = recursive_forecast(SM, sl_fut, FUTURE_STEPS)

    tf_fut = sc_t.inverse_transform(tf_fut_s.reshape(-1, 1)).flatten()
    sf_fut = sc_s.inverse_transform(sf_fut_s.reshape(-1, 1)).flatten()
    hf_fut = tf_fut + sf_fut

    sec("🔮 Forecast 10 Periode ke Depan")
    tail = min(60, n)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(dates[-tail:], y_full[-tail:], color=PAL["actual"], lw=1.5, label="Actual (tail)")
    ax.plot(future_dates, hf_fut, color=PAL["future"], lw=2, ls="--", label="Future Forecast", marker="o", markersize=5)
    ax.axvline(dates[-1], color="#64748b", lw=1, ls=":", alpha=.8)
    ax.set_title("Future Forecast – 10 Periode ke Depan", color="#e2e8f0")
    ax.set_ylabel("SST (°C)")
    tight_ylim(ax, [y_full[-tail:], hf_fut])
    ax.legend()
    ax.grid(True, lw=.4)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    sec("📋 Tabel Forecast – Sebelum & Sesudah Denormalisasi")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Sebelum Denormalisasi (nilai ternormalisasi)**")
        df_norm = pd.DataFrame({
            "Periode": range(1, FUTURE_STEPS + 1),
            "Tanggal": future_dates,
            "Trend (norm)": np.round(tf_fut_s, 5),
            "Seasonal (norm)": np.round(sf_fut_s, 5),
        })
        st.dataframe(df_norm.style.format({"Trend (norm)": "{:.5f}", "Seasonal (norm)": "{:.5f}"}),
                     use_container_width=True)

    with col_b:
        st.markdown("**Setelah Denormalisasi (SST °C)**")
        df_denorm = pd.DataFrame({
            "Periode": range(1, FUTURE_STEPS + 1),
            "Tanggal": future_dates,
            "Trend (°C)": np.round(tf_fut, 4),
            "Seasonal (°C)": np.round(sf_fut, 4),
            "SST Pred (°C)": np.round(hf_fut, 4),
        })
        st.dataframe(df_denorm.style
                     .format({"Trend (°C)": "{:.4f}", "Seasonal (°C)": "{:.4f}", "SST Pred (°C)": "{:.4f}"})
                     .background_gradient(subset=["SST Pred (°C)"], cmap="Blues"),
                     use_container_width=True)

    st.download_button("⬇ Download Future Forecast CSV",
                       df_denorm.to_csv(index=False).encode(),
                       "future_forecast.csv", "text/csv")

    narasi(
        f"Model memproyeksikan SST selama **{FUTURE_STEPS} periode ke depan** menggunakan metode "
        f"recursive forecast berbasis window terakhir data. Nilai prediksi berkisar antara "
        f"**{hf_fut.min():.4f}°C** hingga **{hf_fut.max():.4f}°C**. "
        "Perlu diperhatikan bahwa akurasi prediksi rekursif cenderung menurun seiring bertambahnya "
        "horizon forecast karena akumulasi error pada setiap langkah."
    )
