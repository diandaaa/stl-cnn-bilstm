"""
Dashboard Analisis Time Series – CNN-BiLSTM + STL Decomposition
Backend  : PyTorch (kompatibel Python 3.14 / Streamlit Cloud)
Run      : streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import random, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

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
.metric-card{
    background:linear-gradient(135deg,#1a2236,#151c30);
    border:1px solid #2d3a56;border-radius:12px;
    padding:1rem 1.2rem;text-align:center;margin-bottom:.5rem;
}
.metric-card .label{font-size:.7rem;color:#64748b;letter-spacing:.12em;text-transform:uppercase;}
.metric-card .value{font-size:1.45rem;font-family:'Space Mono',monospace;color:#38bdf8;font-weight:700;}
.metric-card .sub{font-size:.72rem;color:#94a3b8;margin-top:2px;}
.badge-ok {background:#064e3b;color:#34d399;padding:3px 10px;border-radius:999px;font-size:.78rem;}
.badge-err{background:#450a0a;color:#f87171;padding:3px 10px;border-radius:999px;font-size:.78rem;}
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor":"#0e1525","axes.facecolor":"#0e1525",
    "axes.edgecolor":"#2d3a56","axes.labelcolor":"#94a3b8",
    "xtick.color":"#64748b","ytick.color":"#64748b",
    "text.color":"#e2e8f0","grid.color":"#1e2d45","grid.alpha":.6,
    "legend.facecolor":"#111827","legend.edgecolor":"#2d3a56",
    "legend.fontsize":8,"font.family":"monospace",
})
PAL = dict(actual="#38bdf8",train="#34d399",val="#fbbf24",
           test="#f87171",trend="#a78bfa",season="#fb923c",resid="#94a3b8")

# ═════════════════════════════════════════════════════════════════════════════
# PYTORCH MODEL
# ═════════════════════════════════════════════════════════════════════════════
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cpu")   # Streamlit Cloud → CPU only

class CNNBiLSTM(nn.Module):
    def __init__(self, lookback, conv_filters, kernel_size,
                 lstm_units, dense_units, dropout=0.0):
        super().__init__()
        # Conv1D causal → pad left only
        self.pad   = nn.ConstantPad1d((kernel_size - 1, 0), 0)
        self.conv  = nn.Conv1d(1, conv_filters, kernel_size)
        self.relu  = nn.ReLU()
        self.bilstm= nn.LSTM(conv_filters, lstm_units,
                             batch_first=True, bidirectional=True)
        self.drop  = nn.Dropout(dropout)
        self.fc1   = nn.Linear(lstm_units * 2, dense_units)
        self.fc2   = nn.Linear(dense_units, 1)

    def forward(self, x):                 # x: (B, L, 1)
        x = x.permute(0, 2, 1)           # (B, 1, L)
        x = self.relu(self.conv(self.pad(x)))  # (B, C, L)
        x = x.permute(0, 2, 1)           # (B, L, C)
        out, _ = self.bilstm(x)
        x = self.drop(out[:, -1, :])     # last timestep
        x = self.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


def train_model(model, X_tr, y_tr, X_val, y_val,
                epochs, batch_size, lr, patience=20):
    opt  = torch.optim.Adam(model.parameters(), lr=lr)
    sched= torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)
    loss_fn = nn.MSELoss()

    Xt = torch.tensor(X_tr,  dtype=torch.float32)
    yt = torch.tensor(y_tr,  dtype=torch.float32)
    Xv = torch.tensor(X_val, dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.float32)

    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=False)
    best_val, best_state, wait = float("inf"), None, 0
    hist_tr, hist_val = [], []

    for ep in range(epochs):
        model.train()
        ep_loss = []
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb.unsqueeze(-1))
            l = loss_fn(pred, yb)
            l.backward(); opt.step()
            ep_loss.append(l.item())
        tr_loss = float(np.mean(ep_loss))

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv.unsqueeze(-1)), yv).item()

        hist_tr.append(tr_loss); hist_val.append(val_loss)
        sched.step(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
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
    with torch.no_grad():
        for _ in range(steps):
            x = torch.tensor([[w[-len(last_window):]]], dtype=torch.float32)
            x = x.permute(0, 2, 1)   # wrong shape fix below
            # correct: (1, L, 1)
            xin = torch.tensor(w[-len(last_window):],
                                dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
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
    return np.mean(np.abs((yt[m]-yp[m])/yt[m]))*100

def build_dataset(arr, lb):
    X, y = [], []
    for i in range(lb, len(arr)):
        X.append(arr[i-lb:i]); y.append(arr[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def calc_metrics(yt, yp):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {"RMSE": np.sqrt(mean_squared_error(yt, yp)),
            "MAE":  mean_absolute_error(yt, yp),
            "MAPE (%)": mape(yt, yp),
            "R²":   r2_score(yt, yp)}

def fungsi_spektral(x):
    x = np.asarray(x, dtype=float); n = len(x); k = round((n-1)/2)
    t = np.arange(1, n+1); pg = np.zeros(k)
    for i in range(1, k+1):
        w = (2*np.pi*i)/n
        a = (2/n)*np.sum(x*np.cos(w*t))
        b = (2/n)*np.sum(x*np.sin(w*t))
        pg[i-1] = a**2 + b**2
    km  = np.argmax(pg)+1
    per = int(round((2*np.pi)/((2*np.pi*km)/n)))
    Th  = np.max(pg)/np.sum(pg); Tt = 0.13135
    return per, Th, Tt, Th>Tt, pg

def fdGPH(x, bw=0.5):
    import statsmodels.api as sm
    x  = np.asarray(x, dtype=float) - np.mean(x); n = len(x)
    m  = int(np.floor(n**bw)); j = np.arange(1, m+1)
    lam= 2*np.pi*j/n
    fv = np.fft.fft(x)
    I  = (1/(2*np.pi*n))*np.abs(fv[j])**2
    Y  = np.log(I); Xm = sm.add_constant(np.log(4*(np.sin(lam/2)**2)))
    mo = sm.OLS(Y, Xm).fit()
    return -mo.params[1], mo

def generate_random_sst(n=3650, seed=42):
    rng = np.random.default_rng(seed); t = np.arange(n)
    trend    = 28.5 + 0.0003*t
    seasonal = 1.2*np.sin(2*np.pi*t/365 + 0.5) + 0.4*np.sin(4*np.pi*t/365)
    noise    = rng.normal(0, 0.15, n)
    for i in range(1, n): noise[i] += 0.55*noise[i-1]
    sst   = trend + seasonal + noise
    dates = pd.date_range("2015-01-01", periods=n, freq="D")
    return pd.DataFrame({"tgl": dates.strftime("%m/%d/%Y"), "sst": np.round(sst,5)})

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌊 SST Forecast")
    st.markdown("**CNN-BiLSTM + STL Decomposition**")
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
        gen_n    = st.slider("Jumlah hari", 365, 5000, 3650, 365)
        gen_seed = st.number_input("Seed data", value=42)

    date_col = st.text_input("Nama kolom tanggal", value="tgl")
    sst_col  = st.text_input("Nama kolom target",  value="sst")
    st.divider()

    st.markdown("### 🔧 Data Split")
    train_r = st.slider("Train ratio", 0.50, 0.95, 0.90, 0.01)
    val_r   = st.slider("Val ratio",   0.01, 0.20, 0.05, 0.01)
    test_r  = round(1-train_r-val_r, 4)
    st.markdown(f"**Test ratio (auto):** `{max(test_r,0):.2f}`")

    st.markdown("### 📅 STL")
    auto_period   = st.checkbox("Auto-detect period (spektral)", value=True)
    manual_period = 180
    if not auto_period:
        manual_period = st.number_input("Period manual", 2, 730, 180)
    stl_robust = st.checkbox("STL robust", value=True)

    st.markdown("### 🧠 Trend Model")
    t_conv_f = st.slider("Conv1D filters",  8,  128, 32,  8)
    t_kern   = st.slider("Kernel size",     2,  15,  5,   1)
    t_lstm   = st.slider("BiLSTM units",    16, 256, 64,  16)
    t_drop   = st.slider("Dropout",         0.0, 0.5, 0.2, 0.05)
    t_dense  = st.slider("Dense units",     8,  128, 32,  8)
    t_lr     = st.number_input("LR trend",  value=0.0007, format="%.4f")

    st.markdown("### 🧠 Seasonal Model")
    s_conv_f = st.slider("Conv1D filters (S)", 8,  128, 64,  8)
    s_kern   = st.slider("Kernel size (S)",    2,  15,  5,   1)
    s_lstm   = st.slider("BiLSTM units (S)",   16, 256, 64,  16)
    s_dense  = st.slider("Dense units (S)",    4,  64,  16,  4)
    s_lr     = st.number_input("LR seasonal",  value=0.0005, format="%.4f")

    st.markdown("### ⚙️ Training")
    lookback   = st.slider("Lookback",    30, 365, 180, 10)
    epochs     = st.slider("Max epochs",  10, 500, 250, 10)
    batch_size = st.selectbox("Batch size", [16,32,64,128], index=2)
    seed       = st.number_input("Random seed", value=42)

    st.markdown("### 🔬 GPH")
    gph_bw = st.slider("GPH bandwidth exp", 0.3, 0.9, 0.5, 0.05)

    st.divider()
    run_btn = st.button("▶  Run Analysis", use_container_width=True, type="primary")

# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("# 🌊 SST Time Series Analysis Dashboard")
st.markdown("*Hybrid CNN-BiLSTM + STL Decomposition · Sea Surface Temperature Forecasting*")

if not run_btn:
    st.info("👈 Pilih sumber data & atur parameter di sidebar, lalu klik **▶ Run Analysis**.")
    with st.expander("📋 Format CSV yang dibutuhkan"):
        s = generate_random_sst(10, 42)
        st.dataframe(s, use_container_width=True)
        st.caption("Kolom minimal: tanggal (`tgl`) + nilai SST (`sst`). Nama bisa diubah di sidebar.")
        st.download_button("⬇ Download contoh CSV (10 baris)",
                           s.to_csv(index=False).encode(), "contoh_data.csv","text/csv")
    st.stop()

# ── set seed ──────────────────────────────────────────────────────────────────
torch.manual_seed(int(seed)); np.random.seed(int(seed)); random.seed(int(seed))

# ── load data ─────────────────────────────────────────────────────────────────
if data_source == "🎲 Generate Data Contoh":
    df_raw = generate_random_sst(int(gen_n), int(gen_seed))
    st.success(f"✅ Data sintetis: {len(df_raw):,} baris")
else:
    if uploaded is None:
        st.error("⚠️ Upload file CSV terlebih dahulu."); st.stop()
    df_raw = pd.read_csv(uploaded)

if date_col not in df_raw.columns or sst_col not in df_raw.columns:
    st.error(f"Kolom `{date_col}` atau `{sst_col}` tidak ada.\n"
             f"Kolom tersedia: `{list(df_raw.columns)}`"); st.stop()

try:
    from statsmodels.tsa.seasonal import STL
    from sklearn.preprocessing import MinMaxScaler
except ImportError as e:
    st.error(f"Import error: {e}"); st.stop()

df = df_raw.copy()
df[date_col] = pd.to_datetime(df[date_col], dayfirst=False, infer_datetime_format=True)
df = df.sort_values(date_col).set_index(date_col)
y_full = df[sst_col].values.astype(float)
dates  = df.index; n = len(y_full)

n_train = int(n*train_r); n_val = int(n*val_r); n_test = n - n_train - n_val
if n_test <= 0:
    st.error("Test set kosong. Kurangi train/val ratio."); st.stop()

y_trainval = y_full[:n_train+n_val]

# ═════════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════════
t0,t1,t2,t3,t4,t5 = st.tabs([
    "📊 Data Overview","🔬 STL Decomposition","📈 Spectral & GPH",
    "🤖 Model Training","🎯 Forecast Results","📋 Metrics Summary",
])

# ─── TAB 0 · DATA OVERVIEW ───────────────────────────────────────────────────
with t0:
    c1,c2,c3,c4 = st.columns(4)
    mcard(c1,"Total Points",f"{n:,}")
    mcard(c2,"Train",f"{n_train:,}",f"{train_r*100:.0f}%")
    mcard(c3,"Val",  f"{n_val:,}",  f"{val_r*100:.0f}%")
    mcard(c4,"Test", f"{n_test:,}", f"{(n_test/n)*100:.1f}%")

    fig,ax = plt.subplots(figsize=(14,3.8))
    for sl,col,lbl in [
        (slice(None,n_train),          PAL["train"],f"Train ({n_train})"),
        (slice(n_train,n_train+n_val), PAL["val"],  f"Val ({n_val})"),
        (slice(n_train+n_val,None),    PAL["test"], f"Test ({n_test})"),
    ]:
        ax.plot(dates[sl],y_full[sl],color=col,lw=1.2,label=lbl)
        ax.fill_between(dates[sl],y_full[sl],alpha=.1,color=col)
    ax.axvline(dates[n_train],       color=PAL["val"], lw=1.5,ls="--",alpha=.8)
    ax.axvline(dates[n_train+n_val], color=PAL["test"],lw=1.5,ls="--",alpha=.8)
    ax.set_title("Data Split Visualization"); ax.set_ylabel("SST (°C)")
    ax.legend(); ax.grid(True,lw=.5)
    st.pyplot(fig,use_container_width=True); plt.close(fig)

    ca,cb = st.columns([1,2])
    with ca:
        st.markdown("**Statistik Deskriptif**")
        st.dataframe(df[[sst_col]].describe().round(4),use_container_width=True)
    with cb:
        fig2,ax2=plt.subplots(figsize=(7,3))
        ax2.hist(y_full,bins=60,color=PAL["actual"],alpha=.8,edgecolor="#0e1525")
        ax2.set_title("Distribusi SST"); ax2.set_xlabel("SST (°C)"); ax2.set_ylabel("Frekuensi")
        ax2.grid(True,lw=.5)
        st.pyplot(fig2,use_container_width=True); plt.close(fig2)

    st.download_button("⬇ Download data yang digunakan",
                       df[[sst_col]].reset_index().to_csv(index=False).encode(),
                       "data_used.csv","text/csv")

# ─── TAB 1 · STL ─────────────────────────────────────────────────────────────
with t1:
    with st.spinner("Running STL decomposition..."):
        periode = fungsi_spektral(y_trainval)[0] if auto_period else manual_period
        stl_res = STL(y_trainval, period=periode, robust=stl_robust).fit()
    st.success(f"STL selesai · Period = **{periode}** hari")

    fig,axes=plt.subplots(4,1,figsize=(14,9),sharex=True)
    for ax,(name,val,col) in zip(axes,[
        ("Observed",stl_res.observed,PAL["actual"]),
        ("Trend",   stl_res.trend,   PAL["trend"]),
        ("Seasonal",stl_res.seasonal,PAL["season"]),
        ("Residual",stl_res.resid,   PAL["resid"]),
    ]):
        ax.plot(val,color=col,lw=1.2)
        ax.fill_between(range(len(val)),val,alpha=.12,color=col)
        ax.set_ylabel(name,fontsize=9); ax.grid(True,lw=.5)
    axes[-1].set_xlabel("Index")
    fig.suptitle(f"STL Decomposition (period={periode})",fontsize=12)
    plt.tight_layout()
    st.pyplot(fig,use_container_width=True); plt.close(fig)

    vo=np.var(stl_res.observed)
    c1,c2,c3=st.columns(3)
    mcard(c1,"Variance – Trend",   f"{(1-np.var(stl_res.observed-stl_res.trend)/vo)*100:.1f}%")
    mcard(c2,"Variance – Seasonal",f"{(1-np.var(stl_res.observed-stl_res.seasonal)/vo)*100:.1f}%")
    mcard(c3,"Variance – Residual",f"{np.var(stl_res.resid)/vo*100:.1f}%")

# ─── TAB 2 · SPECTRAL & GPH ──────────────────────────────────────────────────
with t2:
    st.markdown("### 📈 Analisis Spektral – Komponen Musiman")
    per_sp,Th,Tt,mus,pg = fungsi_spektral(y_trainval)
    c1,c2,c3=st.columns(3)
    mcard(c1,"Dominant Period",f"{per_sp} hari")
    mcard(c2,"T-hitung",       f"{Th:.5f}")
    mcard(c3,"T-tabel",        f"{Tt:.5f}")
    badge='<span class="badge-ok">✓ Pola Musiman Terdeteksi</span>' if mus else '<span class="badge-err">✗ Tidak Musiman</span>'
    st.markdown(f"**Kesimpulan:** {badge}",unsafe_allow_html=True)

    fig,ax=plt.subplots(figsize=(14,3.5))
    ax.plot(pg,color=PAL["season"],lw=1.2)
    ax.fill_between(range(len(pg)),pg,alpha=.15,color=PAL["season"])
    ix=np.argmax(pg)
    ax.axvline(ix,color="#f87171",lw=1.5,ls="--",label=f"Peak @ idx={ix}")
    ax.set_title("Periodogram (Seasonal Component)")
    ax.set_xlabel("Frequency Index"); ax.set_ylabel("Power")
    ax.legend(); ax.grid(True,lw=.5)
    st.pyplot(fig,use_container_width=True); plt.close(fig)

    st.divider()
    st.markdown("### 📐 GPH Estimator – Long Memory Komponen Trend")
    with st.spinner("Menghitung GPH..."):
        d_gph,gph_model = fdGPH(stl_res.trend, bw=gph_bw)

    if d_gph<0:     mc,ms="Anti-persistent","d < 0: mean-reverting kuat"
    elif d_gph<0.5: mc,ms="Long Memory (stasioner)","0 < d < 0.5"
    elif d_gph<1.0: mc,ms="Long Memory (non-stasioner)","0.5 ≤ d < 1"
    else:           mc,ms="Non-stasioner kuat","d ≥ 1"
    c1,c2,c3=st.columns(3)
    mcard(c1,"GPH d estimate",f"{d_gph:.4f}")
    mcard(c2,"BW exponent",   f"{gph_bw}")
    mcard(c3,"Memory Class",  mc,ms)

    st.markdown("""
| Nilai d | Interpretasi |
|---------|-------------|
| d < 0 | Anti-persistent |
| d ≈ 0 | Short memory (ARMA) |
| 0 < d < 0.5 | Long memory, stasioner |
| 0.5 ≤ d < 1 | Long memory, non-stasioner |
| d ≥ 1 | Non-stasioner kuat |
""")
    n_tv=len(stl_res.trend); m_=int(np.floor(n_tv**gph_bw))
    j_=np.arange(1,m_+1); lam_=2*np.pi*j_/n_tv
    fv_=np.fft.fft(stl_res.trend-np.mean(stl_res.trend))
    I_=(1/(2*np.pi*n_tv))*np.abs(fv_[j_])**2
    logI=np.log(I_); logX=np.log(4*(np.sin(lam_/2)**2))
    fig,ax=plt.subplots(figsize=(7,4))
    ax.scatter(logX,logI,s=12,alpha=.6,color=PAL["trend"],label="Periodogram points")
    xl=np.linspace(logX.min(),logX.max(),100)
    ax.plot(xl,gph_model.params[0]+gph_model.params[1]*xl,
            color="#f87171",lw=2,label=f"GPH fit (d={d_gph:.4f})")
    ax.set_xlabel("log 4sin²(λ/2)"); ax.set_ylabel("log I(λ)")
    ax.set_title("GPH Log-Periodogram Regression")
    ax.legend(); ax.grid(True,lw=.5)
    st.pyplot(fig,use_container_width=True); plt.close(fig)

# ─── TAB 3 · MODEL TRAINING ──────────────────────────────────────────────────
with t3:
    trend_tv  = stl_res.trend
    season_tv = stl_res.seasonal
    t_arr=trend_tv[:n_train];    t_val_arr=trend_tv[n_train:n_train+n_val]
    s_arr=season_tv[:n_train];   s_val_arr=season_tv[n_train:n_train+n_val]

    sc_t=MinMaxScaler().fit(t_arr.reshape(-1,1))
    sc_s=MinMaxScaler().fit(s_arr.reshape(-1,1))
    t_tr_s=sc_t.transform(t_arr.reshape(-1,1)).flatten().astype(np.float32)
    t_vl_s=sc_t.transform(t_val_arr.reshape(-1,1)).flatten().astype(np.float32)
    s_tr_s=sc_s.transform(s_arr.reshape(-1,1)).flatten().astype(np.float32)
    s_vl_s=sc_s.transform(s_val_arr.reshape(-1,1)).flatten().astype(np.float32)

    Xtt,ytt=build_dataset(t_tr_s,lookback)
    Xvt,yvt=build_dataset(np.concatenate([t_tr_s[-lookback:],t_vl_s]),lookback)
    Xts,yts=build_dataset(s_tr_s,lookback)
    Xvs,yvs=build_dataset(np.concatenate([s_tr_s[-lookback:],s_vl_s]),lookback)

    col1,col2=st.columns(2)
    with col1:
        st.markdown("#### 🔵 Trend Model")
        prog=st.progress(0,text="Training…")
        TM=CNNBiLSTM(lookback,t_conv_f,t_kern,t_lstm,t_dense,t_drop)
        ht_tr,ht_val=train_model(TM,Xtt,ytt,Xvt,yvt,epochs,batch_size,t_lr)
        prog.progress(100,text=f"Done · {len(ht_tr)} epochs")
        fig,ax=plt.subplots(figsize=(6,3))
        ax.plot(ht_tr, color=PAL["train"],lw=1.5,label="Train")
        ax.plot(ht_val,color=PAL["val"],  lw=1.5,ls="--",label="Val")
        ax.set_title("Trend Loss"); ax.legend(); ax.grid(True,lw=.5)
        st.pyplot(fig,use_container_width=True); plt.close(fig)

    with col2:
        st.markdown("#### 🟠 Seasonal Model")
        prog2=st.progress(0,text="Training…")
        SM=CNNBiLSTM(lookback,s_conv_f,s_kern,s_lstm,s_dense,dropout=0.0)
        hs_tr,hs_val=train_model(SM,Xts,yts,Xvs,yvs,epochs,batch_size,s_lr)
        prog2.progress(100,text=f"Done · {len(hs_tr)} epochs")
        fig,ax=plt.subplots(figsize=(6,3))
        ax.plot(hs_tr, color=PAL["season"],lw=1.5,label="Train")
        ax.plot(hs_val,color=PAL["val"],   lw=1.5,ls="--",label="Val")
        ax.set_title("Seasonal Loss"); ax.legend(); ax.grid(True,lw=.5)
        st.pyplot(fig,use_container_width=True); plt.close(fig)

    st.session_state.update(dict(
        trained=True, TM=TM, SM=SM, sc_t=sc_t, sc_s=sc_s,
        t_tr_s=t_tr_s,t_vl_s=t_vl_s,s_tr_s=s_tr_s,s_vl_s=s_vl_s,
        Xtt=Xtt,Xvt=Xvt,Xts=Xts,Xvs=Xvs,
        trend_tv=trend_tv,season_tv=season_tv,
    ))

# ─── TAB 4 · FORECAST RESULTS ────────────────────────────────────────────────
with t4:
    if "trained" not in st.session_state:
        st.info("Jalankan training terlebih dahulu (tab 🤖)."); st.stop()

    TM=st.session_state["TM"]; SM=st.session_state["SM"]
    sc_t=st.session_state["sc_t"]; sc_s=st.session_state["sc_s"]
    t_tr_s=st.session_state["t_tr_s"]; t_vl_s=st.session_state["t_vl_s"]
    s_tr_s=st.session_state["s_tr_s"]; s_vl_s=st.session_state["s_vl_s"]
    Xtt=st.session_state["Xtt"]; Xvt=st.session_state["Xvt"]
    Xts=st.session_state["Xts"]; Xvs=st.session_state["Xvs"]
    trend_tv=st.session_state["trend_tv"]; season_tv=st.session_state["season_tv"]

    tp_tr=sc_t.inverse_transform(predict_model(TM,Xtt).reshape(-1,1)).flatten()
    tp_vl=sc_t.inverse_transform(predict_model(TM,Xvt).reshape(-1,1)).flatten()
    sp_tr=sc_s.inverse_transform(predict_model(SM,Xts).reshape(-1,1)).flatten()
    sp_vl=sc_s.inverse_transform(predict_model(SM,Xvs).reshape(-1,1)).flatten()

    tl=np.concatenate([t_tr_s,t_vl_s])[-lookback:]
    sl=np.concatenate([s_tr_s,s_vl_s])[-lookback:]
    with st.spinner("Recursive forecast pada test set…"):
        tp_te=sc_t.inverse_transform(recursive_forecast(TM,tl,n_test).reshape(-1,1)).flatten()
        sp_te=sc_s.inverse_transform(recursive_forecast(SM,sl,n_test).reshape(-1,1)).flatten()

    h_tr=tp_tr+sp_tr; h_vl=tp_vl+sp_vl; h_te=tp_te+sp_te

    def fplot(title,actual,act_dates,segs):
        fig,ax=plt.subplots(figsize=(14,3.8))
        ax.plot(act_dates,actual,color=PAL["actual"],lw=1.2,label="Actual")
        for d,y,col,lbl in segs:
            ax.plot(d,y,color=col,lw=1.2,ls="--",label=lbl)
        ax.set_title(title); ax.legend(); ax.grid(True,lw=.5)
        st.pyplot(fig,use_container_width=True); plt.close(fig)

    fplot("Trend – Actual vs Predicted", trend_tv, dates[:n_train+n_val],[
        (dates[lookback:n_train],       tp_tr,PAL["train"],"Train"),
        (dates[n_train:n_train+n_val],  tp_vl,PAL["val"],  "Val"),
        (dates[n_train+n_val:],         tp_te,PAL["test"], "Test"),
    ])
    fplot("Seasonal – Actual vs Predicted", season_tv, dates[:n_train+n_val],[
        (dates[lookback:n_train],       sp_tr,PAL["train"],"Train"),
        (dates[n_train:n_train+n_val],  sp_vl,PAL["val"],  "Val"),
        (dates[n_train+n_val:],         sp_te,PAL["test"], "Test"),
    ])

    fig,ax=plt.subplots(figsize=(14,4))
    ax.plot(dates,y_full,color=PAL["actual"],lw=1.2,label="Actual SST")
    ax.plot(dates[lookback:n_train],      h_tr,color=PAL["train"],lw=1.2,ls="--",label="Hybrid Train")
    ax.plot(dates[n_train:n_train+n_val], h_vl,color=PAL["val"],  lw=1.2,ls="--",label="Hybrid Val")
    ax.plot(dates[n_train+n_val:],        h_te,color=PAL["test"], lw=1.5,ls="--",label="Hybrid Test")
    ax.set_title("Hybrid Reconstruction – Full Series"); ax.legend(); ax.grid(True,lw=.5)
    st.pyplot(fig,use_container_width=True); plt.close(fig)

    fig,ax=plt.subplots(figsize=(12,3.8))
    ax.plot(dates[n_train+n_val:],y_full[n_train+n_val:],color=PAL["actual"],lw=1.5,label="Actual")
    ax.plot(dates[n_train+n_val:],h_te,                  color=PAL["test"],  lw=1.5,ls="--",label="Predicted")
    ax.fill_between(dates[n_train+n_val:],y_full[n_train+n_val:],h_te,alpha=.1,color=PAL["test"])
    ax.set_title("TEST SET – Actual vs Predicted"); ax.legend(); ax.grid(True,lw=.5)
    st.pyplot(fig,use_container_width=True); plt.close(fig)

    df_pred=pd.DataFrame({
        "date":dates[n_train+n_val:],"actual":y_full[n_train+n_val:],
        "predicted_hybrid":h_te,"predicted_trend":tp_te,"predicted_seasonal":sp_te,
    })
    st.download_button("⬇ Download Hasil Prediksi Test",
                       df_pred.to_csv(index=False).encode(),"test_predictions.csv","text/csv")

    st.session_state.update(dict(
        tp_tr=tp_tr,tp_vl=tp_vl,tp_te=tp_te,
        sp_tr=sp_tr,sp_vl=sp_vl,sp_te=sp_te,
        h_tr=h_tr,h_vl=h_vl,h_te=h_te,
    ))

# ─── TAB 5 · METRICS ─────────────────────────────────────────────────────────
with t5:
    if "h_te" not in st.session_state:
        st.info("Jalankan forecast terlebih dahulu."); st.stop()

    tp_tr=st.session_state["tp_tr"]; tp_vl=st.session_state["tp_vl"]; tp_te=st.session_state["tp_te"]
    sp_tr=st.session_state["sp_tr"]; sp_vl=st.session_state["sp_vl"]; sp_te=st.session_state["sp_te"]
    h_tr=st.session_state["h_tr"];   h_vl=st.session_state["h_vl"];   h_te=st.session_state["h_te"]
    trend_tv=st.session_state["trend_tv"]; season_tv=st.session_state["season_tv"]

    t_tr_true=trend_tv[lookback:n_train];     t_vl_true=trend_tv[n_train:n_train+n_val]
    s_tr_true=season_tv[lookback:n_train];    s_vl_true=season_tv[n_train:n_train+n_val]

    all_m={}
    def show_block(title,rows):
        st.markdown(f"### {title}")
        for lbl,(yt,yp) in rows.items():
            m=calc_metrics(yt,yp); all_m[f"{title.split()[-1]} – {lbl}"]=m
            cols=st.columns(4)
            mcard(cols[0],f"{lbl} RMSE",  f"{m['RMSE']:.4f}")
            mcard(cols[1],f"{lbl} MAE",   f"{m['MAE']:.4f}")
            mcard(cols[2],f"{lbl} MAPE",  f"{m['MAPE (%)']:.2f}%")
            mcard(cols[3],f"{lbl} R²",    f"{m['R²']:.4f}")
            st.markdown("")

    show_block("🔵 Trend",   {"Train":(t_tr_true,tp_tr),"Val":(t_vl_true,tp_vl)})
    show_block("🟠 Seasonal",{"Train":(s_tr_true,sp_tr),"Val":(s_vl_true,sp_vl)})
    show_block("🌊 Hybrid",  {
        "Train":(y_full[lookback:n_train],h_tr),
        "Val":  (y_full[n_train:n_train+n_val],h_vl),
        "TEST": (y_full[n_train+n_val:],h_te),
    })

    st.markdown("### 📋 Summary Table")
    sdf=pd.DataFrame(all_m).T.round(4)
    st.dataframe(
        sdf.style
           .background_gradient(subset=["RMSE","MAE","MAPE (%)"],cmap="RdYlGn_r")
           .background_gradient(subset=["R²"],cmap="RdYlGn"),
        use_container_width=True,
    )
    st.download_button("⬇ Download Metrics CSV",
                       sdf.to_csv().encode(),"metrics_summary.csv","text/csv")
