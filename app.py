"""
Dashboard Analisis Time Series – CNN-BiLSTM + STL Decomposition
Backend  : PyTorch CPU-optimised
Run      : streamlit run app.py
"""
import warnings; warnings.filterwarnings("ignore")
import streamlit as st
import random, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="SST · CNN-BiLSTM+STL", page_icon="🌊",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@700&family=DM+Sans:wght@400;600&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
h1,h2,h3{font-family:'Space Mono',monospace;}
.sec-title{
    font-family:'Space Mono',monospace;font-size:.9rem;font-weight:700;color:#e2e8f0;
    background:linear-gradient(90deg,#1e3a5f,transparent);
    border-left:4px solid #38bdf8;padding:.4rem .8rem;
    margin:1.1rem 0 .6rem;border-radius:0 6px 6px 0;
}
.metric-card{
    background:linear-gradient(135deg,#1a2236,#151c30);
    border:1px solid #2d3a56;border-radius:10px;
    padding:.9rem 1rem;text-align:center;margin-bottom:.4rem;
}
.metric-card .lbl{font-size:.68rem;color:#94a3b8;letter-spacing:.1em;text-transform:uppercase;}
.metric-card .val{font-size:1.35rem;font-family:'Space Mono',monospace;color:#38bdf8;font-weight:700;}
.metric-card .sub{font-size:.7rem;color:#64748b;margin-top:1px;}
.badge-ok{background:#064e3b;color:#34d399;padding:2px 9px;border-radius:999px;font-size:.76rem;}
.badge-err{background:#450a0a;color:#f87171;padding:2px 9px;border-radius:999px;font-size:.76rem;}
.narasi{background:#111827;border:1px solid #1e3a5f;border-radius:8px;
    padding:.75rem 1rem;color:#cbd5e1;font-size:.86rem;line-height:1.6;margin:.5rem 0 .9rem;}
</style>""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor":"#0e1525","axes.facecolor":"#0e1525",
    "axes.edgecolor":"#2d3a56","axes.labelcolor":"#94a3b8",
    "xtick.color":"#64748b","ytick.color":"#64748b",
    "text.color":"#e2e8f0","grid.color":"#1e2d45","grid.alpha":.55,
    "legend.facecolor":"#111827","legend.edgecolor":"#2d3a56",
    "legend.fontsize":8,"font.family":"monospace",
    "axes.titlecolor":"#e2e8f0","axes.titlesize":10,"axes.titleweight":"bold",
})
PAL = dict(actual="#38bdf8",train="#34d399",val="#fbbf24",
           test="#f87171",trend="#a78bfa",season="#fb923c",
           resid="#94a3b8",future="#f0abfc")

def sec(t):  st.markdown(f'<div class="sec-title">{t}</div>', unsafe_allow_html=True)
def narasi(t): st.markdown(f'<div class="narasi">💡 {t}</div>', unsafe_allow_html=True)
def mcard(col, lbl, val, sub=""):
    col.markdown(f"""<div class="metric-card">
        <div class="lbl">{lbl}</div><div class="val">{val}</div><div class="sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

def tight_ylim(ax, arrays, pad=0.12):
    all_v = np.concatenate([np.asarray(a).flatten() for a in arrays if len(a)>0])
    lo,hi = np.nanmin(all_v),np.nanmax(all_v); r=(hi-lo)*pad
    ax.set_ylim(lo-r, hi+r)

# ═══════════════════════════════════════════════════════════════
# PYTORCH MODEL  (ringan, CPU-friendly)
# ═══════════════════════════════════════════════════════════════
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class CNNBiLSTM(nn.Module):
    def __init__(self, lb, cf, ks, lu, du, drop=0.0):
        super().__init__()
        self.pad    = nn.ConstantPad1d((ks-1,0), 0)
        self.conv   = nn.Conv1d(1, cf, ks)
        self.relu   = nn.ReLU()
        self.bilstm = nn.LSTM(cf, lu, batch_first=True, bidirectional=True)
        self.drop   = nn.Dropout(drop)
        self.fc1    = nn.Linear(lu*2, du)
        self.fc2    = nn.Linear(du, 1)
    def forward(self, x):                  # x: (B,L,1)
        x = x.permute(0,2,1)
        x = self.relu(self.conv(self.pad(x)))
        x = x.permute(0,2,1)
        out,_ = self.bilstm(x)
        x = self.drop(out[:,-1,:])
        return self.fc2(self.relu(self.fc1(x))).squeeze(-1)


def train_model(model, Xtr, ytr, Xvl, yvl, epochs, bs, lr, patience=15):
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=6, factor=0.5, verbose=False)
    crit  = nn.MSELoss()
    Xt=torch.tensor(Xtr,dtype=torch.float32); yt=torch.tensor(ytr,dtype=torch.float32)
    Xv=torch.tensor(Xvl,dtype=torch.float32); yv=torch.tensor(yvl,dtype=torch.float32)
    loader=DataLoader(TensorDataset(Xt,yt), batch_size=bs, shuffle=False,
                      num_workers=0, pin_memory=False)
    best_val,best_w,wait=float("inf"),None,0
    h_tr,h_vl=[],[]
    for _ in range(epochs):
        model.train()
        ep=[]
        for xb,yb in loader:
            opt.zero_grad(set_to_none=True)
            l=crit(model(xb.unsqueeze(-1)),yb)
            l.backward(); opt.step(); ep.append(l.item())
        tl=float(np.mean(ep))
        model.eval()
        with torch.no_grad():
            vl=crit(model(Xv.unsqueeze(-1)),yv).item()
        h_tr.append(tl); h_vl.append(vl); sched.step(vl)
        if vl < best_val-1e-7:
            best_val=vl; best_w={k:v.clone() for k,v in model.state_dict().items()}; wait=0
        else:
            wait+=1
            if wait>=patience: break
    if best_w: model.load_state_dict(best_w)
    return h_tr,h_vl


def predict_arr(model, X):
    """Batch predict, returns flat numpy array."""
    model.eval()
    with torch.no_grad():
        t=torch.tensor(X,dtype=torch.float32)
        return model(t.unsqueeze(-1)).numpy().flatten()


def sliding_predict(model, series_norm, lookback):
    """
    One-step-ahead prediction dengan sliding window dari data ASLI (teacher forcing).
    Jauh lebih akurat dari recursive karena tidak ada error akumulasi.
    Dipakai untuk train, val, DAN test set.
    """
    preds=[]
    for i in range(lookback, len(series_norm)):
        w = torch.tensor(series_norm[i-lookback:i], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        model.eval()
        with torch.no_grad():
            preds.append(model(w).item())
    return np.array(preds, dtype=np.float32)


def recursive_forecast(model, window, steps):
    """Recursive untuk FUTURE forecast (tidak ada data aktual)."""
    model.eval()
    w=list(window.copy()); out=[]
    lb=len(window)
    with torch.no_grad():
        for _ in range(steps):
            xin=torch.tensor(w[-lb:],dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            p=model(xin).item(); out.append(p); w.append(p)
    return np.array(out, dtype=np.float32)

# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════
def mape_fn(yt,yp):
    yt,yp=np.array(yt),np.array(yp); m=yt!=0
    return np.mean(np.abs((yt[m]-yp[m])/yt[m]))*100

def mae_fn(yt,yp):
    return np.mean(np.abs(np.array(yt)-np.array(yp)))

def build_dataset(arr, lb):
    X,y=[],[]
    for i in range(lb,len(arr)):
        X.append(arr[i-lb:i]); y.append(arr[i])
    return np.array(X,dtype=np.float32), np.array(y,dtype=np.float32)

def fungsi_spektral(x):
    x=np.asarray(x,dtype=float); n=len(x); k=round((n-1)/2)
    t=np.arange(1,n+1); pg=np.zeros(k)
    for i in range(1,k+1):
        w=(2*np.pi*i)/n
        a=(2/n)*np.sum(x*np.cos(w*t))
        b=(2/n)*np.sum(x*np.sin(w*t))
        pg[i-1]=a**2+b**2
    km=np.argmax(pg)+1
    per=int(round((2*np.pi)/((2*np.pi*km)/n)))
    Th=np.max(pg)/np.sum(pg); Tt=0.13135
    return per,Th,Tt,Th>Tt,pg

def fdGPH(x, bw=0.5):
    import statsmodels.api as sm
    x=np.asarray(x,dtype=float)-np.mean(x); n=len(x)
    m=int(np.floor(n**bw)); j=np.arange(1,m+1); lam=2*np.pi*j/n
    fv=np.fft.fft(x)
    I=(1/(2*np.pi*n))*np.abs(fv[j])**2
    Y=np.log(I); Xm=sm.add_constant(np.log(4*(np.sin(lam/2)**2)))
    mo=sm.OLS(Y,Xm).fit()
    return -mo.params[1], mo

def generate_random_sst(n=4071, seed=42):
    rng=np.random.default_rng(seed); t=np.arange(n)
    trend   = 29.3 + 0.000144*t
    seasonal= 0.30*np.sin(2*np.pi*t/365 - 0.3) + 0.12*np.sin(4*np.pi*t/365)
    noise   = rng.normal(0, 0.09, n)
    for i in range(1,n): noise[i] += 0.62*noise[i-1]
    sst=trend+seasonal+noise
    # clamp mirip data asli: 27.4 - 30.7
    sst=np.clip(sst,27.38,30.67)
    dates=pd.date_range("2015-01-01",periods=n,freq="D")
    # format tanggal sama persis dengan data asli
    return pd.DataFrame({"tgl":dates.strftime("%-m/%-d/%Y"),"sst":np.round(sst,5)})

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌊 SST Forecast")
    st.markdown("**CNN-BiLSTM + STL**")
    st.caption("Backend: PyTorch CPU")
    st.divider()

    st.markdown("### 📂 Data Source")
    data_source=st.radio("Pilih sumber data:",
                         ["📁 Upload CSV","🎲 Generate Data Contoh"])
    uploaded=None
    if data_source=="📁 Upload CSV":
        uploaded=st.file_uploader("Upload CSV",type=["csv"])
        st.caption("Kolom tanggal + kolom SST numerik")
    else:
        gen_n   =st.slider("Jumlah hari",365,5000,4071,365)
        gen_seed=st.number_input("Seed data",value=42)

    date_col=st.text_input("Nama kolom tanggal",value="tgl")
    sst_col =st.text_input("Nama kolom target", value="sst")
    st.divider()

    st.markdown("### 🔧 Data Split")
    train_r=st.slider("Train ratio",0.50,0.95,0.90,0.01)
    val_r  =st.slider("Val ratio",  0.01,0.20,0.05,0.01)
    test_r =round(1-train_r-val_r,4)
    st.markdown(f"**Test ratio (auto):** `{max(test_r,0):.2f}`")

    st.markdown("### 📅 STL")
    auto_period=st.checkbox("Auto-detect period (spektral)",value=True)
    manual_period=365
    if not auto_period:
        manual_period=st.number_input("Period manual",2,730,365)
    stl_robust=st.checkbox("STL robust",value=True)

    st.markdown("### 🧠 Trend Model")
    t_conv_f=st.slider("Conv1D filters", 8,128,16, 8,
        help="Lebih kecil = lebih cepat. Mulai dari 16.")
    t_kern  =st.slider("Kernel size",    2, 15, 3, 1)
    t_lstm  =st.slider("BiLSTM units",  16,128,32,16,
        help="32 sudah cukup untuk SST. 64+ jauh lebih lambat.")
    t_drop  =st.slider("Dropout",      0.0,0.5,0.1,0.05)
    t_dense =st.slider("Dense units",   8,128,16, 8)
    t_lr    =st.number_input("LR trend",value=0.001,format="%.4f")

    st.markdown("### 🧠 Seasonal Model")
    s_conv_f=st.slider("Conv1D filters (S)", 8,128,16, 8)
    s_kern  =st.slider("Kernel size (S)",    2, 15, 3, 1)
    s_lstm  =st.slider("BiLSTM units (S)",  16,128,32,16)
    s_dense =st.slider("Dense units (S)",   8, 64, 16, 8)
    s_lr    =st.number_input("LR seasonal",value=0.001,format="%.4f")

    st.markdown("### ⚙️ Training")
    lookback  =st.slider("Lookback", 7, 90, 30, 7,
        help="⚡ Default 30 hari — jauh lebih cepat dari 180. Hasil biasanya tidak jauh berbeda untuk data harian.")
    epochs    =st.slider("Max epochs",10,300,100,10)
    batch_size=st.selectbox("Batch size",[32,64,128,256],index=1)
    seed      =st.number_input("Random seed",value=42)

    st.divider()
    run_btn=st.button("▶  Run Analysis",use_container_width=True,type="primary")

# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════
st.markdown("# 🌊 SST Time Series Analysis Dashboard")
st.markdown("*Hybrid CNN-BiLSTM + STL · Sea Surface Temperature Forecasting*")

if not run_btn:
    st.info("👈 Pilih sumber data & atur parameter di sidebar, lalu klik **▶ Run Analysis**.")
    st.markdown("""
    **⚡ Tips agar cepat:**
    - Lookback = **30** (bukan 180)
    - BiLSTM units = **32**
    - Conv1D filters = **16**
    - Max epochs = **100** (early stopping aktif)
    - Batch size = **64**
    """)
    with st.expander("📋 Format CSV yang dibutuhkan"):
        s=generate_random_sst(10,42)
        st.dataframe(s,use_container_width=True)
        st.download_button("⬇ Download contoh CSV",
                           s.to_csv(index=False).encode(),"contoh_data.csv","text/csv")
    st.stop()

# ── seed ──────────────────────────────────────────────────────
torch.manual_seed(int(seed)); np.random.seed(int(seed)); random.seed(int(seed))

try:
    from statsmodels.tsa.seasonal import STL
    from sklearn.preprocessing import MinMaxScaler
except ImportError as e:
    st.error(f"Import error: {e}"); st.stop()

# ── load data ──────────────────────────────────────────────────
if data_source=="🎲 Generate Data Contoh":
    df_raw=generate_random_sst(int(gen_n),int(gen_seed))
    st.success(f"✅ Data sintetis: {len(df_raw):,} baris")
else:
    if uploaded is None:
        st.error("⚠️ Upload file CSV terlebih dahulu."); st.stop()
    df_raw=pd.read_csv(uploaded)

if date_col not in df_raw.columns or sst_col not in df_raw.columns:
    st.error(f"Kolom `{date_col}` atau `{sst_col}` tidak ada. Tersedia: {list(df_raw.columns)}"); st.stop()

df=df_raw.copy()
df[date_col]=pd.to_datetime(df[date_col], dayfirst=False, infer_datetime_format=True)
df=df.sort_values(date_col).set_index(date_col)
y_full=df[sst_col].values.astype(float)
dates=df.index; n=len(y_full)

n_train=int(n*train_r); n_val=int(n*val_r); n_test=n-n_train-n_val
if n_test<=0:
    st.error("Test set kosong. Kurangi train/val ratio."); st.stop()

y_trainval=y_full[:n_train+n_val]

# ── STL ───────────────────────────────────────────────────────
with st.spinner("Running STL decomposition..."):
    periode=fungsi_spektral(y_trainval)[0] if auto_period else manual_period
    stl_res=STL(y_trainval, period=periode, robust=stl_robust).fit()

trend_tv  = stl_res.trend
season_tv = stl_res.seasonal

# ─────────────────────────────────────────────────────────────
# SCALING
# Kunci: fit scaler pada SELURUH train+val agar test tidak out-of-range
# ─────────────────────────────────────────────────────────────
sc_t=MinMaxScaler().fit(trend_tv.reshape(-1,1))
sc_s=MinMaxScaler().fit(season_tv.reshape(-1,1))

# Normalized full trainval
t_tv_s = sc_t.transform(trend_tv.reshape(-1,1)).flatten().astype(np.float32)
s_tv_s = sc_s.transform(season_tv.reshape(-1,1)).flatten().astype(np.float32)

# Split after normalization
t_tr_s = t_tv_s[:n_train];  t_vl_s = t_tv_s[n_train:]
s_tr_s = s_tv_s[:n_train];  s_vl_s = s_tv_s[n_train:]

# Build train/val datasets
Xtt,ytt = build_dataset(t_tr_s, lookback)
Xvt,yvt = build_dataset(np.concatenate([t_tr_s[-lookback:], t_vl_s]), lookback)
Xts,yts = build_dataset(s_tr_s, lookback)
Xvs,yvs = build_dataset(np.concatenate([s_tr_s[-lookback:], s_vl_s]), lookback)

# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════
t0,t1,t2,t3,t4,t5 = st.tabs([
    "📊 Data Overview",
    "🔬 STL & Karakteristik",
    "🤖 Model Training",
    "🎯 Forecast Results",
    "📋 Metrics",
    "🔮 Future Forecast",
])

# ─── TAB 0 · DATA OVERVIEW ───────────────────────────────────
with t0:
    c1,c2,c3,c4=st.columns(4)
    mcard(c1,"Total Points",f"{n:,}")
    mcard(c2,"Train",f"{n_train:,}",f"{train_r*100:.0f}%")
    mcard(c3,"Val",  f"{n_val:,}",  f"{val_r*100:.0f}%")
    mcard(c4,"Test", f"{n_test:,}", f"{(n_test/n)*100:.1f}%")

    sec("📈 Visualisasi Data & Split")
    fig,ax=plt.subplots(figsize=(14,3.8))
    for sl,col,lbl in [
        (slice(None,n_train),          PAL["train"],f"Train ({n_train})"),
        (slice(n_train,n_train+n_val), PAL["val"],  f"Val ({n_val})"),
        (slice(n_train+n_val,None),    PAL["test"], f"Test ({n_test})"),
    ]:
        ax.plot(dates[sl],y_full[sl],color=col,lw=1.1,label=lbl)
        ax.fill_between(dates[sl],y_full[sl],alpha=.07,color=col)
    ax.axvline(dates[n_train],       color=PAL["val"], lw=1.3,ls="--",alpha=.8)
    ax.axvline(dates[n_train+n_val], color=PAL["test"],lw=1.3,ls="--",alpha=.8)
    ax.set_title("Data Split Visualization"); ax.set_ylabel("SST (°C)")
    tight_ylim(ax,[y_full])
    ax.legend(); ax.grid(True,lw=.4)
    st.pyplot(fig,use_container_width=True); plt.close(fig)

    sec("📋 Tabel Data – Sebelum & Sesudah Normalisasi")
    sc_full=MinMaxScaler().fit(y_full.reshape(-1,1))
    y_norm=sc_full.transform(y_full.reshape(-1,1)).flatten()
    col_a,col_b=st.columns(2)
    with col_a:
        st.markdown("**Data Asli (10 baris pertama)**")
        raw_df=df[[sst_col]].reset_index().head(10).copy()
        raw_df.columns=["Tanggal","SST (°C)"]
        st.dataframe(raw_df.style.format({"SST (°C)":"{:.5f}"}),use_container_width=True)
    with col_b:
        st.markdown("**Sebelum & Sesudah Normalisasi**")
        norm_df=pd.DataFrame({
            "Tanggal":dates[:10],
            "SST (°C)":np.round(y_full[:10],5),
            "SST Norm [0,1]":np.round(y_norm[:10],5),
        })
        st.dataframe(norm_df.style.format({
            "SST (°C)":"{:.5f}","SST Norm [0,1]":"{:.5f}"}),
            use_container_width=True)

    sec("📊 Statistik Deskriptif")
    st.dataframe(df[[sst_col]].describe().T.round(4),use_container_width=True)
    st.download_button("⬇ Download data",
        df[[sst_col]].reset_index().to_csv(index=False).encode(),"data_used.csv","text/csv")

# ─── TAB 1 · STL & KARAKTERISTIK ─────────────────────────────
with t1:
    st.success(f"STL selesai · Period = **{periode}** hari")

    sec("🔬 STL Decomposition")
    fig,axes=plt.subplots(4,1,figsize=(14,10),sharex=True)
    for ax,(name,val,col) in zip(axes,[
        ("Observed", stl_res.observed, PAL["actual"]),
        ("Trend",    trend_tv,         PAL["trend"]),
        ("Seasonal", season_tv,        PAL["season"]),
        ("Residual", stl_res.resid,    PAL["resid"]),
    ]):
        ax.plot(val,color=col,lw=1.2)
        ax.fill_between(range(len(val)),val,alpha=.1,color=col)
        ax.set_ylabel(name,fontsize=9,color="#e2e8f0"); ax.grid(True,lw=.4)
        tight_ylim(ax,[val])
    axes[-1].set_xlabel("Index")
    fig.suptitle(f"STL Decomposition (period={periode})",fontsize=11)
    plt.tight_layout()
    st.pyplot(fig,use_container_width=True); plt.close(fig)

    vo=np.var(stl_res.observed)
    ve_t=(1-np.var(stl_res.observed-trend_tv)/vo)*100
    ve_s=(1-np.var(stl_res.observed-season_tv)/vo)*100
    ve_r=np.var(stl_res.resid)/vo*100
    c1,c2,c3=st.columns(3)
    mcard(c1,"Variance – Trend",   f"{ve_t:.1f}%")
    mcard(c2,"Variance – Seasonal",f"{ve_s:.1f}%")
    mcard(c3,"Variance – Residual",f"{ve_r:.1f}%")

    # ── TREND: GPH ──────────────────────────────────────────
    sec("📐 Karakteristik Komponen Trend – GPH")
    with st.spinner("Menghitung GPH..."):
        d_gph,_=fdGPH(trend_tv, bw=0.5)

    if d_gph<0:     mc,ms="Anti-persistent","d < 0: mean-reverting kuat"
    elif d_gph<0.5: mc,ms="Long Memory – Stasioner","0 < d < 0.5"
    elif d_gph<1.0: mc,ms="Long Memory – Non-Stasioner","0.5 ≤ d < 1"
    else:           mc,ms="Non-Stasioner Kuat","d ≥ 1"
    c1,c2=st.columns(2)
    mcard(c1,"GPH d estimate",f"{d_gph:.4f}")
    mcard(c2,"Memory Class",  mc,ms)

    slope=(trend_tv[-1]-trend_tv[0])/len(trend_tv)
    arah ="meningkat" if slope>0 else "menurun"
    t_rng=trend_tv.max()-trend_tv.min()

    if d_gph<0:
        mem_n=(f"Nilai d={d_gph:.4f} (d<0) → **anti-persistent**: guncangan pada tren "
               "langsung berbalik arah, tren bergerak sangat mulus tanpa memori jangka panjang.")
    elif d_gph<0.5:
        mem_n=(f"Nilai d={d_gph:.4f} (0<d<0.5) → **long memory stasioner**: tren memiliki "
               "ketergantungan jangka panjang namun tetap stasioner. Fluktuasi masa lalu "
               "masih berpengaruh signifikan terhadap nilai masa depan.")
    elif d_gph<1.0:
        mem_n=(f"Nilai d={d_gph:.4f} (0.5≤d<1) → **long memory non-stasioner**: tren "
               "sangat persisten dan tidak stasioner. Diferensiasi fraksional disarankan.")
    else:
        mem_n=(f"Nilai d={d_gph:.4f} (d≥1) → **non-stasioner kuat**, serupa random walk. "
               "Integrasi orde tinggi terindikasi.")

    narasi(f"Komponen tren secara umum **{arah}** dengan rentang "
           f"**{t_rng:.4f}°C** ({trend_tv.min():.4f}–{trend_tv.max():.4f}°C). {mem_n}")

    # ── SEASONAL: SPEKTRAL ───────────────────────────────────
    sec("📈 Karakteristik Komponen Musiman – Spektral")
    per_sp,Th,Tt,mus,pg=fungsi_spektral(y_trainval)
    c1,c2,c3=st.columns(3)
    mcard(c1,"Dominant Period",f"{per_sp} hari")
    mcard(c2,"T-hitung",       f"{Th:.5f}")
    mcard(c3,"T-tabel",        f"{Tt:.5f}")
    badge=('<span class="badge-ok">✓ Pola Musiman Terdeteksi</span>' if mus
           else '<span class="badge-err">✗ Tidak Musiman</span>')
    st.markdown(f"**Kesimpulan Uji:** {badge}",unsafe_allow_html=True)

    fig,ax=plt.subplots(figsize=(14,3))
    ax.plot(pg,color=PAL["season"],lw=1.2)
    ax.fill_between(range(len(pg)),pg,alpha=.13,color=PAL["season"])
    ix=np.argmax(pg)
    ax.axvline(ix,color="#f87171",lw=1.5,ls="--",label=f"Peak @ idx={ix}")
    ax.set_title("Periodogram – Komponen Musiman")
    ax.set_xlabel("Frequency Index"); ax.set_ylabel("Power")
    ax.legend(); ax.grid(True,lw=.4)
    st.pyplot(fig,use_container_width=True); plt.close(fig)

    s_amp=season_tv.max()-season_tv.min(); s_std=season_tv.std()
    if mus:
        mus_n=(f"Uji spektral mendeteksi **pola musiman signifikan** "
               f"(T-hitung={Th:.5f} > T-tabel={Tt:.5f}). "
               f"Periode dominan **{per_sp} hari**.")
    else:
        mus_n=(f"Uji spektral **tidak mendeteksi** pola musiman signifikan "
               f"(T-hitung={Th:.5f} < T-tabel={Tt:.5f}).")
    narasi(f"{mus_n} Amplitudo musiman **{s_amp:.4f}°C**, "
           f"std **{s_std:.4f}°C** — "
           f"intensitas {'cukup signifikan' if s_amp>0.5 else 'relatif kecil'} "
           f"terhadap nilai SST keseluruhan.")

# ─── TAB 2 · MODEL TRAINING ──────────────────────────────────
with t2:
    sec("🤖 Training CNN-BiLSTM")
    col1,col2=st.columns(2)

    with col1:
        st.markdown("#### 🔵 Trend Model")
        prog=st.progress(0,text="Training…")
        TM=CNNBiLSTM(lookback,t_conv_f,t_kern,t_lstm,t_dense,t_drop)
        ht_tr,ht_val=train_model(TM,Xtt,ytt,Xvt,yvt,epochs,batch_size,t_lr)
        prog.progress(100,text=f"Done · {len(ht_tr)} epochs")
        fig,ax=plt.subplots(figsize=(6,3))
        ax.plot(ht_tr, color=PAL["train"],lw=1.4,label="Train")
        ax.plot(ht_val,color=PAL["val"],  lw=1.4,ls="--",label="Val")
        ax.set_title("Trend – Loss Curve"); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(True,lw=.4)
        st.pyplot(fig,use_container_width=True); plt.close(fig)

    with col2:
        st.markdown("#### 🟠 Seasonal Model")
        prog2=st.progress(0,text="Training…")
        SM=CNNBiLSTM(lookback,s_conv_f,s_kern,s_lstm,s_dense,dropout=0.0)
        hs_tr,hs_val=train_model(SM,Xts,yts,Xvs,yvs,epochs,batch_size,s_lr)
        prog2.progress(100,text=f"Done · {len(hs_tr)} epochs")
        fig,ax=plt.subplots(figsize=(6,3))
        ax.plot(hs_tr, color=PAL["season"],lw=1.4,label="Train")
        ax.plot(hs_val,color=PAL["val"],   lw=1.4,ls="--",label="Val")
        ax.set_title("Seasonal – Loss Curve"); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(True,lw=.4)
        st.pyplot(fig,use_container_width=True); plt.close(fig)

    st.session_state.update(dict(
        trained=True, TM=TM, SM=SM, sc_t=sc_t, sc_s=sc_s,
        t_tr_s=t_tr_s, t_vl_s=t_vl_s, s_tr_s=s_tr_s, s_vl_s=s_vl_s,
        t_tv_s=t_tv_s, s_tv_s=s_tv_s,
    ))

# ─── TAB 3 · FORECAST RESULTS ────────────────────────────────
with t3:
    if "trained" not in st.session_state:
        st.info("Jalankan training terlebih dahulu (tab 🤖)."); st.stop()

    TM=st.session_state["TM"]; SM=st.session_state["SM"]
    sc_t=st.session_state["sc_t"]; sc_s=st.session_state["sc_s"]
    t_tv_s=st.session_state["t_tv_s"]; s_tv_s=st.session_state["s_tv_s"]

    # ── Teacher-forcing sliding window untuk train+val+test ──
    # Test: kita punya data aktual dari STL (trainval), tapi untuk test set
    # kita gunakan recursive karena STL tidak dijalankan pada test.
    # Namun untuk train & val, gunakan sliding window dari data asli (akurat).

    with st.spinner("Menghitung prediksi…"):
        # Train: sliding window dari t_tv_s[0:n_train]
        t_pred_tr_s = sliding_predict(TM, t_tv_s[:n_train], lookback)
        t_pred_vl_s = sliding_predict(TM, t_tv_s, lookback)[n_train-lookback:]  # val portion

        s_pred_tr_s = sliding_predict(SM, s_tv_s[:n_train], lookback)
        s_pred_vl_s = sliding_predict(SM, s_tv_s, lookback)[n_train-lookback:]

        # Inverse transform
        tp_tr=sc_t.inverse_transform(t_pred_tr_s.reshape(-1,1)).flatten()
        tp_vl=sc_t.inverse_transform(t_pred_vl_s.reshape(-1,1)).flatten()
        sp_tr=sc_s.inverse_transform(s_pred_tr_s.reshape(-1,1)).flatten()
        sp_vl=sc_s.inverse_transform(s_pred_vl_s.reshape(-1,1)).flatten()

        # Val actual alignment
        t_vl_true=trend_tv[n_train:]
        s_vl_true=season_tv[n_train:]
        # trim ke panjang yang sama
        min_vl=min(len(tp_vl),len(t_vl_true))
        tp_vl=tp_vl[-min_vl:]; sp_vl=sp_vl[-min_vl:]
        t_vl_true=t_vl_true[-min_vl:]; s_vl_true=s_vl_true[-min_vl:]

        # Test: recursive dari last window
        tl_last=t_tv_s[-lookback:]; sl_last=s_tv_s[-lookback:]
        tp_te_s=recursive_forecast(TM, tl_last, n_test)
        sp_te_s=recursive_forecast(SM, sl_last, n_test)
        tp_te=sc_t.inverse_transform(tp_te_s.reshape(-1,1)).flatten()
        sp_te=sc_s.inverse_transform(sp_te_s.reshape(-1,1)).flatten()

    # Hybrid
    h_tr = tp_tr+sp_tr
    h_vl = tp_vl+sp_vl
    h_te = tp_te+sp_te

    # Align train actual
    y_tr_true=y_full[lookback:n_train]
    y_vl_true=y_full[n_train:n_train+min_vl]
    y_te_true=y_full[n_train+n_val:]

    # ── Plots ────────────────────────────────────────────────
    sec("📉 Trend – Actual vs Predicted")
    fig,ax=plt.subplots(figsize=(14,3.4))
    ax.plot(dates[:n_train+n_val],trend_tv,   color=PAL["actual"],lw=1.3,label="Actual Trend",zorder=3)
    ax.plot(dates[lookback:n_train],   tp_tr, color=PAL["train"], lw=1,  ls="--",label="Train Pred")
    ax.plot(dates[n_train:n_train+min_vl],tp_vl,color=PAL["val"],lw=1,  ls="--",label="Val Pred")
    ax.plot(dates[n_train+n_val:],     tp_te, color=PAL["test"],  lw=1.2,ls="--",label="Test Pred")
    ax.set_title("Trend – Actual vs Predicted")
    tight_ylim(ax,[trend_tv,tp_tr,tp_vl,tp_te]); ax.legend(); ax.grid(True,lw=.4)
    st.pyplot(fig,use_container_width=True); plt.close(fig)

    sec("🌊 Seasonal – Actual vs Predicted")
    fig,ax=plt.subplots(figsize=(14,3.4))
    ax.plot(dates[:n_train+n_val],season_tv,  color=PAL["actual"],lw=1.3,label="Actual Seasonal",zorder=3)
    ax.plot(dates[lookback:n_train],   sp_tr, color=PAL["train"], lw=1,  ls="--",label="Train Pred")
    ax.plot(dates[n_train:n_train+min_vl],sp_vl,color=PAL["val"],lw=1,  ls="--",label="Val Pred")
    ax.plot(dates[n_train+n_val:],     sp_te, color=PAL["test"],  lw=1.2,ls="--",label="Test Pred")
    ax.set_title("Seasonal – Actual vs Predicted")
    tight_ylim(ax,[season_tv,sp_tr,sp_vl,sp_te]); ax.legend(); ax.grid(True,lw=.4)
    st.pyplot(fig,use_container_width=True); plt.close(fig)

    sec("🔀 Hybrid Reconstruction – Full Series")
    fig,ax=plt.subplots(figsize=(14,4))
    ax.plot(dates,            y_full, color=PAL["actual"],lw=1.3,label="Actual SST",zorder=3)
    ax.plot(dates[lookback:n_train],  h_tr,color=PAL["train"],lw=1,  ls="--",label="Hybrid Train")
    ax.plot(dates[n_train:n_train+min_vl],h_vl,color=PAL["val"],lw=1,ls="--",label="Hybrid Val")
    ax.plot(dates[n_train+n_val:],    h_te,color=PAL["test"],lw=1.3,ls="--",label="Hybrid Test")
    ax.set_title("Hybrid Reconstruction – Full Series")
    tight_ylim(ax,[y_full,h_tr,h_vl,h_te]); ax.legend(); ax.grid(True,lw=.4)
    st.pyplot(fig,use_container_width=True); plt.close(fig)

    sec("🎯 Test Set – Actual vs Predicted")
    fig,ax=plt.subplots(figsize=(12,3.8))
    ax.plot(dates[n_train+n_val:],y_te_true,color=PAL["actual"],lw=1.5,label="Actual",zorder=3)
    ax.plot(dates[n_train+n_val:],h_te,     color=PAL["test"],  lw=1.5,ls="--",label="Predicted")
    ax.fill_between(dates[n_train+n_val:],y_te_true,h_te,alpha=.1,color=PAL["test"])
    ax.set_title("TEST SET – Actual vs Predicted")
    tight_ylim(ax,[y_te_true,h_te]); ax.legend(); ax.grid(True,lw=.4)
    st.pyplot(fig,use_container_width=True); plt.close(fig)

    sec("📋 Tabel Perbandingan – 10 Data Terakhir Test")
    last10_d  = dates[n_train+n_val:][-10:]
    last10_a  = y_te_true[-10:]
    last10_p  = h_te[-10:]
    df_l10=pd.DataFrame({
        "Tanggal":    last10_d,
        "Aktual (°C)":last10_a,
        "Prediksi (°C)":last10_p,
        "Error":      last10_a-last10_p,
        "APE (%)":    np.abs((last10_a-last10_p)/last10_a)*100,
    })
    st.dataframe(df_l10.style.format({
        "Aktual (°C)":"{:.4f}","Prediksi (°C)":"{:.4f}",
        "Error":"{:.4f}","APE (%)":"{:.2f}%"})
        .background_gradient(subset=["APE (%)"],cmap="RdYlGn_r"),
        use_container_width=True)

    st.download_button("⬇ Download Hasil Prediksi Test",
        pd.DataFrame({"date":dates[n_train+n_val:],"actual":y_te_true,
                      "predicted":h_te}).to_csv(index=False).encode(),
        "test_predictions.csv","text/csv")

    st.session_state.update(dict(
        h_tr=h_tr, h_vl=h_vl, h_te=h_te,
        y_tr_true=y_tr_true, y_vl_true=y_vl_true, y_te_true=y_te_true,
        tl_last=tl_last, sl_last=sl_last,
        tp_te_s=tp_te_s, sp_te_s=sp_te_s,
    ))

# ─── TAB 4 · METRICS ─────────────────────────────────────────
with t4:
    if "h_te" not in st.session_state:
        st.info("Jalankan forecast terlebih dahulu."); st.stop()

    h_tr=st.session_state["h_tr"]; h_vl=st.session_state["h_vl"]; h_te=st.session_state["h_te"]
    y_tr_true=st.session_state["y_tr_true"]
    y_vl_true=st.session_state["y_vl_true"]
    y_te_true=st.session_state["y_te_true"]

    sec("📊 Evaluasi Model Hybrid – MAPE & MAE")
    sets={"Training":(y_tr_true,h_tr),"Validation":(y_vl_true,h_vl),"Testing":(y_te_true,h_te)}
    cols_m=st.columns(3); results={}
    for (lbl,(yt,yp)),col in zip(sets.items(),cols_m):
        mp=mape_fn(yt,yp); ma=mae_fn(yt,yp)
        results[lbl]={"MAPE (%)":mp,"MAE":ma}
        col.markdown(f"**{lbl}**")
        mcard(col,"MAPE",f"{mp:.2f}%")
        mcard(col,"MAE", f"{ma:.4f} °C")

    sec("📋 Summary Table")
    sdf=pd.DataFrame(results).T.round(4)
    st.dataframe(sdf.style.background_gradient(
        subset=["MAPE (%)","MAE"],cmap="RdYlGn_r"),use_container_width=True)
    st.download_button("⬇ Download Metrics CSV",
        sdf.to_csv().encode(),"metrics_summary.csv","text/csv")

    te_mape=results["Testing"]["MAPE (%)"]
    te_mae =results["Testing"]["MAE"]
    kual=("sangat baik (< 1%)" if te_mape<1 else "baik (1–5%)" if te_mape<5
          else "cukup (5–10%)" if te_mape<10 else "perlu perbaikan (> 10%)")
    tr_mape=results["Training"]["MAPE (%)"]
    narasi(
        f"Model hybrid menghasilkan MAPE testing **{te_mape:.2f}%** dan MAE **{te_mae:.4f}°C** "
        f"— tergolong **{kual}**. "
        f"Selisih train ({tr_mape:.2f}%) vs test "
        f"({'menunjukkan potensi overfitting' if te_mape-tr_mape>5 else 'wajar, generalisasi memadai'})."
    )

# ─── TAB 5 · FUTURE FORECAST ─────────────────────────────────
with t5:
    if "tl_last" not in st.session_state:
        st.info("Jalankan forecast terlebih dahulu di tab 🎯."); st.stop()

    TM=st.session_state["TM"]; SM=st.session_state["SM"]
    sc_t=st.session_state["sc_t"]; sc_s=st.session_state["sc_s"]
    tl_last=st.session_state["tl_last"]; sl_last=st.session_state["sl_last"]
    tp_te_s=st.session_state["tp_te_s"]; sp_te_s=st.session_state["sp_te_s"]

    # extend window dengan prediksi test
    tl_fut=np.concatenate([tl_last, tp_te_s])[-lookback:].astype(np.float32)
    sl_fut=np.concatenate([sl_last, sp_te_s])[-lookback:].astype(np.float32)

    STEPS=10
    freq_g=pd.infer_freq(dates[:50]) or "D"
    fut_dates=pd.date_range(dates[-1],periods=STEPS+1,freq=freq_g)[1:]

    with st.spinner("Menghitung future forecast…"):
        tf_s=recursive_forecast(TM, tl_fut, STEPS)
        sf_s=recursive_forecast(SM, sl_fut, STEPS)

    tf=sc_t.inverse_transform(tf_s.reshape(-1,1)).flatten()
    sf=sc_s.inverse_transform(sf_s.reshape(-1,1)).flatten()
    hf=tf+sf

    sec("🔮 Forecast 10 Periode ke Depan")
    tail=min(90,n)
    fig,ax=plt.subplots(figsize=(14,4))
    ax.plot(dates[-tail:],y_full[-tail:], color=PAL["actual"],lw=1.5,label="Actual (tail)")
    ax.plot(fut_dates,hf,                color=PAL["future"],lw=2,ls="--",
            label="Future Forecast",marker="o",markersize=5)
    ax.axvline(dates[-1],color="#64748b",lw=1,ls=":",alpha=.7)
    ax.set_title("Future Forecast – 10 Periode ke Depan"); ax.set_ylabel("SST (°C)")
    tight_ylim(ax,[y_full[-tail:],hf]); ax.legend(); ax.grid(True,lw=.4)
    st.pyplot(fig,use_container_width=True); plt.close(fig)

    sec("📋 Tabel Forecast – Sebelum & Sesudah Denormalisasi")
    ca,cb=st.columns(2)
    with ca:
        st.markdown("**Sebelum Denormalisasi**")
        st.dataframe(pd.DataFrame({
            "Periode":range(1,STEPS+1),"Tanggal":fut_dates,
            "Trend (norm)":np.round(tf_s,5),"Seasonal (norm)":np.round(sf_s,5),
        }).style.format({"Trend (norm)":"{:.5f}","Seasonal (norm)":"{:.5f}"}),
        use_container_width=True)
    with cb:
        st.markdown("**Setelah Denormalisasi (°C)**")
        df_out=pd.DataFrame({
            "Periode":range(1,STEPS+1),"Tanggal":fut_dates,
            "Trend (°C)":np.round(tf,4),"Seasonal (°C)":np.round(sf,4),
            "SST Pred (°C)":np.round(hf,4),
        })
        st.dataframe(df_out.style
            .format({"Trend (°C)":"{:.4f}","Seasonal (°C)":"{:.4f}","SST Pred (°C)":"{:.4f}"})
            .background_gradient(subset=["SST Pred (°C)"],cmap="Blues"),
            use_container_width=True)
    st.download_button("⬇ Download Future Forecast CSV",
        df_out.to_csv(index=False).encode(),"future_forecast.csv","text/csv")

    narasi(
        f"Proyeksi SST selama **{STEPS} periode ke depan**: "
        f"**{hf.min():.4f}–{hf.max():.4f}°C**. "
        "Akurasi prediksi rekursif cenderung menurun seiring bertambahnya horizon "
        "karena akumulasi error pada setiap langkah."
    )
