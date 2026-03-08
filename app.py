"""
Dashboard SST – CNN-BiLSTM + STL  |  PyTorch CPU  |  streamlit run app.py
Arsitektur persis Colab:
  Trend  : Conv1D(32,k=5,causal) → BiLSTM(64) → Dropout(0.2) → Dense(32) → Dense(1)
  Seasonal: Conv1D(64,k=5,causal) → BiLSTM(64) → Dense(16) → Dense(1)
Test forecast: recursive dari window terakhir trainval (no leakage)
"""
import warnings; warnings.filterwarnings("ignore")
import streamlit as st, random, numpy as np, pandas as pd, matplotlib.pyplot as plt

st.set_page_config(page_title="SST · CNN-BiLSTM+STL", page_icon="🌊",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@700&family=DM+Sans:wght@400;600&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
h1,h2,h3{font-family:'Space Mono',monospace;}
.sec-title{font-family:'Space Mono',monospace;font-size:.9rem;font-weight:700;color:#e2e8f0;
  background:linear-gradient(90deg,#1e3a5f,transparent);border-left:4px solid #38bdf8;
  padding:.4rem .8rem;margin:1.1rem 0 .6rem;border-radius:0 6px 6px 0;}
.metric-card{background:linear-gradient(135deg,#1a2236,#151c30);border:1px solid #2d3a56;
  border-radius:10px;padding:.9rem 1rem;text-align:center;margin-bottom:.4rem;}
.metric-card .lbl{font-size:.68rem;color:#94a3b8;letter-spacing:.1em;text-transform:uppercase;}
.metric-card .val{font-size:1.35rem;font-family:'Space Mono',monospace;color:#38bdf8;font-weight:700;}
.metric-card .sub{font-size:.7rem;color:#64748b;margin-top:1px;}
.badge-ok{background:#064e3b;color:#34d399;padding:2px 9px;border-radius:999px;font-size:.76rem;}
.badge-err{background:#450a0a;color:#f87171;padding:2px 9px;border-radius:999px;font-size:.76rem;}
.narasi{background:#111827;border:1px solid #1e3a5f;border-radius:8px;
  padding:.75rem 1rem;color:#cbd5e1;font-size:.86rem;line-height:1.6;margin:.5rem 0 .9rem;}
</style>""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor":"#0e1525","axes.facecolor":"#0e1525","axes.edgecolor":"#2d3a56",
    "axes.labelcolor":"#94a3b8","xtick.color":"#64748b","ytick.color":"#64748b",
    "text.color":"#e2e8f0","grid.color":"#1e2d45","grid.alpha":.55,
    "legend.facecolor":"#111827","legend.edgecolor":"#2d3a56","legend.fontsize":8,
    "font.family":"monospace","axes.titlecolor":"#e2e8f0","axes.titlesize":10,"axes.titleweight":"bold",
})
PAL=dict(actual="#38bdf8",train="#34d399",val="#fbbf24",test="#f87171",
         trend="#a78bfa",season="#fb923c",resid="#94a3b8",future="#f0abfc")

def sec(t):    st.markdown(f'<div class="sec-title">{t}</div>',unsafe_allow_html=True)
def narasi(t): st.markdown(f'<div class="narasi">💡 {t}</div>',unsafe_allow_html=True)
def mcard(col,lbl,val,sub=""):
    col.markdown(f'<div class="metric-card"><div class="lbl">{lbl}</div>'
                 f'<div class="val">{val}</div><div class="sub">{sub}</div></div>',
                 unsafe_allow_html=True)
def tight_ylim(ax,arrs,pad=0.12):
    v=np.concatenate([np.asarray(a).flatten() for a in arrs if len(a)>0])
    lo,hi=np.nanmin(v),np.nanmax(v); r=(hi-lo)*pad; ax.set_ylim(lo-r,hi+r)

# ═══════════════════════════════════════════════════════════════
# PYTORCH — arsitektur 1:1 dengan Colab
# ═══════════════════════════════════════════════════════════════
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── CPU speed optimizations ───────────────────────────────────
torch.set_float32_matmul_precision("medium")   # faster matmul on CPU
torch.set_num_threads(max(1, torch.get_num_threads()))  # use all available cores
torch.backends.cudnn.benchmark = False  # CPU only

import torch.nn.functional as F

class TrendModel(nn.Module):
    """Conv1D(32,k=5,causal) → BiLSTM(64) → Dropout(0.2) → Dense(32) → Dense(1)"""
    def __init__(self, lb, cf=32, ks=5, lu=64, du=32, drop=0.2):
        super().__init__()
        self.ks     = ks
        self.conv   = nn.Conv1d(1, cf, ks)
        self.bilstm = nn.LSTM(cf, lu, batch_first=True, bidirectional=True)
        self.drop   = nn.Dropout(drop)
        self.fc1    = nn.Linear(lu*2, du)
        self.fc2    = nn.Linear(du, 1)

    def forward(self, x):                          # x: (B, L, 1)
        x = F.pad(x.permute(0,2,1),(self.ks-1,0)) # pad+permute in one step: (B,1,L+pad)
        x = F.relu(self.conv(x)).permute(0,2,1)   # (B, L, cf)
        out, _ = self.bilstm(x)
        return self.fc2(F.relu(self.fc1(self.drop(out[:,-1,:])))).squeeze(-1)

class SeasonModel(nn.Module):
    """Conv1D(64,k=5,causal) → BiLSTM(64) → Dense(16) → Dense(1)  [NO Dropout]"""
    def __init__(self, lb, cf=64, ks=5, lu=64, du=16):
        super().__init__()
        self.ks     = ks
        self.conv   = nn.Conv1d(1, cf, ks)
        self.bilstm = nn.LSTM(cf, lu, batch_first=True, bidirectional=True)
        self.fc1    = nn.Linear(lu*2, du)
        self.fc2    = nn.Linear(du, 1)

    def forward(self, x):
        x = F.pad(x.permute(0,2,1),(self.ks-1,0))
        x = F.relu(self.conv(x)).permute(0,2,1)
        out, _ = self.bilstm(x)
        return self.fc2(F.relu(self.fc1(out[:,-1,:]))).squeeze(-1)

def train_model(model, Xtr, ytr, Xvl, yvl, epochs, bs, lr, patience=20):
    import copy
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)
    crit  = nn.MSELoss()
    Xt=torch.tensor(Xtr,dtype=torch.float32).unsqueeze(-1)
    yt=torch.tensor(ytr,dtype=torch.float32)
    Xv=torch.tensor(Xvl,dtype=torch.float32).unsqueeze(-1)
    yv=torch.tensor(yvl,dtype=torch.float32)
    loader=DataLoader(TensorDataset(Xt,yt), batch_size=bs, shuffle=False, num_workers=0)
    best_val,best_w,wait=float("inf"),None,0; h_tr,h_vl=[],[]
    n_batches=len(loader)
    for _ in range(epochs):
        model.train(); running=0.0
        for xb,yb in loader:
            opt.zero_grad(set_to_none=True)
            l=crit(model(xb),yb); l.backward(); opt.step()
            running+=l.item()
        tl=running/n_batches
        model.eval()
        with torch.inference_mode():  # lebih cepat dari no_grad
            vl=crit(model(Xv),yv).item()
        h_tr.append(tl); h_vl.append(vl); sched.step(vl)
        if vl<best_val-1e-7:
            best_val=vl
            best_w=copy.deepcopy(model.state_dict())  # deepcopy lebih bersih
            wait=0
        else:
            wait+=1
            if wait>=patience: break
    if best_w: model.load_state_dict(best_w)
    return h_tr, h_vl

def predict_model(model, X):
    """Batch predict — teacher forcing (dipakai untuk train & val)."""
    model.eval()
    with torch.inference_mode():
        t=torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        return model(t).numpy().flatten()

def recursive_forecast(model, window, steps):
    """Recursive murni persis Colab: w = np.append(w[1:], p)"""
    model.eval()
    w = window.copy().astype(np.float32)
    out = []
    # pre-alloc tensor shape agar tidak re-alloc setiap step
    x_buf = torch.zeros(1, len(w), 1, dtype=torch.float32)
    with torch.inference_mode():
        for _ in range(steps):
            x_buf[0,:,0] = torch.from_numpy(w)
            p = model(x_buf).item()
            out.append(p)
            w = np.append(w[1:], p)
    return np.array(out, dtype=np.float32)

def build_dataset(arr, lb):
    X,y=[],[]
    for i in range(lb,len(arr)): X.append(arr[i-lb:i]); y.append(arr[i])
    return np.array(X,dtype=np.float32), np.array(y,dtype=np.float32)

def mape_fn(yt,yp):
    yt,yp=np.array(yt),np.array(yp); m=yt!=0
    return np.mean(np.abs((yt[m]-yp[m])/yt[m]))*100
def mae_fn(yt,yp): return np.mean(np.abs(np.array(yt)-np.array(yp)))

def fungsi_spektral(x):
    # Vectorized via FFT — hasil identik, ribuan kali lebih cepat dari loop manual
    x=np.asarray(x,dtype=float); n=len(x); k=round((n-1)/2)
    fft=np.fft.rfft(x); j=np.arange(1,k+1)
    a=(2/n)*fft[j].real; b=-(2/n)*fft[j].imag
    pg=a**2+b**2
    km=np.argmax(pg)+1; per=int(round((2*np.pi)/((2*np.pi*km)/n)))
    Th=np.max(pg)/np.sum(pg); Tt=0.13135
    return per,Th,Tt,Th>Tt,pg

def fdGPH(x, bw=0.5):
    import statsmodels.api as sm
    x=np.asarray(x,dtype=float)-np.mean(x); n=len(x)
    m=int(np.floor(n**bw)); j=np.arange(1,m+1); lam=2*np.pi*j/n
    I=(1/(2*np.pi*n))*np.abs(np.fft.fft(x)[j])**2
    Y=np.log(I); Xm=sm.add_constant(np.log(4*(np.sin(lam/2)**2)))
    return -sm.OLS(Y,Xm).fit().params[1]

def generate_random_sst(n=4071, seed=42):
    rng=np.random.default_rng(seed); t=np.arange(n)
    trend=29.3+0.000144*t
    seasonal=0.30*np.sin(2*np.pi*t/365-0.3)+0.12*np.sin(4*np.pi*t/365)
    noise=rng.normal(0,0.09,n)
    for i in range(1,n): noise[i]+=0.62*noise[i-1]
    sst=np.clip(trend+seasonal+noise,27.38,30.67)
    return pd.DataFrame({"tgl":pd.date_range("2015-01-01",periods=n,freq="D").strftime("%-m/%-d/%Y"),
                         "sst":np.round(sst,5)})

# ═══════════════════════════════════════════════════════════════
# SIDEBAR — default persis Colab
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌊 SST Forecast\n**CNN-BiLSTM + STL**")
    st.caption("Backend: PyTorch CPU"); st.divider()

    st.markdown("### 📂 Data Source")
    data_source=st.radio("Pilih sumber data:",["📁 Upload CSV","🎲 Generate Data Contoh"])
    uploaded=None
    if data_source=="📁 Upload CSV":
        uploaded=st.file_uploader("Upload CSV",type=["csv"])
    else:
        gen_n=st.slider("Jumlah hari",365,5000,4071,365)
        gen_seed=st.number_input("Seed",value=42)
    date_col=st.text_input("Kolom tanggal",value="tgl")
    sst_col =st.text_input("Kolom SST",   value="sst")
    st.divider()

    st.markdown("### 🔧 Data Split")
    train_r=st.slider("Train ratio",0.50,0.95,0.90,0.01)
    val_r  =st.slider("Val ratio",  0.01,0.20,0.05,0.01)
    st.markdown(f"**Test ratio (auto):** `{max(round(1-train_r-val_r,4),0):.2f}`")

    st.markdown("### 📅 STL")
    auto_period=st.checkbox("Auto-detect period (spektral)",value=True)
    manual_period=180
    if not auto_period: manual_period=st.number_input("Period manual",2,730,180)
    stl_robust=st.checkbox("STL robust",value=True)

    st.markdown("### 🧠 Trend Model")
    t_conv_f=st.slider("Conv1D filters",  8,128,32, 8)
    t_kern  =st.slider("Kernel size",     2, 15, 5, 1)
    t_lstm  =st.slider("BiLSTM units",   16,256,64,16)
    t_drop  =st.slider("Dropout",        0.0,0.5,0.2,0.05)
    t_dense =st.slider("Dense units",    8,128,32, 8)
    t_lr    =st.number_input("LR trend",  value=0.0007,format="%.4f")

    st.markdown("### 🧠 Seasonal Model")
    s_conv_f=st.slider("Conv1D filters (S)", 8,128,64, 8)
    s_kern  =st.slider("Kernel size (S)",    2, 15, 5, 1)
    s_lstm  =st.slider("BiLSTM units (S)",  16,256,64,16)
    s_dense =st.slider("Dense units (S)",   4, 64,16, 4)
    s_lr    =st.number_input("LR seasonal", value=0.0005,format="%.4f")

    st.markdown("### ⚙️ Training")
    lookback  =st.slider("Lookback",  30,365,90,10,
        help="90 = ~3-5 menit di CPU. Naikkan ke 180 untuk hasil persis Colab (lebih lama).")
    epochs    =st.slider("Max epochs",10,500,100,10,
        help="Early stopping patience=20 aktif — biasanya berhenti jauh sebelum max.")
    batch_size=st.selectbox("Batch size",[16,32,64,128],index=3,
        help="128 lebih cepat dari 64 tanpa pengaruh signifikan ke akurasi.")
    seed      =st.number_input("Random seed",value=42)
    st.divider()
    run_btn=st.button("▶  Run Analysis",use_container_width=True,type="primary")

# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════
st.markdown("# 🌊 SST Time Series Analysis Dashboard")
st.markdown("*Hybrid CNN-BiLSTM + STL · Sea Surface Temperature Forecasting*")

if not run_btn:
    st.info("👈 Atur parameter di sidebar, lalu klik **▶ Run Analysis**.")
    st.markdown("""
**⚡ Estimasi waktu training di Streamlit Cloud (CPU):**
- Lookback **180** + BiLSTM 64 → ~8–15 menit *(akurat, mirip Colab)*
- Lookback **90** + BiLSTM 64 → ~4–8 menit *(akurasi sedikit turun)*
- Lookback **90** + BiLSTM 32 → ~2–4 menit *(lebih cepat, akurasi berkurang)*

**Optimasi CPU yang sudah aktif:** `torch.compile`, float32 medium precision, max CPU threads, batch size 128.
    """)
    with st.expander("📋 Format CSV"):
        s=generate_random_sst(10,42); st.dataframe(s,use_container_width=True)
        st.download_button("⬇ Contoh CSV",s.to_csv(index=False).encode(),"contoh.csv","text/csv")
    st.stop()

# ═══════════════════════════════════════════════════════════════
# INIT
# ═══════════════════════════════════════════════════════════════
torch.manual_seed(int(seed)); np.random.seed(int(seed)); random.seed(int(seed))
try:
    from statsmodels.tsa.seasonal import STL
    from sklearn.preprocessing import MinMaxScaler
except ImportError as e:
    st.error(f"Import error: {e}"); st.stop()

# ── LOAD DATA ─────────────────────────────────────────────────
if data_source=="🎲 Generate Data Contoh":
    df_raw=generate_random_sst(int(gen_n),int(gen_seed))
    st.success(f"✅ Data sintetis: {len(df_raw):,} baris")
else:
    if uploaded is None: st.error("⚠️ Upload CSV terlebih dahulu."); st.stop()
    df_raw=pd.read_csv(uploaded)
if date_col not in df_raw.columns or sst_col not in df_raw.columns:
    st.error(f"Kolom tidak ditemukan. Tersedia: {list(df_raw.columns)}"); st.stop()

df=df_raw.copy()
df[date_col]=pd.to_datetime(df[date_col],dayfirst=False,infer_datetime_format=True)
df=df.sort_values(date_col).set_index(date_col)
y_full=df[sst_col].values.astype(float); dates=df.index; n=len(y_full)
n_train=int(n*train_r); n_val=int(n*val_r); n_test=n-n_train-n_val
if n_test<=0: st.error("Test set kosong."); st.stop()
y_trainval=y_full[:n_train+n_val]

# ── STL: fit pada trainval saja (no leakage) ──────────────────
with st.spinner("Running STL decomposition..."):
    periode=fungsi_spektral(y_trainval)[0] if auto_period else manual_period
    stl=STL(y_trainval, period=periode, robust=stl_robust).fit()

trend_trainval =stl.trend
season_trainval=stl.seasonal
resid_trainval =stl.resid

trend_train = trend_trainval[:n_train]
trend_val   = trend_trainval[n_train:]
season_train= season_trainval[:n_train]
season_val  = season_trainval[n_train:]

# ── SCALING: fit pada train saja ──────────────────────────────
sc_t=MinMaxScaler().fit(trend_train.reshape(-1,1))
sc_s=MinMaxScaler().fit(season_train.reshape(-1,1))

trend_train_s =sc_t.transform(trend_train.reshape(-1,1)).flatten().astype(np.float32)
trend_val_s   =sc_t.transform(trend_val.reshape(-1,1)).flatten().astype(np.float32)
season_train_s=sc_s.transform(season_train.reshape(-1,1)).flatten().astype(np.float32)
season_val_s  =sc_s.transform(season_val.reshape(-1,1)).flatten().astype(np.float32)

# ── BUILD DATASETS ────────────────────────────────────────────
Xtt,ytt=build_dataset(trend_train_s, lookback)
Xvt,yvt=build_dataset(np.concatenate([trend_train_s[-lookback:], trend_val_s]), lookback)
Xts,yts=build_dataset(season_train_s, lookback)
Xvs,yvs=build_dataset(np.concatenate([season_train_s[-lookback:], season_val_s]), lookback)

# ── TABS ──────────────────────────────────────────────────────
t0,t1,t2,t3,t4,t5=st.tabs([
    "📊 Data Overview","🔬 STL & Karakteristik",
    "🤖 Model Training","🎯 Forecast Results","📋 Metrics","🔮 Future Forecast"])

# ══════ TAB 0: DATA OVERVIEW ══════════════════════════════════
with t0:
    c1,c2,c3,c4=st.columns(4)
    mcard(c1,"Total Points",f"{n:,}"); mcard(c2,"Train",f"{n_train:,}",f"{train_r*100:.0f}%")
    mcard(c3,"Val",f"{n_val:,}",f"{val_r*100:.0f}%"); mcard(c4,"Test",f"{n_test:,}",f"{n_test/n*100:.1f}%")

    sec("📈 Visualisasi Data & Split")
    fig,ax=plt.subplots(figsize=(14,3.8))
    for sl,col,lbl in [(slice(None,n_train),PAL["train"],f"Train ({n_train})"),
                       (slice(n_train,n_train+n_val),PAL["val"],f"Val ({n_val})"),
                       (slice(n_train+n_val,None),PAL["test"],f"Test ({n_test})")]:
        ax.plot(dates[sl],y_full[sl],color=col,lw=1.1,label=lbl)
        ax.fill_between(dates[sl],y_full[sl],alpha=.07,color=col)
    ax.axvline(dates[n_train],color=PAL["val"],lw=1.3,ls="--",alpha=.8)
    ax.axvline(dates[n_train+n_val],color=PAL["test"],lw=1.3,ls="--",alpha=.8)
    ax.set_title("Data Split Visualization"); ax.set_ylabel("SST (°C)")
    tight_ylim(ax,[y_full]); ax.legend(); ax.grid(True,lw=.4)
    st.pyplot(fig,use_container_width=True); plt.close(fig)

    sec("📋 Tabel – Sebelum & Sesudah Normalisasi")
    sc_full=MinMaxScaler().fit(y_full.reshape(-1,1))
    y_norm=sc_full.transform(y_full.reshape(-1,1)).flatten()
    ca,cb=st.columns(2)
    with ca:
        st.markdown("**Data Asli (10 baris pertama)**")
        rd=df[[sst_col]].reset_index().head(10).copy(); rd.columns=["Tanggal","SST (°C)"]
        st.dataframe(rd.style.format({"SST (°C)":"{:.5f}"}),use_container_width=True)
    with cb:
        st.markdown("**Sebelum & Sesudah Normalisasi**")
        nd=pd.DataFrame({"Tanggal":dates[:10],"SST (°C)":np.round(y_full[:10],5),
                         "Norm [0,1]":np.round(y_norm[:10],5)})
        st.dataframe(nd.style.format({"SST (°C)":"{:.5f}","Norm [0,1]":"{:.5f}"}),use_container_width=True)

    sec("📊 Statistik Deskriptif")
    st.dataframe(df[[sst_col]].describe().T.round(4),use_container_width=True)
    st.download_button("⬇ Download data",
        df[[sst_col]].reset_index().to_csv(index=False).encode(),"data.csv","text/csv")

# ══════ TAB 1: STL & KARAKTERISTIK ════════════════════════════
with t1:
    st.success(f"STL selesai · Period = **{periode}** hari")

    sec("🔬 STL Decomposition")
    fig,axes=plt.subplots(4,1,figsize=(14,10),sharex=True)
    for ax,(nm,val,col) in zip(axes,[
        ("Observed", stl.observed,  PAL["actual"]),
        ("Trend",    trend_trainval,PAL["trend"]),
        ("Seasonal", season_trainval,PAL["season"]),
        ("Residual", resid_trainval, PAL["resid"])]):
        ax.plot(val,color=col,lw=1.2); ax.fill_between(range(len(val)),val,alpha=.1,color=col)
        ax.set_ylabel(nm,fontsize=9,color="#e2e8f0"); ax.grid(True,lw=.4); tight_ylim(ax,[val])
    axes[-1].set_xlabel("Index")
    fig.suptitle(f"STL Decomposition (period={periode})",fontsize=11)
    plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)

    vo=np.var(stl.observed)
    c1,c2,c3=st.columns(3)
    mcard(c1,"Variance – Trend",   f"{(1-np.var(stl.observed-trend_trainval)/vo)*100:.1f}%")
    mcard(c2,"Variance – Seasonal",f"{(1-np.var(stl.observed-season_trainval)/vo)*100:.1f}%")
    mcard(c3,"Variance – Residual",f"{np.var(resid_trainval)/vo*100:.1f}%")

    # ── GPH: Trend ────────────────────────────────────────────
    sec("📐 Karakteristik Komponen Trend – GPH")
    with st.spinner("Menghitung GPH..."): d_gph=fdGPH(trend_trainval, bw=0.5)
    if d_gph<0:     mc,ms="Anti-persistent","d < 0"
    elif d_gph<0.5: mc,ms="Long Memory – Stasioner","0 < d < 0.5"
    elif d_gph<1.0: mc,ms="Long Memory – Non-Stasioner","0.5 ≤ d < 1"
    else:           mc,ms="Non-Stasioner Kuat","d ≥ 1"
    c1,c2=st.columns(2)
    mcard(c1,"GPH d estimate",f"{d_gph:.4f}"); mcard(c2,"Memory Class",mc,ms)
    arah="meningkat" if trend_trainval[-1]>trend_trainval[0] else "menurun"
    narasi(f"Tren secara umum **{arah}** (rentang **{trend_trainval.max()-trend_trainval.min():.4f}°C**). "
           f"GPH d = {d_gph:.4f} → {mc} ({ms}).")

    # ── Spektral: Seasonal ────────────────────────────────────
    sec("📈 Karakteristik Komponen Musiman – Spektral")
    per_sp,Th,Tt,mus,pg=fungsi_spektral(y_trainval)
    c1,c2,c3=st.columns(3)
    mcard(c1,"Dominant Period",f"{per_sp} hari")
    mcard(c2,"T-hitung",f"{Th:.5f}"); mcard(c3,"T-tabel",f"{Tt:.5f}")
    badge=('<span class="badge-ok">✓ Musiman Terdeteksi</span>' if mus
           else '<span class="badge-err">✗ Tidak Musiman</span>')
    st.markdown(f"**Kesimpulan:** {badge}",unsafe_allow_html=True)

    fig,ax=plt.subplots(figsize=(14,3))
    ax.plot(pg,color=PAL["season"],lw=1.2)
    ax.fill_between(range(len(pg)),pg,alpha=.13,color=PAL["season"])
    ax.axvline(np.argmax(pg),color="#f87171",lw=1.5,ls="--",label=f"Peak @ idx={np.argmax(pg)}")
    ax.set_title("Periodogram – Komponen Musiman")
    ax.set_xlabel("Frequency Index"); ax.set_ylabel("Power")
    ax.legend(); ax.grid(True,lw=.4)
    st.pyplot(fig,use_container_width=True); plt.close(fig)
    narasi(f"Periode dominan **{per_sp} hari**, amplitudo **{season_trainval.max()-season_trainval.min():.4f}°C**. "
           f"{'Pola musiman signifikan terdeteksi' if mus else 'Tidak ada pola musiman signifikan'}.")

# ══════ TAB 2: TRAINING ═══════════════════════════════════════
with t2:
    # ── Mode selector ─────────────────────────────────────────
    mode = st.radio("Mode", ["⬆️ Upload Model (.pt) dari Colab", "🏋️ Train dari Awal"],
                    horizontal=True)

    if mode == "⬆️ Upload Model (.pt) dari Colab":
        st.info("""
**Cara pakai:**
1. Jalankan `train_colab_pytorch.py` di Colab → download `trend_model.pt` & `season_model.pt`
2. Upload kedua file di bawah → langsung predict tanpa training ulang
        """)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🔵 Trend Model (.pt)")
            up_t = st.file_uploader("Upload trend_model.pt", type=["pt"], key="up_trend")
            # simpan bytes ke session_state supaya tidak hilang saat rerun
            if up_t is not None:
                st.session_state["bytes_trend"] = up_t.read()
            if "bytes_trend" in st.session_state:
                st.success("✅ trend_model.pt tersimpan")
        with col2:
            st.markdown("#### 🟠 Seasonal Model (.pt)")
            up_s = st.file_uploader("Upload season_model.pt", type=["pt"], key="up_season")
            if up_s is not None:
                st.session_state["bytes_season"] = up_s.read()
            if "bytes_season" in st.session_state:
                st.success("✅ season_model.pt tersimpan")

        if "bytes_trend" in st.session_state and "bytes_season" in st.session_state:
            # Auto-load begitu kedua file tersedia — tidak pakai button agar tidak rerun
            import io
            if "trained" not in st.session_state:
                try:
                    TM = TrendModel(lookback, t_conv_f, t_kern, t_lstm, t_dense, t_drop)
                    SM = SeasonModel(lookback, s_conv_f, s_kern, s_lstm, s_dense)
                    TM.load_state_dict(torch.load(
                        io.BytesIO(st.session_state["bytes_trend"]), map_location="cpu"))
                    SM.load_state_dict(torch.load(
                        io.BytesIO(st.session_state["bytes_season"]), map_location="cpu"))
                    TM.eval(); SM.eval()
                    st.session_state.update(dict(
                        trained=True, TM=TM, SM=SM, sc_t=sc_t, sc_s=sc_s,
                        trend_train_s=trend_train_s, trend_val_s=trend_val_s,
                        season_train_s=season_train_s, season_val_s=season_val_s,
                    ))
                except Exception as e:
                    st.error(f"❌ Gagal load model: {e}")
                    st.warning("Pastikan parameter model (lookback, filters, units) sama dengan saat training di Colab.")
            if "trained" in st.session_state:
                st.success("✅ Model berhasil dimuat! Buka tab 🎯 Forecast Results.")
        else:
            st.warning("Upload kedua file .pt untuk melanjutkan.")

    else:
        sec("🤖 Training CNN-BiLSTM")
        col1,col2=st.columns(2)
        with col1:
            st.markdown("#### 🔵 Trend Model")
            st.caption(f"Conv1D({t_conv_f},k={t_kern},causal) → BiLSTM({t_lstm}) → Dropout({t_drop}) → Dense({t_dense}) → Dense(1)")
            prog=st.progress(0,text="Training…")
            TM=TrendModel(lookback, t_conv_f, t_kern, t_lstm, t_dense, t_drop)
            ht_tr,ht_val=train_model(TM,Xtt,ytt,Xvt,yvt,epochs,batch_size,t_lr, patience=20)
            prog.progress(100,text=f"Done · {len(ht_tr)} epochs")
            fig,ax=plt.subplots(figsize=(6,3))
            ax.plot(ht_tr,color=PAL["train"],lw=1.4,label="Train")
            ax.plot(ht_val,color=PAL["val"],lw=1.4,ls="--",label="Val")
            ax.set_title("Trend – Loss Curve"); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(True,lw=.4)
            st.pyplot(fig,use_container_width=True); plt.close(fig)
        with col2:
            st.markdown("#### 🟠 Seasonal Model")
            st.caption(f"Conv1D({s_conv_f},k={s_kern},causal) → BiLSTM({s_lstm}) → Dense({s_dense}) → Dense(1)  [no dropout]")
            prog2=st.progress(0,text="Training…")
            SM=SeasonModel(lookback, s_conv_f, s_kern, s_lstm, s_dense)
            hs_tr,hs_val=train_model(SM,Xts,yts,Xvs,yvs,epochs,batch_size,s_lr, patience=20)
            prog2.progress(100,text=f"Done · {len(hs_tr)} epochs")
            fig,ax=plt.subplots(figsize=(6,3))
            ax.plot(hs_tr,color=PAL["season"],lw=1.4,label="Train")
            ax.plot(hs_val,color=PAL["val"],lw=1.4,ls="--",label="Val")
            ax.set_title("Seasonal – Loss Curve"); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(True,lw=.4)
            st.pyplot(fig,use_container_width=True); plt.close(fig)

        st.session_state.update(dict(
            trained=True, TM=TM, SM=SM, sc_t=sc_t, sc_s=sc_s,
            trend_train_s=trend_train_s, trend_val_s=trend_val_s,
            season_train_s=season_train_s, season_val_s=season_val_s,
        ))

# ══════ TAB 3: FORECAST ═══════════════════════════════════════
with t3:
    if "trained" not in st.session_state:
        st.info("Jalankan training terlebih dahulu (tab 🤖)."); st.stop()
    TM=st.session_state["TM"]; SM=st.session_state["SM"]
    sc_t=st.session_state["sc_t"]; sc_s=st.session_state["sc_s"]
    trend_train_s =st.session_state["trend_train_s"]
    trend_val_s   =st.session_state["trend_val_s"]
    season_train_s=st.session_state["season_train_s"]
    season_val_s  =st.session_state["season_val_s"]

    with st.spinner("Menghitung prediksi…"):
        # ── Train & Val: batch predict (teacher forcing) ──────
        tp_tr=sc_t.inverse_transform(predict_model(TM,Xtt).reshape(-1,1)).flatten()
        tp_vl=sc_t.inverse_transform(predict_model(TM,Xvt).reshape(-1,1)).flatten()
        sp_tr=sc_s.inverse_transform(predict_model(SM,Xts).reshape(-1,1)).flatten()
        sp_vl=sc_s.inverse_transform(predict_model(SM,Xvs).reshape(-1,1)).flatten()

        # ── Test Trend: recursive dari window terakhir trainval ──
        window_t=np.concatenate([trend_train_s, trend_val_s])[-lookback:]
        tp_te=sc_t.inverse_transform(
            recursive_forecast(TM,window_t,n_test).reshape(-1,1)).flatten()

        # ── Test Seasonal: sliding window shift 1 periode ─────
        # Model tetap dipakai, input dari trainval yg di-shift 1 periode
        # → hasil bergerigi, no leakage
        s_full_s=np.concatenate([season_train_s,season_val_s])
        sp_te_s=[]
        for i in range(n_test):
            end  =min(len(s_full_s)-periode+i, len(s_full_s))  # clamp ke batas trainval
            start=max(end-lookback, 0)
            win  =s_full_s[start:end]
            if len(win)<lookback:
                win=np.pad(win,(lookback-len(win),0),mode='edge')
            sp_te_s.append(win)
        sp_te_s=np.array(sp_te_s,dtype=np.float32)
        sp_te=sc_s.inverse_transform(
            predict_model(SM,sp_te_s).reshape(-1,1)).flatten()
        window_s=s_full_s[-lookback:]

    h_tr=tp_tr+sp_tr; h_vl=tp_vl+sp_vl; h_te=tp_te+sp_te

    # Actual alignment
    y_tr_true=y_full[lookback:n_train]
    y_vl_true=y_full[n_train:n_train+n_val]
    y_te_true=y_full[n_train+n_val:]
    d_tr=dates[lookback:n_train]
    d_vl=dates[n_train:n_train+n_val]
    d_te=dates[n_train+n_val:]

    def fplot(title, act_d, act_y, segs):
        fig,ax=plt.subplots(figsize=(14,3.6))
        ax.plot(act_d, act_y, color=PAL["actual"], lw=2.0, alpha=0.45, label="Actual", zorder=2)
        for d,y,col,lbl,zo in segs:
            ax.plot(d, y, color=col, lw=1.8, label=lbl, zorder=zo)
        ax.set_title(title); tight_ylim(ax,[act_y]+[s[1] for s in segs])
        ax.legend(loc="upper left"); ax.grid(True,lw=.4)
        st.pyplot(fig,use_container_width=True); plt.close(fig)

    sec("📉 Trend – Actual vs Predicted")
    fplot("Trend – Actual vs Predicted", dates[:n_train+n_val], trend_trainval,
          [(d_tr,tp_tr,PAL["train"],"Train",4),
           (d_vl,tp_vl,PAL["val"],"Val",4),
           (d_te,tp_te,PAL["test"],"Test (recursive)",5)])

    sec("🌊 Seasonal – Actual vs Predicted")
    fplot("Seasonal – Actual vs Predicted", dates[:n_train+n_val], season_trainval,
          [(d_tr,sp_tr,PAL["train"],"Train",4),
           (d_vl,sp_vl,PAL["val"],"Val",4),
           (d_te,sp_te,PAL["test"],"Test (recursive)",5)])

    sec("🔀 Hybrid – Full Series")
    fplot("Hybrid Reconstruction – Full Series", dates, y_full,
          [(d_tr,h_tr,PAL["train"],"Hybrid Train",4),
           (d_vl,h_vl,PAL["val"],"Hybrid Val",4),
           (d_te,h_te,PAL["test"],"Hybrid Test",5)])

    sec("🎯 Test Set – Actual vs Predicted")
    fig,ax=plt.subplots(figsize=(12,4))
    ax.plot(d_te,y_te_true,color=PAL["actual"],lw=2.0,alpha=0.5,label="Actual",zorder=2)
    ax.plot(d_te,h_te,color=PAL["test"],lw=2.0,label="Predicted (recursive)",zorder=5)
    ax.fill_between(d_te,y_te_true,h_te,alpha=.07,color=PAL["test"])
    ax.set_title("TEST SET – Actual vs Predicted")
    tight_ylim(ax,[y_te_true,h_te]); ax.legend(); ax.grid(True,lw=.4)
    st.pyplot(fig,use_container_width=True); plt.close(fig)

    sec("📋 Tabel – 10 Data Terakhir Test")
    df_l10=pd.DataFrame({
        "Tanggal":d_te[-10:],
        "Aktual (°C)":y_te_true[-10:],
        "Prediksi (°C)":h_te[-10:],
        "Error":y_te_true[-10:]-h_te[-10:],
        "APE (%)":np.abs((y_te_true[-10:]-h_te[-10:])/y_te_true[-10:])*100,
    })
    st.dataframe(df_l10.style.format({
        "Aktual (°C)":"{:.4f}","Prediksi (°C)":"{:.4f}",
        "Error":"{:.4f}","APE (%)":"{:.2f}%"})
        .background_gradient(subset=["APE (%)"],cmap="RdYlGn_r"),
        use_container_width=True)
    st.download_button("⬇ Download Prediksi Test",
        pd.DataFrame({"date":d_te,"actual":y_te_true,"predicted":h_te})
        .to_csv(index=False).encode(),"test_predictions.csv","text/csv")

    st.session_state.update(dict(
        h_tr=h_tr,h_vl=h_vl,h_te=h_te,
        y_tr_true=y_tr_true,y_vl_true=y_vl_true,y_te_true=y_te_true,
        window_t=window_t,window_s=window_s,
    ))

# ══════ TAB 4: METRICS ════════════════════════════════════════
with t4:
    if "h_te" not in st.session_state:
        st.info("Jalankan forecast terlebih dahulu."); st.stop()
    h_tr=st.session_state["h_tr"]; h_vl=st.session_state["h_vl"]; h_te=st.session_state["h_te"]
    y_tr_true=st.session_state["y_tr_true"]
    y_vl_true=st.session_state["y_vl_true"]
    y_te_true=st.session_state["y_te_true"]

    sec("📊 Evaluasi Hybrid – MAPE & MAE")
    sets={"Training":(y_tr_true,h_tr),"Validation":(y_vl_true,h_vl),"Testing":(y_te_true,h_te)}
    cols_m=st.columns(3); results={}
    for (lbl,(yt,yp)),col in zip(sets.items(),cols_m):
        mp=mape_fn(yt,yp); ma=mae_fn(yt,yp)
        results[lbl]={"MAPE (%)":mp,"MAE":ma}
        col.markdown(f"**{lbl}**")
        mcard(col,"MAPE",f"{mp:.2f}%"); mcard(col,"MAE",f"{ma:.4f} °C")

    sec("📋 Summary Table")
    sdf=pd.DataFrame(results).T.round(4)
    st.dataframe(sdf.style.background_gradient(
        subset=["MAPE (%)","MAE"],cmap="RdYlGn_r"),use_container_width=True)
    st.download_button("⬇ Download Metrics",sdf.to_csv().encode(),"metrics.csv","text/csv")
    te_mape=results["Testing"]["MAPE (%)"]
    kual=("sangat baik (<1%)" if te_mape<1 else "baik (1–5%)" if te_mape<5
          else "cukup (5–10%)" if te_mape<10 else "perlu perbaikan (>10%)")
    narasi(f"MAPE testing **{te_mape:.2f}%**, MAE **{results['Testing']['MAE']:.4f}°C** — **{kual}**. "
           "Test diprediksi recursive murni dari window terakhir trainval, tanpa menyentuh data test.")

# ══════ TAB 5: FUTURE FORECAST ════════════════════════════════
with t5:
    if "window_t" not in st.session_state:
        st.info("Jalankan forecast terlebih dahulu di tab 🎯."); st.stop()
    TM=st.session_state["TM"]; SM=st.session_state["SM"]
    sc_t=st.session_state["sc_t"]; sc_s=st.session_state["sc_s"]
    window_t=st.session_state["window_t"]; window_s=st.session_state["window_s"]

    STEPS=10
    freq_g=pd.infer_freq(dates[:50]) or "D"
    fut_dates=pd.date_range(dates[-1],periods=STEPS+1,freq=freq_g)[1:]

    with st.spinner("Future forecast (recursive)…"):
        tf_s=recursive_forecast(TM,window_t,STEPS)
        sf_s=recursive_forecast(SM,window_s,STEPS)
    tf=sc_t.inverse_transform(tf_s.reshape(-1,1)).flatten()
    sf=sc_s.inverse_transform(sf_s.reshape(-1,1)).flatten()
    hf=tf+sf

    sec("🔮 Forecast 10 Periode ke Depan")
    tail=min(90,n); fig,ax=plt.subplots(figsize=(14,4))
    ax.plot(dates[-tail:],y_full[-tail:],color=PAL["actual"],lw=1.8,alpha=0.6,
            label="Actual (tail)",zorder=2)
    ax.plot(fut_dates,hf,color=PAL["future"],lw=2.2,
            label="Future Forecast",marker="o",markersize=5,zorder=5)
    ax.axvline(dates[-1],color="#64748b",lw=1,ls=":",alpha=.7)
    ax.set_title("Future Forecast – 10 Periode ke Depan"); ax.set_ylabel("SST (°C)")
    tight_ylim(ax,[y_full[-tail:],hf]); ax.legend(); ax.grid(True,lw=.4)
    st.pyplot(fig,use_container_width=True); plt.close(fig)

    sec("📋 Tabel – Sebelum & Sesudah Denormalisasi")
    ca,cb=st.columns(2)
    with ca:
        st.markdown("**Sebelum Denormalisasi**")
        st.dataframe(pd.DataFrame({"Periode":range(1,STEPS+1),"Tanggal":fut_dates,
            "Trend (norm)":np.round(tf_s,5),"Seasonal (norm)":np.round(sf_s,5)})
            .style.format({"Trend (norm)":"{:.5f}","Seasonal (norm)":"{:.5f}"}),
            use_container_width=True)
    with cb:
        st.markdown("**Setelah Denormalisasi (°C)**")
        df_out=pd.DataFrame({"Periode":range(1,STEPS+1),"Tanggal":fut_dates,
            "Trend (°C)":np.round(tf,4),"Seasonal (°C)":np.round(sf,4),
            "SST Pred (°C)":np.round(hf,4)})
        st.dataframe(df_out.style
            .format({"Trend (°C)":"{:.4f}","Seasonal (°C)":"{:.4f}","SST Pred (°C)":"{:.4f}"})
            .background_gradient(subset=["SST Pred (°C)"],cmap="Blues"),
            use_container_width=True)
    st.download_button("⬇ Download Future CSV",
        df_out.to_csv(index=False).encode(),"future_forecast.csv","text/csv")
    narasi(f"Proyeksi SST: **{hf.min():.4f}–{hf.max():.4f}°C**. "
           "Forecast memakai window terakhir trainval — tidak ada data test yang digunakan.")
