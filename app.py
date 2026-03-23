"""Dashboard SST – CNN-BiLSTM + STL | PyTorch | streamlit run app.py"""
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
.alert-normal{background:#052e16;border:1px solid #166534;border-radius:8px;
  padding:.75rem 1rem;color:#86efac;font-size:.86rem;line-height:1.6;margin:.5rem 0 .9rem;}
.alert-warn{background:#422006;border:1px solid #92400e;border-radius:8px;
  padding:.75rem 1rem;color:#fde68a;font-size:.86rem;line-height:1.6;margin:.5rem 0 .9rem;}
.alert-crit{background:#450a0a;border:1px solid #991b1b;border-radius:8px;
  padding:.75rem 1rem;color:#fca5a5;font-size:.86rem;line-height:1.6;margin:.5rem 0 .9rem;}
.ref-box{background:#0f172a;border:1px solid #2d3a56;border-radius:8px;
  padding:.6rem 1rem;color:#94a3b8;font-size:.78rem;line-height:1.7;margin:.3rem 0 .8rem;}
</style>""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor":"#0e1525","axes.facecolor":"#0e1525","axes.edgecolor":"#2d3a56",
    "axes.labelcolor":"#94a3b8","xtick.color":"#64748b","ytick.color":"#64748b",
    "text.color":"#e2e8f0","grid.color":"#1e2d45","grid.alpha":.55,
    "legend.facecolor":"#111827","legend.edgecolor":"#2d3a56","legend.fontsize":8,
    "font.family":"monospace","axes.titlecolor":"#e2e8f0","axes.titlesize":10,"axes.titleweight":"bold",
})
PAL=dict(actual="#38bdf8",train="#34d399",val="#fbbf24",test="#f87171",
         trend="#a78bfa",season="#fb923c",resid="#94a3b8",future="#f0abfc",
         normal="#34d399",warn="#fbbf24",crit="#f87171")

def sec(t):    st.markdown(f'<div class="sec-title">{t}</div>',unsafe_allow_html=True)
def narasi(t): st.markdown(f'<div class="narasi">💡 {t}</div>',unsafe_allow_html=True)
def alert_box(level, t):
    css = {"normal":"alert-normal","warn":"alert-warn","crit":"alert-crit"}[level]
    st.markdown(f'<div class="{css}">{t}</div>', unsafe_allow_html=True)
def ref_box(t): st.markdown(f'<div class="ref-box">📚 <b>Referensi:</b> {t}</div>', unsafe_allow_html=True)
def mcard(col,lbl,val,sub=""):
    col.markdown(f'<div class="metric-card"><div class="lbl">{lbl}</div>'
                 f'<div class="val">{val}</div><div class="sub">{sub}</div></div>',
                 unsafe_allow_html=True)
def tight_ylim(ax,arrs,pad=0.12):
    v=np.concatenate([np.asarray(a).flatten() for a in arrs if len(a)>0])
    lo,hi=np.nanmin(v),np.nanmax(v); r=(hi-lo)*pad; ax.set_ylim(lo-r,hi+r)

# ═══════════════════════════════════════════════════════════════
# PYTORCH
# ═══════════════════════════════════════════════════════════════
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.set_float32_matmul_precision("medium")
torch.set_num_threads(max(1, torch.get_num_threads()))

import torch.nn.functional as F

class TrendModel(nn.Module):
    def __init__(self, lb, cf=32, ks=5, lu=64, du=32, drop=0.2):
        super().__init__()
        self.ks     = ks
        self.conv   = nn.Conv1d(1, cf, ks)
        self.bilstm = nn.LSTM(cf, lu, batch_first=True, bidirectional=True)
        self.drop   = nn.Dropout(drop)
        self.fc1    = nn.Linear(lu*2, du)
        self.fc2    = nn.Linear(du, 1)
    def forward(self, x):
        x = F.pad(x.permute(0,2,1),(self.ks-1,0))
        x = F.relu(self.conv(x)).permute(0,2,1)
        out, _ = self.bilstm(x)
        return self.fc2(F.relu(self.fc1(self.drop(out[:,-1,:])))).squeeze(-1)

class SeasonModel(nn.Module):
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
        with torch.inference_mode():
            vl=crit(model(Xv),yv).item()
        h_tr.append(tl); h_vl.append(vl); sched.step(vl)
        if vl<best_val-1e-7:
            best_val=vl; best_w=copy.deepcopy(model.state_dict()); wait=0
        else:
            wait+=1
            if wait>=patience: break
    if best_w: model.load_state_dict(best_w)
    return h_tr, h_vl

def predict_model(model, X):
    model.eval()
    with torch.inference_mode():
        t=torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        return model(t).numpy().flatten()

def recursive_forecast(model, window, steps):
    model.eval()
    w = window.copy().astype(np.float32)
    out = []
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

def _plot_loss(ax, tr, vl, title, color):
    tr_arr, vl_arr = np.array(tr), np.array(vl)
    ep = np.arange(1, len(tr_arr)+1)
    w  = max(5, len(tr_arr)//20)
    ax.plot(ep, tr_arr, color=color,      lw=0.7, alpha=0.2)
    ax.plot(ep, vl_arr, color=PAL["val"], lw=0.7, alpha=0.2)
    if len(tr_arr) >= w:
        ax.plot(ep[w-1:], np.convolve(tr_arr,np.ones(w)/w,'valid'),
                color=color,      lw=2.0, label="Train Loss")
        ax.plot(ep[w-1:], np.convolve(vl_arr,np.ones(w)/w,'valid'),
                color=PAL["val"], lw=2.0, ls="--", label="Val Loss")
    else:
        ax.plot(ep, tr_arr, color=color,      lw=2.0, label="Train Loss")
        ax.plot(ep, vl_arr, color=PAL["val"], lw=2.0, ls="--", label="Val Loss")
    best = int(np.argmin(vl_arr))
    ax.axvline(best+1, color="#64748b", lw=1.2, ls=":", alpha=.8)
    ax.scatter([best+1], [vl_arr[best]], color="gold", s=50, zorder=5)
    ax.annotate(f"best={vl_arr[best]:.5f}\n@ep {best+1}",
                xy=(best+1, vl_arr[best]),
                xytext=(best+1+max(len(vl_arr)//10,3), vl_arr[best]),
                fontsize=7, color="#94a3b8",
                arrowprops=dict(arrowstyle="->", color="#64748b", lw=0.7))
    clip = min(5, len(tr_arr)//10)
    vals = np.concatenate([tr_arr[clip:], vl_arr[clip:]])
    ymin, ymax = np.nanmin(vals), np.nanmax(vals)
    pad = (ymax-ymin)*0.15
    ax.set_ylim(max(0,ymin-pad), ymax+pad)
    ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.legend(loc="upper right"); ax.grid(True, lw=.4)


def fungsi_spektral(x):
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
# FUNGSI INTERPRETASI EKOLOGIS (BARU)
# ═══════════════════════════════════════════════════════════════
def hitung_mmm(y_full, dates):
    """
    Menghitung Maximum Monthly Mean (MMM) dari data historis.
    MMM = nilai tertinggi dari 12 rata-rata bulanan (mengikuti standar NOAA CRW).
    Referensi: NOAA Coral Reef Watch, https://coralreefwatch.noaa.gov/product/5km/methodology.php
    """
    s = pd.Series(y_full, index=dates)
    monthly_mean = s.groupby(s.index.month).mean()
    return monthly_mean.max()

def klasifikasi_status(anomali):
    """
    Mengklasifikasikan status termal berdasarkan anomali SPL terhadap MMM.
    Mengacu pada sistem NOAA Coral Reef Watch HotSpot:
    - Normal   : anomali < 0 (di bawah MMM)
    - Waspada  : 0 <= anomali < 1°C (zona HotSpot ungu NOAA CRW)
    - Kritis   : anomali >= 1°C (ambang bleaching, Glynn & D'Croz 1990)
    Referensi: https://www.aoml.noaa.gov/threats-to-coral/
    """
    if anomali >= 1.0:
        return "🔴 Kritis"
    elif anomali >= 0.0:
        return "🟡 Waspada"
    else:
        return "🟢 Normal"

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌊 SST Forecast\n**CNN-BiLSTM + STL**")
    st.divider()

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
    lookback  =st.slider("Lookback",  30,365,90,10)
    epochs    =st.slider("Max epochs",10,500,100,10)
    batch_size=st.selectbox("Batch size",[16,32,64,128],index=3)
    seed      =st.number_input("Random seed",value=42)
    st.divider()
    run_btn=st.button("▶  Run Analysis",use_container_width=True,type="primary")

# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════
st.markdown("# 🌊 SST Time Series Analysis Dashboard")
st.markdown("*Hybrid CNN-BiLSTM + STL · Sea Surface Temperature Forecasting*")
st.markdown("""
<div style="
    background: linear-gradient(90deg, #1e3a5f, #0e1525);
    border: 1px solid #2d3a56;
    border-left: 4px solid #38bdf8;
    border-radius: 8px;
    padding: 0.55rem 1.1rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
">
    <span style="font-family:'Space Mono',monospace; color:#38bdf8; font-size:.85rem; font-weight:700;">
        🎓 S1 Statistika FMIPA UNPAD
    </span>
    <span style="color:#64748b;">|</span>
    <span style="font-family:'DM Sans',sans-serif; color:#cbd5e1; font-size:.85rem; font-weight:600;">
        👤 Dianda Destin
    </span>
</div>
""", unsafe_allow_html=True)

if not run_btn:
    st.info("👈 Atur parameter di sidebar, lalu klik **▶ Run Analysis**.")
    with st.expander("📋 Format CSV yang diterima"):
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
n_val   = int(n * val_r)
n_test  = int(n * val_r)
n_train = n - n_val - n_test
if n_test<=0: st.error("Test set kosong."); st.stop()
y_trainval=y_full[:n_train+n_val]

# Hitung MMM dari data historis (untuk interpretasi ekologis)
MMM = hitung_mmm(y_full, dates)

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

    sec("📐 Karakteristik Data SST – GPH")
    with st.spinner("Menghitung GPH..."):
        d_gph = fdGPH(y_trainval, bw=0.5)

    if d_gph < 0:    mc, ms = "Anti-persistent",             "d < 0"
    elif d_gph < 0.5:mc, ms = "Long Memory – Stasioner",     "0 < d < 0.5"
    elif d_gph < 1.0:mc, ms = "Long Memory – Non-Stasioner", "0.5 ≤ d < 1"
    else:            mc, ms = "Non-Stasioner Kuat",           "d ≥ 1"

    c1, c2 = st.columns(2)
    mcard(c1, "GPH d estimate", f"{d_gph:.4f}")
    mcard(c2, "Memory Class",   mc, ms)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(y_trainval, color=PAL["actual"], lw=1.3, alpha=.85)
    axes[0].fill_between(range(len(y_trainval)), y_trainval, alpha=.10, color=PAL["actual"])
    axes[0].set_title("Data SST Train+Val\n(input GPH analysis)")
    axes[0].set_ylabel("SST (°C)"); axes[0].set_xlabel("Index")
    axes[0].grid(True, lw=.4)

    bar_col_map = {
        "Anti-persistent":              "#38bdf8",
        "Long Memory – Stasioner":      "#34d399",
        "Long Memory – Non-Stasioner":  "#fbbf24",
        "Non-Stasioner Kuat":           "#f87171",
    }
    bcolor = bar_col_map.get(mc, "#94a3b8")
    axes[1].barh(["GPH d"], [d_gph], color=bcolor, alpha=.85, height=0.4)
    axes[1].axvline(0,    color="#64748b", lw=1.0, ls="--", alpha=.6)
    axes[1].axvline(0.5,  color="#fbbf24", lw=1.2, ls="--", alpha=.8, label="d=0.5 (batas stasioner)")
    axes[1].axvline(1.0,  color="#f87171", lw=1.2, ls="--", alpha=.8, label="d=1.0 (non-stasioner)")
    axes[1].set_xlim(-0.3, max(1.5, d_gph + 0.4))
    axes[1].set_title(f"GPH Estimate: d = {d_gph:.4f}\n{mc}  ({ms})")
    axes[1].set_xlabel("d value"); axes[1].legend(fontsize=8)
    axes[1].grid(True, lw=.4, axis="x")
    axes[1].text(d_gph + 0.04, 0, f"d = {d_gph:.4f}", va="center", fontsize=10, color=bcolor, fontweight="bold")
    fig.suptitle("Karakteristik Data SST – GPH (Geweke-Porter-Hudak)", fontsize=11)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close(fig)

    narasi(f"Analisis GPH pada **data SST Train+Val** ({len(y_trainval):,} titik). "
           f"Nilai d = **{d_gph:.4f}** → **{mc}** ({ms}). "
           f"Rentang SST: **{y_trainval.min():.4f} – {y_trainval.max():.4f}°C**, "
           f"rata-rata **{y_trainval.mean():.4f}°C**.")

    sec("📈 Karakteristik Data SST – Spektral")
    per_sp, Th, Tt, mus, pg = fungsi_spektral(y_trainval)
    c1, c2, c3 = st.columns(3)
    mcard(c1, "Dominant Period", f"{per_sp} hari")
    mcard(c2, "T-hitung",        f"{Th:.5f}")
    mcard(c3, "T-tabel",         f"{Tt:.5f}")
    badge = ('<span class="badge-ok">✓ Musiman Terdeteksi</span>' if mus
             else '<span class="badge-err">✗ Tidak Musiman</span>')
    st.markdown(f"**Kesimpulan:** {badge}", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(pg, color=PAL["season"], lw=1.2)
    ax.fill_between(range(len(pg)), pg, alpha=.13, color=PAL["season"])
    ax.axvline(np.argmax(pg), color="#f87171", lw=1.5, ls="--", label=f"Peak @ idx={np.argmax(pg)}")
    ax.set_title("Periodogram – Data SST Train+Val")
    ax.set_xlabel("Frequency Index"); ax.set_ylabel("Power")
    ax.legend(); ax.grid(True, lw=.4)
    st.pyplot(fig, use_container_width=True); plt.close(fig)

    narasi(f"Analisis spektral pada **data SST Train+Val** ({len(y_trainval):,} titik). "
           f"Periode dominan **{per_sp} hari**. "
           f"{'Pola musiman signifikan terdeteksi (T-hitung > T-tabel).' if mus else 'Tidak ada pola musiman signifikan.'}")

# ══════ TAB 2: TRAINING ═══════════════════════════════════════
with t2:
    mode = st.radio("Mode", ["⬆️ Upload Model (.pt) dari Colab", "🏋️ Train dari Awal"], horizontal=True)

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
            if up_t is not None: st.session_state["bytes_trend"] = up_t.read()
            if "bytes_trend" in st.session_state: st.success("✅ trend_model.pt tersimpan")
        with col2:
            st.markdown("#### 🟠 Seasonal Model (.pt)")
            up_s = st.file_uploader("Upload season_model.pt", type=["pt"], key="up_season")
            if up_s is not None: st.session_state["bytes_season"] = up_s.read()
            if "bytes_season" in st.session_state: st.success("✅ season_model.pt tersimpan")

        if "bytes_trend" in st.session_state and "bytes_season" in st.session_state:
            import io
            if "trained" not in st.session_state:
                try:
                    TM = TrendModel(lookback, t_conv_f, t_kern, t_lstm, t_dense, t_drop)
                    SM = SeasonModel(lookback, s_conv_f, s_kern, s_lstm, s_dense)
                    TM.load_state_dict(torch.load(io.BytesIO(st.session_state["bytes_trend"]), map_location="cpu"))
                    SM.load_state_dict(torch.load(io.BytesIO(st.session_state["bytes_season"]), map_location="cpu"))
                    TM.eval(); SM.eval()
                    st.session_state.update(dict(
                        trained=True, TM=TM, SM=SM, sc_t=sc_t, sc_s=sc_s,
                        trend_train_s=trend_train_s, trend_val_s=trend_val_s,
                        season_train_s=season_train_s, season_val_s=season_val_s,
                    ))
                except Exception as e:
                    st.error(f"❌ Gagal load model: {e}")
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
            _plot_loss(ax,ht_tr,ht_val,"Trend – Loss Curve",PAL["train"])
            st.pyplot(fig,use_container_width=True); plt.close(fig)
            best_t=int(np.argmin(ht_val))
            c1,c2=st.columns(2)
            mcard(c1,"Best Val Loss",f"{min(ht_val):.6f}",f"epoch {best_t+1}")
            mcard(c2,"Final Train Loss",f"{ht_tr[-1]:.6f}",f"{len(ht_tr)} epochs")
        with col2:
            st.markdown("#### 🟠 Seasonal Model")
            st.caption(f"Conv1D({s_conv_f},k={s_kern},causal) → BiLSTM({s_lstm}) → Dense({s_dense}) → Dense(1)  [no dropout]")
            prog2=st.progress(0,text="Training…")
            SM=SeasonModel(lookback, s_conv_f, s_kern, s_lstm, s_dense)
            hs_tr,hs_val=train_model(SM,Xts,yts,Xvs,yvs,epochs,batch_size,s_lr, patience=20)
            prog2.progress(100,text=f"Done · {len(hs_tr)} epochs")
            fig,ax=plt.subplots(figsize=(6,3))
            _plot_loss(ax,hs_tr,hs_val,"Seasonal – Loss Curve",PAL["season"])
            st.pyplot(fig,use_container_width=True); plt.close(fig)
            best_s=int(np.argmin(hs_val))
            c1,c2=st.columns(2)
            mcard(c1,"Best Val Loss",f"{min(hs_val):.6f}",f"epoch {best_s+1}")
            mcard(c2,"Final Train Loss",f"{hs_tr[-1]:.6f}",f"{len(hs_tr)} epochs")

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
        tp_tr=sc_t.inverse_transform(predict_model(TM,Xtt).reshape(-1,1)).flatten()
        tp_vl=sc_t.inverse_transform(predict_model(TM,Xvt).reshape(-1,1)).flatten()
        sp_tr=sc_s.inverse_transform(predict_model(SM,Xts).reshape(-1,1)).flatten()
        sp_vl=sc_s.inverse_transform(predict_model(SM,Xvs).reshape(-1,1)).flatten()

        t_full_s=np.concatenate([trend_train_s, trend_val_s])
        window_t=t_full_s[-lookback:]
        tp_te=sc_t.inverse_transform(
            recursive_forecast(TM,window_t,n_test).reshape(-1,1)).flatten()

        s_full_s=np.concatenate([season_train_s,season_val_s])
        s_tiled =np.concatenate([s_full_s, s_full_s[-periode:]])
        sp_te_s=[]
        for i in range(n_test):
            end_s  =min(len(s_full_s)-periode+i, len(s_tiled))
            start_s=max(end_s-lookback, 0)
            win_s  =s_tiled[start_s:end_s]
            if len(win_s)<lookback:
                win_s=np.pad(win_s,(lookback-len(win_s),0),mode='edge')
            sp_te_s.append(win_s)
        sp_te=sc_s.inverse_transform(
            predict_model(SM,np.array(sp_te_s,dtype=np.float32)).reshape(-1,1)).flatten()
        window_s=s_full_s[-lookback:]

    h_tr=tp_tr+sp_tr; h_vl=tp_vl+sp_vl; h_te=tp_te+sp_te

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
    if "trained" not in st.session_state:
        st.info("Muat model terlebih dahulu di tab 🤖."); st.stop()
    TM=st.session_state["TM"]; SM=st.session_state["SM"]
    sc_t=st.session_state["sc_t"]; sc_s=st.session_state["sc_s"]

    ca_ctrl, cb_ctrl = st.columns([2,3])
    with ca_ctrl:
        STEPS = st.slider("📅 Jumlah hari forecast", min_value=7, max_value=365,
                          value=st.session_state.get("fc_steps_ui", 30), step=1,
                          key="fc_steps_ui")
    with cb_ctrl:
        tail_days = st.slider("📈 Tampilkan data aktual (hari terakhir)",
                              min_value=30, max_value=min(365,n),
                              value=st.session_state.get("fc_tail_ui", min(90,n)), step=10,
                              key="fc_tail_ui")

    freq_g=pd.infer_freq(dates[:50]) or "D"
    fut_dates=pd.date_range(dates[-1],periods=STEPS+1,freq=freq_g)[1:]

    if "stl_full_tf" not in st.session_state:
        with st.spinner("Menghitung STL full data…"):
            from statsmodels.tsa.seasonal import STL as _STL
            _stl=_STL(y_full, period=periode, robust=stl_robust).fit()
            _tf=sc_t.transform(_stl.trend.reshape(-1,1)).flatten().astype(np.float32)
            _sf=sc_s.transform(_stl.seasonal.reshape(-1,1)).flatten().astype(np.float32)
            st.session_state["stl_full_tf"]=_tf
            st.session_state["stl_full_sf"]=_sf
    tf_full_s=st.session_state["stl_full_tf"]
    sf_full_s=st.session_state["stl_full_sf"]

    if st.session_state.get("fc_steps") != STEPS:
        with st.spinner(f"Menghitung forecast {STEPS} hari…"):
            window_t_fc=tf_full_s[-lookback:]
            tf_s=recursive_forecast(TM, window_t_fc, STEPS)
            tf=sc_t.inverse_transform(tf_s.reshape(-1,1)).flatten()
            sf_tiled=np.concatenate([sf_full_s, sf_full_s[-periode:]])
            sf_fc_list=[]
            for i in range(STEPS):
                end  =min(len(sf_full_s)-periode+i, len(sf_tiled))
                start=max(end-lookback, 0)
                win  =sf_tiled[start:end]
                if len(win)<lookback:
                    win=np.pad(win,(lookback-len(win),0),mode='edge')
                sf_fc_list.append(win)
            sf_s=predict_model(SM, np.array(sf_fc_list,dtype=np.float32))
            sf=sc_s.inverse_transform(sf_s.reshape(-1,1)).flatten()
            hf=tf+sf
            st.session_state.update({"fc_steps":STEPS,"fc_tf_s":tf_s,
                                     "fc_tf":tf,"fc_sf_s":sf_s,"fc_sf":sf,"fc_hf":hf})
    tf_s=st.session_state["fc_tf_s"]; tf=st.session_state["fc_tf"]
    sf_s=st.session_state["fc_sf_s"]; sf=st.session_state["fc_sf"]
    hf =st.session_state["fc_hf"]

    # ── HITUNG ANOMALI & STATUS EKOLOGIS ─────────────────────
    anomali_fc = hf - MMM
    status_fc  = [klasifikasi_status(a) for a in anomali_fc]
    n_normal   = sum(1 for s in status_fc if "Normal"  in s)
    n_waspada  = sum(1 for s in status_fc if "Waspada" in s)
    n_kritis   = sum(1 for s in status_fc if "Kritis"  in s)
    max_anomali = anomali_fc.max()
    tgl_kritis  = [str(fut_dates[i].date()) for i,s in enumerate(status_fc) if "Kritis" in s]
    tgl_waspada = [str(fut_dates[i].date()) for i,s in enumerate(status_fc) if "Waspada" in s]

    sec(f"🔮 Forecast {STEPS} Hari ke Depan")

    # Plot dengan garis ambang batas
    fig,ax=plt.subplots(figsize=(14,4))
    ax.plot(dates[-tail_days:],y_full[-tail_days:],color=PAL["actual"],lw=1.8,
            alpha=0.6,label=f"Actual ({tail_days} hari terakhir)",zorder=2)
    ax.plot(fut_dates,hf,color=PAL["future"],lw=2.2,
            label=f"Forecast ({STEPS} hari)",marker="o",markersize=4,zorder=5)
    ax.axhline(MMM,      color="#64748b", lw=1.2, ls=":",  alpha=.8,
               label=f"MMM = {MMM:.2f}°C")
    ax.axhline(MMM+1.0,  color=PAL["crit"], lw=1.5, ls="--", alpha=.85,
               label=f"Ambang Kritis MMM+1°C = {MMM+1:.2f}°C")
    ax.axhline(MMM,      color=PAL["warn"], lw=1.2, ls="--", alpha=.6,
               label=f"Ambang Waspada = MMM = {MMM:.2f}°C")
    ax.axvline(dates[-1],color="#64748b",lw=1,ls=":",alpha=.7,label="Batas data")

    # Warnai area forecast sesuai status
    for i in range(STEPS):
        anom = anomali_fc[i]
        c = PAL["crit"] if anom >= 1.0 else PAL["warn"] if anom >= 0.0 else PAL["normal"]
        ax.axvspan(fut_dates[i], fut_dates[min(i+1, STEPS-1)], alpha=0.07, color=c)

    ax.set_title(f"Future Forecast – {STEPS} Hari ke Depan + Interpretasi Ekologis")
    ax.set_ylabel("SST (°C)")
    tight_ylim(ax,[y_full[-tail_days:],hf,[MMM, MMM+1.0]]); ax.legend(fontsize=7); ax.grid(True,lw=.4)
    st.pyplot(fig,use_container_width=True); plt.close(fig)

    # ── METRIC CARDS EKOLOGIS ────────────────────────────────
    sec("🌡️ Status Termal Ekologis – Ringkasan")
    m1,m2,m3,m4 = st.columns(4)
    mcard(m1, "MMM Historis",     f"{MMM:.3f}°C",   "Maximum Monthly Mean")
    mcard(m2, "Hari Normal 🟢",   f"{n_normal} hari",  f"{n_normal/STEPS*100:.0f}% periode")
    mcard(m3, "Hari Waspada 🟡",  f"{n_waspada} hari", f"{n_waspada/STEPS*100:.0f}% periode")
    mcard(m4, "Hari Kritis 🔴",   f"{n_kritis} hari",  f"{n_kritis/STEPS*100:.0f}% periode")

    # ── ALERT NARASI OTOMATIS ────────────────────────────────
    if n_kritis > 0:
        alert_box("crit",
            f"⚠️ <b>PERINGATAN KRITIS:</b> Terdeteksi <b>{n_kritis} hari</b> dengan SPL melebihi "
            f"ambang bleaching (≥ MMM+1°C = {MMM+1:.2f}°C). "
            f"Anomali maksimum: <b>+{max_anomali:.3f}°C</b>. "
            f"Risiko pemutihan karang tinggi — disarankan aktivasi protokol pemantauan ekosistem segera. "
            f"Hari kritis pertama: <b>{tgl_kritis[0] if tgl_kritis else '-'}</b>.")
    elif n_waspada > 0:
        alert_box("warn",
            f"⚡ <b>ZONA WASPADA:</b> Terdeteksi <b>{n_waspada} hari</b> dengan SPL berada di zona waspada "
            f"(antara MMM dan MMM+1°C = {MMM:.2f}–{MMM+1:.2f}°C). "
            f"Karang mulai termonitor tekanan termal. "
            f"Pemantauan berkala ekosistem terumbu karang disarankan. "
            f"Zona waspada mulai: <b>{tgl_waspada[0] if tgl_waspada else '-'}</b>.")
    else:
        alert_box("normal",
            f"✅ <b>KONDISI NORMAL:</b> Seluruh {STEPS} hari periode forecast berada di bawah MMM ({MMM:.2f}°C). "
            f"Tidak ada indikasi tekanan termal pada ekosistem terumbu karang. "
            f"Kondisi perairan diprediksi aman untuk aktivitas konservasi.")

    ref_box(
        "Ambang bleaching mengacu pada <b>NOAA Coral Reef Watch (CRW)</b>: suhu ≥1°C di atas Maximum Monthly Mean (MMM) "
        "didefinisikan sebagai ambang bleaching (Glynn & D'Croz, 1990). "
        "Zona waspada (0–1°C di atas MMM) mengacu pada HotSpot product NOAA CRW. "
        "Sumber: <a href='https://coralreefwatch.noaa.gov/product/5km/methodology.php' target='_blank'>coralreefwatch.noaa.gov</a> · "
        "<a href='https://www.aoml.noaa.gov/threats-to-coral/' target='_blank'>aoml.noaa.gov</a>"
    )

    # ── GRAFIK ANOMALI ───────────────────────────────────────
    sec("📊 Anomali SPL terhadap MMM Historis")
    fig2, ax2 = plt.subplots(figsize=(14, 3.5))
    bar_colors = [PAL["crit"] if a >= 1.0 else PAL["warn"] if a >= 0.0 else PAL["normal"]
                  for a in anomali_fc]
    ax2.bar(range(STEPS), anomali_fc, color=bar_colors, alpha=0.85, width=0.8)
    ax2.axhline(0,   color=PAL["warn"],   lw=1.5, ls="--", alpha=.8,
                label=f"Ambang Waspada = MMM ({MMM:.2f}°C)")
    ax2.axhline(1.0, color=PAL["crit"],   lw=1.5, ls="--", alpha=.8,
                label="Ambang Kritis = MMM+1°C")
    ax2.axhline(0,   color="#64748b",     lw=0.8, ls="-",  alpha=.5)
    ax2.set_title("Anomali SPL Forecast terhadap MMM Historis")
    ax2.set_xlabel("Hari ke-"); ax2.set_ylabel("Anomali (°C)")
    ax2.legend(fontsize=8); ax2.grid(True, lw=.4, axis="y")

    # Anotasi max anomali
    if max_anomali > 0:
        idx_max = np.argmax(anomali_fc)
        ax2.annotate(f"+{max_anomali:.3f}°C",
                     xy=(idx_max, anomali_fc[idx_max]),
                     xytext=(idx_max, anomali_fc[idx_max]+0.05),
                     fontsize=8, color=PAL["crit"], ha="center", fontweight="bold")
    st.pyplot(fig2, use_container_width=True); plt.close(fig2)

    narasi(
        f"Anomali dihitung terhadap MMM historis = **{MMM:.3f}°C**. "
        f"Batang **hijau** = di bawah MMM (normal), **kuning** = zona waspada (0–1°C di atas MMM), "
        f"**merah** = zona kritis (≥1°C di atas MMM, ambang bleaching NOAA CRW)."
    )

    # ── TABEL LENGKAP ────────────────────────────────────────
    sec("📋 Tabel – Sebelum & Sesudah Denormalisasi + Status Ekologis")
    ca,cb=st.columns(2)
    with ca:
        st.markdown("**Sebelum Denormalisasi**")
        st.dataframe(pd.DataFrame({"Periode":range(1,STEPS+1),"Tanggal":fut_dates,
            "Trend (norm)":np.round(tf_s,5),"Seasonal (norm)":np.round(sf_s,5)})
            .style.format({"Trend (norm)":"{:.5f}","Seasonal (norm)":"{:.5f}"}),
            use_container_width=True)
    with cb:
        st.markdown("**Setelah Denormalisasi + Status Ekologis**")
        df_out=pd.DataFrame({
            "Periode":range(1,STEPS+1),
            "Tanggal":fut_dates,
            "Trend (°C)":np.round(tf,4),
            "Seasonal (°C)":np.round(sf,4),
            "SST Pred (°C)":np.round(hf,4),
            "Anomali vs MMM":np.round(anomali_fc,4),
            "Status":status_fc,
        })
        st.dataframe(df_out.style
            .format({"Trend (°C)":"{:.4f}","Seasonal (°C)":"{:.4f}",
                     "SST Pred (°C)":"{:.4f}","Anomali vs MMM":"{:+.4f}"})
            .background_gradient(subset=["SST Pred (°C)"],cmap="RdYlBu_r"),
            use_container_width=True)

    st.download_button("⬇ Download Future CSV + Status Ekologis",
        df_out.to_csv(index=False).encode(),"future_forecast_ekologis.csv","text/csv")

    narasi(
        f"Proyeksi SST {STEPS} hari: **{hf.min():.4f}–{hf.max():.4f}°C**, "
        f"rata-rata **{hf.mean():.4f}°C**. "
        f"MMM historis = **{MMM:.3f}°C** · "
        f"🟢 Normal: {n_normal} hari · 🟡 Waspada: {n_waspada} hari · 🔴 Kritis: {n_kritis} hari."
    )
