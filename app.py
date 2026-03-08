"""
Dashboard Analisis Time Series – CNN-BiLSTM + STL Decomposition
Run locally  : streamlit run app.py
Deploy       : push ke GitHub → share.streamlit.io
Requirements : requirements.txt  |  .python-version (3.11)
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import random
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

# ── CSS ───────────────────────────────────────────────────────────────────────
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

# ── matplotlib dark theme ─────────────────────────────────────────────────────
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
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def mcard(col, label, value, sub=""):
    col.markdown(f"""<div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        <div class="sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

def mape(yt, yp):
    yt,yp = np.array(yt),np.array(yp)
    m = yt!=0
    return np.mean(np.abs((yt[m]-yp[m])/yt[m]))*100

def build_dataset(arr, lb):
    X,y=[],[]
    for i in range(lb,len(arr)):
        X.append(arr[i-lb:i]); y.append(arr[i])
    return np.array(X),np.array(y)

def recursive_forecast(model,w0,steps):
    w,out=w0.copy(),[]
    for _ in range(steps):
        p=model.predict(w.reshape(1,-1,1),verbose=0)[0,0]
        out.append(p); w=np.append(w[1:],p)
    return np.array(out)

def calc_metrics(yt,yp):
    from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
    return {"RMSE":np.sqrt(mean_squared_error(yt,yp)),
            "MAE":mean_absolute_error(yt,yp),
            "MAPE (%)":mape(yt,yp),
            "R²":r2_score(yt,yp)}

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

def fdGPH(x,bw=0.5):
    import statsmodels.api as sm
    x=np.asarray(x,dtype=float)-np.mean(x); n=len(x)
    m=int(np.floor(n**bw)); j=np.arange(1,m+1)
    lam=2*np.pi*j/n
    fv=np.fft.fft(x)
    I=(1/(2*np.pi*n))*np.abs(fv[j])**2
    Y=np.log(I); Xm=sm.add_constant(np.log(4*(np.sin(lam/2)**2)))
    mo=sm.OLS(Y,Xm).fit()
    return -mo.params[1],mo

def generate_random_sst(n=3650, seed=42):
    """Generate synthetic SST data yang mirip data SST tropis harian."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    # trend pelan naik
    trend = 28.5 + 0.0003*t
    # musiman tahunan ~365 hari
    seasonal = 1.2*np.sin(2*np.pi*t/365 + 0.5) + 0.4*np.sin(4*np.pi*t/365)
    # noise
    noise = rng.normal(0, 0.15, n)
    # autocorrelated noise
    for i in range(1,n):
        noise[i] += 0.55*noise[i-1]
    sst = trend + seasonal + noise
    dates = pd.date_range("2015-01-01", periods=n, freq="D")
    df = pd.DataFrame({"tgl": dates.strftime("%m/%d/%Y"), "sst": np.round(sst,5)})
    return df

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌊 SST Forecast")
    st.markdown("**CNN-BiLSTM + STL Decomposition**")
    st.divider()

    # ── DATA SOURCE ──────────────────────────────────────────────────────────
    st.markdown("### 📂 Data Source")
    data_source = st.radio("Pilih sumber data:",
                           ["📁 Upload CSV", "🎲 Generate Data Contoh"],
                           index=0)

    uploaded = None
    if data_source == "📁 Upload CSV":
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        st.caption("Format: kolom tanggal + kolom SST (nilai numerik)")
    else:
        gen_n    = st.slider("Jumlah hari data",  365, 5000, 3650, 365)
        gen_seed = st.number_input("Seed data", value=42)
        st.caption("Data sintetis mirip SST tropis harian (trend + musiman + noise).")

    date_col = st.text_input("Nama kolom tanggal", value="tgl")
    sst_col  = st.text_input("Nama kolom target",  value="sst")

    st.divider()

    # ── SPLIT ────────────────────────────────────────────────────────────────
    st.markdown("### 🔧 Data Split")
    train_r = st.slider("Train ratio", 0.50, 0.95, 0.90, 0.01)
    val_r   = st.slider("Val ratio",   0.01, 0.20, 0.05, 0.01)
    test_r  = round(1-train_r-val_r, 4)
    st.markdown(f"**Test ratio (auto):** `{max(test_r,0):.2f}`")

    # ── STL ──────────────────────────────────────────────────────────────────
    st.markdown("### 📅 STL")
    auto_period   = st.checkbox("Auto-detect period (spektral)", value=True)
    manual_period = 180
    if not auto_period:
        manual_period = st.number_input("Period manual", 2, 730, 180)
    stl_robust = st.checkbox("STL robust", value=True)

    # ── TREND MODEL ───────────────────────────────────────────────────────────
    st.markdown("### 🧠 Trend Model")
    t_conv_f = st.slider("Conv1D filters",  8,  128, 32,  8)
    t_kern   = st.slider("Kernel size",     2,  15,  5,   1)
    t_lstm   = st.slider("BiLSTM units",    16, 256, 64,  16)
    t_drop   = st.slider("Dropout",         0.0, 0.5, 0.2, 0.05)
    t_dense  = st.slider("Dense units",     8,  128, 32,  8)
    t_lr     = st.number_input("LR trend",  value=0.0007, format="%.4f")

    # ── SEASONAL MODEL ────────────────────────────────────────────────────────
    st.markdown("### 🧠 Seasonal Model")
    s_conv_f = st.slider("Conv1D filters (S)", 8,  128, 64,  8)
    s_kern   = st.slider("Kernel size (S)",    2,  15,  5,   1)
    s_lstm   = st.slider("BiLSTM units (S)",   16, 256, 64,  16)
    s_dense  = st.slider("Dense units (S)",    4,  64,  16,  4)
    s_lr     = st.number_input("LR seasonal",  value=0.0005, format="%.4f")

    # ── TRAINING ──────────────────────────────────────────────────────────────
    st.markdown("### ⚙️ Training")
    lookback   = st.slider("Lookback",    30,  365, 180, 10)
    epochs     = st.slider("Max epochs",  10,  500, 250, 10)
    batch_size = st.selectbox("Batch size", [16,32,64,128], index=2)
    seed       = st.number_input("Random seed", value=42)

    st.markdown("### 🔬 GPH")
    gph_bw = st.slider("GPH bandwidth exp", 0.3, 0.9, 0.5, 0.05)

    st.divider()
    run_btn = st.button("▶  Run Analysis", use_container_width=True, type="primary")

# ═════════════════════════════════════════════════════════════════════════════
# MAIN HEADER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("# 🌊 SST Time Series Analysis Dashboard")
st.markdown("*Hybrid CNN-BiLSTM + STL Decomposition · Sea Surface Temperature Forecasting*")

if not run_btn:
    st.info("👈 Pilih sumber data & atur parameter di sidebar, lalu klik **▶ Run Analysis**.")

    # Preview contoh format data
    with st.expander("📋 Lihat format CSV yang dibutuhkan"):
        sample = generate_random_sst(10, 42)
        st.dataframe(sample, use_container_width=True)
        st.caption("Kolom minimal: tanggal (`tgl`) + nilai SST (`sst`). Nama kolom bisa diubah di sidebar.")

        csv_sample = sample.to_csv(index=False).encode()
        st.download_button("⬇ Download contoh CSV (10 baris)",
                           csv_sample, "contoh_data.csv", "text/csv")
    st.stop()

# ── load data ─────────────────────────────────────────────────────────────────
if data_source == "🎲 Generate Data Contoh":
    df_raw = generate_random_sst(int(gen_n), int(gen_seed))
    st.success(f"✅ Data sintetis dibuat: {len(df_raw):,} baris")
else:
    if uploaded is None:
        st.error("⚠️ Harap upload file CSV terlebih dahulu."); st.stop()
    df_raw = pd.read_csv(uploaded)

if date_col not in df_raw.columns or sst_col not in df_raw.columns:
    st.error(f"Kolom `{date_col}` atau `{sst_col}` tidak ditemukan.\n"
             f"Kolom yang ada: `{list(df_raw.columns)}`\n"
             "Sesuaikan nama kolom di sidebar.")
    st.stop()

# ── parse & sort ──────────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (LSTM, Bidirectional, Dense, Dropout, Conv1D)
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from statsmodels.tsa.seasonal import STL
    from sklearn.preprocessing import MinMaxScaler
except ImportError as e:
    st.error(f"Import error: {e}"); st.stop()

np.random.seed(int(seed)); tf.random.set_seed(int(seed)); random.seed(int(seed))

df = df_raw.copy()
df[date_col] = pd.to_datetime(df[date_col], dayfirst=False, infer_datetime_format=True)
df = df.sort_values(date_col).set_index(date_col)
y_full = df[sst_col].values.astype(float)
dates  = df.index
n      = len(y_full)

n_train = int(n*train_r)
n_val   = int(n*val_r)
n_test  = n - n_train - n_val

if n_test <= 0:
    st.error("Test set kosong. Kurangi train/val ratio."); st.stop()

y_trainval = y_full[:n_train+n_val]

# ═════════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════════
t0,t1,t2,t3,t4,t5 = st.tabs([
    "📊 Data Overview",
    "🔬 STL Decomposition",
    "📈 Spectral & GPH",
    "🤖 Model Training",
    "🎯 Forecast Results",
    "📋 Metrics Summary",
])

# ─── TAB 0 · DATA OVERVIEW ───────────────────────────────────────────────────
with t0:
    c1,c2,c3,c4 = st.columns(4)
    mcard(c1,"Total Points",f"{n:,}")
    mcard(c2,"Train", f"{n_train:,}", f"{train_r*100:.0f}%")
    mcard(c3,"Val",   f"{n_val:,}",   f"{val_r*100:.0f}%")
    mcard(c4,"Test",  f"{n_test:,}",  f"{(n_test/n)*100:.1f}%")

    fig,ax=plt.subplots(figsize=(14,3.8))
    slices = [
        (slice(None,n_train),           PAL["train"], f"Train ({n_train})"),
        (slice(n_train,n_train+n_val),  PAL["val"],   f"Val ({n_val})"),
        (slice(n_train+n_val,None),     PAL["test"],  f"Test ({n_test})"),
    ]
    for sl,col,lbl in slices:
        ax.plot(dates[sl], y_full[sl], color=col, lw=1.2, label=lbl)
        ax.fill_between(dates[sl], y_full[sl], alpha=.1, color=col)
    ax.axvline(dates[n_train],       color=PAL["val"],  lw=1.5,ls="--",alpha=.8)
    ax.axvline(dates[n_train+n_val], color=PAL["test"], lw=1.5,ls="--",alpha=.8)
    ax.set_title("Data Split Visualization"); ax.set_ylabel("SST (°C)")
    ax.legend(); ax.grid(True,lw=.5)
    st.pyplot(fig,use_container_width=True); plt.close(fig)

    c1,c2 = st.columns([1,2])
    with c1:
        st.markdown("**Statistik Deskriptif**")
        st.dataframe(df[[sst_col]].describe().round(4), use_container_width=True)
    with c2:
        fig2,ax2=plt.subplots(figsize=(7,3))
        ax2.hist(y_full, bins=60, color=PAL["actual"], alpha=.8, edgecolor="#0e1525")
        ax2.set_title("Distribusi SST"); ax2.set_xlabel("SST (°C)"); ax2.set_ylabel("Frekuensi")
        ax2.grid(True,lw=.5)
        st.pyplot(fig2,use_container_width=True); plt.close(fig2)

    # download preview data
    csv_dl = df[[sst_col]].reset_index().to_csv(index=False).encode()
    st.download_button("⬇ Download data yang digunakan", csv_dl, "data_used.csv","text/csv")

# ─── TAB 1 · STL ─────────────────────────────────────────────────────────────
with t1:
    with st.spinner("Running STL decomposition..."):
        if auto_period:
            periode,*_ = fungsi_spektral(y_trainval)
        else:
            periode = manual_period
        stl_res = STL(y_trainval, period=periode, robust=stl_robust).fit()

    st.success(f"STL selesai · Period = **{periode}** hari")

    fig,axes=plt.subplots(4,1,figsize=(14,9),sharex=True)
    for ax,(name,val,col) in zip(axes,[
        ("Observed", stl_res.observed, PAL["actual"]),
        ("Trend",    stl_res.trend,    PAL["trend"]),
        ("Seasonal", stl_res.seasonal, PAL["season"]),
        ("Residual", stl_res.resid,    PAL["resid"]),
    ]):
        ax.plot(val,color=col,lw=1.2)
        ax.fill_between(range(len(val)),val,alpha=.12,color=col)
        ax.set_ylabel(name,fontsize=9); ax.grid(True,lw=.5)
    axes[-1].set_xlabel("Index")
    fig.suptitle(f"STL Decomposition  (period={periode})",fontsize=12)
    plt.tight_layout()
    st.pyplot(fig,use_container_width=True); plt.close(fig)

    var_obs=np.var(stl_res.observed)
    c1,c2,c3=st.columns(3)
    mcard(c1,"Variance – Trend",   f"{(1-np.var(stl_res.observed-stl_res.trend)/var_obs)*100:.1f}%")
    mcard(c2,"Variance – Seasonal",f"{(1-np.var(stl_res.observed-stl_res.seasonal)/var_obs)*100:.1f}%")
    mcard(c3,"Variance – Residual",f"{np.var(stl_res.resid)/var_obs*100:.1f}%")

# ─── TAB 2 · SPECTRAL & GPH ──────────────────────────────────────────────────
with t2:
    st.markdown("### 📈 Analisis Spektral – Komponen Musiman")
    with st.spinner("Menghitung periodogram..."):
        per_sp,Th,Tt,mus,pg = fungsi_spektral(y_trainval)

    c1,c2,c3=st.columns(3)
    mcard(c1,"Dominant Period",f"{per_sp} hari")
    mcard(c2,"T-hitung",       f"{Th:.5f}")
    mcard(c3,"T-tabel",        f"{Tt:.5f}")
    badge='<span class="badge-ok">✓ Pola Musiman Terdeteksi</span>' if mus else '<span class="badge-err">✗ Tidak Musiman</span>'
    st.markdown(f"**Kesimpulan:** {badge}", unsafe_allow_html=True)

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

    if d_gph<0:        mc,ms="Anti-persistent","d < 0: mean-reverting kuat"
    elif d_gph<0.5:    mc,ms="Long Memory (stasioner)","0 < d < 0.5"
    elif d_gph<1.0:    mc,ms="Long Memory (non-stasioner)","0.5 ≤ d < 1"
    else:              mc,ms="Non-stasioner kuat","d ≥ 1"

    c1,c2,c3=st.columns(3)
    mcard(c1,"GPH d estimate",f"{d_gph:.4f}")
    mcard(c2,"BW exponent",   f"{gph_bw}")
    mcard(c3,"Memory Class",  mc, ms)

    st.markdown("""
| Nilai d | Interpretasi |
|---------|-------------|
| d < 0 | Anti-persistent (mean-reverting kuat) |
| d ≈ 0 | Short memory (ARMA) |
| 0 < d < 0.5 | Long memory, stasioner |
| 0.5 ≤ d < 1 | Long memory, non-stasioner |
| d ≥ 1 | Non-stasioner kuat |
""")

    # GPH scatter
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
    t_arr = trend_tv[:n_train];   t_val_arr = trend_tv[n_train:n_train+n_val]
    s_arr = season_tv[:n_train];  s_val_arr = season_tv[n_train:n_train+n_val]

    sc_t=MinMaxScaler().fit(t_arr.reshape(-1,1))
    sc_s=MinMaxScaler().fit(s_arr.reshape(-1,1))
    t_tr_s=sc_t.transform(t_arr.reshape(-1,1)).flatten()
    t_vl_s=sc_t.transform(t_val_arr.reshape(-1,1)).flatten()
    s_tr_s=sc_s.transform(s_arr.reshape(-1,1)).flatten()
    s_vl_s=sc_s.transform(s_val_arr.reshape(-1,1)).flatten()

    Xtt,ytt  = build_dataset(t_tr_s,lookback)
    Xvt,yvt  = build_dataset(np.concatenate([t_tr_s[-lookback:],t_vl_s]),lookback)
    Xts,yts  = build_dataset(s_tr_s,lookback)
    Xvs,yvs  = build_dataset(np.concatenate([s_tr_s[-lookback:],s_vl_s]),lookback)
    Xtt=Xtt.reshape(-1,lookback,1); Xvt=Xvt.reshape(-1,lookback,1)
    Xts=Xts.reshape(-1,lookback,1); Xvs=Xvs.reshape(-1,lookback,1)

    def make_trend_model():
        m=Sequential([
            Conv1D(t_conv_f,kernel_size=t_kern,padding="causal",activation="relu",input_shape=(lookback,1)),
            Bidirectional(LSTM(t_lstm)),Dropout(t_drop),
            Dense(t_dense,activation="relu"),Dense(1),
        ])
        m.compile(optimizer=tf.keras.optimizers.Adam(t_lr),loss="mse"); return m

    def make_season_model():
        m=Sequential([
            Conv1D(s_conv_f,kernel_size=s_kern,padding="causal",activation="relu",input_shape=(lookback,1)),
            Bidirectional(LSTM(s_lstm)),
            Dense(s_dense,activation="relu"),Dense(1),
        ])
        m.compile(optimizer=tf.keras.optimizers.Adam(s_lr),loss="mse"); return m

    cbs=[EarlyStopping(patience=20,restore_best_weights=True),
         ReduceLROnPlateau(patience=8,factor=0.5)]

    col1,col2=st.columns(2)

    with col1:
        st.markdown("#### 🔵 Trend Model")
        prog=st.progress(0,text="Training…")
        TM=make_trend_model()
        ht=TM.fit(Xtt,ytt,validation_data=(Xvt,yvt),epochs=epochs,
                  batch_size=batch_size,shuffle=False,callbacks=cbs,verbose=0)
        prog.progress(100,text=f"Done · {len(ht.history['loss'])} epochs")
        fig,ax=plt.subplots(figsize=(6,3))
        ax.plot(ht.history["loss"],    color=PAL["train"],lw=1.5,label="Train")
        ax.plot(ht.history["val_loss"],color=PAL["val"],  lw=1.5,ls="--",label="Val")
        ax.set_title("Trend Loss"); ax.legend(); ax.grid(True,lw=.5)
        st.pyplot(fig,use_container_width=True); plt.close(fig)

    with col2:
        st.markdown("#### 🟠 Seasonal Model")
        prog2=st.progress(0,text="Training…")
        SM=make_season_model()
        hs=SM.fit(Xts,yts,validation_data=(Xvs,yvs),epochs=epochs,
                  batch_size=batch_size,shuffle=False,callbacks=cbs,verbose=0)
        prog2.progress(100,text=f"Done · {len(hs.history['loss'])} epochs")
        fig,ax=plt.subplots(figsize=(6,3))
        ax.plot(hs.history["loss"],    color=PAL["season"],lw=1.5,label="Train")
        ax.plot(hs.history["val_loss"],color=PAL["val"],   lw=1.5,ls="--",label="Val")
        ax.set_title("Seasonal Loss"); ax.legend(); ax.grid(True,lw=.5)
        st.pyplot(fig,use_container_width=True); plt.close(fig)

    st.session_state.update(dict(
        trained=True, TM=TM, SM=SM, sc_t=sc_t, sc_s=sc_s,
        t_tr_s=t_tr_s, t_vl_s=t_vl_s, s_tr_s=s_tr_s, s_vl_s=s_vl_s,
        Xtt=Xtt,Xvt=Xvt,Xts=Xts,Xvs=Xvs,
        trend_tv=trend_tv, season_tv=season_tv,
    ))

# ─── TAB 4 · FORECAST ────────────────────────────────────────────────────────
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

    tp_tr=sc_t.inverse_transform(TM.predict(Xtt,verbose=0)).flatten()
    tp_vl=sc_t.inverse_transform(TM.predict(Xvt,verbose=0)).flatten()
    sp_tr=sc_s.inverse_transform(SM.predict(Xts,verbose=0)).flatten()
    sp_vl=sc_s.inverse_transform(SM.predict(Xvs,verbose=0)).flatten()

    tl=np.concatenate([t_tr_s,t_vl_s])[-lookback:]
    sl=np.concatenate([s_tr_s,s_vl_s])[-lookback:]
    with st.spinner("Recursive forecast pada test set…"):
        tp_te=sc_t.inverse_transform(recursive_forecast(TM,tl,n_test).reshape(-1,1)).flatten()
        sp_te=sc_s.inverse_transform(recursive_forecast(SM,sl,n_test).reshape(-1,1)).flatten()

    h_tr=tp_tr+sp_tr; h_vl=tp_vl+sp_vl; h_te=tp_te+sp_te

    def fplot(title,d_actual,d_actual_dates,segments):
        fig,ax=plt.subplots(figsize=(14,3.8))
        ax.plot(d_actual_dates,d_actual,color=PAL["actual"],lw=1.2,label="Actual")
        for (d,y,col,lbl) in segments:
            ax.plot(d,y,color=col,lw=1.2,ls="--",label=lbl)
        ax.set_title(title); ax.legend(); ax.grid(True,lw=.5)
        st.pyplot(fig,use_container_width=True); plt.close(fig)

    # Trend
    fplot("Trend – Actual vs Predicted",
          trend_tv, dates[:n_train+n_val],
          [(dates[lookback:n_train],    tp_tr, PAL["train"], "Train"),
           (dates[n_train:n_train+n_val],tp_vl, PAL["val"],  "Val"),
           (dates[n_train+n_val:],      tp_te, PAL["test"], "Test")])

    # Seasonal
    fplot("Seasonal – Actual vs Predicted",
          season_tv, dates[:n_train+n_val],
          [(dates[lookback:n_train],    sp_tr, PAL["train"], "Train"),
           (dates[n_train:n_train+n_val],sp_vl, PAL["val"],  "Val"),
           (dates[n_train+n_val:],      sp_te, PAL["test"], "Test")])

    # Hybrid full
    fig,ax=plt.subplots(figsize=(14,4))
    ax.plot(dates,y_full,color=PAL["actual"],lw=1.2,label="Actual SST")
    ax.plot(dates[lookback:n_train],      h_tr,color=PAL["train"],lw=1.2,ls="--",label="Hybrid Train")
    ax.plot(dates[n_train:n_train+n_val], h_vl,color=PAL["val"],  lw=1.2,ls="--",label="Hybrid Val")
    ax.plot(dates[n_train+n_val:],        h_te,color=PAL["test"], lw=1.5,ls="--",label="Hybrid Test")
    ax.set_title("Hybrid Reconstruction – Full Series"); ax.legend(); ax.grid(True,lw=.5)
    st.pyplot(fig,use_container_width=True); plt.close(fig)

    # Test only
    fig,ax=plt.subplots(figsize=(12,3.8))
    ax.plot(dates[n_train+n_val:],y_full[n_train+n_val:],color=PAL["actual"],lw=1.5,label="Actual")
    ax.plot(dates[n_train+n_val:],h_te,                  color=PAL["test"],  lw=1.5,ls="--",label="Predicted")
    ax.fill_between(dates[n_train+n_val:],y_full[n_train+n_val:],h_te,alpha=.1,color=PAL["test"])
    ax.set_title("TEST SET – Actual vs Predicted"); ax.legend(); ax.grid(True,lw=.5)
    st.pyplot(fig,use_container_width=True); plt.close(fig)

    # Download hasil prediksi
    df_pred = pd.DataFrame({
        "date": dates[n_train+n_val:],
        "actual": y_full[n_train+n_val:],
        "predicted_hybrid": h_te,
        "predicted_trend": tp_te,
        "predicted_seasonal": sp_te,
    })
    st.download_button("⬇ Download Hasil Prediksi Test (CSV)",
                       df_pred.to_csv(index=False).encode(),
                       "test_predictions.csv","text/csv")

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

    t_tr_true=trend_tv[lookback:n_train];       t_vl_true=trend_tv[n_train:n_train+n_val]
    s_tr_true=season_tv[lookback:n_train];      s_vl_true=season_tv[n_train:n_train+n_val]

    all_metrics={}
    def show_block(title,rows):
        st.markdown(f"### {title}")
        for lbl,(yt,yp) in rows.items():
            m=calc_metrics(yt,yp); all_metrics[f"{title.split()[-1]} – {lbl}"]=m
            cols=st.columns(4)
            mcard(cols[0],f"{lbl} RMSE",  f"{m['RMSE']:.4f}")
            mcard(cols[1],f"{lbl} MAE",   f"{m['MAE']:.4f}")
            mcard(cols[2],f"{lbl} MAPE",  f"{m['MAPE (%)']:.2f}%")
            mcard(cols[3],f"{lbl} R²",    f"{m['R²']:.4f}")
            st.markdown("")

    show_block("🔵 Trend",   {"Train":(t_tr_true,tp_tr),"Val":(t_vl_true,tp_vl)})
    show_block("🟠 Seasonal",{"Train":(s_tr_true,sp_tr),"Val":(s_vl_true,sp_vl)})
    show_block("🌊 Hybrid",  {"Train":(y_full[lookback:n_train],h_tr),
                               "Val":  (y_full[n_train:n_train+n_val],h_vl),
                               "TEST": (y_full[n_train+n_val:],h_te)})

    st.markdown("### 📋 Summary Table")
    sdf=pd.DataFrame(all_metrics).T.round(4)
    st.dataframe(
        sdf.style
           .background_gradient(subset=["RMSE","MAE","MAPE (%)"],cmap="RdYlGn_r")
           .background_gradient(subset=["R²"],cmap="RdYlGn"),
        use_container_width=True,
    )
    st.download_button("⬇ Download Metrics CSV",
                       sdf.to_csv().encode(),"metrics_summary.csv","text/csv")
