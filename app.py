# ============================================================
# GRID SEARCH CNN-BiLSTM — Google Colab
# Mencari parameter terbaik berdasarkan MAPE Testing
# Trend Model & Seasonal Model dicari secara terpisah
# ============================================================

import warnings; warnings.filterwarnings("ignore")
import random, copy, itertools, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
from IPython.display import display

# ── CONFIG TETAP ─────────────────────────────────────────────
CSV_PATH    = "/content/sst kotak.csv"
DATE_COL    = "tgl"
SST_COL     = "sst"
TRAIN_RATIO = 0.9
VAL_RATIO   = 0.05
LOOKBACK    = 365
EPOCHS      = 300
SEED        = 42

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# ── GRID PARAMETER (maks 3 nilai per parameter) ──────────────
# Trend Model
GRID_TREND = {
    "batch_size": [32,  64,  128],
    "filters":    [16,  32,  64 ],
    "kernel":     [3,   5,   7  ],
    "neurons":    [32,  64,  128],
    "lr":         [0.001, 0.0007, 0.0003],
}

# Seasonal Model
GRID_SEASON = {
    "batch_size": [32,  64,  128],
    "filters":    [32,  64,  128],
    "kernel":     [3,   5,   7  ],
    "neurons":    [32,  64,  128],
    "lr":         [0.001, 0.0005, 0.0003],
}

# ── MODEL ─────────────────────────────────────────────────────
class TrendModel(nn.Module):
    def __init__(self, cf=32, ks=5, lu=64, du=32, drop=0.2):
        super().__init__()
        self.ks=ks; self.conv=nn.Conv1d(1,cf,ks)
        self.bilstm=nn.LSTM(cf,lu,batch_first=True,bidirectional=True)
        self.drop=nn.Dropout(drop)
        self.fc1=nn.Linear(lu*2,du); self.fc2=nn.Linear(du,1)
    def forward(self,x):
        x=F.pad(x.permute(0,2,1),(self.ks-1,0))
        x=F.relu(self.conv(x)).permute(0,2,1)
        out,_=self.bilstm(x)
        return self.fc2(F.relu(self.fc1(self.drop(out[:,-1,:])))).squeeze(-1)

class SeasonModel(nn.Module):
    def __init__(self, cf=64, ks=5, lu=64, du=16):
        super().__init__()
        self.ks=ks; self.conv=nn.Conv1d(1,cf,ks)
        self.bilstm=nn.LSTM(cf,lu,batch_first=True,bidirectional=True)
        self.fc1=nn.Linear(lu*2,du); self.fc2=nn.Linear(du,1)
    def forward(self,x):
        x=F.pad(x.permute(0,2,1),(self.ks-1,0))
        x=F.relu(self.conv(x)).permute(0,2,1)
        out,_=self.bilstm(x)
        return self.fc2(F.relu(self.fc1(out[:,-1,:]))).squeeze(-1)

# ── HELPERS ───────────────────────────────────────────────────
def build_dataset(arr, lb):
    X,y=[],[]
    for i in range(lb,len(arr)): X.append(arr[i-lb:i]); y.append(arr[i])
    return np.array(X,dtype=np.float32), np.array(y,dtype=np.float32)

def train_one(model, Xtr, ytr, Xvl, yvl, lr, bs, epochs=EPOCHS, patience=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=lr)
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)
    crit   = nn.MSELoss()
    Xt = torch.tensor(Xtr).unsqueeze(-1).to(device)
    yt = torch.tensor(ytr).to(device)
    Xv = torch.tensor(Xvl).unsqueeze(-1).to(device)
    yv = torch.tensor(yvl).to(device)
    loader = DataLoader(TensorDataset(Xt,yt), batch_size=bs, shuffle=False)
    best_val, best_w, wait = float("inf"), None, 0
    for ep in range(epochs):
        model.train(); running=0.0
        for xb,yb in loader:
            opt.zero_grad(set_to_none=True)
            l=crit(model(xb),yb); l.backward(); opt.step(); running+=l.item()
        model.eval()
        with torch.inference_mode(): vl=crit(model(Xv),yv).item()
        sched.step(vl)
        if vl < best_val-1e-7: best_val=vl; best_w=copy.deepcopy(model.state_dict()); wait=0
        else:
            wait+=1
            if wait>=patience: break
    if best_w: model.load_state_dict(best_w)
    return model.cpu()

def predict(model, X):
    model.eval()
    with torch.inference_mode():
        return model(torch.tensor(X,dtype=torch.float32).unsqueeze(-1)).numpy().flatten()

def recursive_fc(model, window, steps):
    model.eval()
    w=window.copy().astype(np.float32); out=[]
    buf=torch.zeros(1,len(w),1)
    with torch.inference_mode():
        for _ in range(steps):
            buf[0,:,0]=torch.from_numpy(w)
            p=model(buf).item(); out.append(p); w=np.append(w[1:],p)
    return np.array(out,dtype=np.float32)

def mape_fn(yt,yp):
    yt,yp=np.array(yt),np.array(yp); m=yt!=0
    return float(np.mean(np.abs((yt[m]-yp[m])/yt[m]))*100)

def mae_fn(yt,yp):
    return float(np.mean(np.abs(np.array(yt)-np.array(yp))))

# ── LOAD & PREP DATA ──────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(CSV_PATH)
df[DATE_COL] = pd.to_datetime(df[DATE_COL])
df = df.sort_values(DATE_COL).set_index(DATE_COL)
y_full = df[SST_COL].values.astype(float); n = len(y_full)
n_val   = int(n * VAL_RATIO)
n_test  = int(n * VAL_RATIO)
n_train = n - n_val - n_test
y_trainval = y_full[:n_train+n_val]
print(f"N={n} | Train={n_train} Val={n_val} Test={n_test}")

# Spektral
def fungsi_spektral(x):
    x=np.asarray(x,dtype=float); n=len(x); k=round((n-1)/2)
    fft=np.fft.rfft(x); j=np.arange(1,k+1)
    a=(2/n)*fft[j].real; b=-(2/n)*fft[j].imag; pg=a**2+b**2
    km=np.argmax(pg)+1
    return int(round((2*np.pi)/((2*np.pi*km)/n)))

PERIOD = fungsi_spektral(y_trainval)
print(f"Periode dominan: {PERIOD} hari")

# STL
stl = STL(y_trainval, period=PERIOD, robust=True).fit()
trend_tv  = stl.trend
season_tv = stl.seasonal

sc_t = MinMaxScaler().fit(trend_tv[:n_train].reshape(-1,1))
sc_s = MinMaxScaler().fit(season_tv[:n_train].reshape(-1,1))

t_tv_s = sc_t.transform(trend_tv.reshape(-1,1)).flatten().astype(np.float32)
s_tv_s = sc_s.transform(season_tv.reshape(-1,1)).flatten().astype(np.float32)
t_tr_s = t_tv_s[:n_train]; t_vl_s = t_tv_s[n_train:]
s_tr_s = s_tv_s[:n_train]; s_vl_s = s_tv_s[n_train:]

Xtt,ytt = build_dataset(t_tr_s, LOOKBACK)
Xvt,yvt = build_dataset(np.concatenate([t_tr_s[-LOOKBACK:],t_vl_s]), LOOKBACK)
Xts,yts = build_dataset(s_tr_s, LOOKBACK)
Xvs,yvs = build_dataset(np.concatenate([s_tr_s[-LOOKBACK:],s_vl_s]), LOOKBACK)

t_full_s = np.concatenate([t_tr_s, t_vl_s])
s_full_s = np.concatenate([s_tr_s, s_vl_s])
window_t = t_full_s[-LOOKBACK:]

# Seasonal test windows (tiling, sama persis dengan app.py)
s_tiled = np.concatenate([s_full_s, s_full_s[-PERIOD:]])
sp_te_wins = []
for i in range(n_test):
    end_s   = min(len(s_full_s)-PERIOD+i, len(s_tiled))
    start_s = max(end_s-LOOKBACK, 0)
    win_s   = s_tiled[start_s:end_s]
    if len(win_s)<LOOKBACK: win_s=np.pad(win_s,(LOOKBACK-len(win_s),0),mode='edge')
    sp_te_wins.append(win_s)
sp_te_wins = np.array(sp_te_wins, dtype=np.float32)

y_te_true = y_full[n_train+n_val:]

# ── GRID SEARCH: ONE PARAM AT A TIME ──────────────────────────
# Strategi: variasikan satu parameter, sisanya pakai nilai target.
# Ini jauh lebih efisien dari full grid (3^5 = 243 kombinasi)
# dan tetap menunjukkan pengaruh tiap parameter secara terisolasi.

TARGET_T = dict(batch_size=64, filters=32, kernel=5, neurons=64, lr=0.0007)
TARGET_S = dict(batch_size=64, filters=64, kernel=5, neurons=64, lr=0.0005)

def eval_trend_params(batch_size, filters, kernel, neurons, lr, label=""):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = TrendModel(cf=filters, ks=kernel, lu=neurons, du=max(neurons//2,8))
    model = train_one(model, Xtt, ytt, Xvt, yvt, lr=lr, bs=batch_size)
    tp_te = sc_t.inverse_transform(
        recursive_fc(model, window_t, n_test).reshape(-1,1)).flatten()
    # Untuk hybrid MAPE, pakai seasonal dari target params
    _sm = SeasonModel(cf=TARGET_S["filters"], ks=TARGET_S["kernel"], lu=TARGET_S["neurons"])
    _sm = train_one(_sm, Xts, yts, Xvs, yvs, lr=TARGET_S["lr"], bs=TARGET_S["batch_size"])
    sp_te = sc_s.inverse_transform(predict(_sm, sp_te_wins).reshape(-1,1)).flatten()
    h_te  = tp_te + sp_te
    return {
        "label":      label,
        "batch_size": batch_size,
        "filters":    filters,
        "kernel":     kernel,
        "neurons":    neurons,
        "lr":         lr,
        "MAPE_test":  round(mape_fn(y_te_true, h_te), 4),
        "MAE_test":   round(mae_fn(y_te_true,  h_te), 6),
    }

def eval_season_params(batch_size, filters, kernel, neurons, lr, label=""):
    torch.manual_seed(SEED); np.random.seed(SEED)
    # Trend pakai target params
    _tm = TrendModel(cf=TARGET_T["filters"], ks=TARGET_T["kernel"], lu=TARGET_T["neurons"],
                     du=max(TARGET_T["neurons"]//2,8))
    _tm = train_one(_tm, Xtt, ytt, Xvt, yvt, lr=TARGET_T["lr"], bs=TARGET_T["batch_size"])
    tp_te = sc_t.inverse_transform(
        recursive_fc(_tm, window_t, n_test).reshape(-1,1)).flatten()
    model = SeasonModel(cf=filters, ks=kernel, lu=neurons)
    model = train_one(model, Xts, yts, Xvs, yvs, lr=lr, bs=batch_size)
    sp_te = sc_s.inverse_transform(predict(model, sp_te_wins).reshape(-1,1)).flatten()
    h_te  = tp_te + sp_te
    return {
        "label":      label,
        "batch_size": batch_size,
        "filters":    filters,
        "kernel":     kernel,
        "neurons":    neurons,
        "lr":         lr,
        "MAPE_test":  round(mape_fn(y_te_true, h_te), 4),
        "MAE_test":   round(mae_fn(y_te_true,  h_te), 6),
    }

# ── JALANKAN GRID SEARCH ──────────────────────────────────────
results_trend  = []
results_season = []

param_names = ["batch_size", "filters", "kernel", "neurons", "lr"]

print("\n" + "="*60)
print("GRID SEARCH — TREND MODEL")
print("="*60)
for param in param_names:
    print(f"\n  Variasi {param}:")
    for val in GRID_TREND[param]:
        params = {**TARGET_T, param: val}
        label  = f"{param}={val}"
        print(f"    {label} ...", end=" ", flush=True)
        r = eval_trend_params(**params, label=label)
        results_trend.append({**r, "varied_param": param})
        print(f"MAPE={r['MAPE_test']:.4f}%  MAE={r['MAE_test']:.6f}")

print("\n" + "="*60)
print("GRID SEARCH — SEASONAL MODEL")
print("="*60)
for param in param_names:
    print(f"\n  Variasi {param}:")
    for val in GRID_SEASON[param]:
        params = {**TARGET_S, param: val}
        label  = f"{param}={val}"
        print(f"    {label} ...", end=" ", flush=True)
        r = eval_season_params(**params, label=label)
        results_season.append({**r, "varied_param": param})
        print(f"MAPE={r['MAPE_test']:.4f}%  MAE={r['MAE_test']:.6f}")

# ── RINGKASAN HASIL ───────────────────────────────────────────
df_t = pd.DataFrame(results_trend)
df_s = pd.DataFrame(results_season)

print("\n" + "="*60)
print("HASIL GRID SEARCH — TREND MODEL (urut MAPE ascending)")
print("="*60)
display(df_t.sort_values("MAPE_test").to_string(index=False))

print("\n" + "="*60)
print("HASIL GRID SEARCH — SEASONAL MODEL (urut MAPE ascending)")
print("="*60)
display(df_s.sort_values("MAPE_test").to_string(index=False))

# ── PARAMETER TERBAIK PER KOMPONEN ────────────────────────────
best_t = df_t.loc[df_t["MAPE_test"].idxmin()]
best_s = df_s.loc[df_s["MAPE_test"].idxmin()]

print("\n" + "="*60)
print("PARAMETER TERBAIK")
print("="*60)
print("\nTrend Model:")
for p in param_names:
    mark = " ← varied" if best_t["varied_param"]==p else ""
    print(f"  {p:12s}: {best_t[p]}{mark}")
print(f"  MAPE test : {best_t['MAPE_test']:.4f}%")

print("\nSeasonal Model:")
for p in param_names:
    mark = " ← varied" if best_s["varied_param"]==p else ""
    print(f"  {p:12s}: {best_s[p]}{mark}")
print(f"  MAPE test : {best_s['MAPE_test']:.4f}%")

# ── VISUALISASI ───────────────────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=False)
fig.suptitle("Grid Search — MAPE Testing per Parameter", fontsize=13, y=1.01)

for row, (df_res, title, grid) in enumerate([
    (df_t, "Trend Model",   GRID_TREND),
    (df_s, "Seasonal Model",GRID_SEASON),
]):
    for col, param in enumerate(param_names):
        ax   = axes[row][col]
        sub  = df_res[df_res["varied_param"]==param].copy()
        vals = [str(v) for v in grid[param]]
        mapes= sub.set_index("label").reindex(
            [f"{param}={v}" for v in grid[param]])["MAPE_test"].values

        bars = ax.bar(vals, mapes,
                      color=["#f87171" if m==np.nanmin(mapes) else "#38bdf8"
                             for m in mapes],
                      width=0.5, edgecolor="none")
        for bar, m in zip(bars, mapes):
            if not np.isnan(m):
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+0.001,
                        f"{m:.3f}%", ha="center", va="bottom", fontsize=8)
        ax.set_title(f"{title}\n{param}", fontsize=9)
        ax.set_xlabel(param, fontsize=8)
        ax.set_ylabel("MAPE (%)" if col==0 else "", fontsize=8)
        ax.tick_params(axis='x', labelsize=8)
        ax.grid(True, lw=0.4, axis="y")

plt.tight_layout()
plt.savefig("grid_search_result.png", dpi=120, bbox_inches="tight")
plt.show()
print("\n✅ Plot tersimpan: grid_search_result.png")

# ── DOWNLOAD HASIL ────────────────────────────────────────────
df_t.to_csv("grid_search_trend.csv",  index=False)
df_s.to_csv("grid_search_season.csv", index=False)

from google.colab import files
files.download("grid_search_trend.csv")
files.download("grid_search_season.csv")
files.download("grid_search_result.png")
print("✅ Download selesai")
