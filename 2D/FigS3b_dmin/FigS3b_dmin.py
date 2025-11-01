import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.optimize import curve_fit
import os

# --- styling once (not in loop) ---
plt.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 26,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 22,
    'xtick.major.width': 1.4,
    'ytick.major.width': 1.4,
    'xtick.major.size': 7,
    'ytick.major.size': 7,
    'xtick.minor.width': 1.2,
    'ytick.minor.width': 1.2,
    'xtick.minor.size': 4,
    'ytick.minor.size': 4,
    'axes.linewidth': 1.4,
    'grid.linewidth': 1,
    'grid.alpha': 0.3,
    'text.usetex': False,
    'mathtext.fontset': 'stix',
    'font.family': 'STIXGeneral',
})

in_path  = r"C:\Users\Mathijs Born\Downloads\run3.xlsx"
out_dir  = r"C:\Users\Mathijs Born\OneDrive\Desktop\dmin_3"
os.makedirs(out_dir, exist_ok=True)
df = pd.read_excel(in_path)


import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

VORM_COL = "vorm"
Y_COL    = "H^2/kappa"
idx = df.groupby(VORM_COL)[Y_COL].idxmin()
min_h2_kappa = df.loc[idx.to_numpy()].copy()
KV_COL    = "kappa(m^2)"

A_COL    = "a(µm)"
B_COL    = "b(µm)"
W_COL    = "d(µm)"
L_COL    = "c(µm)"
a_data = np.array(min_h2_kappa[A_COL].to_numpy(float))
b_data = np.array(min_h2_kappa[B_COL].to_numpy(float))
W_data = np.array(min_h2_kappa[W_COL].to_numpy(float))
L_data = np.array(min_h2_kappa[L_COL].to_numpy(float))
Kv_data = np.array(min_h2_kappa[KV_COL].to_numpy(float))

import numpy as np
from scipy.optimize import minimize

def _ellipse_xy(t, a, b):
    # Paramétrisation standard du bord : x = b cos t, y = a sin t
    return np.array([b*np.cos(t), a*np.sin(t)])

def dmin_ellipse_diag_numeric(a, b, L, W, coarse=181, local_starts=6):
    """
    Distance minimale entre les bords de deux ellipses axis-alignées identiques
    (a,b), centrées en (0,0) et (L,W). Retourne un scalaire >= 0.
    Méthode : grille grossière (t,s) puis affinement local (Nelder-Mead).
    """
    # --- 1) grille grossière pour des bons points de départ
    T = np.linspace(0.0, 2*np.pi, coarse, endpoint=False)
    cosT, sinT = np.cos(T), np.sin(T)
    # points ellipse 1
    X1, Y1 = b*cosT, a*sinT
    # points ellipse 2 (décalée de (L,W))
    X2, Y2 = L + b*cosT, W + a*sinT

    # distances au carré pour toutes paires (t_i, s_j) via broadcasting
    # shape (coarse, coarse)
    DX = X1[:,None] - X2[None,:]
    DY = Y1[:,None] - Y2[None,:]
    D2 = DX*DX + DY*DY

    # indices des meilleurs candidats (plus petites distances)
    flat_idx = np.argpartition(D2.ravel(), local_starts)[:local_starts]
    ti, sj = np.unravel_index(flat_idx, D2.shape)
    seeds = [(T[i], T[j]) for i,j in zip(ti, sj)]

    # Ajoute aussi un seed "directionnel" raisonnable
    # (point de l'ellipse le plus proche de la direction du centre)
    # Ce n'est pas critique, mais ça aide parfois.
    if local_starts < 12:
        # direction du vecteur centre-centre
        ang = np.arctan2(W/a, L/b)
        seeds.append((ang, ang))

    # --- 2) affinement local (sans gradients)
    def obj(z):
        t, s = z
        p1 = _ellipse_xy(t, a, b)
        p2 = _ellipse_xy(s, a, b) + np.array([L, W])
        return np.sum((p1 - p2)**2)  # on minimise la distance^2 (plus lisse)

    best = np.inf
    for t0, s0 in seeds:
        res = minimize(obj, x0=np.array([t0, s0]),
                       method="Nelder-Mead",
                       options={"maxiter": 1000, "xatol": 1e-12, "fatol": 1e-12})
        if res.fun < best:
            best = res.fun

    dmin = float(np.sqrt(max(0.0, best)))  # clamp num. safety
    return dmin


dmin_data = np.array([
    dmin_ellipse_diag_numeric(a, b, L, W)
    for a, b, L, W in zip(a_data, b_data, L_data, W_data)
])


kv = Kv_data
dmin = dmin_data/W_data
Wb =2*(W_data-a_data)/W_data

y1 = np.array([])
y2 = np.array([])
x1 = np.array([])
x2 = np.array([])
for i in range(0,len(kv)):
    if float(Wb[i]) < float(dmin[i]):
        x1 = np.append(x1, Wb[i])
        y1 = np.append(y1, kv[i])
    else:
        x2 = np.append(x2, dmin[i])
        y2 = np.append(y2, kv[i])

fig, ax = plt.subplots(figsize=(10, 6.8))
ax.minorticks_on()
ax.yaxis.set_minor_locator(AutoMinorLocator(4))
ax.xaxis.set_minor_locator(AutoMinorLocator(4))
ax.scatter(x1, y1, c="red", s=70, label="$d_{min,1}/W$")
ax.scatter(x2, y2, c="blue", s=70, label="$d_{min,2}/W$")
ax.set_xlabel(r'$d_{min}/W$')
ax.set_ylabel(r"$K_v(m^2)$", rotation=0, labelpad=40)
ax.set_xlim(min(x1.min(), x2.min()), 1.1*max(x1.max(), x2.max()))
ax.set_ylim(min(y1.min(), y2.min()), 1.1*max(y1.max(), y2.max()))
ax.legend()
for label in ax.get_xticklabels():
    if label.get_text() in ('0', '0.00', '0.0'):
        label.set_visible(False)
for label in ax.get_yticklabels():
    if label.get_text() in ('0', '0.00', '0.0'):
        label.set_visible(False)

plt.tight_layout()
out_png = os.path.join(out_dir, f"dminW.png")
plt.savefig(out_png, dpi=400, bbox_inches="tight")
plt.close(fig)

