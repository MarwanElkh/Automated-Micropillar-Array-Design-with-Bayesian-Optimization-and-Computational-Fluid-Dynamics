# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from matplotlib.ticker import AutoMinorLocator


plt.rcParams.update({
    # --- Font sizes ---
    'font.size': 20,          # Base font size
    'axes.labelsize': 26,     # Axis label size
    'xtick.labelsize': 22,    # X tick labels
    'ytick.labelsize': 22,    # Y tick labels
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
    'text.usetex': False,          # <-- turn off external LaTeX
    'mathtext.fontset': 'stix',    # STIX math (Cambria-like)
    'font.family': 'STIXGeneral',  # serif text to match
})


PATH_EXCEL = r"C:\Users\Mathijs Born\Downloads\run4.xlsx"
OUT_2D_PNG = r"C:\Users\Mathijs Born\OneDrive\Desktop\run4H.png"
VORM_COL = "vorm"
A_COL    = "a(µm)"
B_COL    = "b(µm)"
Y_COL    = "H(m)"
AB_GRID_POINTS = 4000
R_MIN, R_MAX = 0.125, 7.849293563579278
A_MIN_DEG, A_MAX_DEG = 18.085231328304236, 168.6
R_N, A_N = 2000, 2000

def alpha_deg_from_ab(a, b):
    L = (np.pi * a * b) / 2.0
    alpha_rad = 2.0 * np.arctan(2.0 / L)
    return np.degrees(alpha_rad)

def r_from_ab(a, b):
    a = np.clip(a, 1e-15, None)
    return b / a

def invalid_mask(alpha_deg_grid, ratio_grid, e=0.5, W=2.0):
    alfa_rad = np.deg2rad(alpha_deg_grid)
    hoek = 0.5 * alfa_rad
    hoek = np.clip(hoek, 1e-9, np.pi/2 - 1e-9)  # stability
    L = W / np.tan(hoek)
    denom = (0.5 * np.pi * ratio_grid) / ((1.0 - e) * W)
    denom = np.clip(denom, 1e-15, None)
    a = np.sqrt(L / denom)
    b = ratio_grid * a
    return (L - b) < 0.0  # same rule as your snippet

def main():
    df = pd.read_excel(PATH_EXCEL)
    for col in [VORM_COL, A_COL, B_COL, Y_COL]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in Excel.")
    df = df.copy()
    df[A_COL] = pd.to_numeric(df[A_COL], errors="coerce")
    df[B_COL] = pd.to_numeric(df[B_COL], errors="coerce")
    df[Y_COL] = pd.to_numeric(df[Y_COL], errors="coerce")
    df = df.dropna(subset=[VORM_COL, A_COL, B_COL, Y_COL])
    if df.empty:
        raise ValueError("No valid rows after cleaning.")
    idx = df.groupby(VORM_COL)[Y_COL].idxmin()
    min_h2_kappa = df.loc[idx.to_numpy()].copy()
    a_data = min_h2_kappa[A_COL].to_numpy(float)
    b_data = min_h2_kappa[B_COL].to_numpy(float)
    y_data = (10**6)*min_h2_kappa[Y_COL].to_numpy(float)
    X_train = np.column_stack([a_data, b_data])
    kernel = C(1.0) * Matern(length_scale=[1, 1], nu=2.5) + WhiteKernel(noise_level=1.0)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True, random_state=0)
    gp.fit(X_train, y_data)
    a_lin = np.linspace(0.637, 1.6, AB_GRID_POINTS)
    b_lin = np.linspace(0.001, 5, AB_GRID_POINTS)
    Ag, Bg = np.meshgrid(a_lin, b_lin, indexing="xy")
    Z_ab = gp.predict(np.column_stack([Ag.ravel(), Bg.ravel()])).reshape(Ag.shape)
    R_samples = r_from_ab(Ag, Bg).ravel()
    A_samples = alpha_deg_from_ab(Ag, Bg).ravel()
    Z_samples = Z_ab.ravel()
    r_lin = np.linspace(R_MIN, R_MAX, R_N)
    a_lin = np.linspace(A_MIN_DEG, A_MAX_DEG, A_N)
    Rg, Ag_deg = np.meshgrid(r_lin, a_lin, indexing="xy")
    pts = np.column_stack([R_samples, A_samples])
    Zg = griddata(pts, Z_samples, (Rg, Ag_deg), method="linear")
    valid = (~invalid_mask(Ag_deg, Rg)) & np.isfinite(Zg)
    if not np.any(valid):
        raise RuntimeError("All grid cells are invalid (mask + NaNs). Shrink mask or adjust (r, alpha) range.")
    zmin = Zg[valid].min()
    zmax = Zg[valid].max()
    if zmin == zmax:
        zmax = zmin + 1e-12
    Zm = np.ma.array(Zg, mask=~valid)
    cmap = plt.get_cmap("coolwarm_r").copy()
    cmap.set_bad(color="white", alpha=1.0)
    levels = np.linspace(zmin, zmax, 40)
    clevels = np.linspace(zmin, zmax, 200)
    norm = Normalize(vmin=zmin, vmax=zmax)
    
    fig, ax1 = plt.subplots(figsize=(10, 6.8))
    ax1.minorticks_on()
    ax1.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax1.xaxis.set_minor_locator(AutoMinorLocator(4))
    cf = ax1.contourf( Ag_deg, Rg, Zm, levels=clevels, cmap=cmap,linewidths=10, norm=norm, antialiased=True)
    ax1.contour(Ag_deg, Rg, Zm, levels=levels, colors="black", linewidths=0.6, antialiased=True, norm = norm)
    r_tr = r_from_ab(a_data, b_data)
    a_tr = alpha_deg_from_ab(a_data, b_data)
    ax1.scatter(a_tr, r_tr, c=y_data, cmap=cmap, norm=norm, edgecolors="black", linewidths=0.6, s=70, label="data")
    ax1.plot([52.9,144],[1,1],c = "green",linewidth=2)
    ax1.scatter([70.3], [1], c=[0.221], cmap=cmap, norm=norm, edgecolors="green", linewidths=1.4, s=70, label="data")
    ax1.scatter([50], [1.71], c=[0.275], cmap=cmap, norm=norm, edgecolors="cyan", linewidths=0.6, s=70, label="data")
    ax1.set_ylabel(r"$b/a$",rotation=0, labelpad=20)
    ax1.set_xlabel(r'$\alpha (^\circ)$')
    ax1.set_ylim(R_MIN,R_MAX)
    ax1.set_xlim(A_MIN_DEG, A_MAX_DEG)
    cbar = plt.colorbar(cf, ax=ax1, pad=0.015)
    cbar.ax.xaxis.set_label_position('bottom')
    cbar.ax.xaxis.set_ticks_position('bottom')
    cbar.ax.set_xlabel(r'$H_{min} \mathrm{(µm)}$', labelpad=12)
    from matplotlib.ticker import FormatStrFormatter
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3g'))
    def deg2rad(x):
        return np.radians(x)
    def rad2deg(x):
        return np.degrees(x)
    ax2 = ax1.secondary_xaxis('top', functions=(deg2rad, rad2deg))
    ax2.set_xlabel(r'$\alpha \, (\mathrm{rad})$', labelpad=5)
    ax2.minorticks_on()
    ax2.xaxis.set_minor_locator(AutoMinorLocator(4))
    plt.tight_layout()
    plt.savefig(OUT_2D_PNG, dpi=400, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
