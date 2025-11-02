import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.optimize import curve_fit
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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

in_path  = r"C:\Users\User\File\run4.xlsx"
out_dir  = r"C:\Users\User\File\kv_v4"
os.makedirs(out_dir, exist_ok=True)
df = pd.read_excel(in_path)
for vorm, g in df.groupby("vorm"):
    y = np.array(g["v_x,av(m/s)"].to_numpy())
    x = -np.array(g["p(Pa/m)"].to_numpy())/np.array(g["µ(Ns/m^2)"].to_numpy())
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    y_pred = model.predict(x.reshape(-1, 1))
    r2 = r2_score(y, y_pred)
    a = model.coef_[0]
    b = model.intercept_
    fig, ax = plt.subplots(figsize=(10, 6.8))
    ax.minorticks_on()
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.scatter(x, y, c="red", s=70, label="Data")
    x_fit_min = x.min()
    x_fit_max = x.max()
    x_fit = np.linspace(x_fit_min, x_fit_max, 1000)
    y_fit = model.predict(x_fit.reshape(-1, 1))
    ax.plot(x_fit, y_fit, c="blue", label="Fit")
    equation = r"$\langle v_x \rangle_m =$" +rf"${a:.4g}$"+r"$ \frac{\Delta P}{\eta \, \Delta L}$"
    ax.text(0.15, 0.95, equation, transform=ax.transAxes, fontsize=20,
            va='top', bbox=dict(fc="white", alpha=0.7))
    R2 = rf"$\mathrm{{R^2}} = {r2:.4f}$"
    ax.text(0.15, 0.83, R2, transform=ax.transAxes, fontsize=20,
            va='top', bbox=dict(fc="white", alpha=0.7))
    ax.set_xlabel(r'$\frac{\Delta P}{\eta \, \Delta L}$')
    ax.set_ylabel(r"$\langle v_x \rangle_m$", rotation=0, labelpad=40)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y_fit.min(), y_fit.max())
    for label in ax.get_xticklabels():
        if label.get_text() in ('0', '0.00', '0.0'):
            label.set_visible(False)
    for label in ax.get_yticklabels():
        if label.get_text() in ('0', '0.00', '0.0'):
            label.set_visible(False)
    plt.tight_layout()
    out_png = os.path.join(out_dir, f"Kv_{vorm}.png")
    plt.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.close(fig)
