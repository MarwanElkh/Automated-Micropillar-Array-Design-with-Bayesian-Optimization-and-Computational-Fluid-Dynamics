import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.optimize import curve_fit
import os

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
in_path  = r"C:\Users\User\File\run3.xlsx"
out_dir  = r"C:\Users\User\File\vanDeemter1D"
os.makedirs(out_dir, exist_ok=True)
df = pd.read_excel(in_path)
def model(x, A, B, C, exp):
    return A + B/x + C*(x**exp)
p0 = [0.01, 0.001, 1, 1.0]
lb = [0.0001, 0.00001, 0.01, 0.99]
ub = [1.00,   0.10,   100.0, 1.01]
bounds = (lb, ub)

for vorm, g in df.groupby("vorm"):
    H_um = (1e6) * g["H(m)"].to_numpy()
    v = g["v_x,av(m/s)"].to_numpy()
    popt, pcov = curve_fit(model, v, H_um, p0=p0, bounds=bounds)
    A, B, C, exp = popt
    y_pred = model(x, A, B, C, exp)
    residuals = H_um - y_pred
    rmse = np.sqrt(np.mean(residuals**2))
    fig, ax = plt.subplots(figsize=(10, 6.8))
    ax.minorticks_on()
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.scatter(v, H_um, c="red", s=70, label="Data")
    x_fit_min = v.min()
    x_fit_max = 0.05
    x_fit = np.linspace(x_fit_min, x_fit_max, 1000)
    y_fit = model(x_fit, A, B, C, exp)
    ax.plot(x_fit, y_fit, c="blue", label="Fit")
    eq = rf"$H(\langle v_x \rangle_m) = {A:.3g} + {B:.3g}/\langle v_x \rangle_m + {C:.3g}\,\langle v_x \rangle_m^{{{exp:.3g}}}$"
    ax.text(0.15, 0.95, eq, transform=ax.transAxes, fontsize=20, va='top', bbox=dict(fc="white", alpha=0.7))
    rmse_text = rf"$\mathrm{{RMSE}} = {rmse:.4f}$"
    ax.text(0.15, 0.83, rmse_text, transform=ax.transAxes, fontsize=20, va='top', bbox=dict(fc="white", alpha=0.7))
    ax.set_xlabel(r'$\langle v_x \rangle_m\ \mathrm{(m/s)}$')
    ax.set_ylabel(r"$H\ (\mu\mathrm{m})$", rotation=0, labelpad=40)
    ax.set_xlim(0, 0.05)
    ax.set_ylim(0, 0.6)
    for label in ax.get_xticklabels():
        if label.get_text() in ('0', '0.00', '0.0'):
            label.set_visible(False)
    for label in ax.get_yticklabels():
        if label.get_text() in ('0', '0.00', '0.0'):
            label.set_visible(False)
    plt.tight_layout()
    out_png = os.path.join(out_dir, f"Deemter_{str(vorm)}.png")
    plt.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.close(fig)


