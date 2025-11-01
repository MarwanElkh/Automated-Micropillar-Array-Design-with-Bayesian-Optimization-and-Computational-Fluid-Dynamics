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

# --- IO ---
in_path  = r"C:\Users\Mathijs Born\Downloads\run4.xlsx"
out_dir  = r"C:\Users\Mathijs Born\OneDrive\Desktop\Deemter4"
os.makedirs(out_dir, exist_ok=True)

# --- read ---
df = pd.read_excel(in_path)
df = df.dropna(subset=["vorm", "H(m)", "v_x,av(m/s)"])

# convert H to µm
H_um = (1e6) * df["H(m)"].to_numpy()
v    = df["v_x,av(m/s)"].to_numpy()
vorms= df["vorm"].to_numpy()
df2  = pd.DataFrame({"vorm": vorms, "H_um": H_um, "v": v})

def model(x, A, B, C, exp):
    return A + B/x + C*(x**exp)

# initial guess + bounds
p0 = [0.01, 0.001, 1, 1.0]
lb = [0.0001, 0.00001, 0.01, 0.99]
ub = [1.00,   0.10,   100.0, 1.01]
bounds = (lb, ub)

results = []

for vorm, g in df2.groupby("vorm"):
    x = g["v"].to_numpy()
    y = g["H_um"].to_numpy()
    order = np.argsort(x)
    x = x[order]; y = y[order]

    # --- fit ---
    popt, pcov = curve_fit(model, x, y, p0=p0, bounds=bounds)
    A, B, C, exp = popt
    perr = np.sqrt(np.diag(pcov))  # parameter std devs

    # --- compute RMSE ---
    y_pred = model(x, A, B, C, exp)
    residuals = y - y_pred
    rmse = np.sqrt(np.mean(residuals**2))

    results.append(
        {"vorm": vorm, "A": A, "B": B, "C": C, "exp": exp,
         "A_se": perr[0], "B_se": perr[1], "C_se": perr[2], "exp_se": perr[3],
         "RMSE": rmse}
    )

    # --- plotting ---
    fig, ax = plt.subplots(figsize=(10, 6.8))
    ax.minorticks_on()
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))

    # data
    ax.scatter(x, y, c="red", s=70, label="Data")

    # smooth curve across observed range
    x_fit_min = max(x.min(), 1e-9)  # stay >0
    x_fit_max = 0.05
    x_fit = np.linspace(x_fit_min, x_fit_max, 1000)
    if np.isfinite(A):
        y_fit = model(x_fit, A, B, C, exp)
        ax.plot(x_fit, y_fit, c="blue", label="Fit")

    # equation text
    if np.isfinite(A):
        eq = rf"$H(\langle v_x \rangle_m) = {A:.3g} + {B:.3g}/\langle v_x \rangle_m + {C:.3g}\,\langle v_x \rangle_m^{{{exp:.3g}}}$"
    else:
        eq = "Fit failed"

    # add equation
    ax.text(0.15, 0.95, eq, transform=ax.transAxes, fontsize=20,
            va='top', bbox=dict(fc="white", alpha=0.7))

    # add RMSE below it
    rmse_text = rf"$\mathrm{{RMSE}} = {rmse:.4f}$"
    ax.text(0.15, 0.83, rmse_text, transform=ax.transAxes, fontsize=20,
            va='top', bbox=dict(fc="white", alpha=0.7))

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
    safe_vorm = str(vorm).replace("/", "_")
    out_png = os.path.join(out_dir, f"Deemter_{safe_vorm}.png")
    plt.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"[{vorm}] saved: {out_png}")

# save parameters + RMSE
pd.DataFrame(results).to_excel(os.path.join(out_dir, "fit_results.xlsx"), index=False)
print("Done.")
