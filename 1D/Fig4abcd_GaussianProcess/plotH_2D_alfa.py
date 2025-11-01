import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from matplotlib.ticker import AutoMinorLocator
import pandas as pd
import math
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

path = r"C:\Users\Mathijs Born\Downloads\run3.xlsx"
out  = r"C:\Users\Mathijs Born\Desktop\run3H.png"
vorm_col = "vorm"
E_col    = "H(m)"
alfa_col = "alfa"

df = pd.read_excel(path)

y_E = []
alfa_data = []

current_vorm = None
min_E = math.inf
min_alfa = None

prev_vorm = None
for _, row in df.iterrows():
    v = int(row[vorm_col])
    E = row[E_col]
    a = row[alfa_col]

    if current_vorm is None:
        # first row initializes state
        current_vorm = v
        min_E = E
        min_alfa = a
    elif v != current_vorm:
        # sanity: vorm must increase by 1
        if prev_vorm is not None and v != prev_vorm + 1:
            raise ValueError(f"vorm sequence broken: saw {prev_vorm} then {v}")

        # push result for the completed vorm
        y_E.append(min_E)
        alfa_data.append(min_alfa)

        # reset for new vorm
        current_vorm = v
        min_E = E
        min_alfa = a
    else:
        # same vorm: track minimum
        if E < min_E:
            min_E = E
            min_alfa = a

    prev_vorm = v

# push the last vorm result
if current_vorm is not None:
    y_E.append(min_E)
    alfa_data.append(min_alfa)
alfa_data = np.array(alfa_data)  
y_E = 10**6*np.array(y_E)  

alfa_min = min(alfa_data)
alfa_max = max(alfa_data)
y_E_min = min(y_E)
y_E_max = max(y_E)
num = 400
X_test = np.linspace(alfa_min, alfa_max, num)
model = GaussianProcessRegressor(kernel=1**2 * Matern(length_scale=1, nu=2.5) + WhiteKernel(noise_level=0.01), n_restarts_optimizer=2,normalize_y=True, random_state=0)
model.fit(alfa_data.reshape(-1, 1), y_E)
Z_pred, stdv_pred = model.predict(X_test.reshape(-1, 1), return_std=True)


fig, ax1 = plt.subplots(figsize=(10, 6.8))
ax1.minorticks_on()
ax1.yaxis.set_minor_locator(AutoMinorLocator(4))
ax1.xaxis.set_minor_locator(AutoMinorLocator(4))
ax1.fill_between(X_test, Z_pred - 2*stdv_pred, Z_pred + 2*stdv_pred, color='lightblue', alpha=0.5, label='95% confidence interval')
ax1.set_xlabel(r'$\alpha (^\circ)$')
ax1.set_ylabel(r'$H_{min} \mathrm{(µm)}$',rotation=0, labelpad=55)
ax1.plot(X_test, Z_pred, 'b', label='Posterior mean')
ax1.scatter(alfa_data, y_E, c='r', s=70, label='Training data')
ax1.set_xlim(alfa_min,alfa_max)
ax1.set_ylim(0.15,0.8)
def deg2rad(x):
    return np.radians(x)

def rad2deg(x):
    return np.degrees(x)
ax2 = ax1.secondary_xaxis('top', functions=(deg2rad, rad2deg))
ax2.set_xlabel(r'$\alpha \, (\mathrm{rad})$', labelpad=5)
ax2.minorticks_on()
ax2.xaxis.set_minor_locator(AutoMinorLocator(4))
plt.savefig(out, dpi=400)
plt.show()
plt.close()
