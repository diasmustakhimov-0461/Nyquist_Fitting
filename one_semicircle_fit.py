
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import os

# ============================================================
# ONE-SEMICIRCLE NYQUIST FITTING
# Equivalent circuit: Rs - (Rct || Cdl)
# ============================================================

# ------------------------------------------------------------
# 1. DATA FILE
# ------------------------------------------------------------
# Change this to your actual CSV filename.
filename = "LDH-M2-6-18h-DS.csv"
# ------------------------------------------------------------
# 2. READ DATA
# ------------------------------------------------------------
if not os.path.exists(filename):
    raise FileNotFoundError(
        f"File not found: {filename}\n"
        "Please place the CSV file in the same folder as this script."
    )

df = pd.read_csv(
    filename,
    sep=";",
    skiprows=1,
    decimal=","
)

print("\nColumn names found in your file:")
print(df.columns.tolist())

# IMPORTANT:
# Change these three names to match your CSV columns.
frequency = df["Frequency (Hz)"].to_numpy(dtype=float)
Zre_exp = df["Zre (ohms)"].to_numpy(dtype=float)
Zim_exp = df["Zim (ohms)"].to_numpy(dtype=float)

# Remove rows containing invalid values
valid = np.isfinite(frequency) & np.isfinite(Zre_exp) & np.isfinite(Zim_exp)
frequency = frequency[valid]
Zre_exp = Zre_exp[valid]
Zim_exp = Zim_exp[valid]

# Sort by frequency
order = np.argsort(frequency)
frequency = frequency[order]
Zre_exp = Zre_exp[order]
Zim_exp = Zim_exp[order]

# ------------------------------------------------------------
# 3. IMPEDANCE MODEL
# ------------------------------------------------------------
# Rs + (Rct || Cdl)
def impedance_model(f, Rs, Rct, Cdl):
    omega = 2 * np.pi * f
    Z_parallel = Rct / (1 + 1j * omega * Rct * Cdl)
    return Rs + Z_parallel

# ------------------------------------------------------------
# 4. INITIAL GUESSES
# ------------------------------------------------------------
# Units:
# Rs, Rct: Ohm
# Cdl: Farad

Rs_guess = max(0.01, np.min(Zre_exp))
Rct_guess = max(1.0, np.max(Zre_exp) - np.min(Zre_exp))
Cdl_guess = 1e-4

p0 = np.log10([Rs_guess, Rct_guess, Cdl_guess])

# ------------------------------------------------------------
# 5. RESIDUAL FUNCTION
# ------------------------------------------------------------
# Parameters are fitted in log10 to keep them positive.
def residuals(log_params):
    Rs = 10 ** log_params[0]
    Rct = 10 ** log_params[1]
    Cdl = 10 ** log_params[2]

    Z_fit = impedance_model(frequency, Rs, Rct, Cdl)

    residual_re = Z_fit.real - Zre_exp
    residual_im = Z_fit.imag - Zim_exp

    return np.concatenate([residual_re, residual_im])

# ------------------------------------------------------------
# 6. OPTIMIZATION
# ------------------------------------------------------------
result = least_squares(
    residuals,
    p0,
    method="trf",
    max_nfev=100000,
    xtol=1e-12,
    ftol=1e-12,
    gtol=1e-12
)

# ------------------------------------------------------------
# 7. FITTED PARAMETERS
# ------------------------------------------------------------
Rs_fit = 10 ** result.x[0]
Rct_fit = 10 ** result.x[1]
Cdl_fit = 10 ** result.x[2]

tau = Rct_fit * Cdl_fit
fmax = 1 / (2 * np.pi * tau)

Z_fit_exp = impedance_model(
    frequency,
    Rs_fit,
    Rct_fit,
    Cdl_fit
)

# ------------------------------------------------------------
# 8. ERROR CALCULATIONS
# ------------------------------------------------------------
residual_complex = Z_fit_exp - (Zre_exp + 1j * Zim_exp)

RMSE = np.sqrt(
    np.mean(np.abs(residual_complex) ** 2)
)

ss_res = np.sum(np.abs(residual_complex) ** 2)
ss_tot = np.sum(
    np.abs((Zre_exp + 1j * Zim_exp) - np.mean(Zre_exp + 1j * Zim_exp)) ** 2
)

R_squared = 1 - ss_res / ss_tot

# ------------------------------------------------------------
# 9. PRINT RESULTS
# ------------------------------------------------------------
print("\n" + "=" * 55)
print("ONE-SEMICIRCLE FITTING RESULTS")
print("=" * 55)

print(f"Rs   = {Rs_fit:.6g} Ohm")
print(f"Rct  = {Rct_fit:.6g} Ohm")
print(f"Cdl  = {Cdl_fit:.6e} F")
print(f"Cdl  = {Cdl_fit * 1e6:.6g} microF")
print(f"tau  = {tau:.6e} s")
print(f"fmax = {fmax:.6g} Hz")
print(f"RMSE = {RMSE:.6g} Ohm")
print(f"R^2  = {R_squared:.6f}")
print(f"Successful optimization: {result.success}")
print("=" * 55)

# ------------------------------------------------------------
# 10. FITTED CURVE
# ------------------------------------------------------------
f_min = np.min(frequency)
f_max = np.max(frequency)

f_plot = np.logspace(
    np.log10(f_min),
    np.log10(f_max),
    2000
)

Z_plot = impedance_model(
    f_plot,
    Rs_fit,
    Rct_fit,
    Cdl_fit
)

# ------------------------------------------------------------
# 11. NYQUIST PLOT
# ------------------------------------------------------------
plt.figure(figsize=(8, 7))

plt.scatter(
    Zre_exp,
    -Zim_exp,
    s=55,
    facecolors="none",
    edgecolors="black",
    linewidths=1.2,
    label="Experimental data",
    zorder=2
)

plt.plot(
    Z_plot.real,
    -Z_plot.imag,
    linewidth=2.5,
    label="One-semicycle RC fit",
    zorder=1
)

plt.xlabel(r"$Z_{\mathrm{re}}\ (\Omega)$", fontsize=16)
plt.ylabel(r"$-Z_{\mathrm{im}}\ (\Omega)$", fontsize=16)

plt.title("Nyquist Plot: One-Semicircle Fitting", fontsize=16)
plt.legend(frameon=False, fontsize=12)
plt.grid(False)
plt.tight_layout()

# Save figure
plt.savefig("one_semicircle_fit.png", dpi=300, bbox_inches="tight")

plt.show()
