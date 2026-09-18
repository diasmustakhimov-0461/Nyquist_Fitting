import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import least_squares


# ============================================================
# 1. DATA FILE
# ============================================================

filename = "LDH-M3-3-18h-DS (96 hours).csv"


# ============================================================
# 2. READ DATA
# ============================================================

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

print("\nColumns found in the file:")
print(df.columns.tolist())


frequency = pd.to_numeric(
    df["Frequency (Hz)"],
    errors="coerce"
).to_numpy()

Zre = pd.to_numeric(
    df["Zre (ohms)"],
    errors="coerce"
).to_numpy()

Zim = pd.to_numeric(
    df["Zim (ohms)"],
    errors="coerce"
).to_numpy()


# Remove invalid values
mask = (
    np.isfinite(frequency)
    & np.isfinite(Zre)
    & np.isfinite(Zim)
    & (frequency > 0)
)

frequency = frequency[mask]
Zre = Zre[mask]
Zim = Zim[mask]

# Nyquist convention:
# experimental imaginary coordinate is -Z''
y_exp = -Zim


# ============================================================
# 3. TWO-TIME-CONSTANT MODEL
#
# Circuit:
#
# Rs - (R1 || CPE1) - (R2 || CPE2)
#
# Z_CPE = 1 / [Q * (j*w)^n]
# ============================================================

def cpe_impedance(Q, n, omega):
    """
    Impedance of a constant phase element:
        Z_CPE = 1 / [Q * (j*w)^n]
    """
    return 1.0 / (Q * (1j * omega) ** n)


def model_impedance(params, frequency):
    """
    Parameters are optimized in logarithmic form:

    params[0] = log(Rs)
    params[1] = log(R1)
    params[2] = log(Q1)
    params[3] = transformed n1
    params[4] = log(R2)
    params[5] = log(Q2)
    params[6] = transformed n2
    """

    Rs = np.exp(params[0])
    R1 = np.exp(params[1])
    Q1 = np.exp(params[2])
    n1 = 1.0 / (1.0 + np.exp(-params[3]))

    R2 = np.exp(params[4])
    Q2 = np.exp(params[5])
    n2 = 1.0 / (1.0 + np.exp(-params[6]))

    omega = 2.0 * np.pi * frequency

    Z_cpe1 = cpe_impedance(Q1, n1, omega)
    Z_cpe2 = cpe_impedance(Q2, n2, omega)

    # Parallel R || CPE impedance
    Z_parallel_1 = 1.0 / (1.0 / R1 + 1.0 / Z_cpe1)
    Z_parallel_2 = 1.0 / (1.0 / R2 + 1.0 / Z_cpe2)

    # Series connection
    Z_total = Rs + Z_parallel_1 + Z_parallel_2

    return Z_total


# ============================================================
# 4. RESIDUALS
# ============================================================

def residuals(params, frequency, Zre_exp, Zim_exp):
    Z_model = model_impedance(params, frequency)

    # Fit both real and imaginary parts
    residual_real = Z_model.real - Zre_exp
    residual_imag = Z_model.imag - Zim_exp

    return np.concatenate([
        residual_real,
        residual_imag
    ])


# ============================================================
# 5. INITIAL PARAMETERS
# ============================================================

# Initial estimates
Rs_initial = max(np.min(Zre), 1e-3)
R_total = max(np.max(Zre) - Rs_initial, 100.0)

R1_initial = R_total * 0.35
R2_initial = R_total * 0.65

Q1_initial = 5e-4
Q2_initial = 5e-4

n1_initial = 0.80
n2_initial = 0.80


def logit(x):
    return np.log(x / (1.0 - x))


initial_params = np.array([
    np.log(Rs_initial),
    np.log(R1_initial),
    np.log(Q1_initial),
    logit(n1_initial),
    np.log(R2_initial),
    np.log(Q2_initial),
    logit(n2_initial)
])


# ============================================================
# 6. FIT
# ============================================================

result = least_squares(
    residuals,
    initial_params,
    args=(frequency, Zre, Zim),
    max_nfev=100000
)

fitted_params = result.x

Rs = np.exp(fitted_params[0])
R1 = np.exp(fitted_params[1])
Q1 = np.exp(fitted_params[2])
n1 = 1.0 / (1.0 + np.exp(-fitted_params[3]))

R2 = np.exp(fitted_params[4])
Q2 = np.exp(fitted_params[5])
n2 = 1.0 / (1.0 + np.exp(-fitted_params[6]))


# ============================================================
# 7. CALCULATE FIT QUALITY
# ============================================================

Z_fit = model_impedance(fitted_params, frequency)

# Fit quality using both real and imaginary components
observed = np.concatenate([Zre, Zim])
predicted = np.concatenate([Z_fit.real, Z_fit.imag])

ss_res = np.sum((observed - predicted) ** 2)
ss_tot = np.sum((observed - np.mean(observed)) ** 2)

R_squared = 1.0 - ss_res / ss_tot

RMSE = np.sqrt(
    np.mean((observed - predicted) ** 2)
)


# ============================================================
# 8. TIME CONSTANTS
# ============================================================

tau1 = (R1 * Q1) ** (1.0 / n1)
tau2 = (R2 * Q2) ** (1.0 / n2)

fmax1 = 1.0 / (2.0 * np.pi * tau1)
fmax2 = 1.0 / (2.0 * np.pi * tau2)


# ============================================================
# 9. PRINT RESULTS
# ============================================================

print("\n" + "=" * 65)
print("TWO-TIME-CONSTANT CPE FIT")
print("=" * 65)

print(f"File = {filename}")
print()
print(f"Rs   = {Rs:.6f} ohm")
print()
print("First time constant:")
print(f"R1   = {R1:.6f} ohm")
print(f"Q1   = {Q1:.8e}")
print(f"n1   = {n1:.6f}")
print(f"tau1 = {tau1:.6f} s")
print(f"fmax1 = {fmax1:.6f} Hz")
print()
print("Second time constant:")
print(f"R2   = {R2:.6f} ohm")
print(f"Q2   = {Q2:.8e}")
print(f"n2   = {n2:.6f}")
print(f"tau2 = {tau2:.6f} s")
print(f"fmax2 = {fmax2:.6f} Hz")
print()
print(f"RMSE = {RMSE:.6f} ohm")
print(f"R_squared = {R_squared:.6f}")
print(f"Fit success = {result.success}")
print(f"Optimization message = {result.message}")


# ============================================================
# 10. PLOT
# ============================================================

sort_idx = np.argsort(frequency)

plt.figure(figsize=(10, 7))

plt.scatter(
    Zre,
    y_exp,
    s=55,
    label="Experimental data"
)

plt.plot(
    Z_fit.real[sort_idx],
    -Z_fit.imag[sort_idx],
    linewidth=2.5,
    label="Two-time-constant CPE fit"
)

plt.xlabel("Z' (ohm)")
plt.ylabel("-Z'' (ohm)")
plt.title(filename)
plt.grid(True, alpha=0.5)
plt.legend()
plt.tight_layout()


# ============================================================
# 11. SAVE PLOT
# ============================================================

results = {
    "file": filename,

    # Series resistance
    "Rs_ohm": Rs,

    # First time constant
    "R1_ohm": R1,
    "Q1_CPE": Q1,
    "n1_CPE": n1,
    "tau1_s": tau1,
    "fmax1_Hz": fmax1,

    # Second time constant
    "R2_ohm": R2,
    "Q2_CPE": Q2,
    "n2_CPE": n2,
    "tau2_s": tau2,
    "fmax2_Hz": fmax2,

    # Fitting quality
    "RMSE_ohm": RMSE,
    "R_squared": R_squared,
    "fit_success": result.success
}

summary = pd.DataFrame([results])

summary.to_csv(
    "fit_results_two_time_constants.csv",
    index=False
)

print("\n" + "=" * 60)
print("Finished.")
print(
    "Results saved to: fit_results_two_time_constants.csv"
)
print(
    "Plot saved to: two_time_constants_fit.png"
)