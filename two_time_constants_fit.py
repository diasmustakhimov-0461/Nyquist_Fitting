import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import least_squares


# ============================================================
# 1. DATA FILES
# ============================================================

filenames = [
    "LDH-M3-3-18h-DS (96 hours).csv",
    "LDH-M3-3-18h-DS (168 hours).csv",
    "LDH-M3-3-18h-DS (48 hours).csv"
]

# List for storing results from all files
all_results = []

# Folder for individual plots
plot_folder = "fit_plots_two_time_constants"
os.makedirs(plot_folder, exist_ok=True)


# ============================================================
# 2. TWO-TIME-CONSTANT MODEL
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
    Parameters are optimized in logarithmic/transformed form:

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
# 3. RESIDUALS
# ============================================================

def residuals(params, frequency, Zre_exp, Zim_exp):
    """
    Residuals include both real and imaginary parts.
    """

    Z_model = model_impedance(params, frequency)

    residual_real = Z_model.real - Zre_exp
    residual_imag = Z_model.imag - Zim_exp

    return np.concatenate([
        residual_real,
        residual_imag
    ])


# ============================================================
# 4. LOGIT TRANSFORMATION
# ============================================================

def logit(x):
    """
    Transforms a value between 0 and 1 into an unrestricted value.
    """
    return np.log(x / (1.0 - x))


# ============================================================
# 5. PROCESS ALL FILES
# ============================================================

for filename in filenames:

    print("\n" + "#" * 75)
    print(f"PROCESSING FILE: {filename}")
    print("#" * 75)

    # ========================================================
    # 5.1 READ DATA
    # ========================================================

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

    # ========================================================
    # 5.2 REMOVE INVALID VALUES
    # ========================================================

    mask = (
        np.isfinite(frequency)
        & np.isfinite(Zre)
        & np.isfinite(Zim)
        & (frequency > 0)
    )

    frequency = frequency[mask]
    Zre = Zre[mask]
    Zim = Zim[mask]

    if len(frequency) == 0:
        print(f"Warning: no valid data found in {filename}")
        continue

    # Nyquist convention:
    # experimental imaginary coordinate is -Z''
    y_exp = -Zim

    # ========================================================
    # 5.3 INITIAL PARAMETERS
    # ========================================================

    Rs_initial = max(np.min(Zre), 1e-3)

    R_total = max(
        np.max(Zre) - Rs_initial,
        100.0
    )

    R1_initial = R_total * 0.35
    R2_initial = R_total * 0.65

    Q1_initial = 5e-4
    Q2_initial = 5e-4

    n1_initial = 0.80
    n2_initial = 0.80

    initial_params = np.array([
        np.log(Rs_initial),
        np.log(R1_initial),
        np.log(Q1_initial),
        logit(n1_initial),
        np.log(R2_initial),
        np.log(Q2_initial),
        logit(n2_initial)
    ])

    # ========================================================
    # 5.4 FIT
    # ========================================================

    result = least_squares(
        residuals,
        initial_params,
        args=(frequency, Zre, Zim),
        max_nfev=100000
    )

    fitted_params = result.x

    # Extract fitted parameters
    Rs = np.exp(fitted_params[0])

    R1 = np.exp(fitted_params[1])
    Q1 = np.exp(fitted_params[2])
    n1 = 1.0 / (1.0 + np.exp(-fitted_params[3]))

    R2 = np.exp(fitted_params[4])
    Q2 = np.exp(fitted_params[5])
    n2 = 1.0 / (1.0 + np.exp(-fitted_params[6]))

    # ========================================================
    # 5.5 CALCULATE FIT QUALITY
    # ========================================================

    Z_fit = model_impedance(
        fitted_params,
        frequency
    )

    # Fit quality using both real and imaginary components
    observed = np.concatenate([
        Zre,
        Zim
    ])

    predicted = np.concatenate([
        Z_fit.real,
        Z_fit.imag
    ])

    ss_res = np.sum(
        (observed - predicted) ** 2
    )

    ss_tot = np.sum(
        (observed - np.mean(observed)) ** 2
    )

    if ss_tot > 0:
        R_squared = 1.0 - ss_res / ss_tot
    else:
        R_squared = np.nan

    RMSE = np.sqrt(
        np.mean(
            (observed - predicted) ** 2
        )
    )

    # ========================================================
    # 5.6 TIME CONSTANTS
    # ========================================================

    tau1 = (R1 * Q1) ** (1.0 / n1)
    tau2 = (R2 * Q2) ** (1.0 / n2)

    fmax1 = 1.0 / (2.0 * np.pi * tau1)
    fmax2 = 1.0 / (2.0 * np.pi * tau2)

    # ========================================================
    # 5.7 PRINT RESULTS
    # ========================================================

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

    # ========================================================
    # 5.8 PLOT
    # ========================================================

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

    # ========================================================
    # 5.9 SAVE INDIVIDUAL PLOT
    # ========================================================

    base_filename = os.path.splitext(filename)[0]

    plot_path = os.path.join(
        plot_folder,
        f"{base_filename}_fit.png"
    )

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Plot saved to: {plot_path}")

    # ========================================================
    # 5.10 STORE RESULTS
    # ========================================================

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

    all_results.append(results)


# ============================================================
# 6. SAVE COMBINED SUMMARY TABLE
# ============================================================

summary = pd.DataFrame(all_results)

summary.to_csv(
    "fit_results_two_time_constants.csv",
    index=False
)

print("\n" + "=" * 70)
print("FINISHED PROCESSING ALL FILES")
print("=" * 70)

print(
    "Results saved to: fit_results_two_time_constants.csv"
)

print(
    f"Individual plots saved in folder: {plot_folder}"
)

print("\nSummary table:")
print(summary.to_string(index=False))