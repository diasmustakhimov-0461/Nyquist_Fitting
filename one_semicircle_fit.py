import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import least_squares


# --------------------------------------------------
# Find all CSV files in the repository
# --------------------------------------------------

csv_files = [
    f for f in glob.glob("*.csv")
    if not os.path.basename(f).startswith("fit_results")
]

if len(csv_files) == 0:
    raise FileNotFoundError(
        "No CSV files found. Please upload your separate CSV files."
    )

print("CSV files found:")
for f in csv_files:
    print("  ", f)


# --------------------------------------------------
# Model: Rs + (Rct || Cdl)
# --------------------------------------------------

def impedance_model(params, frequency):
    Rs, Rct, Cdl = np.exp(params)

    omega = 2 * np.pi * frequency

    Z_parallel = Rct / (1 + 1j * omega * Rct * Cdl)

    return Rs + Z_parallel


# --------------------------------------------------
# Fit one file
# --------------------------------------------------

def fit_one_file(filename):

    print("\n" + "=" * 60)
    print("Processing:", filename)

    df = pd.read_csv(
        filename,
        sep=";",
        skiprows=1,
        decimal=","
    )

    print("Columns found:")
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

    valid = (
        np.isfinite(frequency)
        & np.isfinite(Zre)
        & np.isfinite(Zim)
        & (frequency > 0)
    )

    frequency = frequency[valid]
    Zre = Zre[valid]
    Zim = Zim[valid]

    order = np.argsort(frequency)[::-1]

    frequency = frequency[order]
    Zre = Zre[order]
    Zim = Zim[order]

    Z_exp = Zre + 1j * Zim

    def residuals(params):
        Z_fit = impedance_model(params, frequency)

        return np.concatenate([
            Z_fit.real - Z_exp.real,
            Z_fit.imag - Z_exp.imag
        ])

    Rs_initial = max(0.1, np.min(Zre))
    Rct_initial = max(1.0, np.max(Zre) - np.min(Zre))
    Cdl_initial = 1e-5

    initial_params = np.log([
        Rs_initial,
        Rct_initial,
        Cdl_initial
    ])

    result = least_squares(
        residuals,
        initial_params,
        max_nfev=10000
    )

    Rs, Rct, Cdl = np.exp(result.x)

    Z_fit = impedance_model(result.x, frequency)

    tau = Rct * Cdl
    fmax = 1 / (2 * np.pi * tau)

    rmse = np.sqrt(
        np.mean(np.abs(Z_fit - Z_exp) ** 2)
    )

    ss_res = np.sum(np.abs(Z_exp - Z_fit) ** 2)
    ss_tot = np.sum(
        np.abs(Z_exp - np.mean(Z_exp)) ** 2
    )

    r_squared = 1 - ss_res / ss_tot

    print(f"Rs      = {Rs:.6g} ohm")
    print(f"Rct     = {Rct:.6g} ohm")
    print(f"Cdl     = {Cdl:.6g} F")
    print(f"tau     = {tau:.6g} s")
    print(f"fmax    = {fmax:.6g} Hz")
    print(f"RMSE    = {rmse:.6g} ohm")
    print(f"R²      = {r_squared:.6f}")
    print("Fit success:", result.success)

    os.makedirs("fit_plots", exist_ok=True)

    plot_name = os.path.splitext(
        os.path.basename(filename)
    )[0] + "_fit.png"

    plot_path = os.path.join(
        "fit_plots",
        plot_name
    )

    plt.figure(figsize=(7, 5))

    plt.scatter(
        Zre,
        -Zim,
        label="Experimental data"
    )

    plt.plot(
        Z_fit.real,
        -Z_fit.imag,
        label="Single-semicircle fit"
    )

    plt.xlabel("Z′ (ohm)")
    plt.ylabel("−Z″ (ohm)")
    plt.title(os.path.basename(filename))
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    return {
        "file": filename,
        "Rs_ohm": Rs,
        "Rct_ohm": Rct,
        "Cdl_F": Cdl,
        "tau_s": tau,
        "fmax_Hz": fmax,
        "RMSE_ohm": rmse,
        "R_squared": r_squared,
        "fit_success": result.success
    }


# --------------------------------------------------
# Process all files
# --------------------------------------------------

results = []

for filename in csv_files:
    try:
        results.append(fit_one_file(filename))
    except Exception as error:
        print("\nERROR processing", filename)
        print(error)


# --------------------------------------------------
# Save summary
# --------------------------------------------------

if results:
    summary = pd.DataFrame(results)

    summary.to_csv(
        "fit_results_summary.csv",
        index=False
    )

    print("\n" + "=" * 60)
    print("Finished.")
    print("Results saved to: fit_results_summary.csv")
    print("Plots saved to: fit_plots/")
else:
    print("No files were successfully fitted.")