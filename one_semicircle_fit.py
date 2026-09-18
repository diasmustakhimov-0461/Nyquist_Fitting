import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import least_squares


# --------------------------------------------------
# Find all CSV files
# --------------------------------------------------

csv_files = [
    f for f in glob.glob("*.csv")
    if not os.path.basename(f).startswith("fit_results")
]

if len(csv_files) == 0:
    raise FileNotFoundError(
        "No CSV files found. Please upload your CSV files."
    )

print("CSV files found:")
for f in csv_files:
    print("  ", f)


# --------------------------------------------------
# Model:
# Rs + (Rct || CPE)
#
# Z_CPE = 1 / (Q * (j*w)^n)
# --------------------------------------------------

def impedance_model(params, frequency):

    Rs, Rct, Q, n = np.exp(params)

    omega = 2 * np.pi * frequency

    Z_cpe = 1 / (
        Q * (1j * omega) ** n
    )

    Z_parallel = 1 / (
        1 / Rct + 1 / Z_cpe
    )

    return Rs + Z_parallel


# --------------------------------------------------
# Fit one CSV file
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

    # ----------------------------------------------
    # Initial parameter estimates
    # ----------------------------------------------

    Rs_initial = max(0.1, np.min(Zre))
    Rct_initial = max(1.0, np.max(Zre) - np.min(Zre))

    # Initial CPE parameter
    Q_initial = 1e-4

    # n is constrained indirectly through bounds
    n_initial = 0.85

    initial_params = np.log([
        Rs_initial,
        Rct_initial,
        Q_initial,
        n_initial
    ])

    # ----------------------------------------------
    # Residual function
    # ----------------------------------------------

    def residuals(params):

        Z_fit = impedance_model(params, frequency)

        return np.concatenate([
            Z_fit.real - Z_exp.real,
            Z_fit.imag - Z_exp.imag
        ])

    # ----------------------------------------------
    # Bounds
    #
    # Rs > 0
    # Rct > 0
    # Q > 0
    # 0.1 <= n <= 1.0
    # ----------------------------------------------

    lower_bounds = np.log([
        1e-6,
        1e-6,
        1e-12,
        0.1
    ])

    upper_bounds = np.log([
        1e9,
        1e9,
        1e3,
        1.0
    ])

    result = least_squares(
        residuals,
        initial_params,
        bounds=(lower_bounds, upper_bounds),
        max_nfev=30000
    )

    Rs, Rct, Q, n = np.exp(result.x)

    Z_fit = impedance_model(result.x, frequency)

    # ----------------------------------------------
    # Calculate fit indicators
    # ----------------------------------------------

    tau_approx = (
        Rct * Q
    ) ** (1 / n)

    fmax_approx = 1 / (
        2 * np.pi * tau_approx
    )

    rmse = np.sqrt(
        np.mean(np.abs(Z_fit - Z_exp) ** 2)
    )

    ss_res = np.sum(
        np.abs(Z_exp - Z_fit) ** 2
    )

    ss_tot = np.sum(
        np.abs(Z_exp - np.mean(Z_exp)) ** 2
    )

    r_squared = 1 - ss_res / ss_tot

    print(f"Rs          = {Rs:.6g} ohm")
    print(f"Rct         = {Rct:.6g} ohm")
    print(f"Q           = {Q:.6g}")
    print(f"n           = {n:.6f}")
    print(f"tau_approx   = {tau_approx:.6g} s")
    print(f"fmax_approx  = {fmax_approx:.6g} Hz")
    print(f"RMSE        = {rmse:.6g} ohm")
    print(f"R²          = {r_squared:.6f}")
    print("Fit success:", result.success)

    # ----------------------------------------------
    # Save plot
    # ----------------------------------------------

    os.makedirs("fit_plots_cpe", exist_ok=True)

    plot_name = os.path.splitext(
        os.path.basename(filename)
    )[0] + "_CPE_fit.png"

    plot_path = os.path.join(
        "fit_plots_cpe",
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
        linewidth=2,
        label="Single semicircle + CPE fit"
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
        "Q_CPE": Q,
        "n_CPE": n,
        "tau_approx_s": tau_approx,
        "fmax_approx_Hz": fmax_approx,
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
        results.append(
            fit_one_file(filename)
        )

    except Exception as error:

        print("\nERROR processing", filename)
        print(error)


# --------------------------------------------------
# Save summary
# --------------------------------------------------

if results:

    summary = pd.DataFrame(results)

    summary.to_csv(
        "fit_results_summary_CPE.csv",
        index=False
    )

    print("\n" + "=" * 60)
    print("Finished.")
    print(
        "Results saved to: fit_results_summary_CPE.csv"
    )
    print(
        "Plots saved to: fit_plots_cpe/"
    )

else:

    print("No files were successfully fitted.")