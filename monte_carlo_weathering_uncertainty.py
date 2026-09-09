"""
monte_carlo_weathering_uncertainty.py

How confident can we actually be in the seafloor-weathering-vs-age model
built in Project 2? That deterministic model used single, fixed values
for parameters that are, in reality, uncertain: how much energy a basalt
dissolution reaction needs to get going (activation energy), how deep
the active reaction zone really is, and where exactly liquid-water
chemistry gives way to supercritical fluid.

This script treats those three parameters as uncertain, gives each a
realistic range instead of one fixed number, and runs the model
thousands of times with randomly sampled combinations -- the same logic
used to stress-test an algorithmic trading strategy across thousands of
simulated market scenarios rather than trusting a single backtest.
Where a trading Monte Carlo asks "how much could the strategy's return
vary given uncertain market conditions?", this one asks "how much could
the model's prediction vary given uncertain rock chemistry and Earth
structure?" -- same statistical machinery, different domain.

Author: Ayobami Isola
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# ---------------------------------------------------------------------
# 1. Fixed, well-constrained physical constants (NOT varied)
# ---------------------------------------------------------------------
# These are held fixed because they are comparatively well known from
# independent measurements. Only the three genuinely uncertain
# parameters below (Section 2) are treated as random variables. This
# keeps the analysis interpretable -- varying every possible parameter
# at once would make it impossible to tell which uncertainty actually
# matters (see Figure 4).

SECONDS_PER_MYR = 3.15576e13
KM_PER_CM_YR_MYR = 10.0
KAPPA = 8.0e-7          # thermal diffusivity, m^2/s
T_MANTLE = 1350.0        # deg C
T_SEAWATER = 2.0          # deg C
R_GAS = 8.314            # J/mol/K
FAST_HALF_RATE_CM_YR = 6.0   # for converting age -> distance on a fast ridge

# ---------------------------------------------------------------------
# 2. Uncertain parameters and their ranges
# ---------------------------------------------------------------------
# Each range is deliberately wide, reflecting genuine literature/model
# uncertainty rather than a narrow "best guess":
#
#   Activation energy (Ea):   40-70 kJ/mol
#       Reported activation energies for basalt/basaltic-glass
#       dissolution kinetics vary noticeably between studies and
#       experimental conditions; this range spans that spread.
#
#   Reaction-zone depth (z):  200-1000 m
#       How deep the actively-circulating, weathering-relevant part of
#       the upper crust extends is not a single fixed number -- it
#       depends on local permeability and fracturing, which we have not
#       modelled explicitly.
#
#   Reaction temperature cap: 300-450 deg C
#       The point at which circulating fluid stops behaving as a normal
#       liquid-water weathering agent depends on local pressure and
#       salinity, not a single universal number.

EA_RANGE_KJ = (40.0, 70.0)
Z_RANGE_M = (200.0, 1000.0)
CAP_RANGE_C = (300.0, 450.0)

N_RUNS = 5000
RANDOM_SEED = 42


# ---------------------------------------------------------------------
# 3. The model itself (same physics as Project 2, condensed)
# ---------------------------------------------------------------------

def crustal_temperature(age_myr, z, kappa=KAPPA, t_mantle=T_MANTLE, t_sw=T_SEAWATER):
    """Half-space cooling temperature (deg C) at depth z and age (Myr)."""
    age_s = np.maximum(np.asarray(age_myr, dtype=float) * SECONDS_PER_MYR, 1e-6)
    eta = z / (2.0 * np.sqrt(kappa * age_s))
    return t_sw + (t_mantle - t_sw) * erf(eta)


def reaction_temperature(age_myr, z, cap):
    """Temperature capped at the liquid-water reaction ceiling."""
    return np.minimum(crustal_temperature(age_myr, z), cap)


def normalised_weathering_rate(age_myr, z, ea_kj, cap):
    """Weathering rate relative to its value at the ridge axis (=1.0)."""
    t_k = reaction_temperature(age_myr, z, cap) + 273.15
    t_ridge_k = cap + 273.15
    ea = ea_kj * 1000.0
    rate = np.exp(-ea / (R_GAS * t_k))
    rate_ridge = np.exp(-ea / (R_GAS * t_ridge_k))
    return rate / rate_ridge


def age_at_fraction(age_myr, rate, frac):
    """First age (Myr) at which `rate` drops below `frac` of its axis value."""
    below = np.where(rate < frac)[0]
    return age_myr[below[0]] if len(below) else np.nan


def distance_from_age(age_myr, half_spreading_rate_cm_yr=FAST_HALF_RATE_CM_YR):
    return age_myr * half_spreading_rate_cm_yr * KM_PER_CM_YR_MYR


# ---------------------------------------------------------------------
# 4. Monte Carlo engine
# ---------------------------------------------------------------------

def run_monte_carlo(n_runs=N_RUNS, seed=RANDOM_SEED):
    """
    Draw n_runs random combinations of (Ea, z, cap), compute the
    resulting weathering-rate-vs-age curve for each, and record the
    age (and equivalent fast-ridge distance) at which each curve falls
    to 10% of its own ridge-axis value.

    This mirrors a trading Monte Carlo: instead of one backtest with one
    fixed parameter set, run many scenarios drawn from a realistic
    uncertainty range and look at the SPREAD of outcomes, not just one
    number.
    """
    rng = np.random.default_rng(seed)

    ea_samples = rng.uniform(*EA_RANGE_KJ, n_runs)
    z_samples = rng.uniform(*Z_RANGE_M, n_runs)
    cap_samples = rng.uniform(*CAP_RANGE_C, n_runs)

    age_grid = np.geomspace(1e-4, 5.0, 1500)

    age10_results = np.empty(n_runs)
    curves = np.empty((n_runs, age_grid.size))

    for i in range(n_runs):
        rate = normalised_weathering_rate(age_grid, z_samples[i], ea_samples[i], cap_samples[i])
        curves[i] = rate
        age10_results[i] = age_at_fraction(age_grid, rate, 0.10)

    distance10_results = distance_from_age(age10_results)

    return {
        "age_grid": age_grid,
        "curves": curves,
        "ea_samples": ea_samples,
        "z_samples": z_samples,
        "cap_samples": cap_samples,
        "age10_myr": age10_results,
        "distance10_km": distance10_results,
    }


# ---------------------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------------------

def make_plots(mc, output_dir="."):
    age_grid = mc["age_grid"]
    curves = mc["curves"]
    n_runs = curves.shape[0]

    # --- Figure 1: spaghetti plot of sampled curves -----------------------
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    sample_idx = np.random.default_rng(0).choice(n_runs, size=min(300, n_runs), replace=False)
    for i in sample_idx:
        ax1.semilogy(age_grid, curves[i], color="steelblue", alpha=0.05, linewidth=1)
    # Highlight the Project 2 "base case" for reference
    base_rate = normalised_weathering_rate(age_grid, z=500.0, ea_kj=55.0, cap=400.0)
    ax1.semilogy(age_grid, base_rate, color="firebrick", linewidth=2.2, label="Base case (Project 2 values)")
    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(1e-4, 1.2)
    ax1.set_xlabel("Crustal age (Myr)")
    ax1.set_ylabel("Relative weathering rate (log scale)")
    ax1.set_title(f"Step 2: {min(300, n_runs)} sampled parameter combinations\n(each a different plausible version of the model)")
    ax1.legend()
    ax1.grid(alpha=0.3, which="both")
    fig1.tight_layout()
    fig1.savefig(f"{output_dir}/fig1_spaghetti_curves.png", dpi=150)
    plt.close(fig1)

    # --- Figure 2: histogram of age at 10% ---------------------------------
    age10 = mc["age10_myr"]
    p5, p50, p95 = np.percentile(age10, [5, 50, 95])
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.hist(age10, bins=60, color="seagreen", alpha=0.85, edgecolor="white")
    ax2.axvline(p50, color="black", linewidth=1.5, label=f"Median = {p50:.3f} Myr")
    ax2.axvline(p5, color="gray", linewidth=1, linestyle="--", label=f"5th pct = {p5:.3f} Myr")
    ax2.axvline(p95, color="gray", linewidth=1, linestyle="--", label=f"95th pct = {p95:.3f} Myr")
    ax2.set_xlabel("Age at which weathering rate falls to 10% of axis value (Myr)")
    ax2.set_ylabel(f"Count (out of {n_runs} runs)")
    ax2.set_title("Step 3: Spread of the model's prediction\nacross parameter uncertainty")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(f"{output_dir}/fig2_histogram_age_10pct.png", dpi=150)
    plt.close(fig2)

    # --- Figure 3: same result, converted to distance on a fast ridge -----
    dist10 = mc["distance10_km"]
    dp5, dp50, dp95 = np.percentile(dist10, [5, 50, 95])
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    ax3.hist(dist10, bins=60, color="darkorange", alpha=0.85, edgecolor="white")
    ax3.axvline(dp50, color="black", linewidth=1.5, label=f"Median = {dp50:.1f} km")
    ax3.axvline(dp5, color="gray", linewidth=1, linestyle="--", label=f"5th pct = {dp5:.1f} km")
    ax3.axvline(dp95, color="gray", linewidth=1, linestyle="--", label=f"95th pct = {dp95:.1f} km")
    ax3.set_xlabel(f"Distance from a fast-spreading ridge axis ({FAST_HALF_RATE_CM_YR:.1f} cm/yr half-rate), km")
    ax3.set_ylabel(f"Count (out of {n_runs} runs)")
    ax3.set_title("Same result in map-scale units")
    ax3.legend()
    ax3.grid(alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(f"{output_dir}/fig3_histogram_distance_10pct.png", dpi=150)
    plt.close(fig3)

    # --- Figure 4: sensitivity -- which input drives the uncertainty? -----
    corr_ea = np.corrcoef(mc["ea_samples"], age10)[0, 1]
    corr_z = np.corrcoef(mc["z_samples"], age10)[0, 1]
    corr_cap = np.corrcoef(mc["cap_samples"], age10)[0, 1]

    labels = ["Activation energy\n(Ea, 40-70 kJ/mol)", "Reaction depth\n(z, 200-1000 m)",
              "Reaction temp. cap\n(300-450 °C)"]
    corrs = [corr_ea, corr_z, corr_cap]
    colors = ["#c0392b" if c < 0 else "#2874a6" for c in corrs]

    fig4, ax4 = plt.subplots(figsize=(7, 5))
    bars = ax4.barh(labels, corrs, color=colors)
    ax4.axvline(0, color="black", linewidth=0.8)
    ax4.set_xlabel("Correlation with predicted age-at-10% (Pearson r)")
    ax4.set_title("Step 4: Which uncertain input matters most?")
    ax4.set_xlim(-1, 1)
    for bar, c in zip(bars, corrs):
        ax4.text(c + (0.03 if c >= 0 else -0.03), bar.get_y() + bar.get_height() / 2,
                  f"{c:.2f}", va="center", ha="left" if c >= 0 else "right", fontsize=9)
    ax4.grid(alpha=0.3, axis="x")
    fig4.tight_layout()
    fig4.savefig(f"{output_dir}/fig4_sensitivity_tornado.png", dpi=150)
    plt.close(fig4)

    return {"corr_ea": corr_ea, "corr_z": corr_z, "corr_cap": corr_cap,
            "age10_p5": p5, "age10_p50": p50, "age10_p95": p95,
            "dist10_p5": dp5, "dist10_p50": dp50, "dist10_p95": dp95}


def print_summary(mc, stats):
    age10 = mc["age10_myr"]
    dist10 = mc["distance10_km"]
    print("=" * 66)
    print("MONTE CARLO WEATHERING UNCERTAINTY MODEL - SUMMARY")
    print("=" * 66)
    print(f"Runs: {len(age10)}")
    print(f"Ea sampled from:            {EA_RANGE_KJ[0]:.0f}-{EA_RANGE_KJ[1]:.0f} kJ/mol (uniform)")
    print(f"Reaction depth sampled from: {Z_RANGE_M[0]:.0f}-{Z_RANGE_M[1]:.0f} m (uniform)")
    print(f"Reaction temp cap sampled from: {CAP_RANGE_C[0]:.0f}-{CAP_RANGE_C[1]:.0f} deg C (uniform)")
    print("-" * 66)
    print(f"Age at 10% of axis weathering rate:")
    print(f"    median = {stats['age10_p50']:.3f} Myr   "
          f"90% range = [{stats['age10_p5']:.3f}, {stats['age10_p95']:.3f}] Myr")
    print(f"Equivalent distance on a fast-spreading ridge ({FAST_HALF_RATE_CM_YR:.1f} cm/yr half-rate):")
    print(f"    median = {stats['dist10_p50']:.1f} km   "
          f"90% range = [{stats['dist10_p5']:.1f}, {stats['dist10_p95']:.1f}] km")
    print("-" * 66)
    print("Sensitivity (correlation of each input with the predicted age-at-10%):")
    print(f"    Activation energy (Ea):     r = {stats['corr_ea']:+.2f}")
    print(f"    Reaction depth (z):         r = {stats['corr_z']:+.2f}")
    print(f"    Reaction temperature cap:   r = {stats['corr_cap']:+.2f}")
    dominant = max([("Ea", abs(stats["corr_ea"])), ("reaction depth", abs(stats["corr_z"])),
                     ("reaction temperature cap", abs(stats["corr_cap"]))], key=lambda x: x[1])[0]
    print(f"    -> {dominant} is the strongest driver of the model's uncertainty.")
    print("=" * 66)


if __name__ == "__main__":
    mc = run_monte_carlo()
    stats = make_plots(mc, output_dir=".")
    print_summary(mc, stats)
