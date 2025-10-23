#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import itertools
import pickle
import pandas as pd
import argparse
from pathlib import Path
from tqdm.auto import tqdm
from matplotlib.ticker import MaxNLocator

from pepit_helpers import find_cycle, worst_case_performance
from lyapunov import has_lyapunov, bisection
from plotting import contour_plot, line_plot, standard_textbox, set_matplotlib_style
from utils import dask_grid_compute, dask_parallel_map, nan_greater_than
from theory_helpers import optimal_step_size

# Constants
LABEL_SIZE = 20
TICK_SIZE = 20
L = 1.0
mus = [0.1, 0.25, 0.5]
grid_resolution = 200
epsilon = np.linspace(0.01, 0.99, grid_resolution)
etas = np.linspace(0.01, 2 / (L + mus[0]), grid_resolution)
epsilon_grid, etas_grid = np.meshgrid(epsilon, etas)
tol = 1e-6
upper_bound = 2

# Create figures directory if it doesn't exist
Path("figures").mkdir(exist_ok=True)

def compute_rhos_for_method(method, epsilon_grid, etas_grid, **pepit_kwargs):
    """Compute rho values for a given method."""
    rhos_full = {}
    rhos_fres = {}
    Ps_full = {}
    ps_full = {}
    deltas_grid = 1 - epsilon_grid
    
    for mu in tqdm(mus, total=len(mus)):
        _, P_full, p_full = dask_grid_compute(
            lambda delta, eta: bisection(0, upper_bound, tol, has_lyapunov, eta=eta, delta=delta, L=L, mu=mu, method=method),
            deltas_grid, etas_grid, show_progress=False
        )
        Ps_full[mu] = P_full
        ps_full[mu] = p_full
        
        P_shape = (4, 4) if method == "EF" else (3, 3)
        rhos_full[mu] = dask_grid_compute(
            lambda delta, eta: worst_case_performance(mu=mu, L=L, delta=delta, eta=eta, method=method, 
                                    P=P_full[(deltas_grid == delta) & (etas_grid == eta)].reshape(P_shape), 
                                    p=p_full[(deltas_grid == delta) & (etas_grid == eta)][0]),
            deltas_grid, etas_grid, show_progress=False
        )
        
        rhos_fres[mu] = dask_grid_compute(
            lambda delta, eta: worst_case_performance(mu=mu, L=L, delta=delta, eta=eta, method=method, **pepit_kwargs),
            deltas_grid, etas_grid, show_progress=False
        )

    # Save the data
    with open(f'data/rhos_{method}_full.pkl', 'wb') as f:
        pickle.dump(rhos_full, f)
    with open(f'data/rhos_{method}_simple.pkl', 'wb') as f:
        pickle.dump(rhos_fres, f)
    with open(f'data/Ps_{method}_full.pkl', 'wb') as f:
        pickle.dump(Ps_full, f)
    with open(f'data/ps_{method}_full.pkl', 'wb') as f:
        pickle.dump(ps_full, f)

    return rhos_full, rhos_fres, Ps_full, ps_full

def load_data_for_method(method):
    """Load saved data for a given method, generating it if missing."""
    data_files = [
        f'data/rhos_{method}_full.pkl',
        f'data/rhos_{method}_simple.pkl',
        f'data/Ps_{method}_full.pkl',
        f'data/ps_{method}_full.pkl'
    ]
    if not all(Path(f).exists() for f in data_files):
        print(f"Data for {method} not found. Computing...")
        compute_rhos_for_method(method, epsilon_grid, etas_grid, use_simplified=True)
        print(f"✓ Data for {method} computed and saved")
    with open(f'data/rhos_{method}_full.pkl', 'rb') as f:
        rhos_full = pickle.load(f)
    with open(f'data/rhos_{method}_simple.pkl', 'rb') as f:
        rhos_fres = pickle.load(f)
    with open(f'data/Ps_{method}_full.pkl', 'rb') as f:
        Ps_full = pickle.load(f)
    with open(f'data/ps_{method}_full.pkl', 'rb') as f:
        ps_full = pickle.load(f)
    return rhos_full, rhos_fres, Ps_full, ps_full


def _richtarik_cache_file(kappa, resolution, method):
    return Path('data') / f'richtarik_log_{method.lower()}_kappa_{kappa}_res_{resolution}.pkl'


def _load_or_compute_richtarik_data(kappa, epsilon_vals, eta_vals, method):
    """Load cached Richtárik vs simplified Lyapunov results or compute them."""
    resolution = len(epsilon_vals)
    cache_file = _richtarik_cache_file(kappa, resolution, method)

    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)

        cached_eps = np.asarray(data.get('epsilon_vals'))
        cached_etas = np.asarray(data.get('eta_vals'))
        if (
            cached_eps.shape == epsilon_vals.shape
            and cached_etas.shape == eta_vals.shape
            and np.allclose(cached_eps, epsilon_vals)
            and np.allclose(cached_etas, eta_vals)
        ):
            data['epsilon_vals'] = cached_eps
            data['eta_vals'] = cached_etas
            data['ratio'] = np.asarray(data.get('ratio'))
            return data

    epsilon_grid_local, eta_grid_local = np.meshgrid(epsilon_vals, eta_vals)

    def compute_rho(eps, eta, **kwargs):
        rho, _, _ = bisection(
            0, 1.0, 1e-12, has_lyapunov,
            eta=eta,
            delta=1 - eps,
            L=L,
            mu=L / kappa,
            method=method,
            **kwargs,
        )
        return np.nan if rho is None else rho

    rhos_simplified = dask_grid_compute(
        lambda eps, eta: compute_rho(eps, eta, use_simplified_lyapunov=True),
        epsilon_grid_local,
        eta_grid_local,
        show_progress=True,
    )
    rhos_richtarik = dask_grid_compute(
        lambda eps, eta: compute_rho(eps, eta, use_richtarik=True),
        epsilon_grid_local,
        eta_grid_local,
        show_progress=True,
    )

    ratio = np.full(rhos_simplified.shape, np.nan, dtype=float)
    stable_mask = (
        (rhos_simplified > 0)
        & (rhos_simplified < 1)
        & (rhos_richtarik > 0)
        & (rhos_richtarik < 1)
    )
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio[stable_mask] = (
            np.log(rhos_richtarik[stable_mask])
            / np.log(rhos_simplified[stable_mask])
        )

    data = {
        'kappa': kappa,
        'method': method,
        'epsilon_vals': epsilon_vals,
        'eta_vals': eta_vals,
        'ratio': ratio,
    }

    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)

    return data


def find_best_eta(delta, mu, L, resolution, method='EF', **kwargs):
    """Find the best eta value for given parameters."""
    eta_vals = np.linspace(0.0001, 2/(mu+L), resolution)
    rhos = dask_parallel_map(lambda eta: worst_case_performance(eta=eta, delta=delta, mu=mu, L=L, method=method, **kwargs), eta_vals)
    rhos = np.array([np.nan if rho is None else rho for rho in rhos])
    if np.all(np.isnan(rhos)) or np.all(rhos >= 1):
        return np.nan
    else:
        best_eta = eta_vals[np.nanargmin(rhos)]
        return best_eta


def _compute_cgd_superiority_data():
    print("Computing data for Figure 2...")

    rhos_cgd_optimal = {}
    rhos_ef_optimal = {}
    rhos_ef21_optimal = {}

    for mu in mus:
        rhos_cgd_optimal[mu] = []
        rhos_ef_optimal[mu] = []
        rhos_ef21_optimal[mu] = []
        for eps in tqdm(epsilon, total=len(epsilon)):
            eta_cgd = optimal_step_size(mu, L, 1 - eps, 'CGD')
            rho_cgd = worst_case_performance(mu=mu, L=L, delta=1 - eps, eta=eta_cgd, method='CGD', use_simplified=True)
            rhos_cgd_optimal[mu].append(rho_cgd)
            rho_ef = worst_case_performance(mu=mu, L=L, delta=1 - eps, eta=optimal_step_size(mu, L, 1 - eps, 'EF'), method='EF', use_simplified=True)
            rhos_ef_optimal[mu].append(rho_ef)
            rho_ef21 = worst_case_performance(mu=mu, L=L, delta=1 - eps, eta=optimal_step_size(mu, L, 1 - eps, 'EF21'), method='EF21', use_simplified=True)
            rhos_ef21_optimal[mu].append(rho_ef21)

    data = {
        'rhos_cgd_optimal': rhos_cgd_optimal,
        'rhos_ef_optimal': rhos_ef_optimal,
        'rhos_ef21_optimal': rhos_ef21_optimal
    }
    with open('data/cgd_superiority_data.pkl', 'wb') as f:
        pickle.dump(data, f)
    print("✓ Saved computed data for Figure 2")
    return data


def _load_cgd_superiority_data(context="Figure 4"):
    cache_path = Path('data/cgd_superiority_data.pkl')
    if cache_path.exists():
        with cache_path.open('rb') as f:
            print(f"✓ Loaded existing data for {context}")
            return pickle.load(f)
    return _compute_cgd_superiority_data()


def generate_performance_comparison():
    print("Generating Figure 1: performance_comparison.pdf ...")
    # Load or compute data for all methods
    methods = ['CGD', 'EF', 'EF21']
    data = []
    
    for method in methods:
        rhos_full, _, _, _ = load_data_for_method(method)
        cycles = {}
        for mu in mus:
            cycles[mu] = dask_grid_compute(
                lambda delta, eta: find_cycle(mu, L, method=method, delta=delta, eta=eta, n=2),
                1 - epsilon_grid, etas_grid, show_progress=False
            )
        data.append((epsilon_grid, etas_grid, nan_greater_than(rhos_full[mus[0]], 1), cycles[mus[0]]))
    
    contour_plot(data, center=False, cmap='Blues_r', levels=100,
                overlay_kwargs={'colors': ['darkred', 'white'], 'levels': [0, 1]},
                xlabel=r'$\epsilon$', ylabel=r'$\eta$',
                txtbox_kwargs=[standard_textbox('CGD', {'x': 0.75}), 
                              standard_textbox('EF', {'x': 0.82}), 
                              standard_textbox('EF21', {'x': 0.75})],
                increasing_colorbar=False,
                colorbar_label=r'$\rho$',
                label_size=LABEL_SIZE, tick_size=TICK_SIZE,
                figsize=(15, 4), dpi=150,
                return_plt=True,
                save_file='figures/performance_comparison.pdf')
    plt.close()

def generate_rate_comparison():
    print("Generating Figure 2: rate_comparison.pdf ...")

    set_matplotlib_style()

    # Use kappas derived from global mus and L to stay consistent
    kappas = [L / mu for mu in reversed(mus)]  # ascending order e.g., [2, 4, 10]
    epsilons = epsilon  # use global epsilon grid (0.01..0.99)

    def optimal_rate(kappa, eps):
        sqrt_eps = np.sqrt(eps)
        kappa_minus_1 = kappa - 1
        inner_sqrt = (
            (kappa_minus_1) ** 2
            + eps * (kappa_minus_1) ** 2
            + 2 * sqrt_eps * (1 + kappa * (6 + kappa))
        )
        numerator = (sqrt_eps - 1) * kappa_minus_1 * (-1 - sqrt_eps * kappa_minus_1 + kappa + np.sqrt(inner_sqrt))
        denominator = 2 * (1 + kappa) ** 2
        return sqrt_eps - numerator / denominator

    def richtarik_rate(kappa, eps):
        # Assumes L=1 globally as in this script
        theta = 1 - np.sqrt(eps)
        # Avoid divide-by-zero at eps -> 1 by relying on global eps range (0.01..0.99)
        beta = eps / (1 - np.sqrt(eps))
        gamma1 = theta / (2 * (1 / kappa))  # 1/mu with L=1
        gamma2 = 1 / (L * (1 + np.sqrt(2 * beta / theta)))
        gamma = np.minimum(gamma1, gamma2)
        return 1 - gamma / kappa

    fig, axes = plt.subplots(1, len(kappas), figsize=(15, 4), dpi=150, sharey=True)
    axes_array = np.atleast_1d(axes)

    for idx, (ax, kappa) in enumerate(zip(axes_array, kappas)):
        opt_rates = optimal_rate(kappa, epsilons)
        richtarik_rates = richtarik_rate(kappa, epsilons)

        # Plot both curves
        ax.plot(epsilons, opt_rates, label='Optimal rate', color='blue', linewidth=3)
        ax.plot(epsilons, richtarik_rates, label=r"Richt\'ar\'ik et al.", color='red', linewidth=3)

        # Labels and styling
        ax.set_xlabel(r'$\epsilon$', fontsize=LABEL_SIZE)
        if idx == 0:
            ax.set_ylabel(r'$\rho$', fontsize=LABEL_SIZE)
        ax.grid(True, which='both', linestyle=':', linewidth=0.7)
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=TICK_SIZE)

        # Kappa textbox
        ax.text(**standard_textbox(f'$\\kappa = {int(round(kappa))}$', {'x': 0.05, 'y': 0.15}), transform=ax.transAxes)

        # Legend in lower right
        ax.legend(fontsize=14, loc='lower right')

    fig.tight_layout()
    fig.savefig('figures/rate_comparison.pdf', bbox_inches='tight')
    plt.close(fig)

def generate_rate_log_complexity_fig12():
    print("Generating Figure 3: rate_log_complexity.pdf ...")

    set_matplotlib_style()

    kappas = [L / mu for mu in reversed(mus)]
    epsilons = epsilon

    def optimal_rate(kappa, eps):
        sqrt_eps = np.sqrt(eps)
        kappa_minus_1 = kappa - 1
        inner_sqrt = (
            (kappa_minus_1) ** 2
            + eps * (kappa_minus_1) ** 2
            + 2 * sqrt_eps * (1 + kappa * (6 + kappa))
        )
        numerator = (sqrt_eps - 1) * kappa_minus_1 * (
            -1 - sqrt_eps * kappa_minus_1 + kappa + np.sqrt(inner_sqrt)
        )
        denominator = 2 * (1 + kappa) ** 2
        return sqrt_eps - numerator / denominator

    def richtarik_rate(kappa, eps):
        theta = 1 - np.sqrt(eps)
        beta = eps / (1 - np.sqrt(eps))
        gamma1 = theta / (2 * (1 / kappa))
        gamma2 = 1 / (L * (1 + np.sqrt(2 * beta / theta)))
        gamma = np.minimum(gamma1, gamma2)
        return 1 - gamma / kappa

    ratio_by_kappa = {}
    global_min, global_max = np.inf, -np.inf
    for kappa in kappas:
        opt = optimal_rate(kappa, epsilons)
        rich = richtarik_rate(kappa, epsilons)

        ratio = np.full_like(opt, np.nan, dtype=float)
        mask = (opt > 0) & (opt < 1) & (rich > 0) & (rich < 1)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio[mask] = np.log(rich[mask]) / np.log(opt[mask])

        ratio_by_kappa[kappa] = ratio

        finite = np.isfinite(ratio)
        if finite.any():
            global_min = min(global_min, np.nanmin(ratio[finite]))
            global_max = max(global_max, np.nanmax(ratio[finite]))

    if not np.isfinite(global_min) or not np.isfinite(global_max):
        global_min, global_max = 1.0, 1.1

    span = global_max - global_min
    padding = 0.05 * span if span > 1e-6 else 0.02
    y_lower = max(0.0, global_min - padding)
    y_upper = global_max + padding

    fig, axes = plt.subplots(1, len(kappas), figsize=(15, 4), dpi=150, sharex=True, sharey=True)
    axes_array = np.atleast_1d(axes)

    # Use LaTeX with text accents for Richtárik in text mode
    ylabel = "$\\log \\rho_{\\text{Richt\\'ar\\'ik}} / \\log \\rho_{\\star}$"

    for ax, kappa in zip(axes_array, kappas):
        ratio = ratio_by_kappa[kappa]
        line_plot(
            [(epsilons, ratio, {"color": "blue", "linewidth": 3.0})],
            ax=ax,
            plt_legend=False,
            xlabel=r'$\epsilon$',
            ylabel=ylabel if ax is axes_array[0] else None,
            label_size=LABEL_SIZE,
            tick_size=TICK_SIZE,
            txtbox_kwargs=standard_textbox(f'$\\kappa = {int(round(kappa))}$', {"x": 0.95, "y": 0.15, "ha": "right", "va": "top"}),
            return_plt=True,
        )
        ax.set_ylim(y_lower, y_upper)
        # Mark global minimum, mirroring Figure 3's style
        finite_mask = np.isfinite(ratio)
        if finite_mask.any():
            min_idx = int(np.nanargmin(ratio))
            min_x = float(epsilons[min_idx])
            min_y = float(ratio[min_idx])

            x_min, x_max = ax.get_xlim()
            x_frac = 0.0 if x_max == x_min else (min_x - x_min) / (x_max - x_min)
            y_frac = 0.0 if y_upper == y_lower else (min_y - y_lower) / (y_upper - y_lower)

            ax.axhline(min_y, xmin=0.0, xmax=np.clip(x_frac, 0.0, 1.0), color='#009E73', linestyle='--', linewidth=2.0, alpha=0.8)
            ax.axvline(min_x, ymin=0.0, ymax=np.clip(y_frac, 0.0, 1.0), color='#009E73', linestyle='--', linewidth=2.0, alpha=0.8)
            ax.plot(min_x, min_y, marker='*', color='#009E73', markersize=15, zorder=5)
            # Show numeric label for kappa != 2 only
            if int(round(kappa)) != 2:
                ax.text(
                    -0.035,
                    np.clip(y_frac, 0.0, 1.0),
                    f'{min_y:.2f}',
                    color='#009E73',
                    fontsize=TICK_SIZE,
                    va='center',
                    ha='right',
                    transform=ax.transAxes,
                )

        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    fig.savefig('figures/rate_log_complexity.pdf', bbox_inches='tight')
    plt.close(fig)

def generate_cgd_vs_ef_rate_plot():
    print("Generating Figure 4: cgd_vs_ef_rate_comparison.pdf ...")

    set_matplotlib_style()

    data = _load_cgd_superiority_data("Figure 13")
    rhos_cgd_optimal = data['rhos_cgd_optimal']
    rhos_ef_optimal = data['rhos_ef_optimal']

    fig, axes = plt.subplots(1, len(mus), figsize=(15, 4), dpi=150, sharex=True, sharey=True)
    axes_array = np.atleast_1d(axes)

    for ax, mu in zip(axes_array, reversed(mus)):
        kappa = int(round(L / mu))
        curve_cgd = np.asarray(rhos_cgd_optimal[mu], dtype=float)
        curve_ef = np.asarray(rhos_ef_optimal[mu], dtype=float)

        # Mask out unstable values for visual clarity
        mask_cgd = (curve_cgd > 0) & (curve_cgd < 1)
        mask_ef = (curve_ef > 0) & (curve_ef < 1)
        curve_cgd_plot = np.where(mask_cgd, curve_cgd, np.nan)
        curve_ef_plot = np.where(mask_ef, curve_ef, np.nan)

        line_plot(
            [
                (epsilon, curve_cgd_plot, {'color': 'blue', 'label': 'CGD', 'linewidth': 3}),
                (epsilon, curve_ef_plot, {'color': 'red', 'label': 'EF/EF21', 'linewidth': 3}),
            ],
            ax=ax,
            plt_legend=True,
            xlabel=r'$\epsilon$',
            ylabel=r'$\rho$' if ax is axes_array[0] else None,
            label_size=LABEL_SIZE,
            tick_size=TICK_SIZE,
            txtbox_kwargs=standard_textbox(f'$\\kappa = {kappa}$', {'x': 0.05, 'y': 0.15}),
            return_plt=True,
        )

        # Always put legend at lower right
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='lower right', fontsize=LABEL_SIZE)

        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        ax.grid(True, which='both', linestyle=':', linewidth=0.7)

    fig.tight_layout()
    fig.savefig('figures/cgd_vs_ef_rate_comparison.pdf', bbox_inches='tight')
    plt.close(fig)

def generate_ef_equal_ef21():
    print("Generating Table 1: ef_equal_ef21.tex ...")
    rhos_ef_full, _, _, _ = load_data_for_method('EF')
    rhos_ef21_full, _, _, _ = load_data_for_method('EF21')
    
    diff_filtered = {}
    for mu in mus:
        mask = rhos_ef_full[mu] > 1
        diff_filtered[mu] = (rhos_ef_full[mu] - rhos_ef21_full[mu]).copy()
        diff_filtered[mu][mask] = np.nan
    
    max_diffs = {f'$\\kappa = {int(round(L / mu))}$': np.nanmax(diff_filtered[mu]) for mu in mus}
    df = pd.DataFrame([max_diffs], index=['Absolute error'])

    # Normalize column names to integer kappas and sort ascending (2, 4, 10)
    def _kappa_int(col):
        try:
            val_str = col.split('=')[1].strip().split('$')[0]
            return int(round(float(val_str)))
        except Exception:
            return 0
    rename_map = {c: f'$\\kappa = {_kappa_int(c)}$' for c in df.columns}
    df = df.rename(columns=rename_map)
    df = df.reindex(sorted(df.columns, key=_kappa_int), axis=1)

    body = df.to_latex(index=True, escape=False, float_format=lambda x: f'{x:.2e}')
    with open('figures/ef_equal_ef21.tex', 'w') as f:
        f.write(body)

def generate_tightness_table(method):
    table_map = {"CGD": "Table 2: rhos_cgd_tightness.tex ...", "EF": "Table 3: rhos_ef_tightness.tex ...", "EF21": "Table 4: rhos_ef21_tightness.tex ..."}
    print(f"Generating {table_map.get(method, method)}")
    
    # Try to load existing data
    try:
        with open(f'data/tightness_{method.lower()}_data.pkl', 'rb') as f:
            max_diffs = pickle.load(f)
            print(f"✓ Loaded existing data for {table_map.get(method, method)}")
    except FileNotFoundError:
        print(f"Computing data for {table_map.get(method, method)}...")
        rhos_full, rhos_simple, _, _ = load_data_for_method(method)
        
        max_diffs = {}
        for mu in reversed(mus):  # Using reversed(mus) to get kappa in increasing order
            if method == "CGD":
                # For CGD, compute over entire grid as before
                mask = rhos_full[mu] > 1
                diff_filtered = (rhos_full[mu] - rhos_simple[mu]).copy()
                diff_filtered[mask] = np.nan
                max_diffs[f'$\\kappa = {int(round(L / mu))}$'] = np.nanmax(diff_filtered)
            else:
                # For EF and EF21, compute only at optimal step sizes
                diffs = []
                for eps in epsilon:
                    delta = 1 - eps
                    eta_opt = optimal_step_size(mu, L, delta, method)
                    # Find closest eta in grid
                    eta_idx = np.abs(etas - eta_opt).argmin()
                    eta = etas[eta_idx]
                    
                    # Get rho values at this point
                    rho_full = rhos_full[mu][eta_idx, np.where(epsilon == eps)[0][0]]
                    rho_simple = rhos_simple[mu][eta_idx, np.where(epsilon == eps)[0][0]]
                    
                    if rho_full <= 1:  # Only consider stable points
                        diffs.append(abs(rho_full - rho_simple))
                
                max_diffs[f'$\\kappa = {int(round(L / mu))}$'] = max(diffs) if diffs else np.nan
        
        # Save the computed data
        with open(f'data/tightness_{method.lower()}_data.pkl', 'wb') as f:
            pickle.dump(max_diffs, f)
        print(f"✓ Saved computed data for {table_map.get(method, method)}")
    
    df = pd.DataFrame([max_diffs], index=['Absolute error'])
    # Normalize column names to integer kappas (avoid 2.0, 4.0, 10.0) and sort by kappa
    def _kappa_int(col):
        try:
            val_str = col.split('=')[1].strip().split('$')[0]
            return int(round(float(val_str)))
        except Exception:
            return 0
    rename_map = {c: f'$\\kappa = {_kappa_int(c)}$' for c in df.columns}
    df = df.rename(columns=rename_map)
    df = df.reindex(sorted(df.columns, key=_kappa_int), axis=1)
    body = df.to_latex(index=True, escape=False, float_format=lambda x: f'{x:.2e}')
    with open(f'figures/rhos_{method.lower()}_tightness.tex', 'w') as f:
        f.write(body)

def generate_cgd_superiority():
    print("Generating Figure 4: cgd_superiority_mu_{mu}.pdf ...")

    data = _load_cgd_superiority_data("Figure 2")
    rhos_cgd_optimal = data['rhos_cgd_optimal']
    rhos_ef_optimal = data['rhos_ef_optimal']
    rhos_ef21_optimal = data['rhos_ef21_optimal']
    
    # Generate plots
    fig, axes = plt.subplots(1, len(mus), figsize=(15, 4), dpi=150, sharex=True)
    for mu, ax in zip(mus, axes):
        line_plot([(epsilon, rhos_cgd_optimal[mu], {'color': 'blue', 'label': 'CGD'}), 
                  (epsilon, rhos_ef_optimal[mu], {'color': 'red', 'label': 'EF'}),
                  (epsilon, rhos_ef21_optimal[mu], {'color': 'green', 'label': 'EF21', 'linestyle': '--'})], 
                  ax=ax,
                  txtbox_kwargs=standard_textbox(f'$\\kappa = {int(round(L / mu))}$', {'x': 0.65, 'y': 0.15}),
                  plt_legend=True, xlabel=r'$\epsilon$', ylabel=r'$\rho$', 
                  label_size=LABEL_SIZE, tick_size=TICK_SIZE, return_plt=True,
                  save_file=f'figures/cgd_superiority_mu_{mu}.pdf')
    plt.close()

def generate_performance_plot(method):
    fig_map = {"CGD": "Figure 6: performance_cgd.pdf ...", "EF": "Figure 7: performance_ef.pdf ...", "EF21": "Figure 8: performance_ef21.pdf ..."}
    print(f"Generating {fig_map.get(method, method)}")
    rhos_full, _, _, _ = load_data_for_method(method)
    cycles = {}
    for mu in mus:
        cycles[mu] = dask_grid_compute(
            lambda delta, eta: find_cycle(mu, L, method=method, delta=delta, eta=eta, n=2),
            1 - epsilon_grid, etas_grid, show_progress=False
        )
    
    rhos_clean = {}
    for mu in mus:
        mask = rhos_full[mu] > 1
        rhos_clean[mu] = rhos_full[mu].copy()
        rhos_clean[mu][mask] = np.nan
    
    vmin = np.nanmin(rhos_clean[mus[-1]])
    vmax = np.nanmax(rhos_clean[mus[0]])
    
    lineplot_data = None
    if method in ['EF', 'EF21']:
        lineplot_data = [(epsilon, optimal_step_size(mu, L, 1 - epsilon, method=method), 
                         {'color': 'blue', 'label': r'$\eta_{\star}$'}) for mu in reversed(mus)]
    
    overlay_kwargs = {'colors': ['darkred', 'white'], 'levels': [0, 1]}
    contour_plot([(epsilon_grid, etas_grid, rhos_clean[mu], cycles[mu]) for mu in reversed(mus)],
                lineplot_data=lineplot_data,
                vmin=vmin, vmax=vmax,
                levels=100,
                cmap='Blues_r', center=False,
                txtbox_kwargs=[standard_textbox(f'$\\kappa = {int(round(L / mu))}$', {'x': 0.90, 'ha': 'right'}) for mu in reversed(mus)],
                overlay_kwargs=overlay_kwargs,
                increasing_colorbar=False,
                colorbar_label=r'$\rho$',
                label_size=LABEL_SIZE, tick_size=TICK_SIZE,
                xlabel=r'$\epsilon$', ylabel=r'$\eta$',
                figsize=(15, 4), dpi=150,
                return_plt=True,
                save_file=f'figures/performance_{method.lower()}.pdf')
    plt.close()

def generate_multiple_iterations(method):
    fig_map = {"EF": "Figure 9: ef_multiple_iterations_delta_{delta}.pdf ...", "EF21": "Figure 10: ef21_multiple_iterations_delta_{delta}.pdf ..."}
    print(f"Generating {fig_map.get(method, method)}")
    n_iters = 10
    deltas = [0.25, 0.5, 0.75]
    rhos = {}
    
    for delta in tqdm(deltas):
        rhos[delta] = np.zeros(n_iters)
        for n in range(1, n_iters + 1):
            rho = worst_case_performance(mu=mus[0], L=L, delta=delta, 
                                      eta=optimal_step_size(mus[0], L, delta, method), 
                                      method=method, n_iterations=n, use_simplified=True)
            rhos[delta][n-1] = rho
    
    fig, axes = plt.subplots(1, len(deltas), figsize=(15, 4), dpi=150, sharex=True)
    for delta, ax in zip(deltas, axes):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        line_plot([(range(1, n_iters+1), rhos[delta], {'color': 'blue', 'label': 'PEPit'}), 
                  (range(1,n_iters+1), [rhos[delta][0]**i for i in range(1,n_iters+1)], 
                  {'color': 'red', 'label': '$\\rho^k$', 'linestyle': '--'})], 
                  ax=ax,
                  txtbox_kwargs=standard_textbox(f'$\\epsilon = {1 - delta}$', {'x': 0.05, 'y': 0.15}),
                  plt_legend=True, xlabel='$k$', ylabel=r'$\rho$', 
                  label_size=LABEL_SIZE, tick_size=TICK_SIZE, return_plt=True,
                  save_file=f'figures/{method.lower()}_multiple_iterations_delta_{delta}.pdf')
    plt.close()

def generate_best_etas():
    print("Generating Figure 11: best_etas_{mu}_{L}.pdf ...")
    gridsearch_resolution = 300
    Ls = [1, 5, 10]
    
    # Load or compute best etas
    try:
        with open('data/best_etas.pkl', 'rb') as f:
            best_etas = pickle.load(f)
    except FileNotFoundError:
        best_etas = {(mu, L): np.zeros(len(epsilon)) for mu in mus for L in Ls}
        combs = itertools.product(mus, Ls, enumerate(epsilon))
        for mu, L, (i, eps) in tqdm(combs, total=len(mus) * len(Ls) * len(epsilon)):
            best_etas[(mu, L)][i] = find_best_eta(1 - eps, mu, L, gridsearch_resolution, method='EF', use_simplified=True)
        with open('data/best_etas.pkl', 'wb') as f:
            pickle.dump(best_etas, f)
    
    # Generate plots
    fig, axes = plt.subplots(len(mus), len(Ls), figsize=(15, 10), dpi=150, sharex=True)
    for i, mu in enumerate(mus):
        for j, L in enumerate(Ls):
            ax = axes[i, j]
            data = [(epsilon, best_etas[(mu, L)], {'color': 'blue'}), 
                   (epsilon, optimal_step_size(mu, L, 1 - epsilon, 'EF'), 
                    {'color': 'red', 'linestyle': '--'})]
            props = dict(boxstyle='round', facecolor=(0.93, 0.93, 0.93), alpha=1.0, 
                        linewidth=0.5, edgecolor='gray')
            txtbox_args = {'x': 0.65, 'y': 0.92, 's': f'$L = {L}$\n$\\mu = {mu}$', 
                          'ha': 'left', 'va': 'top', 'bbox': props, 'fontsize': 20}
            line_plot(data, ax=ax, figsize=(5,4), dpi=150, txtbox_kwargs=txtbox_args,
                     xlabel=r'$\epsilon$', ylabel=r'$\eta$', 
                     label_size=LABEL_SIZE, tick_size=TICK_SIZE, return_plt=True,
                     save_file=f'figures/best_etas_{mu}_{L}.pdf')
    plt.close()

def generate_richtarik_log_complexity():
    print("Generating Figure 12: richtarik_log_comparison.pdf ...")

    method = 'EF21'
    kappas = [2, 4, 10]
    grid_resolution_local = 100
    epsilon_vals = np.linspace(0.01, 0.99, grid_resolution_local)

    contour_data = []
    lineplot_data = []
    txtbox_kwargs = []

    for kappa in kappas:
        mu = L / kappa
        eta_max = 1.5 / (L + mu)
        eta_vals = np.linspace(0.05, eta_max, grid_resolution_local)
        cache = _load_or_compute_richtarik_data(kappa, epsilon_vals, eta_vals, method)
        cached_eps = cache['epsilon_vals']
        cached_etas = cache['eta_vals']
        ratio = cache['ratio']

        epsilon_grid_local, eta_grid_local = np.meshgrid(cached_eps, cached_etas)

        contour_data.append((epsilon_grid_local, eta_grid_local, ratio))
        lineplot_data.append((
            cached_eps,
            optimal_step_size(mu, L, 1 - cached_eps, method=method),
            {'color': 'blue', 'label': r'$\\eta_{\star}$', 'linestyle': '--'},
        ))
        txtbox_kwargs.append(standard_textbox(f'$\\kappa = {kappa}$', {'x': 0.65}))

    contour_plot(
        contour_data,
        cmap='Reds_r',
        levels=100,
        center=False,
        lineplot_data=lineplot_data,
        txtbox_kwargs=txtbox_kwargs,
        increasing_colorbar=True,
        colorbar_label=r'$\log \rho^{\prime} / \log \rho_{\star}$',
        xlabel=r'$\epsilon$',
        ylabel=r'$\eta$',
        figsize=(15, 4),
        dpi=150,
        save_file='figures/richtarik_log_comparison.pdf',
        label_size=LABEL_SIZE,
        tick_size=TICK_SIZE,
        return_plt=True,
    )
    plt.close()


def generate_optimal_contraction_comparison():
    print("Generating Figure 5: complexity_cgd_vs_ef.pdf ...")

    data = _load_cgd_superiority_data("complexity_cgd_vs_ef")
    rhos_cgd_optimal = data['rhos_cgd_optimal']
    rhos_ef_optimal = data['rhos_ef_optimal']

    ratio_data = {}
    peak_points = {}
    global_min = np.inf
    global_max = -np.inf

    for mu in mus:
        rho_cgd = np.asarray(rhos_cgd_optimal[mu], dtype=float)
        rho_ef = np.asarray(rhos_ef_optimal[mu], dtype=float)

        ratio = np.full_like(rho_cgd, np.nan, dtype=float)
        valid_mask = (rho_cgd > 0) & (rho_cgd < 1) & (rho_ef > 0) & (rho_ef < 1)
        ratio[valid_mask] = np.log(rho_ef[valid_mask]) / np.log(rho_cgd[valid_mask])

        ratio_data[mu] = ratio

        finite_mask = np.isfinite(ratio)
        if finite_mask.any():
            finite_vals = ratio[finite_mask]
            global_min = min(global_min, np.nanmin(finite_vals))
            global_max = max(global_max, np.nanmax(finite_vals))

            peak_idx = np.nanargmin(ratio)
            peak_points[mu] = (float(epsilon[peak_idx]), float(ratio[peak_idx]))
        else:
            peak_points[mu] = None

    if not np.isfinite(global_min) or not np.isfinite(global_max):
        global_min, global_max = 1.0, 1.1

    span = global_max - global_min
    padding = 0.05 * span if span > 1e-6 else 0.02
    y_lower = max(0.0, global_min - padding)
    y_upper = global_max + padding

    set_matplotlib_style()
    fig, axes = plt.subplots(1, len(mus), figsize=(15, 4), dpi=150, sharex=True, sharey=True)
    axes_array = np.atleast_1d(axes)

    curve_label = r'$\log \rho_{\mathrm{EF}} / \log \rho_{\mathrm{CGD}}$'

    for ax, mu in zip(axes_array, reversed(mus)):
        ratio = ratio_data[mu]

        line_plot(
            [(epsilon, ratio, {'color': 'blue', 'label': curve_label, 'linewidth': 3.0})],
            ax=ax,
            plt_legend=False,
            xlabel=r'$\epsilon$',
            ylabel=curve_label if ax is axes_array[0] else None,
            label_size=LABEL_SIZE,
            tick_size=TICK_SIZE,
            txtbox_kwargs=standard_textbox(
                f'$\\kappa = {int(round(L / mu))}$',
                {'x': 0.95, 'y': 0.95, 'ha': 'right', 'va': 'top'}
            ),
            return_plt=True,
        )

        ax.set_ylim(y_lower, y_upper)

        x_min, x_max = ax.get_xlim()
        peak = peak_points[mu]

        if peak is not None:
            peak_x, peak_y = peak

            x_frac = 0.0 if x_max == x_min else (peak_x - x_min) / (x_max - x_min)
            y_frac = 0.0 if y_upper == y_lower else (peak_y - y_lower) / (y_upper - y_lower)

            ax.axhline(
                peak_y,
                xmin=0.0,
                xmax=np.clip(x_frac, 0.0, 1.0),
                color='#009E73',
                linestyle='--',
                linewidth=2.0,
                alpha=0.8,
            )
            ax.axvline(
                peak_x,
                ymin=0.0,
                ymax=np.clip(y_frac, 0.0, 1.0),
                color='#009E73',
                linestyle='--',
                linewidth=2.0,
                alpha=0.8,
            )

            ax.text(
                -0.035,
                np.clip(y_frac, 0.0, 1.0),
                f'{peak_y:.2f}',
                color='#009E73',
                fontsize=TICK_SIZE,
                va='center',
                ha='right',
                transform=ax.transAxes,
            )

            ax.plot(peak_x, peak_y, marker='*', color='#009E73', markersize=15, zorder=5)

        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    fig.savefig('figures/complexity_cgd_vs_ef.pdf', bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Generate figures and tables for NeurIPS paper')
    parser.add_argument('experiment', type=str, help='Name of the experiment to run or "all" to run all experiments')
    args = parser.parse_args()

    # Create data directory if it doesn't exist
    Path("data").mkdir(exist_ok=True)

    # Map experiment names to functions
    experiment_functions = {
        "Figure 1": generate_performance_comparison,
        "Table 1": generate_ef_equal_ef21,
        "Table 2": lambda: generate_tightness_table("CGD"),
        "Figure 2": generate_rate_comparison,
        "Figure 3": generate_rate_log_complexity_fig12,
        "Figure 4": generate_cgd_vs_ef_rate_plot,
        "Figure 5": generate_optimal_contraction_comparison,
        "Figure 6": lambda: generate_performance_plot("CGD"),
        "Figure 7": lambda: generate_performance_plot("EF"),
        "Figure 8": lambda: generate_performance_plot("EF21"),
        "Table 3": lambda: generate_tightness_table("EF"),
        "Table 4": lambda: generate_tightness_table("EF21"),
        "Figure 9": lambda: generate_multiple_iterations("EF"),
        "Figure 10": lambda: generate_multiple_iterations("EF21"),
        "Figure 11": generate_best_etas,
        "Figure 12": generate_richtarik_log_complexity,
    }

    if args.experiment == "all":
        print("Running all experiments...")
        for exp_name, exp_func in experiment_functions.items():
            exp_func()
            print(f"✓ Successfully generated {exp_name}")
        print("\nAll experiments completed!")
    elif args.experiment not in experiment_functions:
        print(f"Error: Unknown experiment '{args.experiment}'")
        print("Available experiments:")
        for exp in experiment_functions.keys():
            print(f"  - {exp}")
        print("  - all (runs all experiments)")
        return
    else:
        # Run the selected experiment
        experiment_functions[args.experiment]()

if __name__ == "__main__":
    main()
