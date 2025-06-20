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
from plotting import contour_plot, line_plot, standard_textbox
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

def generate_ef_equal_ef21():
    print("Generating Table 1: ef_equal_ef21.tex ...")
    rhos_ef_full, _, _, _ = load_data_for_method('EF')
    rhos_ef21_full, _, _, _ = load_data_for_method('EF21')
    
    diff_filtered = {}
    for mu in mus:
        mask = rhos_ef_full[mu] > 1
        diff_filtered[mu] = (rhos_ef_full[mu] - rhos_ef21_full[mu]).copy()
        diff_filtered[mu][mask] = np.nan
    
    max_diffs = {f'$\\kappa = {L / mu}$': np.nanmax(diff_filtered[mu]) for mu in mus}
    df = pd.DataFrame([max_diffs], index=['Absolute error'])
    
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
                max_diffs[f'$\\kappa = {L / mu}$'] = np.nanmax(diff_filtered)
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
                
                max_diffs[f'$\\kappa = {L / mu}$'] = max(diffs) if diffs else np.nan
        
        # Save the computed data
        with open(f'data/tightness_{method.lower()}_data.pkl', 'wb') as f:
            pickle.dump(max_diffs, f)
        print(f"✓ Saved computed data for {table_map.get(method, method)}")
    
    df = pd.DataFrame([max_diffs], index=['Absolute error'])
    # Sort by kappa value (L/mu)
    df = df.reindex(sorted(df.columns, key=lambda x: float(x.split('=')[1].strip().split('$')[0])), axis=1)
    body = df.to_latex(index=True, escape=False, float_format=lambda x: f'{x:.2e}')
    with open(f'figures/rhos_{method.lower()}_tightness.tex', 'w') as f:
        f.write(body)

def generate_cgd_superiority():
    print("Generating Figure 2: cgd_superiority_mu_{mu}.pdf ...")
    
    # Try to load existing data
    try:
        with open('data/cgd_superiority_data.pkl', 'rb') as f:
            data = pickle.load(f)
            best_etas_cgd = data['best_etas_cgd']
            rhos_cgd_optimal = data['rhos_cgd_optimal']
            rhos_ef_optimal = data['rhos_ef_optimal']
            rhos_ef21_optimal = data['rhos_ef21_optimal']
            print("✓ Loaded existing data for Figure 2")
    except FileNotFoundError:
        print("Computing data for Figure 2...")
        # Compute best etas for CGD
        best_etas_cgd = {}
        for mu in mus:
            best_etas_cgd[mu] = np.zeros_like(epsilon)
            for i, eps in enumerate(tqdm(epsilon, total=len(epsilon))):
                best_etas_cgd[mu][i] = find_best_eta(1 - eps, mu, L, 100, method='CGD', use_simplified=True)
        
        # Compute optimal rhos for all methods
        rhos_cgd_optimal = {}
        rhos_ef_optimal = {}
        rhos_ef21_optimal = {}
        
        for mu in mus:
            rhos_cgd_optimal[mu] = []
            rhos_ef_optimal[mu] = []
            rhos_ef21_optimal[mu] = []
            for i, eps in enumerate(tqdm(epsilon, total=len(epsilon))):
                rho_cgd = worst_case_performance(mu=mu, L=L, delta=1 - eps, eta=best_etas_cgd[mu][i], method='CGD', use_simplified=True)
                rhos_cgd_optimal[mu].append(rho_cgd)
                rho_ef = worst_case_performance(mu=mu, L=L, delta=1 - eps, eta=optimal_step_size(mu, L, 1 - eps, 'EF'), method='EF', use_simplified=True)
                rhos_ef_optimal[mu].append(rho_ef)
                rho_ef21 = worst_case_performance(mu=mu, L=L, delta=1 - eps, eta=optimal_step_size(mu, L, 1 - eps, 'EF21'), method='EF21', use_simplified=True)
                rhos_ef21_optimal[mu].append(rho_ef21)
        
        # Save the computed data
        data = {
            'best_etas_cgd': best_etas_cgd,
            'rhos_cgd_optimal': rhos_cgd_optimal,
            'rhos_ef_optimal': rhos_ef_optimal,
            'rhos_ef21_optimal': rhos_ef21_optimal
        }
        with open('data/cgd_superiority_data.pkl', 'wb') as f:
            pickle.dump(data, f)
        print("✓ Saved computed data for Figure 2")
    
    # Generate plots
    fig, axes = plt.subplots(1, len(mus), figsize=(15, 4), dpi=150, sharex=True)
    for mu, ax in zip(mus, axes):
        line_plot([(epsilon, rhos_cgd_optimal[mu], {'color': 'blue', 'label': 'CGD'}), 
                  (epsilon, rhos_ef_optimal[mu], {'color': 'red', 'label': 'EF'}),
                  (epsilon, rhos_ef21_optimal[mu], {'color': 'green', 'label': 'EF21', 'linestyle': '--'})], 
                  ax=ax,
                  txtbox_kwargs=standard_textbox(f'$\\kappa = {L / mu}$', {'x': 0.65, 'y': 0.15}),
                  plt_legend=True, xlabel=r'$\epsilon$', ylabel=r'$\rho$', 
                  label_size=LABEL_SIZE, tick_size=TICK_SIZE, return_plt=True,
                  save_file=f'figures/cgd_superiority_mu_{mu}.pdf')
    plt.close()

def generate_performance_plot(method):
    fig_map = {"CGD": "Figure 3: performance_cgd.pdf ...", "EF": "Figure 4: performance_ef.pdf ...", "EF21": "Figure 5: performance_ef21.pdf ..."}
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
                txtbox_kwargs=[standard_textbox(f'$\\kappa = {L / mu}$', {'x': 0.65}) for mu in reversed(mus)],
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
    fig_map = {"EF": "Figure 6: ef_multiple_iterations_delta_{delta}.pdf ...", "EF21": "Figure 7: ef21_multiple_iterations_delta_{delta}.pdf ..."}
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
    print("Generating Figure 8: best_etas_{mu}_{L}.pdf ...")
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
        "Figure 2": generate_cgd_superiority,
        "Figure 3": lambda: generate_performance_plot("CGD"),
        "Figure 4": lambda: generate_performance_plot("EF"),
        "Figure 5": lambda: generate_performance_plot("EF21"),
        "Table 3": lambda: generate_tightness_table("EF"),
        "Table 4": lambda: generate_tightness_table("EF21"),
        "Figure 6": lambda: generate_multiple_iterations("EF"),
        "Figure 7": lambda: generate_multiple_iterations("EF21"),
        "Figure 8": generate_best_etas
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
