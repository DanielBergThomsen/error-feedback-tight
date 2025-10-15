# Tight analyses of first-order methods with error feedback
This is the official repo for the paper available on [arXiv](https://arxiv.org/abs/2506.05271).
We make the following contributions in the paper:
- tight analyses of **EF and EF21** with contractive compression operators,
on smooth strongly convex functions in the **single-worker case** ($n=1$) 📈
- lower bounds matching the upper bounds proven above ⚖️
- **optimal** step size settings for EF and EF21 ⚡
- show that EF and EF21 have the exact same convergence rate within the space of allowable Lyapunov functions 🔄
- introduce a methodology for finding *simple*, yet *optimal*, Lyapunov functions to analyze 🔬


This repository contains the code to reproduce all figures and tables from our paper. To facilitate reproducibility, we have implemented:

- A command-line interface for generating individual figures/tables or all results at once
- Automatic data caching to avoid recomputing results unnecessarily

The repository additionally allows you to verify the proofs written in the paper in two ways:
1. Global validation using PEPit: two Jupyter notebook have been provided which numerically certify the statements given in Theorem 1 and 2 of the paper.
2. Local validation of upper bounds: WolframScripts have been provided to verify the first parts of the proofs in Theorem 1 and 2. Similarly, scripts have been adapted for the random compressor setting as well.

## Setup

To get started, we recommend creating a new virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

## Running Experiments

The experiments can be run in two ways:

1. Generate all figures and tables at once:
```bash
python experiments.py all
```

2. Generate individual figures or tables:
```bash
# Generate Figure 1 (performance comparison)
python experiments.py "Figure 1"

# Generate Table 1 (EF vs EF21 comparison)
python experiments.py "Table 1"
```

Note: 
- Running all experiments from scratch with `grid_resolution = 200` takes quite a while. You can set it to e.g. 50 and still get decent looking plots. Subsequent runs are pretty much immediate because of the saved data.
- You should delete the data folder if you make any changes to the code.

## Verifying proofs
Note that the paper itself contains direct links to the files used to certify the theorems. This is the easiest way of navigating our certificates.


In order to verify the proof steps written in WolframScript, you should install the Wolfram Engine, which is available for free at https://www.wolfram.com/engine/. After following the install instructions for your platform, you can verify the rates we show by simply running e.g.
```bash
wolframscript -script certificates/EF/upper_bound_CAS.wls
```
in the root of the project directory. You can do the same for the other `*.wls` files in the subdirectories of the `certificates` directory.

