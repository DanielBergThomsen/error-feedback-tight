"""Helper functions for theoretical analysis of optimization methods.

This module provides functions for computing optimal parameters and Lyapunov functions
for various optimization methods including Error Feedback (EF), EF21, and Compressed Gradient
Descent (CGD).
"""

import numpy as np


def optimal_lyapunov(delta, method):
    """Computes the optimal Lyapunov matrix and scalar for a given method.

    This function returns the optimal Lyapunov function parameters for different
    optimization methods. The Lyapunov function is used to prove convergence
    guarantees for the respective methods.

    Args:
        delta: Compression parameter between 0 and 1.
        method: Optimization method to use ('EF', 'EF21', or 'CGD').

    Returns:
        tuple: (P, p) where:
            - P: Lyapunov matrix (None for CGD)
            - p: Lyapunov scalar

    Raises:
        ValueError: If an invalid method is specified.
    """
    if method == 'EF':
        P = np.array([[1, -1], [-1, 1 + 1 / np.sqrt(1 - delta)]])
        p = 0.0
    elif method == 'EF21':
        P = np.array([[1 + np.sqrt(1 - delta), -1], [-1, 1]])
        p = 0.0
    elif method == 'CGD':
        P = None
        p = 1.0
    else:
        raise ValueError(f"Method '{method}' not supported")

    return P, p


def optimal_step_size(mu, L, delta, method):
    """Computes the optimal step size for a given optimization method.

    This function calculates the optimal step size for different optimization
    methods based on the strong convexity parameter (mu), smoothness parameter (L),
    and compression parameter (delta).

    Args:
        mu: Strong convexity parameter.
        L: Smoothness parameter.
        delta: Compression parameter between 0 and 1.
        method: Optimization method to use ('EF' or 'EF21').

    Returns:
        float: Optimal step size for the specified method.

    Raises:
        ValueError: If an invalid method is specified.
    """
    standard_step_size = 2 / (L + mu)
    
    if method in ['EF', 'EF21']:
        return standard_step_size * (delta / (1 + np.sqrt(1 - delta))**2)
    elif method == 'CGD':
        return 2 / ((1-np.sqrt(1 - delta)) * mu + (1+np.sqrt(1 - delta)) * L)
    else:
        raise ValueError(f"Method '{method}' not supported")