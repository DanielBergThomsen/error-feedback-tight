"""Helper functions for PEPit-based performance analysis of optimization algorithms.

This module provides functions for analyzing the performance of various optimization
algorithms using PEPit (Performance Estimation Problem) framework.
"""

import numpy as np
from PEPit import PEP, Point
from PEPit.functions import SmoothStronglyConvexFunction

from utils import compress
from theory_helpers import optimal_lyapunov


def find_cycle(mu, L, eta, delta, method, n, eps=1e-3, realistic_steps=0):
    """Finds if a cycle exists in the optimization trajectory.

    Args:
        mu: Strong convexity parameter.
        L: Smoothness parameter.
        eta: Step size parameter.
        delta: Compression parameter between 0 and 1.
        method: Optimization method to use ('CGD', 'EF', or 'EF21').
        n: Number of iterations to check for cycle.
        eps: Tolerance for cycle detection.
        realistic_steps: Number of realistic steps to perform before cycle detection.

    Returns:
        bool: True if a cycle is found within tolerance eps, False otherwise.

    Raises:
        ValueError: If an invalid method is specified.
    """
    problem = PEP()
    func = problem.declare_function(function_class=SmoothStronglyConvexFunction, mu=mu, L=L)

    # Initialize points and gradients
    x0 = problem.set_initial_point()
    g0 = func.gradient(x0)
    
    # Initialize error terms based on method
    if method == 'CGD' or realistic_steps > 0:
        e0 = 0*Point() if method != 'EF21' else compress(g0, problem, delta)
    elif method == 'EF':
        e0 = Point(name='e0')
    elif method == 'EF21':
        c0 = Point(name='c0')
    else:
        raise ValueError(f"Invalid method: {method}")

    # Implement optimization method
    if method == 'CGD':
        c0 = compress(g0, problem, delta)
        e = 0
        x = x0 - eta * c0
    elif method == 'EF':
        c0 = compress(eta * g0 + e0, problem, delta)
        e = eta * g0 + e0 - c0
        x = x0 - c0
    elif method == 'EF21':
        x = x0 - eta * c0
        g1 = func.gradient(x)
        c = c0 + compress(g1 - c0, problem, delta)

    # Perform realistic steps if specified
    for i in range(realistic_steps - 1):
        if method == 'CGD':
            g = func.gradient(x)
            c = compress(g, problem, delta)
            e = 0
            x = x - eta * c
        elif method == 'EF':
            g = func.gradient(x)
            c = compress(eta * g + e, problem, delta)
            e = eta * g + e - c
            x = x - c
        elif method == 'EF21':
            x = x - eta * c
            g = func.gradient(x)
            c = c + compress(g - c, problem, delta)

    # Set initial distance condition
    if realistic_steps == 0:
        problem.set_initial_condition((x - x0)**2 >= 1)

    # Store cycle points
    x_cycle = x
    if method == 'CGD':
        e_cycle = 0
    elif method == 'EF':
        e_cycle = e
    elif method == 'EF21':
        c_cycle = c

    # Perform cycle iterations
    xk = x
    if method == 'EF' or method == 'CGD':
        ek = e
    elif method == 'EF21':
        ck = c

    for i in range(n - 1):
        if method == 'CGD':
            g = func.gradient(xk)
            c = compress(g, problem, delta)
            ek = 0
            xk = xk - eta * c
        elif method == 'EF':
            g = func.gradient(xk)
            c = compress(eta * g + ek, problem, delta)
            ek = eta * g + ek - c
            xk = xk - c
        elif method == 'EF21':
            xk = xk - eta * ck
            g = func.gradient(xk)
            ck = ck + compress(g - ck, problem, delta)

        if i == 0 and realistic_steps != 0:
            problem.set_initial_condition((xk - x_cycle)**2 >= 1)

    # Set performance metric based on method and steps
    if realistic_steps > 0:
        if method == 'CGD':
            problem.set_performance_metric(-(xk - x_cycle)**2)
        elif method == 'EF':
            problem.set_performance_metric(-(xk - x_cycle)**2 - (ek - e_cycle)**2)
        elif method == 'EF21':
            problem.set_performance_metric(-(xk - x_cycle)**2 - (ck - c_cycle)**2)
    else:
        if method == 'CGD':
            problem.set_performance_metric(-(xk - x0)**2)
        elif method == 'EF':
            problem.set_performance_metric(-(xk - x0)**2 - (ek - e0)**2)
        elif method == 'EF21':
            problem.set_performance_metric(-(xk - x0)**2 - (ck - c0)**2)

    metric = problem.solve(return_primal_or_dual="primal", solver="MOSEK", verbose=0)
    return abs(metric) < eps


def worst_case_performance(mu, L, eta, delta, method, n_iterations=1, P=None, p=None, use_simplified=False):
    """Computes the worst-case performance of an optimization method.

    Args:
        mu: Strong convexity parameter.
        L: Smoothness parameter.
        eta: Step size parameter.
        delta: Compression parameter between 0 and 1.
        method: Optimization method to use ('CGD', 'EF', or 'EF21').
        n_iterations: Number of iterations to perform.
        P: Lyapunov matrix (optional).
        p: Lyapunov scalar (optional).
        use_simplified: Whether to use simplified Lyapunov function.

    Returns:
        float: Worst-case performance metric (convergence rate) or None if computation fails.

    Raises:
        ValueError: If an invalid method is specified.
    """
    if P is not None and np.isnan(P).any():
        return np.nan

    # Setup PEP problem
    problem = PEP()
    func = problem.declare_function(function_class=SmoothStronglyConvexFunction, mu=mu, L=L)
    xs = func.stationary_point(name='xs')
    fs = func(xs)
    fs.set_name('fs')
    x0 = problem.set_initial_point(name='x0')
    g0 = func.gradient(x0, name='g0')

    # Run optimization method
    x = x0
    g = g0
    for i in range(n_iterations):
        if method == 'CGD':
            c = compress(g, problem, delta, name=f'c{i}')
            if i == 0:
                c0 = c
            x = x - eta * c
            g = func.gradient(x, name=f'g{i+1}')
        elif method == 'EF':
            if i == 0:
                e = Point(name=f'e{i}')
                c = compress(eta * g + e, problem, delta, name=f'c{i}')
                c0, e0 = c, e
            else:
                c = compress(eta * g + e, problem, delta, name=f'c{i}')
            e = eta * g + e - c
            x = x - c
            g = func.gradient(x, name=f'g{i+1}')
        elif method == 'EF21':
            if i == 0:
                c = Point(name=f'c{i}')
                c0 = c
            x = x - eta * c
            g = func.gradient(x, name=f'g{i+1}')
            co = compress(g - c, problem, delta, name=f'compressed term {i+1}')
            c = c + co
        else:
            raise ValueError(f'Method "{method}" not supported.')

    # Compute function values
    f0 = func(x0)
    f = func(x)

    # Set up performance metric based on available information
    if P is not None:
        if method == "CGD":
            init_parameters = np.array([x0 - xs, g0, c0])
            g = func.gradient(x, name=f'g{n_iterations}')
            c = compress(g, problem, delta, name=f'c{n_iterations}')
            parameters = np.array([x - xs, g, c])
        elif method == "EF":
            init_parameters = np.array([x0 - xs, g0, c0, e0])
            g = func.gradient(x, name=f'g{n_iterations}')
            c = compress(eta * g + e, problem, delta, name=f'c{n_iterations}')
            parameters = np.array([x - xs, g, c, e])
        elif method == "EF21":
            init_parameters = np.array([x0 - xs, g0, c0])
            parameters = np.array([x - xs, g, c])
        
        initial_condition = lya_quad(P, p, init_parameters, f0, fs)
        problem.set_initial_condition(initial_condition <= 1, name='initial_condition')
        performance_metric = lya_quad(P, p, parameters, f, fs)
        problem.set_performance_metric(performance_metric)
    elif use_simplified:
        P, p = optimal_lyapunov(delta, method)
        if method == 'EF':
            init_parameters = np.array([x0 - xs, e0])
            parameters = np.array([x - xs, e])
        elif method == 'EF21':
            init_parameters = np.array([g0, c0])
            parameters = np.array([g, c])
        elif method == 'CGD':
            init_parameters = np.array([0, 0])
            parameters = np.array([0, 0])
            P = np.zeros((2, 2))
        
        problem.set_initial_condition(lya_quad(P, p, init_parameters, f0, fs) <= 1, name='initial_condition')
        problem.set_performance_metric(lya_quad(P, p, parameters, f, fs))

    try:
        rho = problem.solve(return_primal_or_dual="dual", solver="MOSEK", verbose=0, 
                          dimension_reduction_heuristic='logdet10')
    except Exception as e:
        print(e)
        return None

    return rho


def lya_quad(P, p, params, fx, fs):
    """Computes the quadratic Lyapunov function value.

    Args:
        P: Lyapunov matrix.
        p: Lyapunov scalar.
        params: Array of parameters to evaluate.
        fx: Current function value.
        fs: Stationary point function value.

    Returns:
        float: Value of the quadratic Lyapunov function.
    """
    lya = p * (fx - fs)
    for i in range(len(P)):
        for j in range(len(P)):
            if isinstance(params[i], Point) and isinstance(params[j], Point):
                lya += P[i, j] * params[i] * params[j]
    
    return lya
