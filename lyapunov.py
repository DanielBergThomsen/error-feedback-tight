'''
This file contains the code for the core of this approach, which is the function that checks if
a Lyapunov function exists for a desired rate of convergence.

The implementation was based on the following repo by Baptiste Goujaud: 
https://github.com/bgoujaud/cycles
'''

import cvxpy as cp
import numpy as np
from interpolation_conditions import interpolation_combination, compression_interpolation


def has_lyapunov(rho, eta, delta,
                 function_class='smooth strongly convex', mu=None, L=None, 
                 method='EF', zero_coefs=None, use_residual=True, 
                 log_det_iterations=0, log_det_delta=1e-6, 
                 use_simplified_lyapunov=False,
                 use_richtarik=False,
                 backup_solver=None):
    """Checks if a Lyapunov function exists for a desired rate of convergence.

    Args:
        rho: Desired rate of convergence.
        eta: Step size parameter.
        delta: Compression parameter between 0 and 1.
        function_class: Type of function class to consider. Options:
            - 'smooth strongly convex'
            - 'lipschitz strongly monotone operator'
            - 'strongly monotone operator'
            - 'cocoercive operator'
        mu: Strong convexity/monotonicity parameter.
        L: Smoothness/Lipschitz parameter.
        method: Optimization method to use. Options: 'CGD', 'EF', 'EF21'.
        zero_coefs: List of tuples (i,j) specifying which coefficients to set to zero.
        use_residual: Whether to use residual terms in the Lyapunov function.
        log_det_iterations: Number of iterations for log determinant optimization.
        log_det_delta: Small constant for numerical stability in log determinant.
        use_simplified_lyapunov: Whether to use simplified Lyapunov function structure.
        backup_solver: Backup solver to use if primary solver fails.

    Returns:
        Tuple containing:
            - bool: Whether a valid Lyapunov function exists
            - np.ndarray: Matrix P of the Lyapunov function if found, otherwise NaN
            - float: Scalar p of the Lyapunov function if found, otherwise NaN
    """
    # Initialize basis vectors
    d = 6
    basis_vectors = np.eye(d)
    x0, g0, e0, c0, g1, c1 = list(basis_vectors)
    xs = np.zeros(d)
    gs = np.zeros(d)
    
    # Initialize function value basis vectors
    f_basis_vectors = np.eye(2)
    f0, f1 = list(f_basis_vectors)
    fs = np.zeros(2)

    # Start maintaining list of supplementary constraints
    supplementary_constraints = []

    # Setup based on method
    if method == 'CGD':
        x1 = x0 - eta * c0
        supplementary_constraints.extend([compression_interpolation(c0, g0, delta)])

        # Lyapunov (note terms are aggregated in multi-worker case)
        P = cp.Variable((3, 3), symmetric=True)
        p = cp.Variable((1,))
        VP = np.array([x0 - xs, g0, c0]).T @ P @ np.array([x0 - xs, g0, c0])
        VP_plus = np.array([x1 - xs, g1, c1]).T @ P @ np.array([x1 - xs, g1, c1])
        Vp = np.array([f0 - fs]).T @ p
        Vp_plus = np.array([f1 - fs]).T @ p

    elif method == 'EF':
        x1 = x0 - c0
        e1 = e0 + eta * g0 - c0
        supplementary_constraints.extend([compression_interpolation(c0, e0 + eta * g0, delta)])

        # Lyapunov setup for EF
        P = cp.Variable((4, 4), symmetric=True)
        p = cp.Variable((1,))
        VP = np.array([x0 - xs, g0, c0, e0]).T @ P @ np.array([x0 - xs, g0, c0, e0])
        VP_plus = np.array([x1 - xs, g1, c1, e1]).T @ P @ np.array([x1 - xs, g1, c1, e1])
        Vp = np.array([f0 - fs]).T @ p
        Vp_plus = np.array([f1 - fs]).T @ p

    elif method == 'EF21':
        x1 = x0 - eta * e0
        e1 = e0 + c0
        supplementary_constraints.extend([compression_interpolation(c0, g1 - e0, delta)])

        # Lyapunov setup for EF21
        P = cp.Variable((3, 3), symmetric=True)
        p = cp.Variable((1,))
        VP = np.array([x0 - xs, g0, e0]).T @ P @ np.array([x0 - xs, g0, e0])
        VP_plus = np.array([x1 - xs, g1, e1]).T @ P @ np.array([x1 - xs, g1, e1])
        Vp = np.array([f0 - fs]).T @ p
        Vp_plus = np.array([f1 - fs]).T @ p

    else:
        raise ValueError(f"Invalid method: {method}")

    # Setup problem constraints
    constraints = [p >= 0, P >> 0]

    # Add interpolation conditions
    matrix_combinations = []
    vector_combinations = []
    
    # Interpolation conditions
    points = [(xs, gs, fs), (x0, g0, f0), (x1, g1, f1)]
    matrix_comb, vector_comb, dual = interpolation_combination(points, function_class, mu, L)
    matrix_combinations.append(matrix_comb)
    vector_combinations.append(vector_comb)
    constraints.append(dual >= 0)

    supplementary_term = cp.sum([cp.Variable((1,)) * matrix for matrix in supplementary_constraints])
    constraints.append(VP_plus - rho * VP << cp.sum(matrix_combinations) + supplementary_term)
    constraints.append(Vp_plus - rho * Vp <= cp.sum(vector_combinations))

    if use_richtarik:
        constraints.append(cp.trace(P) + p == 1)
        constraints.extend([P[0, i] == 0 for i in range(3)] + [P[i, 0] == 0 for i in range(3)])
        constraints.extend([P[1, 1] == P[2, 2]])
        constraints.extend([P[1, 2] == -P[2, 2]])
        constraints.extend([P[2, 1] == -P[2, 2]])
    else:

        # Apply custom Lyapunov function constraints
        if not use_residual:
            constraints.extend([p == 0, cp.trace(P) == 1])
        else:
            constraints.append(cp.trace(P) + p == 1)

        # Apply sparsity constraints if specified
        if zero_coefs is not None:
            for i, j in zero_coefs:
                constraints.extend([P[i, j] == 0, P[j, i] == 0])

        # Apply method-specific simplified Lyapunov constraints
        if method == 'EF' and use_simplified_lyapunov:
            constraints.extend([
                P[0, 0] == 1,
                P[0, 3] == -1,
                P[3, 0] == -1,
                P[3, 3] >= 0
            ])
        elif method == 'EF21' and use_simplified_lyapunov:
            constraints.extend([
                P[1, 1] >= 0,
                P[2, 2] == 1 - P[1, 1],
                P[1, 2] == -P[2, 2],
                P[2, 1] == -P[2, 2]
            ])

    # Solve the optimization problem
    prob = cp.Problem(cp.Minimize(0), constraints)
    P_value = np.full_like(P, np.nan)
    p_value = np.nan

    try:
        value = logdet_solve(prob, P.shape[0], log_det_iterations, log_det_delta, solver="MOSEK")
        P_value = P.value if value == 0 else P_value
        p_value = p.value[0] if value == 0 else p_value
    except cp.error.SolverError:
        if backup_solver is not None:
            value = logdet_solve(prob, P.shape[0], log_det_iterations, log_det_delta, solver=backup_solver)
            P_value = P.value
            p_value = p.value[0]
        else:
            return False, P_value, p_value

    return value == 0, P_value, p_value


def logdet_solve(problem, dim, log_det_iterations, log_det_delta, **kwargs):
    """Solves the optimization problem using log determinant barrier method.

    Args:
        problem: CVXPY problem to solve.
        dim: Dimension of the problem.
        log_det_iterations: Number of iterations for log determinant optimization.
        log_det_delta: Small constant for numerical stability.
        **kwargs: Additional solver parameters.

    Returns:
        float: Solution value (0 if feasible, infinity otherwise).
    """
    Pk = np.eye(dim)
    for _ in range(log_det_iterations):
        W = np.linalg.inv(Pk + log_det_delta * np.eye(Pk.shape[0]))
        problem = cp.Problem(cp.Minimize(cp.trace(W @ P)), problem.constraints)
        problem.solve(**kwargs)
        Pk = Pk.value
    return problem.solve(**kwargs)


def bisection(l, r, tol, solving_fun, *args, **kwargs):
    """Performs bisection search to find the optimal rate of convergence.

    Args:
        l: Left bound of the search interval.
        r: Right bound of the search interval.
        tol: Tolerance for the search.
        solving_fun: Function to solve at each iteration.
        *args: Additional positional arguments for solving_fun.
        **kwargs: Additional keyword arguments for solving_fun.

    Returns:
        Tuple containing:
            - float: Optimal rate of convergence if found, None otherwise
            - np.ndarray: Matrix P of the Lyapunov function if found, None otherwise
            - float: Scalar p of the Lyapunov function if found, None otherwise
    """
    r_init = r
    while r - l > tol:
        m = (l + r) / 2
        feasible, P, pf = solving_fun(m, *args, **kwargs)
        if feasible:
            r = m
        else:
            l = m

    if r == r_init and not feasible:
        return None, None, None
    _, P, pf = solving_fun(r, *args, **kwargs)
    return r, P, pf


if __name__ == "__main__":
    mu = 0.5
    L = 1.0
    from theory_helpers import optimal_step_size
    delta = 0.50
    eta = optimal_step_size(mu, L, delta=delta, method='CGD')
    t = has_lyapunov(rho=0.99, eta=eta, delta=delta, mu=mu, L=L, method = 'CGD', use_residual=True)
    print(t[0])