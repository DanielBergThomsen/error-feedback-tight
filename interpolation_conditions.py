'''
This file contains interpolation conditions for different function classes. It also contains
methods to generate conditions for a set of points automatically.

The code originates from https://github.com/bgoujaud/cycles/blob/master/tools/interpolation_conditions.py
'''

from math import inf
import cvxpy as cp


def inner_product(u, v):
    """Computes the inner product matrix between two vectors.

    Args:
        u: First input vector.
        v: Second input vector.

    Returns:
        A symmetric matrix representing the inner product between u and v.
    """
    matrix = u.reshape(-1, 1) * v.reshape(1, -1)
    return (matrix + matrix.T) / 2


def square(u):
    """Computes the square of a vector using inner product.

    Args:
        u: Input vector.

    Returns:
        A matrix representing the square of the input vector.
    """
    return inner_product(u, u)


def smooth_strongly_convex_interpolation_i_j(pointi, pointj, mu, L):
    """Computes interpolation conditions for smooth strongly convex functions between two points.

    Args:
        pointi: Tuple containing (xi, gi, fi) for the first point.
        pointj: Tuple containing (xj, gj, fj) for the second point.
        mu: Strong convexity parameter.
        L: Smoothness parameter.

    Returns:
        Tuple containing:
            - G: Matrix representing the interpolation condition
            - F: Scalar representing the function value difference

    Raises:
        ValueError: If L or mu is None.
    """
    if L is None or mu is None:
        raise ValueError("L and mu must be provided for smooth strongly convex interpolation")

    xi, gi, fi = pointi
    xj, gj, fj = pointj

    G = inner_product(gj, xi - xj) + 1 / (2 * L) * square(gi - gj) + mu / (2 * (1 - mu / L)) * square(
        xi - xj - 1 / L * gi + 1 / L * gj)
    F = fj - fi

    return G, F


def lipschitz_operator_interpolation_i_j(pointi, pointj, L):
    """Computes interpolation conditions for Lipschitz operators between two points.

    Args:
        pointi: Tuple containing (xi, gi, _) for the first point.
        pointj: Tuple containing (xj, gj, _) for the second point.
        L: Lipschitz constant.

    Returns:
        Tuple containing:
            - G: Matrix representing the interpolation condition
            - F: Always returns 0 for this operator type

    Raises:
        ValueError: If L is None.
    """
    if L is None:
        raise ValueError("L must be provided for lipschitz operator interpolation")

    xi, gi, _ = pointi
    xj, gj, _ = pointj

    G = square(gi - gj) - L ** 2 * square(xi - xj)
    F = 0

    return G, F


def strongly_monotone_operator_interpolation_i_j(pointi, pointj, mu):
    """Computes interpolation conditions for strongly monotone operators between two points.

    Args:
        pointi: Tuple containing (xi, gi, _) for the first point.
        pointj: Tuple containing (xj, gj, _) for the second point.
        mu: Strong monotonicity parameter.

    Returns:
        Tuple containing:
            - G: Matrix representing the interpolation condition
            - F: Always returns 0 for this operator type

    Raises:
        ValueError: If mu is None.
    """
    if mu is None:
        raise ValueError("mu must be provided for strongly monotone operator interpolation")

    xi, gi, _ = pointi
    xj, gj, _ = pointj

    G = mu * square(xi - xj) - inner_product(gi - gj, xi - xj)
    F = 0

    return G, F


def cocoercive_operator_interpolation_i_j(pointi, pointj, L):
    """Computes interpolation conditions for cocoercive operators between two points.

    Args:
        pointi: Tuple containing (xi, gi, _) for the first point.
        pointj: Tuple containing (xj, gj, _) for the second point.
        L: Cocoercivity parameter.

    Returns:
        Tuple containing:
            - G: Matrix representing the interpolation condition
            - F: Always returns 0 for this operator type

    Raises:
        ValueError: If L is None.
    """
    if L is None:
        raise ValueError("L must be provided for cocoercive operator interpolation")

    xi, gi, _ = pointi
    xj, gj, _ = pointj

    G = square(gi - gj) - L * inner_product(xi - xj, gi - gj)
    F = 0

    return G, F


def interpolation(list_of_points, function_class, mu=None, L=None):
    """Generates interpolation conditions for a set of points based on the function class.

    Args:
        list_of_points: List of tuples, each containing (x, g, f) for a point.
        function_class: String specifying the function class. Must be one of:
            - "smooth strongly convex"
            - "lipschitz strongly monotone operator"
            - "strongly monotone operator"
            - "cocoercive operator"
        mu: Strong convexity/monotonicity parameter.
        L: Smoothness/Lipschitz parameter.

    Returns:
        Tuple containing:
            - list_of_matrices: List of matrices representing interpolation conditions
            - list_of_vectors: List of vectors representing function value differences
    """
    list_of_matrices = []
    list_of_vectors = []

    for i, pointi in enumerate(list_of_points):
        for j, pointj in enumerate(list_of_points):
            if i != j:
                if function_class == "smooth strongly convex":
                    G, F = smooth_strongly_convex_interpolation_i_j(pointi, pointj, mu, L)
                    list_of_matrices.append(G)
                    list_of_vectors.append(F)
                elif function_class == "lipschitz strongly monotone operator":
                    G, F = lipschitz_operator_interpolation_i_j(pointi, pointj, L)
                    list_of_matrices.append(G)
                    list_of_vectors.append(F)

                    G, F = strongly_monotone_operator_interpolation_i_j(pointi, pointj, mu)
                    list_of_matrices.append(G)
                    list_of_vectors.append(F)

                elif function_class == "strongly monotone operator":
                    assert L == inf
                    G, F = strongly_monotone_operator_interpolation_i_j(pointi, pointj, mu)
                    list_of_matrices.append(G)
                    list_of_vectors.append(F)

                elif function_class == "cocoercive operator":
                    assert mu == 0
                    G, F = cocoercive_operator_interpolation_i_j(pointi, pointj, L)
                    list_of_matrices.append(G)
                    list_of_vectors.append(F)

    return list_of_matrices, list_of_vectors


def interpolation_combination(list_of_points, function_class, mu=None, L=None):
    """Creates a convex combination of interpolation conditions.

    Args:
        list_of_points: List of tuples, each containing (x, g, f) for a point.
        function_class: String specifying the function class.
        mu: Strong convexity/monotonicity parameter.
        L: Smoothness/Lipschitz parameter.

    Returns:
        Tuple containing:
            - matrix_combination: Convex combination of interpolation matrices
            - vector_combination: Convex combination of interpolation vectors
            - dual: CVXPY variable representing the combination weights
    """
    list_of_matrices, list_of_vectors = interpolation(list_of_points, function_class, mu, L)
    nb_constraints = len(list_of_matrices)
    dual = cp.Variable((nb_constraints,))
    matrix_combination = cp.sum([dual[i] * list_of_matrices[i] for i in range(nb_constraints)])
    vector_combination = cp.sum([dual[i] * list_of_vectors[i] for i in range(nb_constraints)])

    return matrix_combination, vector_combination, dual


def compression_interpolation(compression_point, approximated_object, delta):
    """Computes interpolation conditions for compression operators.
    Note that the compressor is contractive and encodes the condition:
    ||x - c||^2 <= (1 - delta) ||x||^2

    Args:
        compression_point: The point to which the object is being compressed.
        approximated_object: The object being approximated.
        delta: Compression parameter between 0 and 1.

    Returns:
        Matrix representing the compression interpolation condition.

    Raises:
        ValueError: If delta is not between 0 and 1.
    """
    if delta < 0 or delta > 1:
        raise ValueError("delta must be between 0 and 1")
    return square(approximated_object - compression_point) - (1 - delta) * square(approximated_object)