"""
This module is contains functions used to calculated propreties of the statistical
distribution of Fock states at the ouput of a Gaussian boson sampler. No conjectures
are made here.
"""

from math import factorial
from itertools import product, chain, combinations
from sympy import MatrixSymbol, conjugate, symbols, eye, zeros, expand
import numpy as np
from thewalrus.reference import hafnian

# pylint: disable=C0103


def ordered_ladder(m):
    """
    Returns a list of coefficient C_n ordering the ladder operator in the following formula
    (ab)^m = sum_{n=1}^m C_n a^n b^n

    Args:
        m (int) : The power of the photon state number

    Returns:
        (list) : A list of all the coefficient C_n in order
    """
    coef = []
    for n in range(1, m + 1):
        coef.append(
            int(
                sum(
                    [
                        (-1) ** (n - k) * k**m / factorial(n - k) / factorial(k)
                        for k in range(1, n + 1)
                    ]
                )
            )
        )
    return coef


def partition(collection):
    """
    Returns the partition of all element of a given collection.

    Args:
        collection (iterable) : A  collection of elements to be partitioned.

    Returns:
        (iterable) : An iterable of all the partitions made from the collection
    """
    if len(collection) == 1:
        yield [collection]
        return

    first = collection[0]
    for smaller in partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n + 1 :]
        # put `first` in its own subset
        yield [[first]] + smaller


def symmetric_M(m):
    """
    Returns a symmetric symbolic matrix of the given size.

    Arg:
        m (int) : number of modes of the Gaussian boson sampler.

    Returns:
        (numpy.ndarray) : symmetric symbolic matix of size m.
    """
    matrix_M = zeros(m, m)
    buffer_M = MatrixSymbol("M", m, m)

    # M is symmetric
    for i in range(m - 1):
        for j in range(i + 1, m):
            matrix_M[j, i] = buffer_M[i, j]
            matrix_M[i, j] = buffer_M[i, j]

    # The diagonal
    for i in range(m):
        matrix_M[i, i] = buffer_M[i, i]

    return np.array(matrix_M)


def hermitian_N(m, N_nature="identity"):
    """
    Returns an hermitian symbolic matrix of the given size.

    Arg:
        m (int) : number of modes of the Gaussian boson sampler.
        nature (string) : "general" if the matrix is hermitian.
                        "diagonal" if the matrix is diagonal.
                        "identify" if the matrix is proportional to the identity.

    Returns:
        (numpy.ndarray) : symmetric symbolic matix of size m.
    """
    if N_nature == "general":
        matrix_N = zeros(m, m)
        N_buffer = MatrixSymbol("N", m, m)

        for i in range(m - 1):
            for j in range(i + 1, m):
                matrix_N[j, i] = conjugate(N_buffer[i, j])
                matrix_N[i, j] = N_buffer[i, j]

        for i in range(m):
            matrix_N[i, i] = N_buffer[i, i]

    elif N_nature == "diagonal":
        matrix_N = zeros(m, m)
        N_buffer = symbols("n0:%d" % m)
        for i in range(m):
            matrix_N[i, i] = N_buffer[i]

    elif N_nature == "identity":
        matrix_N = eye(m, m)
        N_buffer = symbols("n")
        matrix_N = N_buffer * matrix_N

    else:
        raise ValueError('Possible values for N_nature are "general", "diagonal" and "identity"')

    return np.array(matrix_N)


def moment_coefficient_calculator(vector_J, vector_K):
    """
    Returns the coefficient associated with the values of j and k given.

    Args:
        vector_J (list) : List of values of j indices for summation. Ex. [1,0,3,4,1]
        vector_K (list) : List of values of k indices for summation. Ex. [1,2,5,4,3]
            Should always be lower than vector_j

    Returns:
        (int) : The coefficient associated with the values of j and k given.
    """
    coef = 1  # multiplication of coefficients from all modes

    for s in range(len(vector_J)):
        js = vector_J[s]
        ks = vector_K[s]
        cs = 0  # single mode coefficient

        for l in range(1, js + 1):
            cs += (-1) ** (js - l) * l**ks / factorial(js - l) / factorial(l)

        coef *= cs if cs != 0 else 1  # to take into acount the possibility that js is 0

    return int((-1) ** sum(vector_J) * coef)


def moment_calculator(vector_K, N_nature="identity", displacement=False):
    """
    Returns the Gaussian boson sampler moment in the Fock basis for a
    specific vector of powers of the photon number.

    Arg:
        vector_K (list): Vector of powers of the moment. Ex. [1,2,5,4,3]
        nature (string) : "general" if the matrix is hermitian.
                        "diagonal" if the matrix is diagonal.
                        "identity" if the matrix is proportional to the identity.
        displacement (bool): Boolean stating if there is displacement or not. False by default.

    Returns:
        (sympy.core.mul.Mul) : Symbolic moment of given power.
    """
    dummy = [list(range(1, i + 1)) if i != 0 else [0] for i in vector_K]
    indice = product(*dummy)

    m = len(vector_K)
    matrix_M = symmetric_M(m)
    matrix_N = hermitian_N(m, N_nature)

    gamma = symbols("L0:%d" % m) if displacement else m * [0]
    gamma = np.array(gamma)

    moment = 0

    for vector_J in indice:
        slicing = []

        for s, j in enumerate(vector_J):  # Ex [1,2,0,4,0,1] -> [0,1,1,3,3,3,3,5] used for slicing
            slicing.extend(j * [s])

        # build B outside and do a double slicing on local_B instead?
        local_M = matrix_M[slicing][:, slicing]
        local_N = matrix_N[slicing][:, slicing]

        local_B = -np.block([[local_M, local_N], [conjugate(local_N), conjugate(local_M)]])

        local_gamma = gamma[slicing]
        if displacement:
            conjugate_gamma = [conjugate(i) for i in local_gamma]
            local_gamma = np.append(conjugate_gamma, -local_gamma)
        else:
            local_gamma = 2 * local_gamma

        np.fill_diagonal(local_B, local_gamma)

        coef = moment_coefficient_calculator(vector_J, vector_K)
        moment += coef * hafnian(local_B, loop=displacement)

    return moment


def cumulant_calculator(order, N_nature="identity", displacement=False):  # Ex order = 3
    """
    Returns the Gaussian boson sampler cumulant of the Fock basis for the
    given order

    Args:
        order (int) : The order of the cumulant
        nature (string) : "general" if the matrix is hermitian.
                        "diagonal" if the matrix is diagonal.
                        "identity" if the matrix is proportional to the identity.
        displacement (bool): Boolean stating if there is displacement or not. False by default.

    Returns:
        (sympy.core.mul.Mul) : Symbolic cumulant of given order.
    """
    indices = list(range(order))
    part = partition(indices)
    cumulant = 0
    matrix_M = symmetric_M(order)
    matrix_N = hermitian_N(order, N_nature)

    gamma = symbols("L0:%d" % order) if displacement else order * [0]
    gamma = np.array(gamma)

    for p in part:  # Ex p = [[1,3],[2]]
        size = len(p) - 1
        cum = factorial(size) * (-1) ** size  # prefactor

        for b in p:  # Ex b = [1,3]
            local_M = matrix_M[b][:, b]
            local_N = matrix_N[b][:, b]

            local_B = -np.block([[local_M, local_N], [conjugate(local_N), conjugate(local_M)]])

            local_gamma = gamma[b]
            if displacement:
                conjugate_gamma = [conjugate(i) for i in local_gamma]
                local_gamma = np.append(conjugate_gamma, -local_gamma)
            else:
                local_gamma = 2 * local_gamma

            np.fill_diagonal(local_B, local_gamma)

            cum *= (-1) ** len(b) * hafnian(local_B, loop=displacement)  # equation (10.1.3)

        cumulant += cum

    return expand(cumulant)


def hafnian_diagonal_block(matrix_M, diagonal_N):
    """
    Returns the hafnian of a four block matrix, A tilde, composed of,
    from left to right, top to bottom, M, N, N*, M*.
    This can be used to calculate the hafnian of A tilde
    only when N is diagonal.
    This is useful only for the case where there is no displacement
    since we would otherwise require a loop hafnian.
    matrix_M and matrix_N need to be square, and of same dimension.

    Args:
        matrix_M (numpy.ndarray): Symetric matrix.
        diagonal_N (numpy.ndarray): Elements of the diagonal of N.

    Returns:
        (sympy.core.mul.Mul): Symbolic hafnian of the block matrix A tilde.
    """
    m = len(diagonal_N)
    n = list(range(m))
    ps = chain.from_iterable(combinations(n, r) for r in range(len(n) + 1))  # power set
    ps = [list(set) for set in ps]

    haf = 0
    for s in ps:
        # hafnian of odd size is 0
        if len(s) % 2 == 0:
            local_M = matrix_M[s][:, s]
            local_haf = hafnian(local_M)
            # the permanent of an empty matrix is 1
            local_N = np.delete(diagonal_N, s) if len(s) < m else [1]
            haf += np.prod(local_N) * local_haf * conjugate(local_haf)
    return expand(haf)


def cumulant_calculator_block(order, N_nature="diagonal"):
    """
    Returns the Gaussian boson sampler cumulant of the Fock basis for the
    given order for N diagonal only. This function leverages a simplify
    equation to calculate the hafnian of a N diagonal block matrix.
    The matrix N is either diagonal or proportional to the indentity.
    There can be no displacement or else the calculation would require
    the calculation of loop hafnians.

    Args:
        order (int) : The order of the cumulant
        nature (string) : "diagonal" if the matrix is diagonal.
                        "identity" if the matrix is proportional to the identity.
        displacement (bool): Boolean stating if there is displacement or not. False by default.

    Returns:
        (sympy.core.mul.Mul) : Symbolic cumulant of given order.
    """
    indices = list(range(order))
    part = partition(indices)
    cumulant = 0
    matrix_M = symmetric_M(order)

    if N_nature == "diagonal":
        diagonal_N = np.array(symbols("n0:%d" % order))
    elif N_nature == "identity":
        diagonal_N = np.array(order * [symbols("n")])
    else:
        raise ValueError('Possible values for N_nature are "diagonal" and "identity"')

    for p in part:  # Ex p = [[1,3],[2]]
        size = len(p) - 1
        cum = factorial(size) * (-1) ** size  # prefactor

        for b in p:  # Ex b = [1,3]
            local_M = matrix_M[b][:, b]
            local_N = diagonal_N[b]

            cum *= hafnian_diagonal_block(local_M, local_N)  # equation (10.1.7)

        cumulant += cum

    return expand(cumulant)


def cumulant_calculator_speculative(order):  # Ex order = 3
    """
    Returns the symbolic cumulant of given order of the Gaussian boson sampler's photon number.
        The GBS is assumed to have no displacement. The input state is the squeezed vacuum with
        the same parameter r for all modes. It is speculated here that, in this specific case,
        the cumulant is indepant of parameter n which is therefor set to 0.

    Arg:
        order (int) : Order of the cumulant to be calculated.

    Returns:
        (sympy.core.mul.Mul) : Symbolic cumulant of given order.
    """
    indices = list(range(order))
    part = partition(indices)
    cumulant = 0
    M = symmetric_M(order)

    for p in part:  # Ex p = [[1,3],[2]]
        size = len(p) - 1
        cum = factorial(size) * (-1) ** size  # prefactor

        for b in p:  # Ex b = [1,3]
            local_M = M[b][:, b]
            haf = hafnian(local_M)
            cum *= haf * conjugate(haf)

        cumulant += cum

    return expand(cumulant)
