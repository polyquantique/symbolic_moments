"""
This module contains functions used to calculated propreties of the statistical
distribution of Fock states at the ouput of a Gaussian boson sampler. No conjectures
are made here.
"""

from math import factorial
from itertools import product
from sympy import MatrixSymbol, conjugate, symbols, expand
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
    M_matrix = MatrixSymbol("M", m + 1, m + 1)
    return (np.triu(M_matrix, 0) + np.triu(M_matrix, 1).T)[1:, 1:]


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
    if N_nature == "identity":
        N_matrix = np.diag([symbols("n")] * m)

    elif N_nature == "diagonal":
        N_matrix = np.diag(symbols("n1:%d" % (m + 1)))

    elif N_nature == "general":
        N_matrix = MatrixSymbol("N", m + 1, m + 1)
        N_matrix = (np.triu(N_matrix, 0) + np.triu(N_matrix, 1).T.conj())[1:, 1:]

    else:
        raise ValueError(
            'Possible values for N_nature are "general", "diagonal" and "identity"'
        )

    return N_matrix


def symmetric_A(m):
    """
    Returns a symmetric symbolic matrix of the given size.

    Arg:
        m (int) : number of modes of the Gaussian boson sampler.

    Returns:
        (numpy.ndarray) : symmetric symbolic matix of size m.
    """
    m = 2 * m
    A_matrix = MatrixSymbol("A", m + 1, m + 1)
    return (np.triu(A_matrix, 0) + np.triu(A_matrix, 1).T)[1:, 1:]


def moment_coefficients(vector_J, vector_K):
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

    for s, _ in enumerate(vector_J):
        js = vector_J[s]
        ks = vector_K[s]
        cs = 0  # single mode coefficient

        for l in range(1, js + 1):
            cs += (-1) ** (js - l) * l**ks / factorial(js - l) / factorial(l)

        coef *= cs if cs != 0 else 1  # to take into acount the possibility that js is 0

    return int(coef)


def moment(vector_K, N_nature="general", displacement=True):
    """
    Returns the photon-number moment for a specific vector of powers.
    This is the most general version of the moment calculation.
    It allows for repetition of powers. There is no constraint on N
    except that it is hermitian and there is no dipslacement constrait.
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
    matrix_A = np.block(
        [[conjugate(matrix_M), matrix_N], [conjugate(matrix_N), matrix_M]]
    )

    gamma = symbols("alpha1:%d" % (m + 1)) if displacement else m * [0]
    gamma = np.array(gamma)
    gamma_conj = np.concatenate((gamma.conj(), gamma))

    moment_val = 0

    for vector_J in indice:
        slicing = []

        for s, j in enumerate(
            vector_J
        ):  # Ex [1,2,0,4,0,1] -> [0,1,1,3,3,3,3,5] used for slicing
            slicing.extend(j * [s])

        slicing = slicing + [i + m for i in slicing]
        local_A = matrix_A[slicing][:, slicing]
        local_gamma = gamma_conj[slicing]
        np.fill_diagonal(local_A, local_gamma)

        coef = moment_coefficients(vector_J, vector_K)
        moment_val += coef * hafnian(local_A, loop=displacement)

    return moment_val


def cumulant(vector_K, N_nature="general", displacement=True):
    """
    Returns the photon-number cumulant for the given order of the
    most general case. The case where there can be displacement,
    N is a general hermitian matrix and the vector K allows for
    repetition of powers.

    Args:
        vector_K (list): Vector of powers of photon-number operators. Ex. [1,2,5,4,3]
        nature (string) : "general" if the matrix is hermitian.
                        "diagonal" if the matrix is diagonal.
                        "identity" if the matrix is proportional to the identity.
        displacement (bool): Boolean stating if there is displacement or not. False by default.

    Returns:
        (sympy.core.mul.Mul) : Symbolic cumulant of given order.
    """
    order = len(vector_K)

    power = []
    for s, j in enumerate(vector_K):  # Ex [1,2,0,4,0,1] -> [0,1,1,3,3,3,3,5]
        power.extend(j * [s])
    partyy = partition(power)

    cumulant_val = 0
    for party in partyy:
        size = len(party) - 1
        cum = factorial(size) * (-1) ** size  # prefactor
        for part in party:
            buffer = [0] * order
            for p in part:
                buffer[p] += 1
            cum *= moment(buffer, N_nature, displacement)
        cumulant_val += cum

    return expand(cumulant_val)


def montrealer(M):
    """
    Returns the montrealer of a matrix

    Args:
        matrix_M (numpy.ndarray): A matrix

    Returns:
        (sympy.core.mul.Mul) : The Montrealer of the given matrix
    """
    order = len(M)
    if order % 2 != 0:
        return 0
    indices = list(range(order))
    part = partition(indices)
    cumulant_val = 0
    for p in part:  # Ex p = [[1,3],[2]]
        # Check there are no partitions that odd length parts.
        check_weight = True
        for i in p:
            if len(i) % 2 != 0:
                check_weight = False
                break
        if check_weight:
            size = len(p) - 1
            cum = factorial(size) * (-1) ** size  # prefactor
            for b in p:  # Ex b = [1,3]
                local_M = M[b][:, b]
                haf = hafnian(local_M)
                cum *= haf * conjugate(haf)

            cumulant_val += cum

    return cumulant_val
