"""
This module contains functions used to calculated
statistical properties of gaussian states.
"""

import numba
import numpy as np
from sympy import MatrixSymbol, symbols, conjugate, simplify, expand
from itertools import product, permutations
from functools import reduce
from operator import mul
from math import factorial
from thewalrus.reference import hafnian
from thewalrus.charpoly import powertrace
from thewalrus._hafnian import nb_ix
from thewalrus.quantum import Xmat

# pylint: disable=C0103


def photon_number_moment(A, zeta, modes):
    """Returns the photon number moment of the modes in the Gaussian state.

    Args:
            A (array): square even-sized complex-symmetric matrix representing the covariance of the Gaussian state
            zeta (array): even-sized complex vector representing the displacement of the Gaussian state
            modes (dict): the specific modes and order of the moment

    Returns:
            (scalar) [NOTE we should check the type]: the moment
    """
    order = int(A.shape[0] / 2)

    keys = [i - 1 for i, j in modes.items() if j != 0]
    keys = keys + [i + order for i in keys]
    A = A[keys][:, keys]
    zeta = zeta[keys]

    power = [i for i in modes.values() if i != 0]
    m = len(power)
    dummy = [list(range(1, i + 1)) for i in power]
    indice = product(*dummy)

    moment_val = 0

    for vector_J in indice:
        slicing = []

        for s, j in enumerate(vector_J):  # Ex [1,2,0,4,0,1] -> [0,1,1,3,3,3,3,5] used for slicing
            slicing.extend(j * [s])

        slicing = slicing + [i + m for i in slicing]
        local_A = A[slicing][:, slicing]
        local_zeta = zeta[slicing]
        np.fill_diagonal(local_A, local_zeta.conj())

        coef = photon_number_moment_coefficients(vector_J, power)
        moment_val += coef * hafnian(local_A, loop=not all(v == 0 for v in zeta))

    return simplify(moment_val)


def photon_number_cumulant(A, zeta, modes):
    """Returns the photon number cumulant of the modes in the Gaussian state.

    Args:
            A (array): square even-sized complex-symmetric matrix representing the covariance of the Gaussian state
            zeta (array): even-sized complex vector representing the displacement of the Gaussian state
            modes (dict): the specific modes and order of the cumulant

    Returns:
            (scalar) [NOTE we should check the type]: the moment
    """
    # modes = {3:4, 1:2}
    # This means I want n_3^4, n_1^2
    slicing = []
    for key, value in modes.items():
        slicing.extend(value * [key])

    cumulant = 0
    for party in partition(slicing):
        size = len(party) - 1
        coef = factorial(size) * (-1) ** size  # prefactor

        for part in party:
            modes_partition = {}

            for i in part:
                if i in modes_partition.keys():
                    modes_partition[i] += 1
                else:
                    modes_partition[i] = 1

            coef *= photon_number_moment(A, zeta, modes_partition)

        cumulant += coef

    return expand(cumulant)


# This one is optional but nice to have
def gspm(s):
    """Generator for the set of perfect matching permutations that appear in a Gaussian state cumulant.

    Args:
        s (tuple): an input tuple

    Returns:
        generator: the set of perfect matching permutations of the tuple s
    """
    # NOTE that typically s = [1,2,3...,n; n+1,n+2,....2n]
    # NOTE IDeally we return a *generator*
    # NOTE look at pmp(s) from the Walrus


# This one is optional but nice to have
def pmpr(s):
    """Generates the set of perfect matchings that cannot be built as products of lower order perfect matchings.

    Args:
        s (tuple): as tuple

    Returns:
        generator: the set of perfect matching permutations of the tuple s
    """
    m = len(s) // 2
    local_mapper = lambda x: mapper(x, s)
    return map(local_mapper, product(permutations(range(1, m)), bitstrings(m)))


def montrealer(A):
    """Calculates the Montrealer of a square symmetric matrix of even size.

    Args:
            A (array): square even-sized complex-symmetric matrix representing the covariance of the Gaussian state.

    Returns:
            (complex): the value of the montrealer

    """


def montrealer(A):
    """Calculates the Montrealer of a square symmetric matrix of even size.

    Args:
            A (array): square even-sized complex-symmetric matrix representing the covariance of the Gaussian state.

    Returns:
            (complex): the value of the montrealer

    """
    tot_sum = 0
    m = len(A)
    iter_set = pmpr(tuple(range(m)))
    for i in iter_set:
        result = 1

        for j in i:
            result = result * A[j]

        tot_sum = tot_sum + result
    return tot_sum


def loop_montrealer(A, zeta):
    """Calculates the loop Montrealer of a square symmetric matrix of even size.

    Args:
            A (array): square even-sized complex-symmetric matrix representing the covariance of the Gaussian state.
            zeta (array): even-sized complex vector representing the displacement of the Gaussian state

    Returns:
            (complex): the value of the montrealer

    """
    # MAKE SUR A AND ZETA ARE THE SAME SIZE
    equation = 0
    # making the original 2 rows matrix
    m = int(np.shape(A)[0] / 2)
    original = np.arange(1, 2 * m + 1).reshape(2, m)
    # only the conjugate of zeta is used
    zeta_conj = zeta.conj()

    # initial graphs
    graph1 = list(range(1, m)) + [m + 1]
    graph2 = list(range(m + 2, 2 * m + 1)) + [m]

    # loop over all bistrings and all permutations
    for bit in bitstrings(m):
        for perm in permutations(range(1, m)):
            B = np.copy(original)

            for i, j in enumerate(bit):
                if int(j):
                    buffer = B[0, i]
                    B[0, i] = B[1, i]
                    B[1, i] = buffer

            B = B[:, [0] + list(perm)]  # first column stays in place always

            # dictionary mapping
            dico = {j: i + 1 for i, j in enumerate(B.reshape(1, 2 * m)[0])}
            new_mapping = {dico[i]: dico[j] for i, j in zip(graph1, graph2)}

            term = 1
            for i, j in new_mapping.items():
                term *= A[i - 1, j - 1]

            equation += term

            for key_1, val_1 in new_mapping.items():
                term = zeta_conj[key_1 - 1] * zeta_conj[val_1 - 1]

                for key_2, val_2 in new_mapping.items():
                    if key_1 != key_2:
                        term *= A[key_2 - 1, val_2 - 1]

                equation += term

    return equation


def laurentienne(M):  # This is technically the "old" montrealer
    """Calculates the laurentienne of a square symmetric matrix.

    Args:
            M (array): square complex-symmetric matrix representing the phase-sensitive quadrature moments of the Gaussian state.

    Returns:
            (complex): the value of the laurentienne
    """
    order = len(M)

    # The Laurentienne of odd sized matrix is 0
    if order % 2 != 0:
        return 0

    indices = list(range(order))
    part = partition(indices)
    laurent = 0
    for p in part:  # Ex p = [[1,3],[2]]
        # Check there are no partitions of odd length parts.
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

            laurent += cum

    return expand(laurent)


def lavalois(N):
    """Calculates the lavalois of a square hermitian matrix.

    Args:
            M (array): square hermitrian matrix representing the phase-insensitive quadrature moments of the Gaussian state.

    Returns:
            (complex): the value of the lavalois
    """
    order = N.shape[0]
    laval = 0
    for party in partition(list(range(order))):
        size = len(party) - 1
        coef = factorial(size) * (-1) ** size  # prefactor

        for part in party:
            coef *= permanent(N[part][:, part])

        laval += coef

    return expand(laval)


def symmetric_A(n, initial_index=0):
    """Return a symmetric symbolic matrix of size 2n with entries index by A.

    Args:
            n (int): number of modes
            initial_index (int): initial value for indexing

    Returns:
            (array): a symbolic array

    """
    n = 2 * n
    A_matrix = MatrixSymbol("A", n + initial_index, n + initial_index)
    return (np.triu(A_matrix, 0) + np.triu(A_matrix, 1).T)[initial_index:, initial_index:]


def block_A(n, initial_index=0):
    """Return a block symmetric symbolic matrix of size 2n with entries index by A.

    Args:
            n (int): number of modes
            initial_index (int): initial value for indexing

    Returns:
            (array): a symbolic array

    """
    matrix_M = symmetric_M(n, initial_index)
    matrix_N = hermitian_N(n, initial_index)
    return np.block([[matrix_M.conj(), matrix_N], [matrix_N.conj(), matrix_M]])


def symmetric_M(n, initial_index=0):
    """
    Returns a symmetric symbolic matrix M of size n.

    Arg:
        n (int) : size of the matrix.
        initial_index (int) : inital index of the matrix

    Returns:
        (numpy.ndarray) : symmetric symbolic matix of size n.
    """
    M_matrix = MatrixSymbol("M", n + initial_index, n + initial_index)
    return (np.triu(M_matrix, 0) + np.triu(M_matrix, 1).T)[initial_index:, initial_index:]


def hermitian_N(n, initial_index=0):
    """
    Returns an hermitian symbolic matrix N of size n.

    Arg:
        n (int) : size of the matrix.
        initial_index (int) : inital index of the matrix

    Returns:
        (numpy.ndarray) : symmetric symbolic matix of size n.
    """
    N_matrix = MatrixSymbol("N", n + initial_index, n + initial_index)
    return (np.triu(N_matrix, 0) + np.triu(N_matrix, 1).T.conj())[initial_index:, initial_index:]


def diagonal_N(n, initial_index=0):
    """
    Returns a diagonal symbolic matrix N of size n.

    Arg:
        n (int) : size of the matrix.
        initial_index (int) : inital index of the matrix

    Returns:
        (numpy.ndarray) : symmetric symbolic matix of size n.
    """
    return np.diag(symbols("n" + str(initial_index) + ":%d" % (n + initial_index)))


def mapper(x, objects=None):
    """Helper function to turn a permutation and bistring into an element of pmpr"""
    (perm, bit) = x
    m = len(bit)
    graph1 = list(range(0, m - 1)) + [m]
    graph2 = list(range(m + 1, 2 * m)) + [m - 1]
    Blist = [list(range(m)), list(range(m, 2 * m))]
    for i, j in enumerate(bit):
        if int(j):
            (Blist[0][i], Blist[1][i]) = (Blist[1][i], Blist[0][i])
    Blist = [[Blist[0][i] for i in tuple((0,) + perm)], [Blist[1][i] for i in tuple((0,) + perm)]]
    [X, Y] = Blist
    Blistflat = X + Y
    dico_list = {j: i + 1 for i, j in enumerate(Blistflat)}
    # dico_list = {j-1: i for i, j in enumerate(Blistflat)}
    if objects is None:
        new_mapping_list = {dico_list[i] - 1: dico_list[j] - 1 for i, j in zip(graph1, graph2)}
    else:
        new_mapping_list = {
            objects[dico_list[i] - 1]: objects[dico_list[j] - 1] for i, j in zip(graph1, graph2)
        }
    return tuple(new_mapping_list.items())


def bitstrings(n):
    """
    Returns the bistrings from 0 to n/2

    Args:
        n (int) : Twice the highest bitstring value.

    Returns:
        (iterable) : An iterable of all bistrings.
    """
    for binary in map("".join, product("01", repeat=n - 1)):
        yield "0" + binary


def partition(collection):
    """
    Returns the partition of all element of a given collection.

    Args:
        collection (iterable) : A collection of elements to be partitioned.

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


def permanent(A):
    """
    Returns the permanent of the entry matrix.

    Args:
        A (array) : A matrix.

    Returns:
        (complex): the permanent of matrix A.
    """
    n = len(A)
    r = range(n)
    s = permutations(r)
    prod = lambda lst: reduce(mul, lst, 1)
    return sum(prod(A[i][sigma[i]] for i in r) for sigma in s)


def photon_number_moment_coefficients(vector_J, vector_K):
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

        coef *= cs

    return int(coef)


@numba.jit(nopython=True, cache=True)
def dec2bin(num, n):
    """Finds the positions where there is a one in the binary digit representation of the number

    Args:
        num (int): integer
        n (int): number of bits

    Returns:
        (array): positions where the ones are located
    """
    digits = np.zeros((n), dtype=type(num))
    nn = num
    counter = -1
    while nn >= 1:  # Should be >= 1, not > 1.
        digits[counter] = nn % 2
        counter -= 1
        nn //= 2
    return np.nonzero(digits)[0]


@numba.jit(nopython=True, cache=True)
def nb_block(X):
    """Numba compatible version of numpy.block"""
    xtmp1 = np.concatenate(X[0], axis=1)
    xtmp2 = np.concatenate(X[1], axis=1)
    return np.concatenate((xtmp1, xtmp2), axis=0)


@numba.jit(nopython=True, cache=True)
def montrealer_numba(A):
    """Calculates the Montrealer of a square symmetric matrix of even size.
    Note that this only works for numerical arrays.

    Args:
            A (array): square even-sized complex-symmetric matrix representing the covariance of the Gaussian state.

    Returns:
            (complex): the value of the montrealer

    """
    n = len(A) // 2
    Sigma = nb_block(
        (
            (A[n : 2 * n, 0:n], A[n : 2 * n, n : 2 * n]),
            (A[0:n, 0:n], A[0:n, n : 2 * n]),
        )
    )
    tot_num = 2**n
    val = np.complex128(0)
    for p in numba.prange(tot_num):
        pos = dec2bin(p, n)
        lenpos = len(pos)
        pos = np.concatenate((pos, n + pos))
        submat = nb_ix(Sigma, pos, pos)
        if lenpos % 2 == 1:
            sign = 1
        else:
            sign = -1
        val += (sign) * powertrace(submat, n + 1)[-1]
    return (-1) ** (n + 1) * val / (2 * n)
