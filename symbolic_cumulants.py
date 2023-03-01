"""
This module contains functions used to calculated 
statistical properties of gaussian states.
"""

import numpy as np
from sympy import MatrixSymbol, symbols

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
    # modes = {3:4, 1:2}
    # This means I want n_3^4, n_1^2


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


# This one is optional but nice to have
def gspm(s):
    r"""Generator for the set of perfect matching permutations that appear in a Gaussian state cumulant.

    Args:
        s (tuple): an input tuple

    Returns:
        generator: the set of perfect matching permutations of the tuple s
    """
    # NOTE that typically s = [1,2,3...,n; n+1,n+2,....2n]
    # NOTE IDeally we return a *generator*


def montrealer(A):  # This is technically the "new" montrealer
    r"""Calculates the Montrealer of a square symmetric matrix of even size.

    Args:
            A (array): square even-sized complex-symmetric matrix representing the covariance of the Gaussian state.

    Returns:
            (complex): the value of the montrealer

    """


def laurentienne(M):  # This is technically the "old" montrealer
    r"""Calculates the laurentienne of a square symmetric matrix.

    Args:
            M (array): square complex-symmetric matrix representing the phase-sensitive quadrature moments of the Gaussian state.

    Returns:
            (complex): the value of the laurentienne
    """
    # If size of matrix equals odd return 0


def lavalois(N):
    r"""Calculates the lavalois of a square hermitian matrix.

    Args:
            M (array): square hermitrian matrix representing the phase-insensitive quadrature moments of the Gaussian state.

    Returns:
            (complex): the value of the lavalois
    """


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
    return np.block(
        [[matrix_M.conj(), matrix_N], [matrix_N.conj(), matrix_M]]
    )


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
    return np.diag(symbols("n"+str(initial_index)+":%d" % (n + initial_index)))