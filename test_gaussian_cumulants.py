"""
Unit tests for symbolic_cumulants.py
"""

import pytest
import symbolic_cumulants as gbs
import numpy as np
from thewalrus.random import random_covariance
from thewalrus.quantum import Qmat, Xmat
from sympy import symbols
from functools import reduce
from math import factorial
from scipy.stats import unitary_group


@pytest.mark.parametrize("n", [1,2,3,4])
def test_montrealer_agrees_with_cumulants(n):
	"""Checks that the montrealer and the cumulant function agree"""
	A = gbs.symmetric_A(n)
	zeta = np.zeros(2*n) #no displacement
	modes = {i:1 for i in range(1,n+1)} #no repetition
	assert gbs.montrealer(A) == gbs.photon_number_cumulant(A, zeta, modes)


@pytest.mark.parametrize("n", [2,3,4,5])
def test_laurentienne_agrees_with_cumulants(n):
	"""Checks that the laurentienne and the cumulant function agree"""
	M = gbs.symmetric_M(n)
	N = gbs.diagonal_N(n)
	A = np.block([[M.conj(), N], [N.conj(), M]])
	zeta = np.zeros(2*n)
	modes = {i:1 for i in range(1,n+1)}
	assert gbs.laurentienne(M) == gbs.photon_number_cumulant(A, zeta, modes)


@pytest.mark.parametrize("n", [2,3,4,5])
def test_lavalois_agrees_with_cumulants(n):
	"""Checks that the lavalois and the cumulant function agree"""
	N = gbs.hermitian_N(n)
	M = np.zeros((n,n))
	A = np.block([[M.conj(), N], [N.conj(), M]])
	zeta = np.zeros(2*n)
	modes = {i:1 for i in range(1,n+1)}
	assert gbs.lavalois(N) == gbs.photon_number_cumulant(A, zeta, modes)


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7])
def test_montrealer_agrees_with_laurentienne(n):
    """Checks that the montrealer agrees with the laurentienne"""
    M = gbs.symmetric_M(n)
    N = gbs.diagonal_N(n)
    A = np.block([[M.conj(), N], [N.conj(), M]])
    assert gbs.laurentienne(M) == gbs.montrealer(A)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_montrealer_agrees_with_lavalois(n):
    """Checks that the montrealer agrees with the lavalois"""
    N = gbs.hermitian_N(n)
    M = np.zeros((n, n))
    A = np.block([[M.conj(), N], [N.conj(), M]])

    if n == 1:
        laval = np.conjugate(gbs.lavalois(N))
    else:
        laval = gbs.lavalois(N)

    assert laval == gbs.montrealer(A)


@pytest.mark.parametrize("n", [1,2,3,4])
def test_loopmontrealer_agrees_cumulant(n):
	"""Checks that the loopmontrealer agrees with the cumulant"""
	zeta = symbols("alpha1:%d" % (n + 1))
	zeta = np.array(zeta)
	zeta = np.concatenate((zeta, zeta.conj()))
	A = gbs.symmetric_A(n, initial_index=1)
	modes = {i:1 for i in range(1,n+1)} # no repetition
	
	lmtl = gbs.loop_montrealer(A, zeta)
	cum = gbs.photon_number_cumulant(A, zeta, modes)
	assert lmtl == cum


@pytest.mark.parametrize("n", [2,3,4,5])
def test_moment_number_of_term(n):
    """Checks that the moment has (2n-1)!! terms"""
    """No displacement, no repetition"""
    A = gbs.symmetric_A(n)
    zeta = np.zeros(2*n) #no displacement
    modes = {i:1 for i in range(1,n+1)} #no repetition
    cumu = gbs.photon_number_moment(A, zeta, modes)
    assert len(cumu.args) == reduce(int.__mul__, range(2*n-1, 0, -2))


@pytest.mark.parametrize("n", [1,2,3,4])
def test_montrealer_number_of_term(n):
	"""Checks that the montrealer has (2n-2)!! terms"""
	A = gbs.symmetric_A(n)
	mtl = gbs.montrealer(A)
	terms = reduce(int.__mul__, range(2*n-2, 0, -2)) if n>1 else 3
	assert len(mtl.args) == terms


@pytest.mark.parametrize("n", [1,2,3,4])
def test_loopmontrealer_number_of_term(n):
	"""Checks that the loopmontrealer has (n+1)(2n-2)!! terms"""
	zeta = symbols("alpha1:%d" % (n + 1))
	zeta = np.array(zeta)
	zeta = np.concatenate((zeta, zeta.conj()))
	A = gbs.symmetric_A(n, initial_index=1)
	loopmtl = gbs.loop_montrealer(A, zeta)
	terms = (n+1)*reduce(int.__mul__, range(2*n-2, 0, -2)) if n>1 else 2
	assert len(loopmtl.args) == terms


@pytest.mark.parametrize("n", [1,3,4,5,6,7,8])
def test_laurentienne_number_of_term(n):
    """Checks that the laurentienne has (n-1)! terms if even, else 0"""
    M = gbs.symmetric_M(n)
    laur = gbs.laurentienne(M)
    if (n%2): #odd order
        assert laur == 0
    else:
        assert len(laur.args) == factorial(n-1)


@pytest.mark.parametrize("n", [3,4,5,6])
def test_lavalois_number_of_term(n):
	"""Checks that the lavalois has (n-1)! terms"""
	N = gbs.hermitian_N(n)
	laval = gbs.lavalois(N)
	assert len(laval.args) == factorial(n-1)


@pytest.mark.parametrize("n", [1,2,3,4,5])
def size_of_gspm(n):
    """Checks that the number of elements in gspm is precisely 2^n n!"""
    # Check that the number of elements in gspm is precisely 2^n n! or 2^{n-1} (n-1)!


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_symmetric_A(n):
    """Check that the A you get is symmetric"""
    A = gbs.symmetric_A(n)
    assert np.array_equal(A, A.T)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_shape_symmetric_A(n):
    """Check that the A you get has the correct shape"""
    A = gbs.symmetric_A(n)
    assert A.shape[0] == A.shape[1]
    assert A.shape[0] == 2 * n


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test__block_A(n):
    """Check that the A you get is a block matrix"""
    A = gbs.block_A(n)
    M = A[n : 2 * n, n : 2 * n]
    M_conj = A[:n, :n]
    N = A[:n, n : 2 * n]
    N_conj = A[n : 2 * n, :n]
    assert np.array_equal(M.conj(), M_conj)
    assert np.array_equal(N.conj(), N_conj)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_symmetric_block_A(n):
    """Check that the A you get is symmetric"""
    A = gbs.block_A(n)
    A_conj = A.T
    for i in range(n):
        A_conj[i, n + i] = np.conjugate(A_conj[i, n + i])
    assert np.array_equal(A, A_conj)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_shape_block_A(n):
    """Check that the A you get has the correct shape"""
    A = gbs.block_A(n)
    assert A.shape[0] == A.shape[1]
    assert A.shape[0] == 2 * n


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_symmetric_M(n):
    """Check that the M you get is symmetric"""
    M = gbs.symmetric_M(n)
    assert np.array_equal(M, M.T)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_shape_M(n):
    """Check that the M you get has the correct shape"""
    M = gbs.symmetric_M(n)
    assert M.shape[0] == M.shape[1]
    assert M.shape[0] == n


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_hermitian_N(n):
    """Check that the N you get is hermitian"""
    N = gbs.hermitian_N(n)
    N_conjugate = N.T.conj()
    for i in range(n):
        N_conjugate[i, i] = np.conjugate(N_conjugate[i, i])
    assert np.array_equal(N, N_conjugate)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_shape_hermitian_N(n):
    """Check that the N you get has the correct shape"""
    N = gbs.hermitian_N(n)
    assert N.shape[0] == N.shape[1]
    assert N.shape[0] == n


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_diagonal_N(n):
    """Check that the N you get is diagonal"""
    N = gbs.diagonal_N(n)
    np.fill_diagonal(N, 0)
    assert np.sum(N) == 0


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_shape_diagonal_N(n):
    """Check that the N you get has the correct shape"""
    N = gbs.diagonal_N(n)
    assert N.shape[0] == N.shape[1]
    assert N.shape[0] == n


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_montrealers_agree(n):
    """Checks that the montrealer numba agrees with the montrealer"""
    covmat = random_covariance(n)
    Q = Qmat(covmat) - np.identity(2 * n)
    A = Xmat(n) @ Q
    assert np.allclose(gbs.montrealer(A), gbs.montrealer_numba(A))


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7])
def test_laurentienne_numba_agrees_with_laurentienne(n):
    """Checks that the laurentienne numba agrees with the laurentienne"""
    U = unitary_group.rvs(n)
    M = U @ np.diag(np.random.rand(n)) @ U.T
    laur_numba = round(gbs.laurentienne_numba(M).real, 12)
    laur = round(complex(gbs.laurentienne(M)).real,12)
    assert laur_numba == laur


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7])
def test_lavalois_numba_agrees_with_lavalois(n):
    """Checks that the lavalois numba agrees with the lavalois"""
    U = unitary_group.rvs(n)
    N = U.conj() @ np.diag(np.random.rand(n)) @ U.T
    lav_numba = round(gbs.lavalois_numba(N).real, 12)
    lav = round(complex(gbs.lavalois(N)).real,12)
    assert lav_numba == lav