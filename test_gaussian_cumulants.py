"""
Unit tests for symbolic_cumulants.py
"""

import pytest
import symbolic_cumulants as gbs
import numpy as np

@pytest.mark.parametrize("n", [1,2,3,4,5])
def test_montrealer_agrees_with_cumulants(n):
	"""Checks that the montrealer and the cumulant function agree"""
	# Make an A with parameter n (of size 2n)
	# You pass it to cumulant with no repetions and zero displacement
	# You pass it to the montrealer

	#assert result_from_montrealer == result_from_cumulant


@pytest.mark.parametrize("n", [2,3,4,5,6,7])
def test_montrealer_agrees_with_laurentienne(n):
	"""Checks that the montrealer agrees with the laurentienne"""
	M = gbs.symmetric_M(n)
	N = gbs.diagonal_N(n)
	A = np.block([[M.conj(), N], [N.conj(), M]])
	assert gbs.laurentienne(M) == gbs.montrealer(A)


@pytest.mark.parametrize("n", [1,2,3,4,5])
def test_montrealer_agrees_with_lavalois(n):
	"""Checks that the montrealer agrees with the lavalois"""
	N = gbs.hermitian_N(n)
	M = np.zeros((n,n))
	A = np.block([[M.conj(), N], [N.conj(), M]])

	if n == 1:
		laval = np.conjugate(gbs.lavalois(N))
	else:
		laval = gbs.lavalois(N)

	assert laval == gbs.montrealer(A)


@pytest.mark.parametrize("n", [1,2,3,4,5])
def size_of_gspm(n):
	"""Checks that the number of elements in gspm is precisely 2^n n!"""
	# Check that the number of elements in gspm is precisely 2^n n! or 2^{n-1} (n-1)!


@pytest.mark.parametrize("n", [1,2,3,4,5])
def test_symmetric_A(n):
	"""Check that the A you get is symmetric"""
	A = gbs.symmetric_A(n)
	assert np.array_equal(A, A.T)


@pytest.mark.parametrize("n", [1,2,3,4,5])
def test_shape_symmetric_A(n):
	"""Check that the A you get has the correct shape"""
	A = gbs.symmetric_A(n)
	assert A.shape[0] == A.shape[1]
	assert A.shape[0] == 2*n


@pytest.mark.parametrize("n", [1,2,3,4,5])
def test__block_A(n):
	"""Check that the A you get is a block matrix"""
	A = gbs.block_A(n)
	M = A[n:2*n, n:2*n]
	M_conj = A[:n,:n]
	N = A[:n,n:2*n]
	N_conj = A[n:2*n,:n]
	assert np.array_equal(M.conj(), M_conj)
	assert np.array_equal(N.conj(), N_conj)


@pytest.mark.parametrize("n", [1,2,3,4,5])
def test_symmetric_block_A(n):
	"""Check that the A you get is symmetric"""
	A = gbs.block_A(n)
	A_conj = A.T
	for i in range(n):
		A_conj[i, n+i] = np.conjugate(A_conj[i, n+i])
	assert np.array_equal(A, A_conj)


@pytest.mark.parametrize("n", [1,2,3,4,5])
def test_shape_block_A(n):
	"""Check that the A you get has the correct shape"""
	A = gbs.block_A(n)
	assert A.shape[0] == A.shape[1]
	assert A.shape[0] == 2*n


@pytest.mark.parametrize("n", [1,2,3,4,5])
def test_symmetric_M(n):
	"""Check that the M you get is symmetric"""
	M = gbs.symmetric_M(n)
	assert np.array_equal(M, M.T)


@pytest.mark.parametrize("n", [1,2,3,4,5])
def test_shape_M(n):
	"""Check that the M you get has the correct shape"""
	M = gbs.symmetric_M(n)
	assert M.shape[0] == M.shape[1]
	assert M.shape[0] == n


@pytest.mark.parametrize("n", [1,2,3,4,5])
def test_hermitian_N(n):
	"""Check that the N you get is hermitian"""
	N = gbs.hermitian_N(n)
	N_conjugate = N.T.conj()
	for i in range(n):
		N_conjugate[i,i] = np.conjugate(N_conjugate[i,i])
	assert np.array_equal(N, N_conjugate)


@pytest.mark.parametrize("n", [1,2,3,4,5])
def test_shape_hermitian_N(n):
	"""Check that the N you get has the correct shape"""
	N = gbs.hermitian_N(n)
	assert N.shape[0] == N.shape[1]
	assert N.shape[0] == n


@pytest.mark.parametrize("n", [1,2,3,4,5])
def test_diagonal_N(n):
	"""Check that the N you get is diagonal"""
	N = gbs.diagonal_N(n)
	np.fill_diagonal(N, 0)
	assert np.sum(N)==0


@pytest.mark.parametrize("n", [1,2,3,4,5])
def test_shape_diagonal_N(n):
	"""Check that the N you get has the correct shape"""
	N = gbs.diagonal_N(n)
	assert N.shape[0] == N.shape[1]
	assert N.shape[0] == n