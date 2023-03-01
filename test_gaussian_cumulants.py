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


@pytest.mark.parametrize("n", [1,2,3,4,5])
def test_montrealer_agrees_with_laurentienne(n):
	"""Checks that the montrealer agrees with the laurentienne"""
	# Call diagonal_N and symmetric_M
	# Make and A matrix with the right structure
	# Compare the results and assert



@pytest.mark.parametrize("n", [1,2,3,4,5])
def test_montrealer_agrees_with_lavalois(n):
	"""Checks that the montrealer agrees with the lavalois"""
	# Call diagonal_N and symmetric_M
	# Make and A matrix with the right structure
	# Compare the results and assert

@pytest.mark.parametrize("n", [1,2,3,4,5])
def size_of_gspm(n):
	"""Checks that the number of elements in gspm is precisely 2^n n!"""
	# Check that the number of elements in gspm is precisely 2^n n! or 2^{n-1} (n-1)!


@pytest.mark.parametrize("n", [1,2,3,4,5])
def test_A_symmetric(n):
	"""Check that the A you get is symmetric"""


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