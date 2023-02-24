

import pytest


####
# How to check equality from sympy
####

@pytest.mark.parametrize("n", [1,2,3,4,5])
def test_montrealer_agrees_with_cumulants(n):
	"""Checks that the montraler and the cumulant function agree"""
	# Make an A with parameter n (of size 2n)
	# You pass it to cumulant with no repetions and zero displacement
	# You pass it to the montrealer
	assert result_from_montrealer == result_from_cumulant


@pytest.mark.parametrize("n", [1,2,3,4,5])
def test_montrealer_agrees_with_laurentienne(n):
	# Call diagonal_N and symmetric_M
	# Make and A matrix with the right structure
	# Compare the results and assert



@pytest.mark.parametrize("n", [1,2,3,4,5])
def test_montrealer_agrees_with_lavalois(n):
	# Call diagonal_N and symmetric_M
	# Make and A matrix with the right structure
	# Compare the results and assert

@pytest.mark.parametrize("n", [1,2,3,4,5])
def size_of_gspm(n):
	# Check that the number of elements in gspm is precisely 2^n n! or 2^{n-1} (n-1)!


@pytest.mark.parametrize("n", [1,2,3,4,5])
def test_A_symmetric(n):
	"""Check that the A you get is symmetric"""
