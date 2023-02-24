




def photon_number_moment(A, zeta, modes):
	""" Returns the photon number moment of the modes in the Gaussian state.

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
	""" Returns the photon number cumulant of the modes in the Gaussian state.

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


def montrealer(A): # This is technically the "new" montrealer
	r""" Calculates the Montrealer of a square symmetric matrix of even size.

	Args:
		A (array): square even-sized complex-symmetric matrix representing the covariance of the Gaussian state.

	Returns:
		(complex): the value of the montrealer

	"""

def laurentienne(M): # This is technically the "old" montrealer
	r""" Calculates the laurentienne of a square symmetric matrix.

	Args:
		M (array): square complex-symmetric matrix representing the phase-sensitive quadrature moments of the Gaussian state.

	Returns:
		(complex): the value of the laurentienne
	"""
	# If size of matrix equals odd return 0

def lavalois(N):
	r""" Calculates the lavalois of a square hermitian matrix.

	Args:
		M (array): square hermitrian matrix representing the phase-insensitive quadrature moments of the Gaussian state.

	Returns:
		(complex): the value of the lavalois
	"""

def symmetric_A(n, initial_index=0):
	""" Return a symbolic matrix of size 2n with entries index by A.

	Args:
		n (int): number of modes
		start_from_zero (int): initial value for indexing

	Returns:
		(array): a symbolic array

	"""

def block_A(n, initial_index=0):
	""" Returns FILL THE GAPS

	Args:
		n (int): number of modes
		start_from_zero (int): initial value for indexing

	Returns:
		(array): a symbolic array

	"""

def symmetric_M(n, initial_index=0):
	""""""


def hermitian_N(n, initial_index=0):
	""""""


def diagonal_N(n, initial_index=0):
	""""""

##################
# Add functions to easily make symmetric A, block A, symmetric M, hermitian N and diagonal N
##################