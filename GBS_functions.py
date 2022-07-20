from math import factorial
from sympy import MatrixSymbol, Matrix, conjugate, simplify, symbols, sinh, eye, zeros
import numpy as np
from thewalrus.reference import hafnian
from itertools import product

###################################################################################
#Equation (9.3.8) gives all the values of C_n
def ordered_ladder(m):
	coef = []
	for n in range(1,m+1):
		coef.append(int(sum([(-1)**(n-k)*k**m/factorial(n-k)/factorial(k) for k in range(1,n+1)])))
	return coef

###################################################################################
#Equation (9.5.7)
def moment_coefficient_calculator(VECTOR_J, VECTOR_K):
    coef = 1 #multiplication of coefficients from all modes

    for s in range(len(VECTOR_J)):
        js = VECTOR_J[s]
        ks = VECTOR_K[s]
        cs = 0 #single mode coefficient

        for l in range(1,js+1):
            cs+=(-1)**(js-l)*l**ks/factorial(js-l)/factorial(l)

        coef*=cs if cs != 0 else 1 #to take into acount the possibility that js is 0
    
    return int((-1)**sum(VECTOR_J)*coef)

##################################################################################
#Equation (9.5.8)
#Displacement is False if there is no displacement
#Squeezing is False if all modes have the same parameter r
def matrix_B_builder(VECTOR_J, displacement=True, N_diagonal=False):
    m = len(VECTOR_J) #number of modes
    J = sum(VECTOR_J)

    if N_diagonal:
        r = symbols('r')
        r = sinh(r)**2
        N = r*eye(m)
    else:
        N = zeros(m,m)
        N_buffer = MatrixSymbol('N', m, m)
        for i in range(m-1):
            for j in range(i+1,m):
                N[j,i] = conjugate(N_buffer[i,j])
                N[i,j] = N_buffer[i,j]

        for i in range(m):
            N[i,i] = N_buffer[i,i]

    M = zeros(m,m)
    M_buffer = MatrixSymbol('M', m, m)
    for i in range(m-1):
        for j in range(i+1,m):
            M[j,i] = M_buffer[i,j]
            M[i,j] = M_buffer[i,j]
    
    for i in range(m):
        M[i,i] = M_buffer[i,i]

    O = conjugate(N)
    P  = conjugate(M)

    gamma = MatrixSymbol('L', m, 1)
    conj_gamma = conjugate(gamma)
    minus_gamma = -gamma

    B = -np.block([[M, N],[O,P]])

    #COLUMN DUPLICATION
    #the first column, deleted later, is kept for size
    B_col = np.array(B[:, 0]).reshape([2*m,1])
    for i in range(2): #the columns are always duplicated in pairs
        for s, j in enumerate(VECTOR_J):
            for _ in range(j): # _ is convention for a variable we dont care about
                B_col = np.append(B_col, B[:, s+i*m].reshape([2*m,1]), axis=1)

    B_col = np.delete(B_col, 0, axis=1) #delete first column

    #ROW DUPLICATION
    #the first row, deleted later, is kept for size
    B_tilde = np.array(B_col[0,:]).reshape([1,2*J])
    for i in range(2):
        for s, j in enumerate(VECTOR_J):
            for _ in range(j):
                B_tilde = np.append(B_tilde, B_col[s+i*m,:].reshape([1,2*J]), axis = 0)
                
    B_tilde = np.delete(B_tilde, 0, axis=0)

    #DISPLACEMENT DIAGONAL
    minus_displacement = []
    conj_displacement = []
    for s, j in enumerate(VECTOR_J):
        for _ in range(j):
            conj_displacement = np.append(conj_displacement, conj_gamma[s])
            minus_displacement = np.append(minus_displacement, minus_gamma[s])

    tot_displacement = np.append(conj_displacement, minus_displacement)

    for i in range(2*J):
        B_tilde[i,i] = tot_displacement[i] if displacement else 0

    return B_tilde

#########################################################################################
#Equation (9.5.6)
def moment_calculator(VECTOR_K, displacement=True, N_diagonal=False):
    dummy = [list(range(1,i+1)) if i!=0 else [0] for i in VECTOR_K]
    indice = product(*dummy)

    moment_list = []
    for VECTOR_J in indice:
        coefficient = moment_coefficient_calculator(VECTOR_J, VECTOR_K)
        B_tilde = matrix_B_builder(VECTOR_J, displacement, N_diagonal)
        moment_list = np.append(moment_list, coefficient*hafnian(B_tilde, loop=True))

    return sum(moment_list)

###########################################################################################
#PARTITION
#taken from : https://stackoverflow.com/questions/19368375/set-partitions-in-python
def partition(collection):
    if len(collection) == 1:
        yield [ collection ]
        return

    first = collection[0]
    for smaller in partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
        # put `first` in its own subset 
        yield [ [ first ] ] + smaller

###########################################################################################
#CUMULANT
def cumulant_calculator(ORDER, displacement=True, N_diagonal=False): #Ex order = 3
    indice = [i+1 for i in  range(ORDER)]
    part = partition(indice)
    cumulant = 0
    
    for p in part: #Ex p = [[1,3],[2]]
        size = len(p)-1
        cum = factorial(size)*(-1)**size #prefactor
        
        for b in p: #Ex b = [1,3]
            VECTOR_K = np.zeros(ORDER, int)

            for k in b:
                VECTOR_K[k-1] = 1 #Ex vector_k = [1,0,1]
            
            cum*=moment_calculator(VECTOR_K, displacement, N_diagonal)
        
        cumulant+=cum

    return simplify(cumulant)