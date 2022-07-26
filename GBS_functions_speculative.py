from itertools import product
from math import factorial
from sympy import MatrixSymbol, Matrix, conjugate, simplify, expand, symbols, sinh, eye, zeros
import numpy as np
from thewalrus.reference import hafnian
from GBS_functions import partition
#pylint: disable=C0103 

def symmetric_M(m):
    """
    Returns a symmetric symbolic matrix of the given size
    """
    matrix_M = zeros(m,m)
    buffer_M = MatrixSymbol('M', m, m)
    
    #M is symmetric
    for i in range(m-1):
        for j in range(i+1,m):
            matrix_M[j,i] = buffer_M[i,j]
            matrix_M[i,j] = buffer_M[i,j]
    return np.array(matrix_M)


def speculative_cumulant_calculator(order): #Ex order = 3
    indices = list(range(order))
    part = partition(indices)
    cumulant = 0
    M = symmetric_M(order)

    for p in part: #Ex p = [[1,3],[2]]
        size = len(p)-1
        cum = factorial(size)*(-1)**size #prefactor
        
        for b in p: #Ex b = [1,3]
            local_M = M[b][:,b]
            haf = hafnian(local_M)
            cum*=haf*conjugate(haf)
        
        cumulant+=cum

    return expand(cumulant)



def montrealer(M):
    order = len(M)
    if order%2 != 0:
        return 0
    indices = list(range(order))
    part = partition(indices)
    cumulant = 0
    for p in part: #Ex p = [[1,3],[2]]
        # Check there are no partitions that odd length parts.
        check_weight =  True 
        for i in p:
            if len(i)%2 != 0:
                check_weight = False
                break
        if check_weight:
            size = len(p)-1
            cum = factorial(size)*(-1)**size #prefactor
            for b in p: #Ex b = [1,3]
                local_M = M[b][:,b]
                haf = hafnian(local_M)
                cum*=haf*conjugate(haf)

            cumulant+=cum

    return cumulant
