from itertools import product
from math import factorial
from sympy import MatrixSymbol, Matrix, conjugate, simplify, expand, symbols, sinh, eye, zeros
import numpy as np
from thewalrus.reference import hafnian
#pylint: disable=C0103 

def speculative_matrix_builder(vector_J):
    """
    Returns
    
    Args:
        vector_J (list):

    Returns:
    """
    m = len(vector_J) #number of modes
    J = sum(vector_J)
    matrix_M = zeros(m,m)
    buffer_M = MatrixSymbol('M', m, m)
    
    #M is symmetric
    for i in range(m-1):
        matrix_M[i,i] = buffer_M[i,i]
        for j in range(i+1,m):
            matrix_M[j,i] = buffer_M[i,j]
            matrix_M[i,j] = buffer_M[i,j]
    
    #matrix_M[m-1,m-1] = buffer_M[m-1,m-1]

    #Removes rows and columns
    for s, j in enumerate(reversed(vector_J)):
        if not j:
            matrix_M = np.delete(matrix_M, m-s-1, axis=0)
            matrix_M = np.delete(matrix_M, m-s-1, axis=1)

    #no displacement. Zero diagonal
    for i in range(J):
        matrix_M[i,i] = 0

    return matrix_M

def speculative_moment_calculator(vector_K):
    """
        Returns
    
    Args:
        

    Returns:
    """
    matrix_M = speculative_matrix_builder(vector_K)
    return expand(hafnian(matrix_M)*hafnian(conjugate(matrix_M))) #should i expand?

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

def speculative_cumulant_calculator(order): #Ex order = 3
    indice = [i+1 for i in  range(order)]
    part = partition(indice)
    cumulant = 0
    
    for p in part: #Ex p = [[1,3],[2]]
        size = len(p)-1
        cum = factorial(size)*(-1)**size #prefactor
        
        for b in p: #Ex b = [1,3]
            vector_K = np.zeros(order, int)

            for k in b:
                vector_K[k-1] = 1 #Ex vector_k = [1,0,1]
            
            cum*=speculative_moment_calculator(vector_K)
        
        cumulant+=cum

    return simplify(cumulant)