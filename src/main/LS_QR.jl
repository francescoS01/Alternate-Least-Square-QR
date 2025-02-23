include("QR_factorization.jl")
include("utils/print_matrix.jl")
using LinearAlgebra
using Random
using Printf
 

"""
problem: arg min (x) || Ax - y ||

A dim = m x n
x dim = n x 1
y dim = m x 1
"""


function LS_QR(householder_vectors, R, m, n, y)
    """
    Solves `Rx = Qᵀy` using Householder reflections without explicitly forming `Q`.

    # Arguments
    - `householder_vectors`: Householder vectors from QR factorization.
    - `R`: Upper triangular `n × n` matrix.
    - `m, n`: Dimensions of the original matrix.
    - `y`: Right-hand side vector.

    # Method
    Applies Householder reflectors iteratively to compute `Qᵀy`, then solves the triangular system `Rx = Qᵀy`.

    # Returns
    - `x`: Solution vector.
    """

    for k = 1:n  
        u = householder_vectors[k]  
        # take the subvector of y from k to m where the householder vector is applied
        sub_y = y[k:m] 
        y[k:m] -= 2 * (u' * sub_y) * u  
    end

    # Now solve Rx = Q^T * y (y is now Q^T * y)
    x = R \ y  
    return x
end