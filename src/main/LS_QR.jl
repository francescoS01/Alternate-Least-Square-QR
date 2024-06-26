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



function LS_QR(A, y)
    Q, R = qr_fact(A)
    m, n = size(A)

    # caso matrice A thin 
    R0_dim = n # quadrata
    Q0_row_size = m
    Q0_col_size = n

    # calcolo R0 
    R0 = R[1:R0_dim, 1:R0_dim]
    # calcolo Q0
    Q0 = Q[1:Q0_row_size, 1:Q0_col_size]

    # Risolviamo il sistema lineare Rx = Q^T y per trovare il vettore x
    x = inv(R0) * (Q0'*y)
    return x
end

function LS_QR2(Q, R, m, n, y)
    #Q, R = qr_fact(A)
    #m, n = size(Q*R)

    # caso matrice A thin 
    R0_dim = n # quadrata
    Q0_row_size = m
    Q0_col_size = n

    # calcolo R0 
    R0 = R[1:R0_dim, 1:R0_dim]
    # calcolo Q0
    Q0 = Q[1:Q0_row_size, 1:Q0_col_size]

    # Risolviamo il sistema lineare Rx = Q^T y per trovare il vettore x
    x = inv(R0) * (Q0'*y)
    return x
end



