"""
problem: arg min (x) || Ax - y ||

A dim = m x n
x dim = n x 1
y dim = m x 1


"""
include("QR_factorization.jl")

function LS_QR(A, y)
    Q, R = qr_fact(A)
    m, n = size(A)
    
    # dim di R0 (quadrata) 
    R0_dim = min(m, n)
    R0 = R[1:R0_dim, 1:R0_dim]
    Q0_col_size = 
    Q0_row_size 
