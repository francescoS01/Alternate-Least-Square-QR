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


# function LS_QR(Q, R, m, n, y)

#     # caso matrice A thin 
#     R0_dim = n # quadrata
#     Q0_row_size = m
#     Q0_col_size = n

#     # calcolo R0 
#     R0 = R[1:R0_dim, 1:R0_dim]
#     # calcolo Q0
#     Q0 = Q[1:Q0_row_size, 1:Q0_col_size]

#     # Risolviamo il sistema lineare Rx = Q^T y per trovare il vettore x
#     x = inv(R0) * (Q0'*y)
#     return x
# end

function LS_QR(householder_vectors, R, m, n, y)
    # Applicare i riflettori di Householder su y
    for k = 1:n  # Applica i riflettori in ordine
        u = householder_vectors[k]  # Riflettore di Householder
        # println("Dimensions of u: ", length(u))
        # println("Dimensions of y: ", length(y))
        # La porzione di y sulla quale il riflettore u agisce
        sub_y = y[k:m]  # Porzione di y corrispondente al riflettore

        # Applicazione del riflettore a sub_y: sub_y = sub_y - 2 * (u' * sub_y) * u
        y[k:m] -= 2 * (u' * sub_y) * u  # Applica il riflettore alla porzione corretta di y
    end

    # Ora risolvi Rx = Q^T * y (y è ora Q^T * y)
    x = R \ y  # Risoluzione del sistema triangolare
    return x
end