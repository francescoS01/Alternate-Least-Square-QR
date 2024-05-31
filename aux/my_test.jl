include("../utils/alternate_LSQR.jl")
include("../utils/print_matrix.jl")
include("../utils/low_rank_SVD.jl")
using LinearAlgebra
using Random
using Printf

A = [2 0; 1 0]
k = 1
e = 0.0001 # soglia tra due iterazioni


m, n = size(A) # m x n 

#V_iterative = [1; 0;;]
V_iterative = rand(n, k)
U_iterative = rand(m, k)
err = norm(A - U_iterative * V_iterative')
dif_err = Inf

while dif_err > e
    
    # CASO 1. fiso V e cerco U, m sotto problemi (col di U' e A')
    for i in 1:m
        a_col = copy(A'[:, i])
        U_iterative'[:, i] = LS_QR(copy(V_iterative), copy(a_col))
    end
    
    # CASO 2. fiso U e cerco V, n sotto problemi 
    for s in 1:n
        a_col = copy(A[:, s])
        V_iterative'[:, s] = LS_QR(copy(U_iterative), copy(a_col))
    end
    
    """
    print("ERRORE ", j, " : ", norm(A - U_iterative * V_iterative'))
    print("\n-------\n")
    """
    
    # variazione errore rispetto a iterazione precedente (norma frobenius)
    dif_err = err - sqrt(sum(abs2, (A - U_iterative * V_iterative')))
    # nuovo errore 
    err = sqrt(sum(abs2, (A - U_iterative * V_iterative')))
end	

U = U_iterative	
V = V_iterative





Ak_QR =  U*V'

# chiamata con SVD ( perfect approximation )
Ak_SVD = low_rank(A, k)

# confronto le due soluzioni 
print("\n-------  matrice U*V' con QR  -------\n")
print_matrix(Ak_QR)

print("\n-------    matrice U*V' SVD   -------\n")
print_matrix(Ak_SVD)

print("\n-------        errore         -------\n")
print( sqrt( sum( abs2, (Ak_QR - Ak_SVD) ) ) )