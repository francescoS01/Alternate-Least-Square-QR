using LinearAlgebra
using Random
using Printf
include("print_matrix.jl")


function householder(x)
    """
    Input: a vector x
    Output: a normalized householder vector u 
    """
    
    s = norm(x)
   
    # for numerical stability
    if x[1] >= 0
        s = - s; 
    end
        
    v = x
    # v = x - e1*||x|| ()
    v[1] = v[1] - s  
    u = v / norm(v)
    
    return u
    
end


function qr_fact(A)
    
    m, n = size(A)
    R = A
    Q = Matrix{Float64}(I[1:m, 1:m])
    
    for i = 1:min(m, n)
        # Houselder vector for the i-th column of A
        u = householder(A[i:m, i])
        H = I[i:m, i:m] - 2 * u * u'
        Q[1:m, i:m] = Q[1:m, i:m] * H
        # moltiplico sotto matrice di R che verrà modificata
        R[i:m, i:n] = H * A[i:m, i:n]  
        
    end
    return Q, R
end





"""
A = rand(3,3)
print_matrix(A)
print("-----\n")
#B = copy(A)

Q, R = qr_fact(A)

#print_matrix(Q)
print("-----\n")
#print_matrix(R)
print("-----\n")
print_matrix(Q*R)
"""