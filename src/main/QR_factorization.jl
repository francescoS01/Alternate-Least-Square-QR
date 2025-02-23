using LinearAlgebra
using Random
using Printf
using Base.Threads

# Funzione per calcolare il vettore di Householder
function householder(x)
    """
    Input: a vector x
    Output: a normalized householder vector u 
    """
    s = norm(x)
    if s == 0
        return x 
    end
    if x[1] >= 0
        s = -s  # for numerical stability
    end
    v = copy(x) 
    v[1] -= s  # modify only the first element 
    u = v / norm(v)  # Normalization 
    return u
end


function qr_fact(A)
    """
    # Arguments
    - `A::AbstractMatrix`: The input matrix to be factorized.

    # Returns
    - `householder_vectors::Vector`: A vector containing the Householder vectors used to compute `Q`.
    - `R::AbstractMatrix`: The upper triangular matrix `R` from the QR factorization.

    """
    m, n = size(A)
    R = copy(A)  # Copia di A
    householder_vectors = []  # Vettori di Householder da usare per calcolare Q

    for i = 1:n
        # Estrai il vettore Householder per la colonna i-esima
        u = householder(R[i:m, i])
        
        # Salva il riflettore di Householder
        push!(householder_vectors, u)
        
        # Applica la trasformazione di Householder a R
        R[i:m, :] -= 2 * u * (u' * R[i:m, :])
    end

    return householder_vectors, R
end
