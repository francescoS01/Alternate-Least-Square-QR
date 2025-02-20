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
        return x  # Se la norma è zero, torna il vettore originale
    end
    if x[1] >= 0
        s = -s  # Per stabilità numerica
    end
    v = copy(x)  # Copia per evitare di modificare il vettore originale
    v[1] -= s  # Modifica il primo elemento
    u = v / norm(v)  # Normalizza il vettore
    return u
end


function qr_fact(A)
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
