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

# # Funzione per la fattorizzazione QR
# function qr_fact(A)
#     m, n = size(A)
#     R = copy(A)  # Fai una copia di A in modo che non venga modificata
#     Q = Matrix{Float64}(I, m, m)  # Inizializza Q come una matrice identità

#     for i = 1:min(m, n)
#         # Estrai il vettore Householder per la colonna i-esima
#         u = householder(R[i:m, i])
        
#         # Costruisci la matrice di Householder
#         H = Matrix{Float64}(I, m, m)  # Matrice identità m x m
#         H[i:m, i:m] -= 2 * u * u'  # Modifica solo la parte i:m di H

#         # Aggiorna Q e R
#         Q = Q * H  # Applica H a Q
#         R = H * R  # Applica H a R
#     end

#     return Q, R
# end


# function qr_fact(A)
#     m, n = size(A)
#     R = copy(A)
#     Q = Matrix{Float64}(I, m, m)

#     for i = 1:min(m, n)
#         # Calcolo del vettore di Householder
#         u = householder(R[i:end, i])           # Vettore u per la riflessione
#         u_full = zeros(m)                      # Estensione di u alle dimensioni di R
#         u_full[i:end] = u                      # Riempimento delle posizioni corrette
        
#         # Applica la riflessione a R senza formare H
#         R -= 2 * u_full * (u_full' * R)        
        
#         # Applica la riflessione a Q senza formare H
#         Q -= 2 * (Q * u_full) * u_full'        
#     end

#     return Q, R
# end


# Funzione per la fattorizzazione QR
# function qr_fact(A)
#     m, n = size(A)
#     R = copy(A)  # Fai una copia di A in modo che non venga modificata
#     Q = Matrix{Float64}(I, m, m)  # Inizializza Q come una matrice identità

#     for i = 1:min(m, n)
#         # Estrai il vettore Householder per la colonna i-esima
#         u = householder(R[i:m, i])
        
#         # Applica la trasformazione di Householder a R
#         R[i:m, :] -= 2 * u * (u' * R[i:m, :])
        
#         # Applica la trasformazione di Householder a Q
#         Q[:, i:m] -= 2 * (Q[:, i:m] * u) * u'
#     end

#     return Q, R
# end

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






# OPTIMIZED VERSION
# function qr_fact(A)
#     m, n = size(A)
#     Q = Matrix{Float64}(undef, m, n)
#     R = zeros(Float64, n, n)

#     for j = 1:n
#         v = A[:, j]
#         for i = 1:j-1
#             R[i, j] = Q[:, i]' * A[:, j]
#             v -= R[i, j] * Q[:, i]
#         end
#         R[j, j] = norm(v)
#         Q[:, j] = v / R[j, j]
#     end

#     return Q, R
# end

# function qr_fact(A)
#     return qr(A).Q, qr(A).R
# end

# function qr_fact(A)
#     m, n = size(A)
#     Q = Matrix{Float64}(I, m, m)
#     R = copy(A)

#     for j = 1:n
#         for i = m:-1:(j+1)
#             a = R[i-1, j]
#             b = R[i, j]
#             if b != 0
#                 r = sqrt(a^2 + b^2)
#                 c = a / r
#                 s = -b / r

#                 # Applica la rotazione di Givens a R
#                 for k = j:n
#                     temp = c * R[i-1, k] - s * R[i, k]
#                     R[i, k] = s * R[i-1, k] + c * R[i, k]
#                     R[i-1, k] = temp
#                 end

#                 # Applica la rotazione di Givens a Q
#                 for k = 1:m
#                     temp = c * Q[k, i-1] - s * Q[k, i]
#                     Q[k, i] = s * Q[k, i-1] + c * Q[k, i]
#                     Q[k, i-1] = temp
#                 end
#             end
#         end
#     end

#     return Q, R
# end

# # Funzione per calcolare il vettore di Householder
# function householder(x)
#     """
#     Input: a vector x
#     Output: a normalized householder vector u 
#     """
#     s = norm(x)
#     if s == 0
#         return x  # Se la norma è zero, torna il vettore originale
#     end
#     if x[1] >= 0
#         s = -s  # Per stabilità numerica
#     end
#     v = copy(x)  # Copia per evitare di modificare il vettore originale
#     v[1] -= s  # Modifica il primo elemento
#     u = v / norm(v)  # Normalizza il vettore
#     return u
# end

# # Funzione per la fattorizzazione QR
# function qr_fact(A)
#     m, n = size(A)
#     R = copy(A)  # Fai una copia di A in modo che non venga modificata
#     Q = Matrix{Float64}(I, m, m)  # Inizializza Q come una matrice identità

#     for i = 1:min(m, n)
#         # Estrai il vettore Householder per la colonna i-esima
#         u = householder(R[i:m, i])
        
#         # Costruisci la matrice di Householder
#         H = I - 2 * u * u'  # Matrice di Householder

#         # Aggiorna Q e R
#         Q[:, i:m] = Q[:, i:m] * H  # Applica H a Q
#         R[i:m, :] = H * R[i:m, :]  # Applica H a R
#     end

#     return Q, R
# end



# function qr_fact(A)
#     m, n = size(A)
#     Q = Matrix{Float64}(I, m, m)  # Matrice identità m×m con Float64
#     R = copy(A)

#     for k = 1:n
#         # Estrai il vettore della colonna k da R
#         x = R[k:m, k]

#         # Crea il vettore di Householder
#         e1 = zeros(length(x))
#         e1[1] = norm(x)
#         v = x - e1
#         v = v / norm(v)

#         # Costruisci la matrice di Householder Hk
#         Hk = Matrix{Float64}(I, m, m)  # Assicura che sia di tipo Float64
#         Hk[k:m, k:m] -= 2 * (v * v')

#         # Applica Hk a R e aggiorna Q
#         R = Hk * R
#         Q = Q * Hk
#     end

#     return Q[:, 1:n], R[1:n, :]
# end
