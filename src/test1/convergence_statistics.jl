using LinearAlgebra
using Random
using Printf
using Base
using Plots
using SparseArrays
include("../main/alternate_LSQR.jl")
include("../main/utils/print_matrix.jl")
include("../main/utils/time_gap.jl")
include("../main/low_rank_SVD.jl")
# sizes = 
# [(10,90), (30,90), (80,90), (100,150),      
# (90,10), (90,30), (90,80), (150,100),    
# (50,50),(120,120),(150,150)]

#---------------------------------------------------------------------------------------
# the gap between our algorithm and SVD solution when varyng A dimention

function gap_A_var()
    global m = 50  # variable
    global n = 50  # variable
    k = 10 
    e = 0.001
    LSQR_SVD = []
    # Plot the execution times as the matrix sizes of the 3 methods vary in time_gap
    for _ in 1:11
        A = rand(m, n)
        V_initial = rand(n, k)
        _, gapSVD, _ = time_gap(copy(A), k, e, copy(V_initial), parallel=true)
        push!(LSQR_SVD, Dict(:dim => (m, n), :frobenius_norm => gapSVD))
        global m += 10
        global n += 10
    end
    return LSQR_SVD
end


function gap_A_var_Hilbert_poorly_conditioned()
    global m = 50  # variable
    global n = 50  # variable
    k = 10 
    e = 0.001
    LSQR_SVD = []
    # Plot the execution times as the matrix sizes of the 3 methods vary in time_gap
    for _ in 1:11
        # Hilbert poorly conditioned square matrix
        A = [1/(i+j-1) for i in 1:n, j in 1:m]
        print("\n")
        print_matrix(A)
        print("\n")
        V_initial = rand(n, k)
        _, gapSVD, _ = time_gap(copy(A), k, e, copy(V_initial), parallel=true)
        push!(LSQR_SVD, Dict(:dim => (m, n), :frobenius_norm => gapSVD))
        global m += 10
        global n += 10
    end
    return LSQR_SVD
end


function gap_A_var_Vandermonde_poorly_conditioned()
    global m = 50  # variable
    global n = 50  # variable
    k = 10 
    e = 0.001
    LSQR_SVD = []
    # Plot the execution times as the matrix sizes of the 3 methods vary in time_gap
    for _ in 1:11
        #Vandermonde matrix
        A = [i^(j-1) for i in 1:n, j in 1:m]
        V_initial = rand(n, k)
        _, gapSVD, _ = time_gap(copy(A), k, e, copy(V_initial), parallel=true)
        push!(LSQR_SVD, Dict(:dim => (m, n), :frobenius_norm => gapSVD))
        global m += 10
        global n += 10
    end
    return LSQR_SVD
end


function gap_A_var_square()
    global m = 50  # variable
    global n = 50  # variable
    k = 10 
    e = 0.001
    LSQR_SVD = []
    # Plot the execution times as the matrix sizes of the 3 methods vary in time_gap
    for _ in 1:5
        A = rand(m, n)
        V_initial = rand(n, k)
        _, gapSVD, _ = time_gap(copy(A), k, e, copy(V_initial), parallel=true)
        push!(LSQR_SVD, Dict(:dim => (m, n), :frobenius_norm => gapSVD))
        global m += 40
        global n += 40
    end
    return LSQR_SVD
end


function gap_A_var_thin() # m>n
    sizes = [(125,20), (270,30), (338,50), (578,50), (630,70)]
    k = 10 
    e = 0.001
    LSQR_SVD = []
    
    for i in 1:5
        m, n = sizes[i]
        A = rand(m, n)
        V_initial = rand(n, k)
        _, gapSVD, _ = time_gap(copy(A), k, e, copy(V_initial), parallel=true)
        push!(LSQR_SVD, Dict(:dim => (m, n), :frobenius_norm => gapSVD))
    end
    return LSQR_SVD
end


function gap_A_var_fat() # m<n
    
    sizes = [(20,125), (30,270), (50,338), (50,578), (70,630)]
    k = 10 
    e = 0.001
    LSQR_SVD = []
    
    for i in 1:5
        m, n = sizes[i]
        A = rand(m, n)
        V_initial = rand(n, k)
        _, gapSVD, _ = time_gap(copy(A), k, e, copy(V_initial), parallel=true)
        push!(LSQR_SVD, Dict(:dim => (m, n), :frobenius_norm => gapSVD))
    end
    return LSQR_SVD
end


function gap_A_var_dense_sparse()
    global m = 50  # variable
    global n = 50  # variable
    k = 10 
    e = 0.001
    Dense = []
    Sparse = []
    # Plot the execution times as the matrix sizes of the 3 methods vary in time_gap
    for _ in 1:5
        A = rand(m, n)
        V_initial = rand(n, k)
        _, gapSVD, _ = time_gap(copy(A), k, e, copy(V_initial), parallel=true)
        push!(Dense, Dict(:dim => (m, n), :frobenius_norm => gapSVD))

        random_sparse_matrix = sprand(m, n, 0.2)
        random_sparse_matrix = Matrix(random_sparse_matrix)
        _, gapSVD, _ = time_gap(copy(random_sparse_matrix), k, e, copy(V_initial), parallel=true)
        push!(Sparse, Dict(:dim => (m, n), :frobenius_norm => gapSVD))

        global m += 40
        global n += 40
    end
    return Dense, Sparse
end


function gap_A_var_orth_diag_lower()
    global m = 50  # variable
    global n = 50  # variable
    k = 10 
    e = 0.001
    Orthogonal = []
    Diagonal_out = []
    Lower_tr = []

    for _ in 1:5
        V_initial = rand(n, k)

        # --- Orthogonal ---
        A = rand(m, n)
        Q, _ = qr(copy(A))
        Q = Q[:, 1:n]
        _, gapSVD, _ = time_gap(copy(Q), k, e, copy(V_initial), parallel=true)
        push!(Orthogonal, Dict(:dim => (m, n), :frobenius_norm => gapSVD))

        # --- diagonal --- 
        random_diagonal_matrix = Diagonal(randn(n))        
        _, gapSVD, _ = time_gap(copy(random_diagonal_matrix), k, e, copy(V_initial), parallel=true)
        push!(Diagonal_out, Dict(:dim => (m, n), :frobenius_norm => gapSVD))
        
        # --- triangular ---
        random_lower_triangular_matrix = LowerTriangular(rand(m, m))
        _, gapSVD, _ = time_gap(copy(random_lower_triangular_matrix), k, e, copy(V_initial), parallel=true)
        push!(Lower_tr, Dict(:dim => (m, n), :frobenius_norm => gapSVD))

        global m += 40
        global n += 40
    end
    return Orthogonal, Diagonal_out, Lower_tr
end



#---------------------------------------------------------------------------------------
# the gap between different algorithms solution and even with A matrix varyng the rank k

function gap_k_var()
    m = 50
    n = 50
    global k = 10  # variable
    A = rand(m, n)
    e = 0.01 # da vedere
    Svd_A = []
    LSQR_A = []
    LSQR_SVD = []
    # Plot the execution times as the matrix sizes of the 3 methods vary in time_gap
    for _ in 1:10
        V_initial = rand(n, k)
        # Chiamo SVD e fa gap con A
        _, gapSVD, gapA = time_gap(copy(A), k)
        push!(Svd_A, Dict(:k => k, :frobenius_norm => gapA))
        #print("SVD done\n")
        # _, gapSVD, gapA = time_gap(copy(A), k, e, copy(V_initial), parallel=false)
        # push!(LSQR_seq, Dict(:dim => (m, n), :time => t))
        #print("LSQR_seq done\n")
        # Chiamo metodo parallelo e fa gapA e gapSVD
        _, gapSVD, gapA = time_gap(copy(A), k, e, copy(V_initial), parallel=true)
        push!(LSQR_A, Dict(:k => k, :frobenius_norm => gapA))
        push!(LSQR_SVD, Dict(:k => k, :frobenius_norm => gapSVD))
        #print("LSQR_par done\n")
        global k += 5
    end
    
    return Svd_A, LSQR_A, LSQR_SVD
end



#---------------------------------------------------------------------------------------
# the gap between different algorithms solution and even with A matrix varyng the rank k
# OSS! deicdere se mettere la differenza rea A e il nostro metodo 
function gap_e_var()
    m = 60
    n = 60
    k = 10 
    A = rand(m, n)
    global e = round(0.0001, digits=4)  # variable
    Svd_A = []
    LSQR_A = []
    LSQR_SVD = []
    # Plot the execution times as the matrix sizes of the 3 methods vary in time_gap
    for _ in 1:8
        V_initial = rand(n, k)
        _, gapSVD, gapA = time_gap(copy(A), k, e, copy(V_initial), parallel=true)
        push!(LSQR_SVD, Dict(:e => e, :frobenius_norm => gapSVD))
        global e = e*5
        global e = round(e, digits=4)
    end
    return LSQR_SVD
end