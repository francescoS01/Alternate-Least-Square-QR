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

#--------------------------------------------------------------------------------------------------------------------------

# the gap between our algorithm and SVD solution when varyng A dimention
function gap_A_var()
    global m = 50  # variable
    global n = 50  # variable
    k = 10 
    e = 1e-12
    LSQR_SVD = []
    for _ in 1:11
        A = rand(m, n)
        V_initial = rand(n, k)
        _, gapSVD, _ = time_gap(copy(A), k, e, copy(V_initial), parallel=true)
        push!(LSQR_SVD, Dict(:dim => (m, n), :relative_error => gapSVD))
        global m += 10
        global n += 10
    end
    return LSQR_SVD
end


function gap_A_var_square()
    global m = 50  
    global n = 50  
    k = 10 
    e = 1e-12
    LSQR_SVD = []
    for _ in 1:5
        A = rand(m, n)
        V_initial = rand(n, k)
        _, gapSVD, _ = time_gap(copy(A), k, e, copy(V_initial), parallel=true)
        push!(LSQR_SVD, Dict(:dim => (m, n), :relative_error => gapSVD))
        global m += 40
        global n += 40
    end
    return LSQR_SVD
end

function gap_A_var_thin() # m>n
    sizes = [(50,10), (90,30), (130,50), (170,70), (210,90)]
    k = 10 
    e = 1e-12
    LSQR_SVD = []
    
    for i in 1:5
        m, n = sizes[i]
        A = rand(m, n)
        V_initial = rand(n, k)
        _, gapSVD, _ = time_gap(copy(A), k, e, copy(V_initial), parallel=true)
        push!(LSQR_SVD, Dict(:dim => (m, n), :relative_error => gapSVD))
    end
    return LSQR_SVD
end



function gap_A_var_fat() # m<n
    sizes = [(10,50), (30,90), (50,130), (70,170), (90,210)]
    k = 10 
    e = 1e-12
    LSQR_SVD = []
    
    for i in 1:5
        m, n = sizes[i]
        A = rand(m, n)
        V_initial = rand(n, k)
        _, gapSVD, _ = time_gap(copy(A), k, e, copy(V_initial), parallel=true)
        push!(LSQR_SVD, Dict(:dim => (m, n), :relative_error => gapSVD))
    end
    return LSQR_SVD
end


function gap_A_var_dense_sparse()
    global m = 50 
    global n = 50 
    k = 10 
    e = 1e-12
    Dense = []
    Sparse = []
    for _ in 1:10
        A = rand(m, n)
        V_initial = rand(n, k)
        _, gapSVD, _ = time_gap(copy(A), k, e, copy(V_initial), parallel=true)
        push!(Dense, Dict(:dim => (m, n), :relative_error => gapSVD))

        random_sparse_matrix = sprand(m, n, 0.2)
        random_sparse_matrix = Matrix(random_sparse_matrix)
        _, gapSVD, _ = time_gap(copy(random_sparse_matrix), k, e, copy(V_initial), parallel=true)
        push!(Sparse, Dict(:dim => (m, n), :relative_error => gapSVD))
        
        #print("Fatte tutte quelle di dimensione ", m, "*", n, "\n")

        global m += 20
        global n += 20
    end
    return Dense, Sparse
end

function gap_A_var_orth_diag_lower()
    global m = 50  
    global n = 50  
    k = 10 
    e = 1e-12
    Orthogonal = []
    Diagonal_out = []
    Lower_tr = []
    for _ in 1:10
        A = rand(m, n)
        V_initial = rand(n, k)

        Q, _ = qr(copy(A))
        Q = Matrix(Q)
        _, gapSVD, _ = time_gap(copy(Q), k, e, copy(V_initial), parallel=true)
        push!(Orthogonal, Dict(:dim => (m, n), :relative_error => gapSVD))


        random_diagonal_matrix = Diagonal(randn(n))        
        _, gapSVD, _ = time_gap(copy(random_diagonal_matrix), k, e, copy(V_initial), parallel=true)
        push!(Diagonal_out, Dict(:dim => (m, n), :relative_error => gapSVD))

        random_lower_triangular_matrix = LowerTriangular(rand(m, m))
        _, gapSVD, _ = time_gap(copy(random_lower_triangular_matrix), k, e, copy(V_initial), parallel=true)
        push!(Lower_tr, Dict(:dim => (m, n), :relative_error => gapSVD))

        global m += 20
        global n += 20
    end
    return Orthogonal, Diagonal_out, Lower_tr
end




#--------------------------------------------------------------------------------------------------------------------------

# the gap between different algorithms solution and even with A matrix varyng the rank k
function gap_k_var()
    m = 50
    n = 50
    global k = 10 
    e = 1e-12
    Svd_A = []
    LSQR_A = []
    LSQR_SVD = []
    A = rand(m, n)
    for _ in 1:9
        V_initial = rand(n, k)
        _, gapSVD, gapA = time_gap(copy(A), k)
        push!(Svd_A, Dict(:k => k, :relative_error => gapA))
        _, gapSVD, gapA = time_gap(copy(A), k, e, copy(V_initial), parallel=true)
        push!(LSQR_A, Dict(:k => k, :relative_error => gapA))
        push!(LSQR_SVD, Dict(:k => k, :relative_error => gapSVD))
        global k += 5
    end
    
    return Svd_A, LSQR_A, LSQR_SVD
end

function gap_k2_var()
    m = 50
    n = 50
    global k = 10 
    e = 7
    Svd_A = []
    LSQR_A = []
    LSQR_SVD = []
    A = rand(m, n)
    for _ in 1:9
        V_initial = rand(n, k)
        _, gapSVD, gapA = time_gap(copy(A), k)
        push!(Svd_A, Dict(:k => k, :relative_error => gapA))
        _, gapSVD, gapA = time_gap(copy(A), k, e, copy(V_initial), parallel=true)
        push!(LSQR_A, Dict(:k => k, :relative_error => gapA))
        push!(LSQR_SVD, Dict(:k => k, :relative_error => gapSVD))
        global k += 5
    end
    
    return Svd_A, LSQR_A, LSQR_SVD
end

function gap_e_var()
    m = 50
    n = 50
    k = 10 
    A = rand(m, n)
    global e = 1e-14
    Svd_A = []
    LSQR_A = []
    LSQR_SVD = []
    V_initial = rand(n, k)
    for _ in 1:2
        _, gapSVD, gapA = time_gap(copy(A), k, e, copy(V_initial), parallel=true)
        push!(LSQR_SVD, Dict(:e => e, :relative_error => gapSVD))
        global e *= 0.1
    end
    return LSQR_SVD
end


function find_e()
    m = 50
    n = 50
    k = 10 
    A = rand(m, n)
    global e = 1e-11
    gapSVD = Inf
    while (gapSVD>1e-16)
        print("Provo e: ", e)
        print("\n")
        V_initial = rand(n, k)
        t = @elapsed begin
            _, gapSVD, _ = time_gap(copy(A), k, e, copy(V_initial), parallel=true)
        end

        print("Ci ho messo: ", t)
        print("\n")
        print("Errore attuale: ", gapSVD)
        print("\n")
        print("\n")
        global e *= 0.1
        # global e = round(e, digits=4)
    end
    print("gapSVD finale : ", gapSVD)
    print("\n")
    print("e finale : ", e)
    print("\n")
    return
end