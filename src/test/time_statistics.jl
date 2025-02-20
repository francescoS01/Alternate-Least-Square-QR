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
# julia -e 'include("plot_all.jl"); plot_all()'

#--------------------------------------------------------------------------------------------------------------------------

# the time execution varyng the dimentions of the matrix A
function time_A_var()
    global m = 100
    k = 25 # fixed
    e = 1e-12
    Svd_time = []
    LSQR_seq = []
    LSQR_par = []
    global n = 100
    # Plot the execution times as the matrix sizes of the 3 methods vary in time_gap
    for _ in 1:4
        A = rand(m, n)
        V_initial = rand(n, k)
        t,_,_= time_gap(copy(A), k)
        push!(Svd_time, Dict(:dim => (m, n), :time => t))
        #print("SVD done\n")
        t,_,_= time_gap(copy(A), k, e, copy(V_initial), parallel=false)
        push!(LSQR_seq, Dict(:dim => (m, n), :time => t))
        #print("LSQR_seq done\n")
        t,_,_= time_gap(copy(A), k, e, copy(V_initial), parallel=true)
        push!(LSQR_par, Dict(:dim => (m, n), :time => t))
        #print("LSQR_par done\n")
        global m += 50
        # global n += 10
    end
    return Svd_time, LSQR_seq, LSQR_par
end

function time_A_var_square()
    global m = 50  # variable
    global n = 50  # variable
    k = 10 
    e = 1e-12
    Svd_time = []
    LSQR_seq = []
    LSQR_par = []
    # Plot the execution times as the matrix sizes of the 3 methods vary in time_gap
    for _ in 1:5
        A = rand(m, n)
        V_initial = rand(n, k)
        t,_,_= time_gap(copy(A), k)
        push!(Svd_time, Dict(:dim => (m, n), :time => t))
        t,_,_= time_gap(copy(A), k, e, copy(V_initial), parallel=false)
        push!(LSQR_seq, Dict(:dim => (m, n), :time => t))
        t,_,_= time_gap(copy(A), k, e, copy(V_initial), parallel=true)
        push!(LSQR_par, Dict(:dim => (m, n), :time => t))
        global m += 40
        global n += 40
    end
    return Svd_time, LSQR_seq, LSQR_par
end

function time_A_var_thin() # m>n
    #sizes = [(50,10), (90,30), (130,50), (170,70), (210,90)]
    sizes = [(100,25),  (180, 45), (260, 65), (340, 85), (420, 105)]
    k = 10 
    e = 1e-12
    Svd_time = []
    LSQR_seq = []
    LSQR_par = []
    
    for i in 1:5
        m, n = sizes[i]
        A = rand(m, n)
        V_initial = rand(n, k)
        t,_,_= time_gap(copy(A), k)
        push!(Svd_time, Dict(:dim => (m, n), :time => t))
        t,_,_= time_gap(copy(A), k, e, copy(V_initial), parallel=false)
        push!(LSQR_seq, Dict(:dim => (m, n), :time => t))
        t,_,_= time_gap(copy(A), k, e, copy(V_initial), parallel=true)
        push!(LSQR_par, Dict(:dim => (m, n), :time => t))
    end
    return Svd_time, LSQR_seq, LSQR_par
end



function time_A_var_fat() # m<n
    sizes = [(25, 100), (45, 180), (65, 260), (85, 340), (105, 420)]
    k = 10 
    e = 1e-12
    Svd_time = []
    LSQR_seq = []
    LSQR_par = []
    
    for i in 1:5
        m, n = sizes[i]
        A = rand(m, n)
        V_initial = rand(n, k)
        t,_,_= time_gap(copy(A), k)
        push!(Svd_time, Dict(:dim => (m, n), :time => t))
        t,_,_= time_gap(copy(A), k, e, copy(V_initial), parallel=false)
        push!(LSQR_seq, Dict(:dim => (m, n), :time => t))
        t,_,_= time_gap(copy(A), k, e, copy(V_initial), parallel=true)
        push!(LSQR_par, Dict(:dim => (m, n), :time => t))
    end
    return Svd_time, LSQR_seq, LSQR_par
end


function time_A_var_dense_sparse()
    global m = 50  # variable
    global n = 50  # variable
    k = 10 
    e = 1e-12
    Dense = []
    Sparse = []
    # Plot the execution times as the matrix sizes of the 3 methods vary in time_gap
    for _ in 1:10
        A = rand(m, n)
        V_initial = rand(n, k)
        t, _, _ = time_gap(copy(A), k, e, copy(V_initial), parallel=true)
        push!(Dense, Dict(:dim => (m, n), :time => t))

        random_sparse_matrix = sprand(m, n, 0.2)
        random_sparse_matrix = Matrix(random_sparse_matrix)
        t, _, _ = time_gap(copy(random_sparse_matrix), k, e, copy(V_initial), parallel=true)
        push!(Sparse, Dict(:dim => (m, n), :time => t))

        global m += 20
        global n += 20
    end
    return Dense, Sparse
end

function time_A_var_orth_diag_lower()
    global m = 50  # variable
    global n = 50  # variable
    k = 10 
    e = 1e-12
    Orthogonal = []
    Diagonal_out = []
    Lower_tr = []
    # Plot the execution times as the matrix sizes of the 3 methods vary in time_gap
    for _ in 1:10
        A = rand(m, n)
        V_initial = rand(n, k)

        Q, _ = qr(copy(A))
        Q = Matrix(Q)
        t, _, _ = time_gap(copy(Q), k, e, copy(V_initial), parallel=true)
        push!(Orthogonal, Dict(:dim => (m, n), :time => t))


        random_diagonal_matrix = Diagonal(randn(n))        
        t, _, _ = time_gap(copy(random_diagonal_matrix), k, e, copy(V_initial), parallel=true)
        push!(Diagonal_out, Dict(:dim => (m, n), :time => t))

        random_lower_triangular_matrix = LowerTriangular(rand(m, m))
        t, _, _ = time_gap(copy(random_lower_triangular_matrix), k, e, copy(V_initial), parallel=true)
        push!(Lower_tr, Dict(:dim => (m, n), :time => t))

        global m += 20
        global n += 20
    end
    return Orthogonal, Diagonal_out, Lower_tr
end

# the time execution varyng k
function time_k_var()
    m = 50
    n = 50
    global k = 10 # variable 
    e = 1e-12
    Svd_time = []
    LSQR_seq = []
    LSQR_par = []
    # Plot the execution times as the matrix sizes of the 3 methods vary in time_gap
    A = rand(m, n)
    for _ in 1:9
        V_initial = rand(n, k)
        t,_,_= time_gap(copy(A), k)
        push!(Svd_time, Dict(:k => k, :time => t))
        #print("SVD done\n")
        t,_,_= time_gap(copy(A), k, e, copy(V_initial), parallel=false)
        push!(LSQR_seq, Dict(:k => k, :time => t))
        #print("LSQR_seq done\n")
        t,_,_= time_gap(copy(A), k, e, copy(V_initial), parallel=true)
        push!(LSQR_par, Dict(:k => k, :time => t))
        #print("LSQR_par done\n")
        global k += 5
    end
    return Svd_time, LSQR_seq, LSQR_par
end


# # the time execution varyng e
function time_e_var()
    m = 50
    n = 50
    k = 10
    global e = 1e-6
    Svd_time = []
    LSQR_seq = []
    LSQR_par = []
    # Plot the execution times as the matrix sizes of the 3 methods vary in time_gap
    A = rand(m, n)
    V_initial = rand(n, k)
    for _ in 1:8
        t,_,_= time_gap(copy(A), k)
        push!(Svd_time, Dict(:e => e, :time => t))
        #print("SVD done\n")
        t,_,_= time_gap(copy(A), k, e, copy(V_initial), parallel=false)
        push!(LSQR_seq, Dict(:e => e, :time => t))
        #print("LSQR_seq done\n")
        t,_,_= time_gap(copy(A), k, e, copy(V_initial), parallel=true)
        push!(LSQR_par, Dict(:e => e, :time => t))
        #print("LSQR_par done\n")
        global e *= 0.1
        # global e = e*5
        # global e = round(e, digits=4)
    end
    return Svd_time, LSQR_seq, LSQR_par
end