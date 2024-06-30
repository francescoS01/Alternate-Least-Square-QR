using LinearAlgebra
using Random
using Printf
using Base
using Plots
include("../main/alternate_LSQR.jl")
include("../main/utils/print_matrix.jl")
include("../main/utils/time_gap.jl")
include("../main/low_rank_SVD.jl")


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