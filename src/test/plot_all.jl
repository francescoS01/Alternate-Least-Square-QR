using LinearAlgebra
using Random
using Printf
using Base
using Plots
include("../main/alternate_LSQR.jl")
include("../main/utils/print_matrix.jl")
include("../main/utils/time_gap.jl")
include("../main/low_rank_SVD.jl")


# julia -e 'include("plot_all.jl"); plot_all()'

function plot_all()
    global m = 50
    global n = 50

    k = 25 # fixed
    e = 0.1 # fixed
    Svd_time = []
    LSQR_seq = []
    LSQR_par = []
    # Plot the execution times as the matrix sizes of the 3 methods vary in time_gap
    for _ in 1:11
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
        global m += 10
        global n += 10
    end

    # print(Svd_time)
    # print("\n")
    # print(LSQR_seq)
    # print("\n")
    # print(LSQR_par)


    return Svd_time, LSQR_seq, LSQR_par
end

function gap_k_var()
    m = 60
    n = 60
    global k = 10 

    A = rand(m, n)# fixed
    e = 0.01 # fixed
    Svd_A = []
    LSQR_A = []
    LSQR_SVD = []
    # Plot the execution times as the matrix sizes of the 3 methods vary in time_gap
    for _ in 1:11
        V_initial = rand(n, k)
        # Chiamo SVD e fa gapA
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

