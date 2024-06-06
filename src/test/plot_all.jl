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
    global m = 90
    global n = 50

    k = 30 # fixed
    e = 0.1 # fixed
    Svd_time = []
    LSQR_seq = []
    LSQR_par = []
    # Plot the execution times as the matrix sizes of the 3 methods vary in time_gap
    for _ in 1:4
        A = rand(m, n)
        V_initial = rand(k, n)
        t,_= time_gap(copy(A), k)
        push!(Svd_time, Dict(:dim => (m, n), :time => t))
        #print("SVD done\n")
        # t,_= time_gap(copy(A), k, e, copy(V_initial), parallel=false)
        # push!(LSQR_seq, Dict(:dim => (m, n), :time => t))
        #print("LSQR_seq done\n")
        #t,_= time_gap(copy(A), k, e, copy(V_initial), parallel=true)
        #push!(LSQR_par, Dict(:dim => (m, n), :time => t))
        #print("LSQR_par done\n")
        global m += 5
        global n += 5
    end

    # print(Svd_time)
    # print("\n")
    # print(LSQR_seq)
    # print("\n")
    # print(LSQR_par)


    return Svd_time, LSQR_seq, LSQR_par
end




