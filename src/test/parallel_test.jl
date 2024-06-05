using LinearAlgebra
using Random
using Printf
using Base
using Plots

include("../main/alternate_LSQR.jl")
include("../main/utils/print_matrix.jl")
include("../main/utils/timing.jl")
include("../main/low_rank_SVD.jl")

# julia -t 4 parallel_test.jl

m = 50
n = 50

k = 10 # fixed
e = 0.1 # fixed

println("Number of cores used: ", Threads.nthreads())

A = rand(m, n)
V_initial = rand(k, n)
print("LSQR_seq test time: ", total_time(A, k, e, V_initial, parallel=false), "\n")
print("LSQR_par test time: ", total_time(A, k, e, V_initial, parallel=true), "\n")
