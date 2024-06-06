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

m = 100
n = 70

k = 10 # fixed
e = 0.01 # fixed

nt = Threads.nthreads()
println("Number of threads used: ", nt)

A = rand(m, n)
V_initial = rand(k, n)

Ak_SVD, trash = low_rank(A, k)

# U, V = LS_QR_alternate(copy(A), k, e, copy(V_initial))
# Ak_QR =  U*V'
# gap = norm(Ak_SVD - Ak_QR, 2)
# println("\nGap between the SVD and the QR solution: ", gap)

U, V = LS_QR_alternate_parallellized_new(copy(A), k, e, copy(V_initial), nt) 
Ak_QR =  U*V'
gap = norm(Ak_SVD - Ak_QR, 2)


println("\nGap between the SVD and the QR solution: ", gap)

println("\n------------------------------------\n")
print("LSQR_seq test time: ", total_time(copy(A), k, e, copy(V_initial), parallel=false), "\n")
print("LSQR_par_new test time: ", total_time(copy(A), k, e, copy(V_initial), parallel=true), "\n")
#LS_QR_alternate_parallellized_new(A, k, e, copy(V_initial), nt)


