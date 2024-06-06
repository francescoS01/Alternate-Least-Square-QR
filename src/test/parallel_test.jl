using LinearAlgebra
using Random
using Printf
using Base
using Plots

include("../main/alternate_LSQR.jl")
include("../main/utils/print_matrix.jl")
include("../main/utils/time_gap.jl")
include("../main/low_rank_SVD.jl")

# julia -t 4 parallel_test.jl

m = 90
n = 50

k = 30 # fixed
e = 0.1 # fixed

nt = Threads.nthreads()
println("Number of threads used: ", nt)

A = rand(m, n)
V_initial = rand(k, n)

println("SVD test time and gap: ", time_gap(copy(A), k), "\n")
#println("LSQR_seq test time and gap: ", time_gap(copy(A), k, e, copy(V_initial), parallel=false), "\n")
#println("LSQR_par test time and gap: ", time_gap(copy(A), k, e, copy(V_initial), parallel=true), "\n")

#println("\n------------------------------------\n")

#Ak_SVD, _ = low_rank(A, k)

# U, V = LS_QR_alternate(copy(A), k, e, copy(V_initial))
# Ak_QR =  U*V'
# gap = norm(Ak_SVD - Ak_QR, 2)
# println("\nGap between the SVD and the QR solution: ", gap)

# U, V = LS_QR_alternate_parallellized(copy(A), k, e, copy(V_initial), nt) 
# Ak_QR =  U*V'
# gap = norm(Ak_SVD - Ak_QR, 2)
#println("\nGap between the SVD and the parallel QR solution: ", gap)

#println("\n------------------------------------\n")

# print("LSQR_seq test time: ", time_gap(copy(A), k, e, copy(V_initial), parallel=false), "\n")
# print("LSQR_par_new test time: ", time_gap(copy(A), k, e, copy(V_initial), parallel=true), "\n")




