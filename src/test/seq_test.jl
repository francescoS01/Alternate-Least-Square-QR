using LinearAlgebra
using Random
using Printf
using Base
using Plots
include("../main/alternate_LSQR.jl")
include("../main/utils/print_matrix.jl")
include("../main/utils/timing.jl")
include("../main/low_rank_SVD.jl")


m = 50
n = 50

k = 10 # fixed
e = 0.1 # fixed

#print("SVD test time: ", total_time(rand(m, n), k), "\n")
#print("LSQR_seq test time: ", total_time(rand(m, n), k, e, rand(k, n), parallel=false), "\n")
#print("LSQR_par test time: ", total_time(rand(m, n), k, e, rand(k, n), parallel=true), "\n")


LSQR_seq = []
LSQR_par = []
for _ in 1:4
    A = rand(m, n)
    V_initial = rand(k, n)
    push!(LSQR_seq, Dict(:dim => (m, n), :time => total_time(A, k, e, V_initial, parallel=false)))
    #push!(LSQR_par, Dict(:dim => (m, n), :time => total_time(A, k, e, V_initial, parallel=true)))
    global m += 5
    global n += 5
end

dim = [LSQR_seq[i][:dim] for i in 1:length(LSQR_seq)]
times = [LSQR_seq[i][:time] for i in 1:length(LSQR_seq)]

print(dim)
print("\n")
print(times)
print("\n")


dim_strings = ["$(x[1])x$(x[2])" for x in dim]

p = plot(dim_strings, times, label="LSQR", xlabel="Matrix size", ylabel="Time (s)", title="Time for LSQR with different matrix sizes", lw=2, legend=:topleft)
savefig(p, "../../results/LSQR_seq.png")