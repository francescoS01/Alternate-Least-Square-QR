include("../src/main/alternate_LSQR.jl")
include("../src/main/utils/print_matrix.jl")
include("../src/main/low_rank_SVD.jl")
include("../src/main/utils/time_gap.jl")
#import Pkg
#Pkg.add("BenchmarkTools")
#using BenchmarkTools


A = rand(50, 50)
e = 0.001
k = 20
V_initial = rand(50, k)

print("\n-------  A  -------\n")
#print_matrix(A)

Ak_SVD, trash = low_rank_SVD(copy(A), k)

U, V = low_rank_LSQR_parallellized(copy(A), k, e, copy(V_initial), Threads.nthreads()) 
Ak_QR =  U*V'



print("\n-------  Matrice U*V' con QR  -------\n")
#print_matrix(Ak_QR)

print("\n-------    Matrice U*V' SVD   -------\n")
#print_matrix(Ak_SVD)

gap = norm(Ak_SVD - Ak_QR, 2)
println("\nGap between the SVD and the QR solution: ", gap)


















# t,_= time_gap(copy(A), k, e, copy(V_initial), parallel=false)
# print(t)
# print("\n")
# t,_= time_gap(copy(A), k)
# print(t)
# print("\n")
# t,_ =time_gap(copy(A), k, e, copy(V_initial), parallel=true)
# print(t)


# t = @elapsed begin
#     Ak_SVD, trash = low_rank(A, k)
# end
# print(t, "\n")
# @btime low_rank(copy(A), k)
# @time begin
#     Ak_SVD, trash = low_rank(A, k)
# end

# @btime LS_QR_alternate(copy(A), k, e, copy(V_initial)) 
# @time begin
#     U, V = LS_QR_alternate(copy(A), k, e, copy(V_initial)) 
#     Ak_QR =  U*V'
# end

# t = @btime begin
#     Ak_SVD, trash = low_rank(copy($A), $k)
# end


# #print(t)
# print("\n")

# t2 = @btime begin
#     Ak_SVD, trash = low_rank(copy($A), $k)
# end

# #print(t2)