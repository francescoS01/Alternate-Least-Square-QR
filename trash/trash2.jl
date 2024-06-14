include("../main/alternate_LSQR.jl")
include("../main/utils/print_matrix.jl")
include("../main/low_rank_SVD.jl")
include("../main/utils/time_gap.jl")
#import Pkg
#Pkg.add("BenchmarkTools")
using BenchmarkTools


A = rand(30, 50)
e = 0.01
k = 15
V_initial = rand(50, 15)

# t,_= time_gap(copy(A), k, e, copy(V_initial), parallel=false)
# print(t)
# print("\n")
# t,_= time_gap(copy(A), k)
# print(t)
# print("\n")
# t,_ =time_gap(copy(A), k, e, copy(V_initial), parallel=true)
# print(t)


t = @elapsed begin
    Ak_SVD, trash = low_rank(A, k)
end
print(t, "\n")
@btime low_rank(copy(A), k)
@time begin
    Ak_SVD, trash = low_rank(A, k)
end

@btime LS_QR_alternate(copy(A), k, e, copy(V_initial)) 
@time begin
    U, V = LS_QR_alternate(copy(A), k, e, copy(V_initial)) 
    Ak_QR =  U*V'
end

# t = @btime begin
#     Ak_SVD, trash = low_rank(copy($A), $k)
# end


# #print(t)
# print("\n")

# t2 = @btime begin
#     Ak_SVD, trash = low_rank(copy($A), $k)
# end

# #print(t2)