include("../main/alternate_LSQR.jl")
include("../main/utils/print_matrix.jl")
include("../main/low_rank_SVD.jl")
using LinearAlgebra
using Random
using Printf

# generate a random matrix A 70x70
m = 100
n = 50
A = rand(m, n)

k = 10



# ------- STAMPE_TEST -------
# Begin timing and print at the end
print("\nTime for SVD: ")
@time begin
    # chiamata con SVD ( perfect approximation )
    Ak_SVD, trash = low_rank(A, k)
end


# chiamata di QR alternato
e = 0.1
V_initial = rand(k, n)

@time begin
    U, V = LS_QR_alternate(A, k, e, V_initial) 
    Ak_QR =  U*V'
    print("Time for LS_QR_alternate: ")
end


# Calculate the gap between the svd and the qr solution using the Frobenius norm
gap = norm(Ak_SVD - Ak_QR, 2)
println("\nGap between the SVD and the QR solution: ", gap)