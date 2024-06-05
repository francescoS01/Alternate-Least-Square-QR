include("../main/alternate_LSQR.jl")
include("../main/utils/print_matrix.jl")
include("../main/low_rank_SVD.jl")
using LinearAlgebra
using Random
using Printf

# ------- TEST 1 -------

#A = 100*rand(7, 5) # m x n, rank = n (in generale, perchè random fa tutte le colonne linearmente indipendenti, ma ci potrebbero essere casi in cui non lo sono quindi rank = n non è sempre vero)

# Generate a fixed integer matrix A with rank = 5
A = [61 22 14 67 23;
    78 31 11 35 57;
    56 86 14 24 90;
    46 0 70 3 60;
    8 89 58 4 9]


# Generate a random integer matrix A
#Random.seed!(1234)
#A = rand(1:10, 10, 10)

# Generate a random integer matrix A with some columns that are linearly dependent
#A = [1 2 3 4 5; 2 4 6 8 10; 3 6 9 12 15; 4 8 12 16 20; 5 10 15 20 25; 6 12 18 24 30; 7 14 21 28 35; 8 16 24 32 40; 9 18 27 36 45; 10 20 30 40 50]
# rank(A) is 1


#k = rango della matrice che approssima 
k = 2




# ------- TEST 2 -------
# A = [2 0; 0 1]
# print(A)
# k = 1

# ------- TEST 3 -------
# A = [2 0 0; 0 1 0; 0 0 1]
# print(A)
# k = 2



# ------- STAMPE_TEST -------
# Begin timing and print at the end
print("Time for SVD: ")
@time begin
    # chiamata con SVD ( perfect approximation )
    Ak_SVD, trash = low_rank(A, k)
end


# chiamata di QR alternato
e = 0.0001
V_initial = rand(2, 5)
print("Time for LS_QR_alternate: ")
@time begin
    U, V = LS_QR_alternate(A, k, e, V_initial) 
    Ak_QR =  U*V'
end

# Calculate the rank of the matrix A
r = rank(A)
println("\nRank of the matrix A: ", r)
# confronto le due soluzioni 
print("\n-------  Matrice A -------\n")
print_matrix(A)
print("\n-------  Matrice U*V' con QR  -------\n")
print_matrix(Ak_QR)

print("\n-------    Matrice U*V' SVD   -------\n")
print_matrix(Ak_SVD)

#print("\n-------        Errore         -------\n")
#print( sqrt( sum( abs2, (Ak_QR - Ak_SVD) ) ) )

# Calculate the gap between the svd and the qr solution using the Frobenius norm
gap = norm(Ak_SVD - Ak_QR, 2)
println("\nGap between the SVD and the QR solution: ", gap)

# Calculate the gap between the svd and the qr solution using the Frobenius norm
gap = norm(Ak_SVD - Ak_QR, 2) / norm(Ak_SVD, 2)
println("\nNormalized gap between the SVD and the QR solution: ", gap)