include("QR_factorization.jl")
include("utils/print_matrix.jl")
include("LS_QR.jl")
using LinearAlgebra
using Random
using Printf

function low_rank(A, k)
	"""	
	Input: A (matrix) and precision k (rango)
	Output: the matrix A' (approximation of A) with the specified precision
	"""

	U, Σ, V = svd(A)
	# S is a matrix with diagonal = Σ
	S = Diagonal(Σ) 

	Uk = U[:, 1:k]
	Sk = S[1:k, 1:k]
	Vk = V[:, 1:k]
	
	return Uk*Sk*Vk', Vk'

end