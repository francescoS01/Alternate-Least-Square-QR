include("../utils/QR_factorization.jl")
include("../utils/print_matrix.jl")
include("../utils/LS_QR.jl")
using LinearAlgebra
using Random
using Printf


function LS_QR_alternate(A, k, e, V_test)
	
	m, n = size(A) # m x n 

	#V_iterative = rand(n, k)
	
	#print("\n\n\n\n")
	#V_iterative = rand(n, k)
	V_iterative = V_test'
	# PER TEST CHE CI SI FERMA AD UN MINIMI LOCALE (e non globale, come fa SVD)
	# print("\n")
	# V_iterative[1] =  0.00000
	# V_iterative[2] =  1.00000
	# print(V_iterative)
	# print("\n\n\n\n")
	U_iterative = rand(m, k)
	err = norm(A - U_iterative * V_iterative')
	dif_err = Inf
	
	while dif_err > e
		
		# CASO 1. fiso V e cerco U, m sotto problemi (col di U' e A')
		for i in 1:m
			a_col = copy(A'[:, i])
			U_iterative'[:, i] = LS_QR(copy(V_iterative), copy(a_col))
		end
		
		# CASO 2. fiso U e cerco V, n sotto problemi 
		for s in 1:n
			a_col = copy(A[:, s])
			V_iterative'[:, s] = LS_QR(copy(U_iterative), copy(a_col))
		end
		
		"""
		print("ERRORE ", j, " : ", norm(A - U_iterative * V_iterative'))
		print("\n-------\n")
		"""
		
		# variazione errore rispetto a iterazione precedente (norma frobenius)
		dif_err = err - sqrt(sum(abs2, (A - U_iterative * V_iterative')))
		# nuovo errore --> Loss = differenza tra A e U*V' attuale
		err = sqrt(sum(abs2, (A - U_iterative * V_iterative')))

	end	

	return U_iterative,	V_iterative
	
end