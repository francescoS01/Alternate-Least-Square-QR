include("QR_factorization.jl")
include("print_matrix.jl")
include("LS_QR.jl")
using LinearAlgebra
using Random
using Printf


function LS_QR_alternate(A, k)
	
	m, n = size(A)

	V_iterative = rand(n, k)
	U_iterative = rand(m, k)
	
	for t in 1:50 # da cambiare con un criterio di arresto
	
		# CASO 1. fisso V e cerco U, m sottoproblemi  		 arg min || V * U' - A'||) 
		for i in 1:m #m = colonne di U' e A'
			a_col = copy(A'[:, i])
			U_iterative'[:, i] = LS_QR(copy(V_iterative), copy(a_col))
		end
		

		# CASO 2. fiso U e cerco V, n sottoproblemi 		 arg min || U * V' - A ||)
		for i in 1:n
			a_col = copy(A[:, i])
			V_iterative'[:, i] = LS_QR(copy(U_iterative), copy(a_col))
		end

		# ERRORE
		
		print("ERRORE ", t, " : ", norm(A - U_iterative * V_iterative'))
		print("\n-------\n")
		
		
	end		
	
	return U_iterative, V_iterative

end




