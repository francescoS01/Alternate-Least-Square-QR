include("QR_factorization.jl")
include("utils/print_matrix.jl")
include("LS_QR.jl")
using LinearAlgebra
using Random
using Printf
# import Pkg
# Pkg.update("Random")


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
	dif_UV = Inf

	#n_iter = 0 # per contare il numero di iterazioni
	
	while dif_UV > e #dif_err > e

		#n_iter += 1 # per contare il numero di iterazioni

		UV_old = U_iterative * V_iterative'
		
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




		# ----- PARALLELIZZAZIONE -----
		# # CASO 1. fiso V e cerco U, m sotto problemi (col di U' e A')
		# Threads.@threads for i in 1:m
		# 	a_col = copy(A'[:, i])
		# 	U_iterative'[:, i] = LS_QR(copy(V_iterative), copy(a_col))
		# end
		
		# # CASO 2. fiso U e cerco V, n sotto problemi 
		# Threads.@threads for s in 1:n
		# 	a_col = copy(A[:, s])
		# 	V_iterative'[:, s] = LS_QR(copy(U_iterative), copy(a_col))
		# end
		# ------------------------------
		





		
		# print("ERRORE ", j, " : ", norm(A - U_iterative * V_iterative'))
		# print("\n-------\n")
		
		# Differenza tra UV_old e UV_new (norma frobenius) tra due iterazioni
		dif_UV = norm(UV_old - U_iterative * V_iterative', 2)
		# print("\n-------\n")
		# print(dif_UV)
		# print("\n-------\n")
		
		# variazione errore rispetto a iterazione precedente (norma frobenius) rispetto ad A
		dif_err = err - sqrt(sum(abs2, (A - U_iterative * V_iterative')))
		# nuovo errore --> Loss = differenza tra A e U*V' attuale
		err = sqrt(sum(abs2, (A - U_iterative * V_iterative')))

	end	

	# print("Iterazioni totali: ", n_iter) # per contare il numero di iterazioni
	# print("\n") # per contare il numero di iterazioni

	return U_iterative,	V_iterative
	
end



function LS_QR_alternate_parallellized(A, k, e, V_test)

	m, n = size(A)

	V_iterative = V_test'
	U_iterative = rand(m, k)
	err = norm(A - U_iterative * V_iterative')
	dif_err = Inf
	dif_UV = Inf
	
	while dif_UV > e

		UV_old = U_iterative * V_iterative'
		
		#V_clone = copy(V_iterative)
		Threads.@threads for i in 1:m # Parallelize the computation of U
			a_col = A'[:, i]
			U_iterative'[:, i] = LS_QR(copy(V_iterative), a_col)
		end
		
		#U_clone = copy(U_iterative)
		Threads.@threads for s in 1:n # Parallelize the computation of V
			a_col2 = A[:, s]
			V_iterative'[:, s] = LS_QR(copy(U_iterative), a_col2)
		end

		dif_UV = norm(UV_old - U_iterative * V_iterative', 2)
		#dif_err = err - sqrt(sum(abs2, (A - U_iterative * V_iterative')))
		#err = sqrt(sum(abs2, (A - U_iterative * V_iterative')))
	end	

	return U_iterative,	V_iterative
	
end

function U_sub(A_sub_tr, V, k)
	m_sub, n_sub = size(A_sub_tr)
	U_iterative = rand(n_sub, k)
	for i in 1:n_sub
		a_col = copy(A_sub_tr[:, i])
		U_iterative'[:, i] = LS_QR(copy(V), copy(a_col))
	end
	return U_iterative'[:, 1:n_sub]
end

function LS_QR_alternate_parallellized_new(A, k, e, V_test, nt)

	m, n = size(A)

	V_iterative = V_test'
	U_iterative = rand(m, k)
	err = norm(A - U_iterative * V_iterative')
	dif_err = Inf
	dif_UV = Inf
	
	m_sub = ceil(Int, m/nt)

	while dif_UV > e

		UV_old = U_iterative * V_iterative'
		
		Threads.@threads for i in 0:nt-1
			if i == nt-1
				U_iterative'[:, Int(i*m_sub+1):m] = U_sub(copy(A'[:, Int(i*m_sub+1):m]), V_iterative, k)
			else
				U_iterative'[:, Int(i*m_sub+1):Int(i*m_sub+m_sub)] = U_sub(copy(A'[:, Int(i*m_sub+1):Int(i*m_sub+m_sub)]), V_iterative, k)
			end
		end

		for s in 1:n
			a_col = copy(A[:, s])
			V_iterative'[:, s] = LS_QR(copy(U_iterative), copy(a_col))
		end

		dif_UV = norm(UV_old - U_iterative * V_iterative', 2)
		#dif_err = err - sqrt(sum(abs2, (A - U_iterative * V_iterative')))
		#err = sqrt(sum(abs2, (A - U_iterative * V_iterative')))
	end	

	return U_iterative,	V_iterative
	
end

