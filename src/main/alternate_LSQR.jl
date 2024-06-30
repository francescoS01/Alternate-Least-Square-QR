using Distributed
include("QR_factorization.jl")
include("utils/print_matrix.jl")
include("LS_QR.jl")
using LinearAlgebra
using Random
using Printf


function low_rank_LSQR(A, k, e, V_iterative)
	
	m, n = size(A) 

	#V_iterative = V_initial'
	U_iterative = rand(m, k)
	err = norm(A - U_iterative * V_iterative')
	dif_err = Inf
	dif_UV = Inf

	#n_iter = 0 # per contare il numero di iterazioni
	
	while dif_UV > e #dif_err > e
		#n_iter += 1 # per contare il numero di iterazioni

		UV_old = U_iterative * V_iterative'
		
		# CASO 1. fiso V e cerco U, m sotto problemi (col di U' e A')
		Q, R = qr_fact(copy(V_iterative))
    	v_r, v_c = size(V_iterative)
		for i in 1:m
			a_col = copy(A'[:, i])
			U_iterative'[:, i] = LS_QR(copy(Q), copy(R), v_r, v_c, copy(a_col))
		end
	
		# CASO 2. fiso U e cerco V, n sotto problemi 
		Q, R = qr_fact(copy(U_iterative))
		u_r, u_c = size(U_iterative)
		for s in 1:n
			a_col = copy(A[:, s])
			V_iterative'[:, s] = LS_QR(copy(Q), copy(R), u_r, u_c, copy(a_col))
		end
		
		# Differenza tra UV_old e UV_new (norma frobenius) tra due iterazioni
		dif_UV = norm(UV_old - U_iterative * V_iterative', 2)
			
		# variazione errore rispetto a iterazione precedente (norma frobenius) rispetto ad A
		dif_err = err - sqrt(sum(abs2, (A - U_iterative * V_iterative')))

		# nuovo errore --> Loss = differenza tra A e U*V' attuale
		err = sqrt(sum(abs2, (A - U_iterative * V_iterative')))
	end	

	# print("Iterazioni totali: ", n_iter, "\n") # per contare il numero di iterazioni
	return U_iterative,	V_iterative

end



# ------- PARALLELIZED VERSION  -------

function low_rank_LSQR_parallellized(A, k, e, V_iterative, nt)
	
	m, n = size(A)

	#V_iterative = V_initial' # sistemare questione delle dimensioni 
	U_iterative = rand(m, k)
	err = norm(A - U_iterative * V_iterative')
	dif_err = Inf
	dif_UV = Inf
	
	m_sub = ceil(Int, m/nt)
	n_sub = ceil(Int, n/nt)

	while dif_UV > e
		UV_old = U_iterative * V_iterative'

		# CASO 1. fiso V e cerco U, m sotto problemi 
		Q, R = qr_fact(copy(V_iterative))
    	v_r, v_c = size(V_iterative)
		Threads.@threads for i in 0:nt-1
			# for the last portion of matrix 
			if i == nt-1
				U_iterative'[:, Int(i*m_sub+1):m] = U_sub(copy(A'[:, Int(i*m_sub+1):m]), copy(Q), copy(R), v_r, v_c, k)
			# general case 
			else
				U_iterative'[:, Int(i*m_sub+1):Int(i*m_sub+m_sub)] = U_sub(copy(A'[:, Int(i*m_sub+1):Int(i*m_sub+m_sub)]), copy(Q), copy(R), v_r, v_c, k)
			end
		end

		# CASO 2. fiso U e cerco V, n sotto problemi 
		Q, R = qr_fact(copy(U_iterative))
		u_r, u_c = size(U_iterative)
		Threads.@threads for i in 0:nt-1
			# for the last portion of matrix 
			if i == nt-1
				V_iterative'[:, Int(i*n_sub+1):n] = V_sub(copy(A[:, Int(i*n_sub+1):n]), copy(Q), copy(R), u_r, u_c, k)
			# general case
			else
				V_iterative'[:, Int(i*n_sub+1):Int(i*n_sub+n_sub)] = V_sub(copy(A[:, Int(i*n_sub+1):Int(i*n_sub+n_sub)]), copy(Q), copy(R), u_r, u_c, k)
			end	
		end

		dif_UV = norm(UV_old - U_iterative * V_iterative', 2)
		#dif_err = err - sqrt(sum(abs2, (A - U_iterative * V_iterative')))
		#err = sqrt(sum(abs2, (A - U_iterative * V_iterative')))
	end	

	return U_iterative,	V_iterative
end


function U_sub(A_sub_tr, Q_V, R_V, v_r, v_c, k)
	_ , m_sub = size(A_sub_tr)
	U_iterative = rand(m_sub, k)
	for i in 1:m_sub
		a_col = copy(A_sub_tr[:, i])
		U_iterative'[:, i] = LS_QR(copy(Q_V), copy(R_V), v_r, v_c, copy(a_col))
	end
	return U_iterative'
end


function V_sub(A_sub, Q_U, R_U, u_r, u_c, k)
	_ , n_sub = size(A_sub)
	V_iterative = rand(n_sub, k)
	for i in 1:n_sub
		a_col = copy(A_sub[:, i])
		# V_iterative'[:, i] = LS_QR(copy(U), copy(a_col))
		V_iterative'[:, i] = LS_QR(copy(Q_U), copy(R_U), u_r, u_c, copy(a_col))
	end
	return V_iterative'
end