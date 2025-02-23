using Distributed
include("QR_factorization.jl")
include("utils/print_matrix.jl")
include("LS_QR.jl")
using LinearAlgebra
using Random
using Printf




# ------- SEQUENTIAL VERSION  -------

function low_rank_LSQR(A, k, e, V_iterative)
	
	m, n = size(A) 

	U_iterative = rand(m, k)
	dif_UV = Inf

	cycle = 0
	while dif_UV > e 

		cycle += 1
		if cycle > 200
			break
		end
		UV_old = U_iterative * V_iterative'
		
		# CASE 1. fix V and find U, m subproblems (columns of U' and A')
		householder_vectors, R = qr_fact(V_iterative)
    	v_r, v_c = size(V_iterative)
		for i in 1:m
			a_col = A'[:, i]
			U_iterative'[:, i] = LS_QR(householder_vectors, R, v_r, v_c, a_col)
		end
	
		# CASE 2. fix U and find V, n subproblems
		householder_vectors, R = qr_fact(U_iterative)
		u_r, u_c = size(U_iterative)
		for s in 1:n
			a_col = A[:, s]
			V_iterative'[:, s] = LS_QR(householder_vectors, R, u_r, u_c, a_col)
		end
		
		# Difference between UV_old and UV_new (Frobenius norm) between two iterations
		dif_UV = norm(UV_old - U_iterative * V_iterative', 2)
	end	

	return U_iterative,	V_iterative
end


# ------- PARALLELIZED VERSION  -------


function low_rank_LSQR_parallellized(A, k, e, V_iterative, nt)
	
	m, n = size(A)

	U_iterative = rand(m, k)
	dif_UV = Inf
	
	m_sub = ceil(Int, m/nt)
	n_sub = ceil(Int, n/nt)

	cycle = 0
	while dif_UV > e
		UV_old = U_iterative * V_iterative'
		
		cycle += 1
		if cycle > 200
			break
		end

		# CASE 1. fix V and find U, m subproblems
		householder_vectors, R = qr_fact(V_iterative)
    	v_r, v_c = size(V_iterative)
		Threads.@threads for i in 0:nt-1
			# for the last portion of matrix 
			if i == nt-1
				U_iterative'[:, Int(i*m_sub+1):m] = U_sub(A'[:, Int(i*m_sub+1):m], householder_vectors, R, v_r, v_c, k)
			# general case 
			else
				U_iterative'[:, Int(i*m_sub+1):Int(i*m_sub+m_sub)] = U_sub(A'[:, Int(i*m_sub+1):Int(i*m_sub+m_sub)], householder_vectors, R, v_r, v_c, k)
			end
		end

		# CASE 2. fix U and find V, n subproblems
		householder_vectors, R = qr_fact(U_iterative)
		u_r, u_c = size(U_iterative)
		Threads.@threads for i in 0:nt-1
			# for the last portion of matrix 
			if i == nt-1
				V_iterative'[:, Int(i*n_sub+1):n] = V_sub(A[:, Int(i*n_sub+1):n], householder_vectors, R, u_r, u_c, k)
			# general case
			else
				V_iterative'[:, Int(i*n_sub+1):Int(i*n_sub+n_sub)] = V_sub(A[:, Int(i*n_sub+1):Int(i*n_sub+n_sub)], householder_vectors, R, u_r, u_c, k)
			end	
		end

		dif_UV = norm(UV_old - U_iterative * V_iterative', 2)
	end	

	return U_iterative,	V_iterative
end


function U_sub(A_sub_tr, householder_vectors, R_V, v_r, v_c, k)
	_ , m_sub = size(A_sub_tr)
	U_iterative = rand(m_sub, k)
	for i in 1:m_sub
		a_col = A_sub_tr[:, i]
		U_iterative'[:, i] = LS_QR(householder_vectors, R_V, v_r, v_c, a_col)
	end
	return U_iterative'
end


function V_sub(A_sub, householder_vectors, R_U, u_r, u_c, k)
	_ , n_sub = size(A_sub)
	V_iterative = rand(n_sub, k)
	for i in 1:n_sub
		a_col = A_sub[:, i]
		V_iterative'[:, i] = LS_QR(householder_vectors, R_U, u_r, u_c, a_col)
	end
	return V_iterative'
end