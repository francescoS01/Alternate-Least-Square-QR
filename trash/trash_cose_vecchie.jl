
function func()
    return [1, 2, 3, 4, 5]
end

# take the maximum number of threads available in the system and set it as the number of threads to be used by Julia
#JULIA_NUM_THREADS = Sys.CPU_THREADS
#ENV["JULIA_NUM_THREADS"] = 4
#println("Number of threads: ", Sys.CPU_THREADS)
println("Number of threads: ", Threads.nthreads())






# ---------------- LE VECCHIE U SUB E V SUB --------------------
# # U_sub function remains unchanged (no need for parallelization)l
# # V_sub function remains unchanged (no need for parallelization)
# function V_sub(A_sub, U, k)
# 	_ , n_sub = size(A_sub)
# 	V_iterative = rand(n_sub, k)
# 	for i in 1:n_sub
# 		a_col = copy(A_sub[:, i])
# 		V_iterative'[:, i] = LS_QR(copy(U), copy(a_col))
# 	end
# 	return V_iterative'
# end

# # U_sub function remains unchanged (no need for parallelization)
# function U_sub(A_sub_tr, V, k)
# 	_ , m_sub = size(A_sub_tr)
# 	U_iterative = rand(m_sub, k)
# 	for i in 1:m_sub
# 		a_col = copy(A_sub_tr[:, i])
# 		U_iterative'[:, i] = LS_QR(copy(V), copy(a_col))
# 	end
# 	return U_iterative'
# end
#V_sub function remains unchanged (no need for parallelization)










# -------------------- VECCHIA VERSIONE DI LASQR ALTERNATE --------------------

# import Pkg
# Pkg.update("Random")


# function low_rank_LSQR(A, k, e, V_iterative)
	
# 	m, n = size(A) 

# 	#V_iterative = V_initial'
# 	U_iterative = rand(m, k)
# 	err = norm(A - U_iterative * V_iterative')
# 	dif_err = Inf
# 	dif_UV = Inf

# 	#n_iter = 0 # per contare il numero di iterazioni
	
# 	while dif_UV > e #dif_err > e

# 		#n_iter += 1 # per contare il numero di iterazioni

# 		UV_old = U_iterative * V_iterative'
		
# 		# CASO 1. fiso V e cerco U, m sotto problemi (col di U' e A')
# 		for i in 1:m
# 			a_col = copy(A'[:, i])
# 			U_iterative'[:, i] = LS_QR(copy(V_iterative), copy(a_col))
# 		end
	
# 		# CASO 2. fiso U e cerco V, n sotto problemi 
# 		for s in 1:n
# 			a_col = copy(A[:, s])
# 			V_iterative'[:, s] = LS_QR(copy(U_iterative), copy(a_col))
# 		end
		
# 		# Differenza tra UV_old e UV_new (norma frobenius) tra due iterazioni
# 		dif_UV = norm(UV_old - U_iterative * V_iterative', 2)
			
# 		# variazione errore rispetto a iterazione precedente (norma frobenius) rispetto ad A
# 		dif_err = err - sqrt(sum(abs2, (A - U_iterative * V_iterative')))

# 		# nuovo errore --> Loss = differenza tra A e U*V' attuale
# 		err = sqrt(sum(abs2, (A - U_iterative * V_iterative')))

# 	end	

# 	# print("Iterazioni totali: ", n_iter, "\n") # per contare il numero di iterazioni

# 	return U_iterative,	V_iterative

# end







# -------------------- VECCHIA VERSIONE DI LASQR dove viene fatta   qr internamente (cosa stuoida) --------------------
"""
function LS_QR(A, y)
    Q, R = qr_fact(A)
    m, n = size(A)

    # caso matrice A thin 
    R0_dim = n # quadrata
    Q0_row_size = m
    Q0_col_size = n

    # calcolo R0 
    R0 = R[1:R0_dim, 1:R0_dim]
    # calcolo Q0
    Q0 = Q[1:Q0_row_size, 1:Q0_col_size]

    # Risolviamo il sistema lineare Rx = Q^T y per trovare il vettore x
    x = inv(R0) * (Q0'*y)
    return x
end
ù
"""