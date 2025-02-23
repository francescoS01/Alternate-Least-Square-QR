using LinearAlgebra
using Random
using Printf

# Include i file necessari
include("../main/utils/print_matrix.jl")
include("../main/utils/qr_test.jl")
include("../main/QR_factorization.jl")



# Funzione per perturbare leggermente una matrice
function perturb_matrix_with_noise(A, epsilon)
    noise = epsilon * randn(size(A))  # Aggiunge rumore con distribuzione normale
    return A + noise
end

# Funzione per calcolare la differenza tra matrici
function matrix_difference(A, B)
    return norm(A - B)
end


function qr_fact_backward_dem(ill_conditioned_matrices, n, epsilon, our=true)

    # Lista per memorizzare i risultati
    results = []
    
    # Itera su ogni matrice mal condizionata
    for (i, A) in enumerate(ill_conditioned_matrices)
        # Calcola il condizionamento della matrice
        cond_A = cond(A)

        if our
            Q, R = qr_fact(A)
            A_approx = Q * R

        else
            # Calcola la decomposizione QR per la matrice originale
            qr_fac = qr(copy(A))
            Q = Matrix(qr_fac.Q)  # Converti Q in una matrice densa
            R = Matrix(qr_fac.R)  # Converti R in una matrice densa
            A_approx = Q * R
        end
      
        backward_error = norm(A - A_approx, 2) / norm(A, 2)

        
        # Salva i risultati in un dizionario
        result = Dict(
            :index => i,  # Indice della matrice
            :cond_A => cond_A,  # Condizionamento della matrice
            :back_error => backward_error
        )
     
        # Aggiungi il dizionario alla lista dei risultati
        push!(results, result)
    end
    # stampa i risultati
    for result in results
        @printf("Matrix %d: cond(A) = %.2e, backward error = %.2e\n", result[:index], result[:cond_A], result[:back_error])
    end
    
    return results  # Ritorna la lista di risultati
end

function qr_test_lib()
    num_matrices = 6  
    n = 40        
    epsilon = 1e-12    
    ill_conditioned_matrices = create_ill_conditioned_matrix(num_matrices, n)
    return qr_fact_backward_dem(ill_conditioned_matrices, n, epsilon, false) 
end 

function qr_test_our()
    num_matrices = 6
    n = 40        
    epsilon = 1e-12    
    ill_conditioned_matrices = create_ill_conditioned_matrix(num_matrices, n)
    return qr_fact_backward_dem(ill_conditioned_matrices, n, epsilon, true) 
end 

# Esegui il test
qr_test_lib()
