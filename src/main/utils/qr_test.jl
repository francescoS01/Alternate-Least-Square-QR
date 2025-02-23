using LinearAlgebra


function create_ill_conditioned_matrix2(n, num_matrices)
    """
    Generates a list of ill-conditioned matrices.

    # Arguments
    - `n::Int`: The size of the square matrices.
    - `num_matrices::Int`: The number of matrices to generate.

    # Returns
    - `matrices`: A vector containing the ill-conditioned matrices.
    """
    matrices = [] 

    for i in 1:num_matrices
        condition_number = 10^(i * 2)
        U, _ = qr(randn(n, n))
        V, _ = qr(randn(n, n))
        singular_values = [1.0 * (condition_number)^(i / (n - 1)) for i in 0:n-1]
        S = Diagonal(singular_values)
        A = U * S * V'
        push!(matrices, A)
    end 
    
    return matrices
end


function create_ill_conditioned_matrix(num_matrices, n)
    matrices = []  # Lista per memorizzare le matrici mal condizionate

    for i in 2:num_matrices  # Itera su un intervallo da 1 a num_matrices
        # Crescita più rapida del numero di condizionamento
        condition_number = 10^((i - 1) * 3)  # Incremento più rapido per condizionamenti più alti

        # Genera matrici ortogonali casuali U e V usando la decomposizione QR
        U, _ = qr(randn(n, n))
        V, _ = qr(randn(n, n))
        
        # Crea valori singolari con un primo valore fisso a 1 e l'ultimo molto piccolo
        singular_values = [1.0 / (condition_number)^(i / (n - 1)) for i in 0:n-1]
        S = Diagonal(singular_values)
        
        # Costruisci la matrice mal condizionata: A = U * S * V'
        A = U * S * V'
        push!(matrices, A)  # Aggiungi la matrice alla lista
    end 
    
    return matrices
end


function perturb_matrix(A, epsilon)
    """
    Perturbs a given matrix by adding a small value to its (1,1) element.

    # Arguments
    - `A::Matrix`: The input matrix to be perturbed.
    - `epsilon::Number`: The small value to add to the (1,1) element of the matrix.

    # Returns
    - `A_pert`: A new matrix with the perturbed (1,1) element.
    """
    A_pert = copy(A)
    A_pert[1, 1] += epsilon 
    return A_pert
end