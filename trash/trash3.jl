using LinearAlgebra

# Example of a parallel matrix multiplication function
A = rand(1000, 1000)
B = rand(1000, 400)


function parallel_matmul(A, B)
    m, kA = size(A)
    kB, n = size(B)
    if kA != kB
        throw(DimensionMismatch("A's columns must match B's rows"))
    end
    C = zeros(m, n)
    chunk_size = ceil(Int, m / Threads.nthreads()) # Determine chunk size based on the number of threads

    Threads.@threads for t in 1:Threads.nthreads()
        start_row = (t-1) * chunk_size + 1
        end_row = min(t * chunk_size, m)
        for i = start_row:end_row
            for j = 1:n
                for l = 1:kA
                    C[i, j] += A[i, l] * B[l, j]
                end
            end
        end
    end
    return C
end

function matrix_mul_parallel(A, B)
    m, n = size(A)
    p, q = size(B)
    if n != p
        error("Matrix dimensions must agree.")
    end
    C = zeros(m, q)
    Threads.@threads for i in 1:m
        for j in 1:q
            for k in 1:n
                C[i, j] += A[i, k] * B[k, j]
            end
        end
    end
    return C
end



t = @elapsed begin
    C = A * B
end
println("Time: ", t)

t = @elapsed begin
    C = parallel_matmul(A, B)
end
println("Time: ", t)

# t= @elapsed begin
#     C = matrix_mul_parallel(A, B)
# end
# println("Time: ", t)





# t = @elapsed begin
#     C = A .* B
# end
# println("Time: ", t)





