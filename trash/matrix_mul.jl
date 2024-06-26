"""
parallel version of matrix multiplication

"""

function parallel_mul(A, B)
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

# Example usage
# A = rand(1500, 1500)
# B = rand(1500, 1500)
# @time C = parallel_mul(A, B)
# @time C = A * B


