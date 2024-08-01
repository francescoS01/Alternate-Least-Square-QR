using Printf
function print_matrix(A)
    """
    Input: A (matrix) and precision (number of decimal places)
    Output: prints the matrix A with the specified precision
    """
    for m in eachrow(A)
        for element in m
            print(@sprintf("%.5f ", element))
        end
        println()
    end
end