using LinearAlgebra
using Random
using Printf
using Base
using Plots
include("../main/alternate_LSQR.jl")
include("../main/utils/print_matrix.jl")
include("../main/utils/time_gap.jl")
include("../main/low_rank_SVD.jl")


matrix_arr = []
num_matrix = 4
k = 20
i = 3
e = 0.1

V_iterate = rand(65, k)

function create_matrix()
    global matrix_arr
    m = 65
    n = 65
    A = rand(m, n)
    for j in 1:num_matrix
        #A = rand(m, n)
        push!(matrix_arr, A)
    end
end


function iter_update()
    global i += 1
end



function SVD_all()
    #global i 
    global k
    global matrix_arr
    for i in 1:num_matrix
        Ak_SVD, trash = low_rank_SVD(copy(matrix_arr[i]), k)
    end
    return
end



function LS_QR_all()
    global V_iterate
    global k
    global matrix_arr
    m,n = size(matrix_arr[1]) # attenzione
    
    for i in 1:num_matrix    
        U, V = low_rank_LSQR(copy(matrix_arr[i]), k, e , copy(V_iterate)) 
        Ak_QR =  U*V'
    end
    return
end


function LS_QR_parallel_all()
    global V_iterate
    global k
    global matrix_arr
    m,n = size(matrix_arr[1]) # attenzione
    nt = Threads.nthreads()
    for i in 1:num_matrix    
        U, V = low_rank_LSQR_parallellized(copy(matrix_arr[i]), k, e , copy(V_iterate), nt) 
        Ak_QR =  U*V'
    end
    return
end



function LS_QR_parallel_new_all()
    global V_iterate
    global k
    global matrix_arr
    m,n = size(matrix_arr[1]) # attenzione
    nt = Threads.nthreads()
    for i in 1:num_matrix    
        U, V = low_rank_LSQR_parallellized_new(copy(matrix_arr[i]), k, e , copy(V_iterate), nt) 
        Ak_QR =  U*V'
    end
    return
end

function julia_SVD_time()
    global matrix_arr
    global num_matrix
    global k
    total_time = 0
    for i in 1:num_matrix
        t = @elapsed begin
            Ak_SVD, trash = low_rank_SVD(copy(matrix_arr[i]), k)
        end
        println("Time for SVD: ", t)
        total_time += t
    end
    print("Total time for SVD: ", total_time)
    # t = @elapsed begin
    #     Ak_SVD, trash = low_rank_SVD(copy(matrix_arr[1]), k)
    # end
    
    return
end
