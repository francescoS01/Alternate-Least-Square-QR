using LinearAlgebra
using Random
using Printf
using Base
using Plots
include("../main/alternate_LSQR.jl")
include("../main/utils/print_matrix.jl")
include("../main/utils/time_gap.jl")
include("../main/low_rank_SVD.jl")
# julia -e 'include("plot_all.jl"); plot_all()'

#--------------------------------------------------------------------------------------------------------------------------

# the time execution varyng the dimentions of the matrix A
function time_A_var()
    global m = 50
    global n = 50
    k = 25 # fixed
    e = 0.1 # fixed
    Svd_time = []
    LSQR_seq = []
    LSQR_par = []
    # Plot the execution times as the matrix sizes of the 3 methods vary in time_gap
    for _ in 1:5
        A = rand(m, n)
        V_initial = rand(n, k)
        t,_,_= time_gap(copy(A), k)
        push!(Svd_time, Dict(:dim => (m, n), :time => t))
        #print("SVD done\n")
        t,_,_= time_gap(copy(A), k, e, copy(V_initial), parallel=false)
        push!(LSQR_seq, Dict(:dim => (m, n), :time => t))
        #print("LSQR_seq done\n")
        t,_,_= time_gap(copy(A), k, e, copy(V_initial), parallel=true)
        push!(LSQR_par, Dict(:dim => (m, n), :time => t))
        #print("LSQR_par done\n")
        global m += 10
        global n += 10
    end
    return Svd_time, LSQR_seq, LSQR_par
end

function time_A_var_Hilbert_poorly_conditioned()
    global m = 50
    global n = 50
    k = 25 # fixed
    e = 0.1 # fixed
    Svd_time = []
    LSQR_seq = []
    LSQR_par = []
    # Plot the execution times as the matrix sizes of the 3 methods vary in time_gap
    for _ in 1:5
        
        # Hilbert poorly conditioned square matrix
        A = [1/(i+j-1) for i in 1:n, j in 1:m]
        
        V_initial = rand(n, k)
        t,_,_= time_gap(copy(A), k)
        push!(Svd_time, Dict(:dim => (m, n), :time => t))
        #print("SVD done\n")
        t,_,_= time_gap(copy(A), k, e, copy(V_initial), parallel=false)
        push!(LSQR_seq, Dict(:dim => (m, n), :time => t))
        #print("LSQR_seq done\n")
        t,_,_= time_gap(copy(A), k, e, copy(V_initial), parallel=true)
        push!(LSQR_par, Dict(:dim => (m, n), :time => t))
        #print("LSQR_par done\n")
        global m += 10
        global n += 10
    end
    return Svd_time, LSQR_seq, LSQR_par
end

function time_A_var_Vandermonde_poorly_conditioned()
    global m = 50
    global n = 50
    k = 25 # fixed
    e = 0.1 # fixed
    Svd_time = []
    LSQR_seq = []
    LSQR_par = []
    # Plot the execution times as the matrix sizes of the 3 methods vary in time_gap
    for _ in 1:5
        
        #Vandermonde matrix
        A = [i^(j-1) for i in 1:n, j in 1:m]
        
        V_initial = rand(n, k)
        t,_,_= time_gap(copy(A), k)
        push!(Svd_time, Dict(:dim => (m, n), :time => t))
        #print("SVD done\n")
        t,_,_= time_gap(copy(A), k, e, copy(V_initial), parallel=false)
        push!(LSQR_seq, Dict(:dim => (m, n), :time => t))
        #print("LSQR_seq done\n")
        t,_,_= time_gap(copy(A), k, e, copy(V_initial), parallel=true)
        push!(LSQR_par, Dict(:dim => (m, n), :time => t))
        #print("LSQR_par done\n")
        global m += 10
        global n += 10
    end
    return Svd_time, LSQR_seq, LSQR_par
end

#--------------------------------------------------------------------------------------------------------------------------


# the time execution varyng k
function time_k_var()
    m = 50
    n = 50
    global k = 10 # variable 
    e = 0.1 
    Svd_time = []
    LSQR_seq = []
    LSQR_par = []
    # Plot the execution times as the matrix sizes of the 3 methods vary in time_gap
    for _ in 1:8
        A = rand(m, n)
        V_initial = rand(n, k)
        t,_,_= time_gap(copy(A), k)
        push!(Svd_time, Dict(:k => k, :time => t))
        #print("SVD done\n")
        t,_,_= time_gap(copy(A), k, e, copy(V_initial), parallel=false)
        push!(LSQR_seq, Dict(:k => k, :time => t))
        #print("LSQR_seq done\n")
        t,_,_= time_gap(copy(A), k, e, copy(V_initial), parallel=true)
        push!(LSQR_par, Dict(:k => k, :time => t))
        #print("LSQR_par done\n")
        global k += 5
    end
    return Svd_time, LSQR_seq, LSQR_par
end


# # the time execution varyng e
function time_e_var()
    m = 50
    n = 50
    k = 10
    global e = round(0.0001, digits=4)  # variable
    Svd_time = []
    LSQR_seq = []
    LSQR_par = []
    # Plot the execution times as the matrix sizes of the 3 methods vary in time_gap
    for _ in 1:8
        A = rand(m, n)
        V_initial = rand(n, k)
        t,_,_= time_gap(copy(A), k)
        push!(Svd_time, Dict(:e => e, :time => t))
        #print("SVD done\n")
        t,_,_= time_gap(copy(A), k, e, copy(V_initial), parallel=false)
        push!(LSQR_seq, Dict(:e => e, :time => t))
        #print("LSQR_seq done\n")
        t,_,_= time_gap(copy(A), k, e, copy(V_initial), parallel=true)
        push!(LSQR_par, Dict(:e => e, :time => t))
        #print("LSQR_par done\n")
        global e = e*5
        global e = round(e, digits=4)
    end
    return Svd_time, LSQR_seq, LSQR_par
end