include("../alternate_LSQR.jl")
include("../low_rank_SVD.jl")
include("print_matrix.jl")



# A questa funzione vengono passate A, k, e, V_initial. La funzione resituisce il gap tra il metodo e SVD, 
# il gap tra il nostro metodo e A e il tempo di esecuzione del metodo  
function time_gap(A, k, e=nothing, V_initial=nothing; parallel=nothing)

    missing_param = (isnothing(e)) + (isnothing(V_initial)) + (isnothing(parallel))
    #println("Number of missing parameters: ", missing_param)
    
    Ak_SVD=[]
    Ak_QR=[]
    gapSVD = 0
    gapA = 0

    # SVD method case 
    if (missing_param == 3)
        t = @elapsed begin
            Ak_SVD, trash = low_rank_SVD(A, k)
        end
        gapA = norm(A - Ak_SVD, 2) / norm(A, 2)
        #println("SVD entered")
    end

    # our method sequential case (miss only parallel parameter)
    if (missing_param == 1 || (missing_param == 0 && parallel == false))
        t = @elapsed begin
            U, V = low_rank_LSQR(A, k, e, V_initial) 
            Ak_QR =  U*V'
        end
        # Gap calculation
        Ak_SVD, trash = low_rank_SVD(A, k)
        gapSVD = norm(Ak_SVD - Ak_QR, 2) / norm(Ak_SVD, 2)
        gapA = norm(A - Ak_QR, 2) / norm(A, 2)
        #println("LSQR_seq entered")
    end

    # our method parallel case
    if (missing_param == 0 && parallel == true)
        nt=Threads.nthreads()
        t = @elapsed begin
            U, V = low_rank_LSQR_parallellized(A, k, e, V_initial, nt) 
            Ak_QR =  U*V'
        end
        # Gap calculation
        Ak_SVD, trash = low_rank_SVD(A, k)
        gapSVD = norm(Ak_SVD - Ak_QR, 2) / norm(Ak_SVD, 2)
        gapA = norm(A - Ak_QR, 2) / norm(A, 2)
        #println("LSQR_parallel_new entered")
    end

    return t, gapSVD, gapA

end