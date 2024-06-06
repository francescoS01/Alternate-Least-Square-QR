include("../alternate_LSQR.jl")
include("../low_rank_SVD.jl")
include("print_matrix.jl")


function time_gap(A, k, e=nothing, V_initial=nothing; parallel=true)

    Ak_SVD=[]
    Ak_QR=[]

    if (e===nothing && V_initial===nothing)
        t = @elapsed begin
            Ak_SVD, trash = low_rank(A, k)
        end
    else
        if (parallel===true)
            nt=Threads.nthreads()
            t = @elapsed begin
                U, V = LS_QR_alternate_parallellized(A, k, e, V_initial, nt) 
                Ak_QR =  U*V'
            end
            #print("LSQR_parallel_new entered\n")
        else
            t = @elapsed begin
                U, V = LS_QR_alternate(A, k, e, V_initial) 
                Ak_QR =  U*V'
            end
            #print("LSQR_seq entered\n")
        end
    end

    gap = 0
    if (Ak_SVD != [] && Ak_QR != [])
        gap = norm(Ak_SVD - Ak_QR, 2)
    elseif (Ak_SVD == [] && Ak_QR != [])
        Ak_SVD, _ = low_rank(A, k)
        gap = norm(Ak_SVD - Ak_QR, 2)
    end
    # If Ak_SVD == [] && Ak_QR == [] t=0 o t=svd_time, gap always 0 because it is useless to calculate a gap with svd

    return t, gap
end
