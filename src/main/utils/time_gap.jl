include("../alternate_LSQR.jl")
include("../low_rank_SVD.jl")
include("print_matrix.jl")


# function time_gap(A, k, e=nothing, V_initial=nothing; parallel=true)

#     Ak_SVD=[]
#     Ak_QR=[]

#     if (e===nothing && V_initial===nothing)
#         t = @elapsed begin
#             Ak_SVD, trash = low_rank_SVD(A, k)
#         end
#     else
#         if (parallel===true)
#             nt=Threads.nthreads()
#             t = @elapsed begin
#                 U, V = LS_QR_alternate_parallellized(A, k, e, V_initial, nt) 
#                 Ak_QR =  U*V'
#             end
#             #print("LSQR_parallel_new entered\n")
#         else
#             t = @elapsed begin
#                 U, V = LS_QR_alternate(A, k, e, V_initial) 
#                 Ak_QR =  U*V'
#             end
#             #print("LSQR_seq entered\n")
#         end
#     end

#     gap = 0
#     if (Ak_SVD != [] && Ak_QR != [])
#         gap = norm(Ak_SVD - Ak_QR, 2)
#     elseif (Ak_SVD == [] && Ak_QR != [])
#         Ak_SVD, _ = low_rank(A, k)
#         gap = norm(Ak_SVD - Ak_QR, 2)
#     end
#     # If Ak_SVD == [] && Ak_QR == [] t=0 o t=svd_time, gap always 0 because it is useless to calculate a gap with svd

#     return t, gap
# end

function time_gap(A, k, e=nothing, V_initial=nothing; parallel=nothing)

    missing_param = (isnothing(e)) + (isnothing(V_initial)) + (isnothing(parallel))
    #println("Number of missing parameters: ", missing_param)
    
    if (missing_param == 3)
        t = @elapsed begin
            Ak_SVD, trash = low_rank_SVD(A, k)
        end
        #println("SVD entered")
    end
    if (missing_param == 1 || (missing_param == 0 && parallel == false))
        t = @elapsed begin
            U, V = low_rank_LSQR(A, k, e, V_initial) 
            Ak_QR =  U*V'
        end
        #println("LSQR_seq entered")
    end
    if (missing_param == 0 && parallel == true)
        nt=Threads.nthreads()
        t = @elapsed begin
            U, V = low_rank_LSQR_parallellized(A, k, e, V_initial, nt) 
            Ak_QR =  U*V'
        end
        #println("LSQR_parallel_new entered")
    end

    return t
end

# A = [61 22 14 67 23;
#     78 31 11 35 57;
#     56 86 14 24 90;
#     46 0 70 3 60;
#     8 89 58 4 9]
# e = 0.0001
# k = 3
# V_initial = rand(5, 5)


# t = @elapsed begin
#     Ak_SVD, trash = low_rank(copy(A), k)
# end
# println("Final time: ", t)

# println("")

# t = time_gapV2(copy(A), k)
# println("Final time: ", t)

