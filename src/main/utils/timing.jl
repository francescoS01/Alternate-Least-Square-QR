include("../alternate_LSQR.jl")
include("../low_rank_SVD.jl")

# function total_time(A, k, e=nothing, V_initial=nothing; parallel=true)
#     if (e===nothing && V_initial===nothing)
#         t = @elapsed begin
#             Ak_SVD, trash = low_rank(A, k)
#         end
#         print("SVD entered\n")
#     else
#         if (parallel===true)
#             try
#                 t = @elapsed begin
#                     U, V = LS_QR_alternate_parallellized(A, k, e, V_initial) 
#                     Ak_QR =  U*V'
#                 end
#             catch ex
#                 println("Caught an error: ", ex)
#             end
#             print("LSQR_parallel entered\n")
#         else
#             t = @elapsed begin
#                 U, V = LS_QR_alternate(A, k, e, V_initial) 
#                 Ak_QR =  U*V'
#             end
#             print("LSQR_seq entered\n")
#         end
#     end

#     return t
# end

function total_time(A, k, e=nothing, V_initial=nothing; parallel=true)
    if (e===nothing && V_initial===nothing)
        t = @elapsed begin
            Ak_SVD, trash = low_rank(A, k)
        end
        #print("SVD entered\n")
    else
        if (parallel===true)
            nt=Threads.nthreads()
            t = @elapsed begin
                U, V = LS_QR_alternate_parallellized_new(A, k, e, V_initial, nt) 
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

    return t
end