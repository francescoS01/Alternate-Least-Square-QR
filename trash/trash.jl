
function func()
    return [1, 2, 3, 4, 5]
end

# take the maximum number of threads available in the system and set it as the number of threads to be used by Julia
#JULIA_NUM_THREADS = Sys.CPU_THREADS
#ENV["JULIA_NUM_THREADS"] = 4
#println("Number of threads: ", Sys.CPU_THREADS)
println("Number of threads: ", Threads.nthreads())
