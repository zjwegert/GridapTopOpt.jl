
using LSTO_Distributed

h = LSTO_Distributed.OptimiserHistory(Float64,[:L,:J,:C1,:C2,:C3],Dict(:C => [:C1,:C2,:C3]),100,true)

push!(h,(1.0,2.0,3.0,4.0,5.0))

write_history("results/tmp.txt",h)