using DelimitedFiles
using BenchmarkTools

isr=readdlm("isr.csv",',');
returns=readdlm("returns.csv",',');
weig=readdlm("weights.csv",',');

include("backtest.jl")

@benchmark backtest(isr,returns,weig,1000.0,0.0003)
