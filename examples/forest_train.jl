using DeepForest, Flux, ProgressMeter, Plots, StatsPlots
using StatsBase: mode

x = rand(Float32, 2, 1000);
x[1, :] .*= 2*π;
x[1, :] .-= π;
x[2, :] .*= 3;
x[2, :] .-= 1.5;
y = Int64.(x[2, :] .< sin.(x[1, :])) .+ 1;

model = Forest(25, 2, 10, 2, 1.0)

forest_train!(1250, model, x, y)

ŷ = predict(model, x)
scatter(x[1, :], x[2, :], color=ŷ, legend=false)
plot!((x) -> sin(x), xlims=(-π, π), color=:green, linewidth=5)

imp = importance(model, x)
plot_importance(imp, y)