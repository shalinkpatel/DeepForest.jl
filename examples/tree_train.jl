using DeepForest, Flux, ProgressBars, Plots, StatsPlots
using StatsBase: mode

x = rand(Float32, 2, 1000);
x[1, :] .*= 2*π;
x[1, :] .-= π;
x[2, :] .*= 3;
x[2, :] .-= 1.5;
y = Int64.(x[2, :] .< sin.(x[1, :])) .+ 1;

features = Dict(
    7 => [1, 2],
    6 => [1, 2],
    5 => [1, 2],
    4 => [1, 2],
    3 => [1, 2],
    2 => [1, 2],
    1 => [1, 2]
)

model = Node(features, 10, 2, 1)

tree_train!(1000, model, x, y)

ŷ = DeepForest.predict(model, x)
scatter(x[1, :], x[2, :], color=ŷ, legend=false)
plot!((x) -> sin(x), xlims=(-π, π), color=:green, linewidth=5)

imp = importance(model, x)
ρ = sum(values(imp))
for (k, v) ∈ imp
    imp[k] = v / ρ
end
plot_importance(imp, y)