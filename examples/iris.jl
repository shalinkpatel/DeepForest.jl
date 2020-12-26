using DeepForest, Flux, ProgressMeter, Plots, StatsPlots, Random
using StatsBase: mode, mean
gr(fmt=:svg, size=(1500, 1000))

x = Float32.(Flux.Data.Iris.features())
lab = Flux.Data.Iris.labels();
y_unique = unique(lab)
mapping = Dict()
for i in 1:length(y_unique)
    mapping[y_unique[i]] = i
end

y = rand(Int, length(lab))
for i in 1:length(y)
    y[i] = mapping[lab[i]]
end
y = Int.(y)

perm = shuffle(Array(1:size(x, 2)))
x = x[:, perm]
y = y[perm]

features = Dict(
    3 => [1, 3],
    2 => [2, 4],
    1 => [1, 2]
)

n = Node(features, 10, 2, 1)
tree_train!(2500, n, x, y)

imp = importance(n, x)
plot_importance(imp, x)

model = Forest(100, 2, 10, 4, 0.5)
forest_train!(1000, model, x, y)

ŷ = predict(model, x);
@show mean(ŷ .== y)

imp = importance(model, x)
plot_importance(imp, x)
