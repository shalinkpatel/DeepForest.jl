using DeepForest, Flux, ProgressMeter, Plots, StatsPlots, Random, CSV, DataFrames
using StatsBase: mode, mean
gr(fmt=:svg, size=(1500, 1000))

data = CSV.File("wine.data") |> DataFrame |> Array |> transpose |> Array
x = Float32.(data[2:end, :])
y = Int.(data[1, :])

perm = shuffle(Array(1:size(x, 2)))
x = x[:, perm]
y = y[perm]

model = Forest(100, 2, 10, 4, 0.5)
forest_train!(1500, model, x, y)

ŷ = predict(model, x);
@show mean(ŷ .== y)

imp = importance(model, x)
plot_importance(imp, x)