# DeepForest on a Breast Cancer Dataset

Here we set up the dataset (breast) and initialize the model

```julia
using DeepForest, Flux, ProgressMeter, Plots, StatsPlots, Random, CSV, DataFrames
using StatsBase: mode, mean
gr(fmt=:svg, size=(1500, 1000))

data = CSV.File("wdbc.data", header=0) |> DataFrame |> Array
x = Float32.(data[:, 3:end]) |> transpose |> Array
y = Int.(data[:, 2] .== "M") .+ 1

perm = shuffle(Array(1:size(x, 2)))
x = x[:, perm]
y = y[perm]
```




Now we train

```julia
model = Forest(100, 2, 10, 30, 0.25)
forest_train!(5000, model, x, y)
```




Observe the results.

```julia
ŷ = predict(model, x);
@show mean(ŷ .== y)
```

```
mean(ŷ .== y) = 0.9630931458699473
0.9630931458699473
```





We can also compute importances.

```julia
imp = importance(model, x)
plot_importance(imp, x)
```

![](figures/breast_4_1.png)
