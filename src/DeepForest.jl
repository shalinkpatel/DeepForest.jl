module DeepForest

using Flux, ShapML, StatsBase, Plots

include("forest.jl")
include("tree.jl")

export Node, predict

end
