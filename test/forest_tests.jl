using DeepForest, Flux
using StatsBase: mode, mean
using Test

@testset "Forest init" begin
    f = Forest(10, 2, 10, 10, 0.25)
    @test length(f.trees) == 10
    @test length(f.trees[1].subset) == 3
end

@testset "Forest precompute" begin
    x = rand(Float32, 3, 4)
    y = [1, 2, 2, 2]
    f = Forest(10, 2, 10, 3, 2/3)
    DeepForest.precompute!(f, x, y)
    @test f.trees[1].best.first != 1 || f.trees[1].best.second != 1
end

@testset "Forest predict" begin
    x = rand(Float32, 3, 4)
    y = [2, 2, 2, 2]
    f = Forest(10, 2, 10, 3, 2/3)
    DeepForest.precompute!(f, x, y)
    @test predict(f, x) == [2, 2, 2, 2]
end

@testset "Forest loss" begin
    x = rand(Float32, 3, 4)
    y = [1, 2, 2, 2]
    f = Forest(10, 2, 10, 3, 2/3)
    DeepForest.precompute!(f, x, y)

    @test loss!(f, x, y) != 0

    gs = gradient(() -> loss!(f, x, y), params(f))
    @test gs.grads != IdDict()

    FS = DeepForest.ForestSlice(f, Array(1:5))
    gs = gradient(() -> loss!(FS, x, y), params(f))
    @test gs.grads != IdDict()
end

@testset "Forest train" begin
    x = rand(Float32, 2, 1000);
    x[1, :] .*= 2*π;
    x[1, :] .-= π;
    x[2, :] .*= 3;
    x[2, :] .-= 1.5;
    y = Int64.(x[2, :] .< sin.(x[1, :])) .+ 1;

    model = Forest(25, 2, 10, 2, 1.0)

    forest_train!(500, model, x, y)
    @test mean(predict(model, x) .== y) > .9
end

@testset "Forest importance" begin
    x = rand(Float32, 2, 1000);
    x[1, :] .*= 2*π;
    x[1, :] .-= π;
    x[2, :] .*= 3;
    x[2, :] .-= 1.5;
    y = Int64.(x[2, :] .< sin.(x[1, :])) .+ 1;

    model = Forest(25, 2, 10, 2, 1.0)

    forest_train!(500, model, x, y)
    imp = importance(model, x)
    @test imp[2] > 0
    @test imp[1] > 0
end