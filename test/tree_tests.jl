using DeepForest, Flux
using Test

@testset "Leaf precompute" begin
    l = DeepForest.Leaf()
    x = [1.0 2.0; 3.0 4.0]
    y = [1, 1]
    DeepForest.precompute!(l, x, y)
    @test l.pred == 1
    y = [2, 2]
    DeepForest.precompute!(l, x, y)
    @test l.pred == 2
end

@testset "Leaf predict" begin
    l = DeepForest.Leaf()
    x = [1.0 2.0; 3.0 4.0]
    y = [1, 1]
    DeepForest.precompute!(l, x, y)
    @test DeepForest.predict(l, x) == [1, 1]
end

@testset "Node init" begin
    feat_sub = Dict(
        1 => [1, 2]
    )

    n = Node(feat_sub, 5, 1, 1)
    @test n.subset == [1, 2]
    @test typeof(n.left) == DeepForest.Leaf

    feat_sub = Dict(
        1 => [1, 2],
        2 => [2, 3],
        3 => [1, 3]
    )

    n = Node(feat_sub, 5, 2, 1)
    @test n.left.subset == [2, 3]
    @test n.right.subset == [1, 3]
end

@testset "Node parameters" begin
    feat_sub = Dict(
        1 => [1, 2],
        2 => [2, 3],
        3 => [1, 3]
    )

    n = Node(feat_sub, 5, 2, 1)
    @test DeepForest.tree_params(n) != Flux.Params([])
end

@testset "Node precompute" begin
    feat_sub = Dict(
        1 => [1, 2],
        2 => [2, 3],
        3 => [1, 3]
    )

    n = Node(feat_sub, 5, 2, 1)
    x = rand(4, 3)
    y = [1, 2, 2, 2]
    DeepForest.precompute!(n, x, y)
    @test n.best.first != 1 || n.best.second != 1
end

@testset "Node predict" begin
    feat_sub = Dict(
        1 => [1, 2],
        2 => [2, 3],
        3 => [1, 3]
    )

    n = Node(feat_sub, 5, 2, 1)
    x = rand(4, 3)
    y = [1, 2, 2, 2]
    DeepForest.precompute!(n, x, y)
    @test length(predict(n, x)) == 4
    @test length(unique(predict(n, x))) <= 2
end

@testset "Node loss" begin
    feat_sub = Dict(
        1 => [1, 2],
        2 => [2, 3],
        3 => [1, 3]
    )

    n = Node(feat_sub, 5, 2, 1)
    x = rand(4, 3)
    y = [1, 2, 2, 2]
    DeepForest.precompute!(n, x, y)

    @test loss!(n, x, y) != 0

    gs = gradient(() -> loss!(n, x, y), DeepForest.tree_params(n))
    @test gs != IdDict()
end