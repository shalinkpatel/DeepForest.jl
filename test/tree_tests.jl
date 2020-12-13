using DeepForest
using Test

@testset "Leaf precompute" begin
    l = DeepForest.Leaf()
    x = [1.0 2.0; 3.0 4.0]
    y = [0, 0]
    DeepForest.precompute(l, x, y)
    @test l.pred == 0
    y = [1, 1]
    DeepForest.precompute(l, x, y)
    @test l.pred == 1
end