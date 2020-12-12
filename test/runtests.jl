using DeepForest
using Test

@testset "DeepForest.jl" begin
    @test DeepForest.f(2) == 4
    @test DeepForest.f(3) == 6
end
