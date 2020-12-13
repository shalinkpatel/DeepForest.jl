using SafeTestsets

@safetestset "Forest Tests" begin include("forest_tests.jl") end
@safetestset "Tree Tests" begin include("tree_tests.jl") end