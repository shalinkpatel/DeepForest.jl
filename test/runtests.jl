using SafeTestsets

@safetestset "Tree Tests" begin include("tree_tests.jl") end
@safetestset "Forest Tests" begin include("forest_tests.jl") end