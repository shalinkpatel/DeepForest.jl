abstract type TreeNode end

mutable struct Leaf <: TreeNode
    pred :: Int
end

Leaf() = Leaf(0)

function precompute!(l :: Leaf, x :: Array{Float64, 2}, y :: Vector{Int})
    if size(x, 1) != 0
        l.pred = mode(y)
    end
end

