abstract type TreeNode end

# Leaves

mutable struct Leaf <: TreeNode
    pred :: Int
end

Leaf() = Leaf(0)

loss(l :: Leaf, x :: Array{Float64, 2}, y :: Vector{Int}) = 0

function precompute!(l :: Leaf, x :: Array{Float64, 2}, y :: Vector{Int})
    if size(x, 1) != 0
        l.pred = mode(y)
    end
end

function predict(l :: Leaf, x :: Array{Float64, 2})
    fill(l.pred, size(x, 1))
end

# Nodes

mutable struct Node <: TreeNode
    splitter :: Flux.Chain
    subset :: Vector{Int}
    best :: Pair{Int, Int}
    depth :: Int
    impurity :: Float64
    left_split :: BitVector
    right_split :: BitVector
    left :: TreeNode
    right :: TreeNode
end

function Node(feat_sub :: Dict{Int, Vector{Int}}, hidden :: Int, depth :: Int, id :: Int)
    splitter = Chain(
        Dense(length(feat_sub[id]), hidden, leakyrelu),
        Dense(hidden, hidden, leakyrelu),
        Dense(hidden, 2, softmax)
    )

    if depth != 1
        id += 1
        left = Node(feat_sub, hidden, depth - 1, id)
        id += 1
        right = Node(feat_sub, hidden, depth - 1, id)
    else
        left = Leaf()
        right = Leaf()
    end

    Node(splitter, feat_sub[id], Pair(0, 0), depth, 0.0, BitVector(), BitVector(), left, right)
end

function tree_params(n :: Node)
    get_splitter(l :: Leaf) = []
    get_splitter(n :: Node) = [n.splitter, get_splitter(n.left)..., get_splitter(n.right)...]
    Flux.params(get_splitter(n)...)
end
