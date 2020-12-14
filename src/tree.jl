abstract type TreeNode end

# Leaves

mutable struct Leaf <: TreeNode
    pred :: Int
end

Leaf() = Leaf(1)

loss!(l :: Leaf, x :: Array{Float32, 2}, y :: Vector{Int}) = 0

function precompute!(l :: Leaf, x :: Array{Float32, 2}, y :: Vector{Int})
    if size(x, 2) != 0
        l.pred = mode(y)
    end
end

function predict(l :: Leaf, x :: Array{Float32, 2})
    fill(l.pred, size(x, 2))
end

# Nodes

mutable struct Node <: TreeNode
    splitter :: Flux.Chain
    subset :: Vector{Int}
    best :: Pair{Int, Int}
    depth :: Int
    impurity :: Float32
    left_split :: BitVector
    right_split :: BitVector
    left :: TreeNode
    right :: TreeNode
end

Flux.@functor Node

function Node(feat_sub :: Dict{Int, Vector{Int}}, hidden :: Int, depth :: Int, id :: Int)
    splitter = Chain(
        Dense(length(feat_sub[id]), hidden, leakyrelu),
        Dense(hidden, hidden, leakyrelu),
        Dense(hidden, 2),
        softmax
    )

    subset = feat_sub[id]

    if depth != 1
        id += 1
        left = Node(feat_sub, hidden, depth - 1, id)
        id += 1
        right = Node(feat_sub, hidden, depth - 1, id)
    else
        left = Leaf()
        right = Leaf()
    end

    Node(splitter, subset, Pair(1, 1), depth, Float32(0.0), BitVector(), BitVector(), left, right)
end

Flux.trainable(l :: Leaf) = ()
Flux.trainable(n :: Node) = (n.splitter, Flux.trainable(n.left)..., Flux.trainable(n.right)...,)

function precompute!(n :: Node, x :: Array{Float32, 2}, y :: Vector{Int})
    if size(x, 2) != 0
        decision = n.splitter(x[n.subset, :])
        n.left_split = decision[1, :] .>= 0.5
        n.right_split = decision[2, :] .> 0.5

        if length(y[n.left_split]) == 0
            left_best = 1
        else
            left_best = mode(y[n.left_split])
        end

        if length(y[n.right_split]) == 0
            right_best = 1
        else
            right_best = mode(y[n.right_split])
        end

        n.best = Pair(left_best, right_best)
        precompute!(n.left, x[:, n.left_split], y[n.left_split])
        precompute!(n.right, x[:, n.right_split], y[n.right_split])
    end
end

function predict(n :: Node, x :: Array{Float32, 2})
    decision = n.splitter(x[n.subset, :])
    n.left_split = decision[1, :] .>= 0.5
    n.right_split = decision[2, :] .> 0.5
    left_data = x[:, n.left_split]
    right_data = x[:, n.right_split]

    preds = Int.(zeros(size(x, 2)))
    preds[n.left_split] = predict(n.left, left_data)
    preds[n.right_split] = predict(n.right, right_data)

    return preds
end

function loss!(n :: Node, x :: Array{Float32, 2}, y :: Vector{Int})
    loss = 0
    if size(x, 2) != 0
        decision = n.splitter(x[n.subset, :])
        left = decision[1, :]
        right = decision[2, :]

        classes = 1:maximum(y)
        ȳ = Flux.onehotbatch(y, classes)

        left_weight = left' .* ȳ
        right_weight = right' .* ȳ

        left_best = Flux.onehotbatch(fill(n.best.first, length(y)), classes)
        right_best = Flux.onehotbatch(fill(n.best.second, length(y)), classes)

        n.impurity = Flux.crossentropy(left_weight, left_best)
        n.impurity += Flux.crossentropy(right_weight, right_best)

        loss += n.impurity
        loss += loss!(n.left, x, y)
        loss += loss!(n.right, x, y)

        return loss
    else
        return loss
    end
end

function tree_train!(epochs :: Int, n :: Node, x :: Array{Float32, 2}, y :: Vector{Int})
    ps = params(n)
    opt = ADAM()

    pbar = ProgressBar(1:epochs)
    for epoch ∈ pbar
        local training_loss

        DeepForest.precompute!(n, x, y)
        gs = gradient(ps) do
            training_loss = loss!(n, x, y)
            return training_loss
        end
        set_description(pbar, string(@sprintf("Loss: %.4f || Acc: %.4f || ", training_loss, mean(predict(n, x) .== y))))
        Flux.update!(opt, ps, gs)
    end
end