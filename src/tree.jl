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
        Dense(length(feat_sub[id]), hidden, tanh),
        Dense(hidden, hidden, tanh),
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
        n.right_split = decision[1, :] .< 0.5

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
    left_split = decision[1, :] .>= 0.5
    right_split = decision[1, :] .< 0.5
    left_data = x[:, left_split]
    right_data = x[:, right_split]

    preds = Int.(zeros(size(x, 2)))
    preds[left_split] = predict(n.left, left_data)
    preds[right_split] = predict(n.right, right_data)

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

function splitter_importance(n :: Node, x :: Array{Float32, 2})
    val = min(200, size(x, 2))
    function predict_wrapper(n :: Node, data :: DataFrame)
        pred = DataFrame(y_pred = predict(n, Float32.(Array(data)')))
    end

    data_shap = ShapML.shap(explain = DataFrame(x[:, 1:val]'),
        reference = DataFrame(x[:, end-val+1:end]'),
        model = n,
        predict_function = predict_wrapper,
        sample_size = 150,
        parallel = :both,
    )

    s̄ = DataFrames.by(data_shap, [:feature_name], 
        mean_effect = [:shap_effect] => x -> mean(abs.(x.shap_effect)))

    imp = Dict{Int, Float32}()
    for i ∈ 1:length(n.subset)
        @inbounds imp[n.subset[i]] = s̄.mean_effect[i]
    end
    return imp
end

function importance(n :: Node, x :: Array{Float32, 2})
    scores = splitter_importance(n, x)
    for (k, v) ∈ scores
        scores[k] = v * 1/(n.impurity + 0.000001)
    end

    if n.depth > 1
        left_scores = importance(n.left, x[:, n.left_split])
        right_scores = importance(n.right, x[:, n.right_split])

        for d ∈ [left_scores, right_scores]
            for (k, v) ∈ d
                if k ∈ keys(scores)
                    scores[k] += v
                else
                    scores[k] = v
                end
            end
        end
    end
    return scores
end

function plot_importance(imp :: Dict{Int, Float32}, x :: Array{Float32, 2})
    bar(string.(1:size(x, 1)), [imp[x] for x ∈ 1:size(x, 1)], legend=:false, color=1:size(x, 1),
        title="Relative Feature Importance", xlabel="Feature", ylabel="Importance")
end

function tree_train!(epochs :: Int, n :: Node, x :: Array{Float32, 2}, y :: Vector{Int})
    ps = params(n)
    opt = ADAM(0.05)

    pbar = Progress(epochs)
    for epoch ∈ 1:epochs
        local training_loss

        DeepForest.precompute!(n, x, y)
        gs = gradient(ps) do
            training_loss = loss!(n, x, y)
            return training_loss
        end
        next!(pbar; showvalues = [(:loss, training_loss), (:acc, mean(predict(n, x) .== y))])
        Flux.update!(opt, ps, gs)
    end
end