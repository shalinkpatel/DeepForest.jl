mutable struct Forest
    T :: Int
    trees :: Vector{Node}
end

Flux.@functor Forest
trainable(F :: Forest) = (trainable.(F.trees)...,)

mutable struct ForestSlice
    F :: Forest
    slice :: Vector{Int}
end

Flux.@functor ForestSlice
trainable(FS :: ForestSlice) = (trainable.(FS.F.trees[FS.slice])...,)

function Forest(num_trees :: Int, depth :: Int, hidden :: Int, num_feats :: Int, ratio :: Float64, threads :: Int = Threads.nthreads())
    tree_feats = Vector{Dict{Int, Vector{Int}}}()
    n = Int(ceil(ratio * num_feats))
    for i ∈ 1:num_trees
        ctr = 1
        feats = Dict{Int, Vector{Int}}()
        for j ∈ 1:(2^depth-1)
            rg = Array(1:num_feats)
            shuffle!(rg)
            feats[ctr] = rg[1:n]
            ctr += 1
        end
        push!(tree_feats, feats)
    end


    trees = Vector{Node}()
    for i ∈ 1:num_trees
        push!(trees, Node(tree_feats[i], hidden ,depth, 1))
    end

    Forest(threads, trees)
end

function precompute!(FS :: ForestSlice, x :: Array{Float32, 2}, y :: Vector{Int})
    Threads.@threads for n ∈ FS.F.trees[FS.slice]
        precompute!(n, x, y)
    end
end

function predict(FS :: ForestSlice, x :: Array{Float32, 2})
    preds = ones(Int, length(FS.F.trees[FS.slice]), size(x, 2))
    Threads.@threads for i ∈ 1:length(FS.F.trees[FS.slice])
        preds[i, :] = predict(FS.F.trees[FS.slice][i], x)
    end

    final_preds = rand(Int, size(x, 2))
    Threads.@threads for i ∈ 1:size(x, 2)
        if !isempty(preds[:, i])
            final_preds[i] = mode(preds[:, i])
        end
    end

    return final_preds
end

function loss!(FS :: ForestSlice, x :: Array{Float32, 2}, y :: Vector{Int})
    loss = 0
    for n ∈ FS.F.trees[FS.slice]
        loss += loss!(n, x, y)
    end
    return loss
end

function precompute!(F :: Forest, x :: Array{Float32, 2}, y :: Vector{Int})
    FS = ForestSlice(F, 1:length(F.trees))
    precompute!(FS, x, y)
end

function predict(F :: Forest, x ::Array{Float32, 2})
    FS = ForestSlice(F, 1:length(F.trees))
    predict(FS, x)
end

function loss!(F :: Forest, x :: Array{Float32, 2}, y :: Vector{Int})
    FS = ForestSlice(F, 1:length(F.trees))
    loss!(FS, x, y)
end

function importance(F :: Forest, x :: Array{Float32, 2})
    scores = Dict{Int, Float32}()
    pbar = Progress(length(F.trees))
    Threads.@threads for n ∈ F.trees
        tree_imp = importance(n, x)
        for (k, v) ∈ tree_imp
            if k ∈ keys(scores)
                scores[k] += v
            else
                scores[k] = v
            end
        end
        next!(pbar)
    end
    ρ = sum(values(scores))
    for (k, v) ∈ scores
        scores[k] = v / ρ
    end
    return scores
end

function forest_train!(epochs :: Int, F :: Forest, x :: Array{Float32, 2}, y :: Vector{Int})
    slices = Vector{ForestSlice}()
    pbar = Progress(epochs * F.T)
    per_thread = Int(ceil(length(F.trees) / F.T))
    for i ∈ 1:F.T
        FS = ForestSlice(F, Array(((i-1)*per_thread+1):min(length(F.trees), (i*per_thread))))
        push!(slices, FS)
    end

    Threads.@threads for i ∈ 1:F.T
        FS = slices[i]
        ps = params(FS)
        opt = ADAM(0.05)

        for epoch ∈ 1:epochs
            local training_loss

            DeepForest.precompute!(FS, x, y)
            gs = gradient(ps) do
                training_loss = loss!(FS, x, y)
                return training_loss
            end
            next!(pbar; showvalues = [(:loss, training_loss/length(FS.slice)), (:acc, mean(predict(FS, x) .== y))])
            Flux.update!(opt, ps, gs)
        end
    end
end