using CombinatorialLinearOracles
using Test
using Random
using SparseArrays

using Graphs
using GraphsMatching
using HiGHS
import FrankWolfe

@testset "Perfect Matching LMO" begin
    N = Int(1e3)
    Random.seed!(4321)
    g = Graphs.complete_graph(N)
    iter = collect(Graphs.edges(g))
    M = length(iter)
    direction = randn(M)
    lmo = CombinatorialLinearOracles.PerfectMatchingLMO(g)
    v = FrankWolfe.compute_extreme_point(lmo, direction)
    tab = zeros(M)
    is_matching = true
    for i in 1:M
        if (v[i] == 1)
            is_matching = (tab[src(iter[i])] == 0 && tab[dst(iter[i])] == 0)
            if (!is_matching)
                break
            end
            tab[src(iter[i])] = 1
            tab[dst(iter[i])] = 1
        end
    end
    @test is_matching == true
end

@testset "Matching LMO" begin
    N = 200
    Random.seed!(9754)
    g = Graphs.complete_graph(N)
    iter = collect(Graphs.edges(g))
    M = length(iter)
    direction = randn(M)
    lmo = CombinatorialLinearOracles.MatchingLMO(g)
    v = FrankWolfe.compute_extreme_point(lmo, direction)
    adj_mat = spzeros(M, M)
    for i in 1:M
        adj_mat[src(iter[i]), dst(iter[i])] = direction[i]
    end
    match_result = GraphsMatching.maximum_weight_matching(g, HiGHS.Optimizer, adj_mat)
    v_sol = spzeros(M)
    K = length(match_result.mate)
    for i in 1:K
        for j in 1:M
            if (match_result.mate[i] == src(iter[j]) && dst(iter[j]) == i)
                v_sol[j] = 1
            end
        end
    end
    @test v_sol == v
end


@testset "SpanningTreeLMO" begin
    N = 500
    Random.seed!(1645)
    g = Graphs.complete_graph(N)
    iter = collect(Graphs.edges(g))
    M = length(iter)
    direction = randn(M)
    lmo = CombinatorialLinearOracles.SpanningTreeLMO(g)
    v = FrankWolfe.compute_extreme_point(lmo, direction)
    tree = eltype(iter)[]
    for i in 1:M
        if (v[i] == 1)
            push!(tree, iter[i])
        end
    end
    @test Graphs.is_tree(SimpleGraphFromIterator(tree))
end
