using .CombinatorialLinearOracles
using Test
using Random
using SparseArrays
using Graphs, GraphsMatching
using JuMP, Cbc

@testset "Perfect Matching LMO" begin
    N = Int(1e3)
    Random.seed!(4321)
    g = Graphs.complete_graph(N)
    iter = collect(Graphs.edges(g))
    M = length(iter)
    direction = randn(M)
    lmo = CombinatorialLinearOracles.PerfectMatchingLMO(g)
    v = CombinatorialLinearOracles.compute_extreme_point(lmo, direction)
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
    v = CombinatorialLinearOracles.compute_extreme_point(lmo,direction)
    adjMat = zeros(M,M)
    for i in 1:M
        adjMat[src(iter[i]),dst(iter[i])] = direction[i]
    end
    match = GraphsMatching.maximum_weight_matching(g,optimizer_with_attributes(Cbc.Optimizer,"logLevel"=>0),adjMat)
    v_sol = spzeros(M)
    K = length(match.mate)
    for i in 1:K
        for j in 1:M
            if(match.mate[i] == src(iter[j]) && dst(iter[j]) == i)
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
    v = CombinatorialLinearOracles.compute_extreme_point(lmo, direction)
    tree = Array{Edge}(undef, (0,))
    for i in 1:M
        if (v[i] == 1)
            push!(tree, iter[i])
        end
    end
    @test Graphs.is_tree(SimpleGraphFromIterator(tree)) == true
end
