using .CombinatorialLinearOracles
using Test
using Random
using Graphs

@testset "Matching LMO" begin
    N = Int(1e3)
    Random.seed!(4321)
    g = Graphs.complete_graph(N)
    iter = collect(Graphs.edges(g))
    M = length(iter)
    direction = randn(M)
    lmo = CombinatorialLinearOracles.MatchingLMO(g)
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
