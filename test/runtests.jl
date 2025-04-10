import CombinatorialLinearOracles as CO
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
    lmo = CO.PerfectMatchingLMO(g)
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
    @test is_matching
end

@testset "Matching LMO" begin
    N = 200
    Random.seed!(9754)
    g = Graphs.complete_graph(N)
    iter = collect(Graphs.edges(g))
    M = length(iter)
    direction = randn(M)
    lmo = CO.MatchingLMO(g)
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
    lmo = CO.SpanningTreeLMO(g)
    v = FrankWolfe.compute_extreme_point(lmo, direction)
    tree = eltype(iter)[]
    for i in 1:M
        if (v[i] == 1)
            push!(tree, iter[i])
        end
    end
    @test Graphs.is_tree(SimpleGraphFromIterator(tree))
end

@testset "Shortest path" begin
    n = 200
    Random.seed!(42)
    for _ in 1:10
        g = Graphs.random_orientation_dag(Graphs.complete_graph(n))
        src_node = 1
        dst_node = nv(g)
        while !has_path(g, src_node, dst_node) || src_node == dst_node
            src_node = rand(1:nv(g))
            dst_node = rand(1:nv(g))
        end
        lmo = CO.ShortestPathLMO(g, src_node, dst_node)
        for _ in 1:10
            direction = randn(ne(g))
            v = FrankWolfe.compute_extreme_point(lmo, direction)
            @test sum(v) >= 1
        end
    end
    @testset "Shortest path with negative costs" begin
        g = SimpleDiGraph(4)
        add_edge!(g, 1, 2)
        add_edge!(g, 1, 4)
        add_edge!(g, 2, 3)
        add_edge!(g, 3, 4)
        mat = adjacency_matrix(g)
        mat[3,4] = -10
        lmo = CO.ShortestPathLMO(g, 1, 4)
        costs = ones(ne(g))
        idx = findfirst(==(Edge(3,4)), collect(edges(g)))
        costs[idx] = -10
        v = FrankWolfe.compute_extreme_point(lmo, costs)
        @test sum(v) == 3
    end
end
