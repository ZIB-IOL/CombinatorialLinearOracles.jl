"""
MatchingLMO{G}(g::Graphs)

Return a vector v corresponding to edges(g), where if v[i] = 1, 
the edge i is in the maximum weight matching, and if v[i] = 0, the edge i is not in the matching.
"""
struct MatchingLMO{G} <: FrankWolfe.LinearMinimizationOracle
    graph::G
end

function compute_extreme_point(lmo::MatchingLMO, direction::M; v=nothing, kwargs...) where {M}
    N = length(direction)
    v = spzeros(N)
    iter = collect(edges(lmo.graph))
    g = SimpleGraphFromIterator(iter)
    l = nv(g)
    add_vertices!(g, l)
    w = Dict{typeof(iter[1]),typeof(direction[1])}()
    for i in 1:N
        add_edge!(g, src(iter[i]) + l, dst(iter[i]) + l)
        w[iter[i]] = -direction[i]
        w[Edge(src(iter[i]) + l, dst(iter[i]) + l)] = -direction[i]
    end

    for i in 1:l
        add_edge!(g, i, i + l)
        w[Edge(i, i + l)] = 0
    end

    match = GraphsMatching.minimum_weight_perfect_matching(g, w)

    K = length(match.mate)
    for i in 1:K
        for j in 1:N
            if (match.mate[i] == src(iter[j]) && dst(iter[j]) == i)
                v[j] = 1
            end
        end
    end
    return v
end
