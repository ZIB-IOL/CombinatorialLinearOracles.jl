
"""
MatchingLMO{G}(g::Graphs)

Return an incidence vector v of the edges of `g`, ordered as `edges(g)` for a minimum-weight matching.
"""
struct MatchingLMO{G} <: FrankWolfe.LinearMinimizationOracle
    graph::G
end

function FrankWolfe.compute_extreme_point(
    lmo::MatchingLMO,
    direction::M;
    v=nothing,
    kwargs...,
) where {M}
    N = length(direction)
    if v === nothing
        v = spzeros(N)
    else
        v .= 0
    end
    iter = collect(edges(lmo.graph))
    g = SimpleGraphFromIterator(iter)
    l = nv(g)
    add_vertices!(g, l)
    w = Dict{typeof(iter[1]),typeof(direction[1])}()
    for i in 1:N
        add_edge!(g, src(iter[i]) + l, dst(iter[i]) + l)
        w[iter[i]] = direction[i]
        w[Edge(src(iter[i]) + l, dst(iter[i]) + l)] = direction[i]
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


"""
PerfectMatchingLMO{G}(g::Graphs)

Return the incidence vector of a minimum-weight perfect matching, `v` is ordered as `edges(g)`.
The constructor verifies that the graph admits a perfect matching.
"""
struct PerfectMatchingLMO{G} <: FrankWolfe.LinearMinimizationOracle
    graph::G
end

function PerfectMatchingLMO(graph::G) where {G}
    @assert nv(graph) % 2 == 0
    return PerfectMatchingLMO{G}(graph)
end

function FrankWolfe.compute_extreme_point(
    lmo::PerfectMatchingLMO,
    direction::M;
    v=nothing,
    kwargs...,
) where {M}
    N = length(direction)
    if v === nothing
        v = spzeros(N)
    else
        v .= 0
    end
    iter = collect(Graphs.edges(lmo.graph))
    w = Dict{typeof(iter[1]),typeof(direction[1])}()
    for i in 1:N
        w[iter[i]] = direction[i]
    end

    match = GraphsMatching.minimum_weight_perfect_matching(lmo.graph, w)
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
