"""
MatchingLMO{G}(g::Graphs)

Return a vector v corresponding to edges(g), where if v[i] = 1, 
the edge i is in the matching, and if v[i] = 0, the edge i is not in the matching.  
"""
struct MatchingLMO{G} <: FrankWolfe.LinearMinimizationOracle
    graph::G
end

function compute_extreme_point(
    lmo::MatchingLMO,
    direction::M;
    v=nothing,
    kwargs...,
) where {M}
    N = length(direction)
    iter = collect(Graphs.edges(lmo.graph))
    w = Dict{typeof(iter[1]),typeof(direction[1])}()
    for i in 1:N
        w[iter[i]] = direction[i]
    end
    match = GraphsMatching.minimum_weight_perfect_matching(lmo.graph,w)
    K = length(match.mate)
    v = spzeros(N)
    for i in 1:K
        for j in 1:N
            if(match.mate[i] == src(iter[j]) && dst(iter[j]) == i)
                v[j] = 1
            end
        end
    end
    return v
end
