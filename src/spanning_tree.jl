"""
SpanningTreeLMO{G}(g::Graphs)

Return a vector v corresponding to edges(g), where if v[i] = 1, 
the edge i is in the minimum spanning tree, and if v[i] = 0, 
the edge i is not in the minimum spanning tree.  
"""
struct SpanningTreeLMO{G} <: FrankWolfe.LinearMinimizationOracle
    graph::G
end

function FrankWolfe.compute_extreme_point(
    lmo::SpanningTreeLMO,
    direction::M;
    v=nothing,
    kwargs...,
) where {M}
    N = length(direction)
    iter = collect(Graphs.edges(lmo.graph))
    distmx = spzeros(N, N)
    min_weight = minimum(direction)
    # we add a positive offset for negative weights
    # not changing optimal solutions since all trees have the same # of edges
    offset = min_weight > 0 ? zero(min_weight) : 1 - min_weight
    for idx in 1:N
        distmx[src(iter[idx]), dst(iter[idx])] = direction[idx] + offset
        distmx[dst(iter[idx]), src(iter[idx])] = direction[idx] + offset
    end
    span = Graphs.kruskal_mst(lmo.graph, distmx)
    v = spzeros(N)
    for edge in span
        for i in 1:N
            if (src(edge) == src(iter[i]) && dst(edge) == dst(iter[i]))
                v[i] = 1
            end
        end
    end
    return v
end
