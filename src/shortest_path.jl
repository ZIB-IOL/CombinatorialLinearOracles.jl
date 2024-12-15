
struct ShortestPathLMO{T,G} <: FrankWolfe.LinearMinimizationOracle
    graph::G
    src::Int
    dst::Int
    dist_matrix::SparseArrays.SparseMatrixCSC{T,Int}
end

function ShortestPathLMO(graph, src_node, dst_node)
    @assert !Graphs.is_cyclic(graph)
    @assert Graphs.has_path(graph, src_node, dst_node)
    dist_matrix = spzeros(nv(graph), nv(graph))
    return ShortestPathLMO{eltype(dist_matrix), typeof(graph)}(graph, src_node, dst_node, dist_matrix)
end

function FrankWolfe.compute_extreme_point(lmo::ShortestPathLMO, direction; v=falses(ne(lmo.graph)))
    for (idx, edge) in enumerate(edges(lmo.graph))
        lmo.dist_matrix[src(edge), dst(edge)] = direction[idx]
    end
    shortest_path_result = Set(Graphs.a_star(lmo.graph, lmo.src, lmo.dst, lmo.dist_matrix))
    v .= 0
    for (idx, edge) in enumerate(edges(lmo.graph))
        if edge in shortest_path_result
            v[idx] = 1
            delete!(shortest_path_result, edge)
        end
    end
    return v
end
