
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

function FrankWolfe.compute_extreme_point(lmo::ShortestPathLMO, direction)
    for (idx, edge) in enumerate(edges(lmo.graph))
        lmo.dist_matrix[src(edge), dst(edge)] = direction[idx]
    end
    shortest_path_result = Set(Graphs.a_star(lmo.graph, lmo.src, lmo.dst, lmo.dist_matrix))
    # resulting vertex as a BitVector
    indicence_vector = falses(ne(lmo.graph))
    for (idx, edge) in enumerate(edges(lmo.graph))
        if edge in shortest_path_result
            indicence_vector[idx] = 1
            delete!(shortest_path_result, edge)
        end
    end
    return indicence_vector
end
