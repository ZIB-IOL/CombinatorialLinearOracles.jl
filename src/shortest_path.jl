
struct ShortestPathLMO{T,G} <: FrankWolfe.LinearMinimizationOracle
    graph::G
    src::Int
    dst::Int
    dist_matrix::SparseArrays.SparseMatrixCSC{T,Int}
    edge_dict::Dict{Edge{Int},Int}
end

function ShortestPathLMO(graph, src_node, dst_node)
    @assert !Graphs.is_cyclic(graph)
    @assert Graphs.has_path(graph, src_node, dst_node)
    dist_matrix = spzeros(Graphs.nv(graph), Graphs.nv(graph))
    edge_dict = Dict(Graphs.edges(graph) .=> 1:Graphs.ne(graph))
    return ShortestPathLMO{eltype(dist_matrix), typeof(graph)}(graph, src_node, dst_node, dist_matrix, edge_dict)
end

function FrankWolfe.compute_extreme_point(lmo::ShortestPathLMO, direction; v=falses(ne(lmo.graph)))
    for (idx, edge) in enumerate(edges(lmo.graph))
        lmo.dist_matrix[src(edge), dst(edge)] = direction[idx]
    end
    shortest_path_state = bellman_ford_shortest_paths(lmo.graph, lmo.src, lmo.dist_matrix)
    v .= 0
    # src node is the origin
    @assert shortest_path_state.parents[lmo.src] == 0
    node_idx = lmo.dst
    while node_idx != lmo.src
        u_node = shortest_path_state.parents[node_idx]
        v[lmo.edge_dict[Graphs.Edge(u_node, node_idx)]] = 1
        node_idx = u_node
    end
    return v
end
