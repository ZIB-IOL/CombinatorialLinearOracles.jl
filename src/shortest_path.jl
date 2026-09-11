"""
Shortest path LMO copied from https://zib-iol.github.io/Boscia.jl/stable/examples/docs-01-network-design/ based on K. Sharma et al.
\"Network Design for the Traffic Assignment Problem with Mixed-Integer Frank-Wolfe\"
https://github.com/ZIB-IOL/Network_Design_with_Integer_Frank_Wolfe/tree/main
"""
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
    return ShortestPathLMO{eltype(dist_matrix),typeof(graph)}(
        graph,
        src_node,
        dst_node,
        dist_matrix,
        edge_dict,
    )
end

"""
     compute_extreme_point(lmo::ShortestPathLMO,direction;v=falses(ne(lmo.graph)),kwargs...,)

Compute the shortest path using Moore Bellman Ford algorithm and route flow along it.
"""
function FrankWolfe.compute_extreme_point(
    lmo::ShortestPathLMO,
    direction;
    v=falses(ne(lmo.graph)),
    kwargs...,
)
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

"""
     add_demand_to_path!(x, demand, state, origin, destination, link_dic, edge_list, num_zones)
     
Add demand to flow vector following shortest path
"""
function add_demand_to_path!(x, demand, state, origin, destination, link_dic, edge_list, num_zones)
    current = destination
    parent = -1
    edge_count = length(edge_list)
    agg_start = edge_count * num_zones

    while parent != origin && origin != destination && current != 0
        parent = state.parents[current]
        if parent != 0
            link_idx = link_dic[parent, current]
            if link_idx != 0
                x[(destination-1)*edge_count+link_idx] += demand
                x[agg_start+link_idx] += demand
            end
        end
        current = parent
    end
end


"""
     all_or_nothing_assignment(travel_time_vector, net_data, graph, link_dic, edge_list)

route all flow on shortest paths
"""
function all_or_nothing_assignment(travel_time_vector, net_data, graph, link_dic, edge_list)
    num_zones = net_data.num_zones
    edge_count = net_data.num_edges
    travel_time = travel_time_vector[(num_zones*edge_count+1):((num_zones+1)*edge_count)]
    x = zeros(length(travel_time_vector))

    for origin in 1:num_zones
        state = Graphs.dijkstra_shortest_paths(graph, origin)

        for destination in 1:num_zones
            demand = net_data.travel_demand[origin, destination]
            if demand > 0
                add_demand_to_path!(
                    x,
                    demand,
                    state,
                    origin,
                    destination,
                    link_dic,
                    edge_list,
                    num_zones,
                )
            end
        end
    end

    return x
end

"""
     function Boscia.bounded_compute_extreme_point(lmo::ShortestPathLMO,direction,lower_bounds,upper_bounds,int_vars,)

Computes a mixed integer extreme point in the shortest path polytope by using an all or nothing flow assignment.
"""
function Boscia.bounded_compute_extreme_point(
    lmo::ShortestPathLMO,
    direction,
    lower_bounds,
    upper_bounds,
    int_vars,
)
    x = all_or_nothing_assignment(direction, lmo.net_data, lmo.graph, lmo.link_dic, lmo.edge_list)
    for (i, var_idx) in enumerate(int_vars)
        if direction[var_idx] < 0
            x[var_idx] = upper_bounds[i]
        else
            x[var_idx] = lower_bounds[i]
        end
    end
    return x
end

"""
     function Boscia.is_simple_linear_feasible(lmo::ShortestPathLMO, x)

Checks for linear feasibility inside the shortest path polytope
"""
function Boscia.is_simple_linear_feasible(lmo::ShortestPathLMO, x)
    num_zones = lmo.net_data.num_zones
    num_edges = lmo.net_data.num_edges
    return all(x .>= -1e-6)
end
