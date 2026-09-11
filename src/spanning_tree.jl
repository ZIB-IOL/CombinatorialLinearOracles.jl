"""
SpanningTreeLMO{G}(g::Graphs)

Return a vector v corresponding to edges(g), where if v[i] = 1, 
the edge i is in the minimum spanning tree, and if v[i] = 0, 
the edge i is not in the minimum spanning tree.  
"""
struct SpanningTreeLMO{G} <: FrankWolfe.LinearMinimizationOracle
    graph::G
end

"""
     compute_extreme_point(lmo::SpanningTreeLMO,direction::M;v=nothing,kwargs...,)

Computes a minimum spanning tree with respect to weights v.
"""
function FrankWolfe.compute_extreme_point(
    lmo::SpanningTreeLMO,
    direction::M;
    v=nothing,
    kwargs...,
) where {M}
    N = length(direction)
    iter = collect(Graphs.edges(lmo.graph))
    distmx = spzeros(N, N)
    for idx in 1:N
        distmx[src(iter[idx]), dst(iter[idx])] = direction[idx]
        distmx[dst(iter[idx]), src(iter[idx])] = direction[idx]
    end
    span = Graphs.kruskal_mst(lmo.graph, distmx)
    v = spzeros(N)
    for edge in span
        for i in 1:N
            if (src(edge) == src(iter[i]) && dst(edge) == dst(iter[i]))
                v[i] = 1
                break
            end
        end
    end
    return v
end

"""
     Boscia.check_feasibility(lmo::SpanningTreeLMO, lb, ub, int_vars, n)

Checks whether there is a spanning tree with respect to the given upper and lower bounds.
"""
function Boscia.check_feasibility(lmo::SpanningTreeLMO, lb, ub, int_vars, n)
    n_int = length(int_vars)
    num_nodes = nv(lmo.graph)
    edges = collect(Graphs.edges(lmo.graph))
    num_included_edges = 0 #if the number of edges set to 1 exceeds nv - 1 there is a cycle inside the induced graph
    forced_include_graph = SimpleGraph(num_nodes) #The graph with only edges set to 1
    for i in 1:n_int
        if lb[i] ≈ 1
            u, v = Tuple(edges[int_vars[i]])
            add_edge!(forced_include_graph, u, v)
            num_included_edges += 1
            if num_included_edges == nv # there is a cycle inside the induced graph
                return false
            end
        end
    end
    if is_cyclic(forced_include_graph)
        return false
    end
    forced_exclude_graph = SimpleGraph(num_nodes) #The graph where edges set to 0 are removed, we then check if this graph is connected
    is_included = trues(n)
    for i in 1:n_int
        if ub[i] ≈ 0
            is_included[int_vars[i]] = false
        end
    end
    for i in 1:n
        if is_included[i]
            u, v = Tuple(edges[i])
            add_edge!(forced_exclude_graph, u, v)
        end
    end
    if is_connected(forced_exclude_graph)#no spanning trees if the graph is disconnected
        return true
    else
        return false
    end
end

"""
     is_simple_linear_feasible(lmo::SpanningTreeLMO, v)

Checks for feasibility inside the spanning tree polytope
"""
function Boscia.is_simple_linear_feasible(lmo::SpanningTreeLMO, v)
    N = length(v)
    num_nodes = nv(lmo.graph)
    iter = collect(Graphs.edges(lmo.graph))
    tol = 1e-6
    if minimum(v) < -tol #check non-negativity
        return false
    end
    if abs(sum(v) - (num_nodes - 1)) > tol #check fractional edge_count
        return false
    end
    distmx = spzeros(num_nodes, num_nodes)
    for idx in 1:N
        distmx[src(iter[idx]), dst(iter[idx])] = v[idx]
        distmx[dst(iter[idx]), src(iter[idx])] = v[idx]
    end
    _, val = mincut(lmo.graph, distmx) #check min-cut
    if val < (1 - tol)
        return false
    end
    return true
end


"""
     bounded_compute_extreme_point(lmo::SpanningTreeLMO,direction,lb,ub,int_vars;kwargs...,)

Computes a bounded mixed integer extreme point of the spanning tree polytope.
Acyclic subgraphs of a connected graph form a matroid where spanning trees are the bases
Start with edges that are fixed to 1 (lb = 1), exclude edges that are fixed to 0 (ub = 0).
Sort the edges and use a greedy algorithm to obtain a MST use Union-Find to track connected components.
"""
function Boscia.bounded_compute_extreme_point(
    lmo::SpanningTreeLMO,
    direction,
    lb,
    ub,
    int_vars;
    kwargs...,
)

    ne = length(direction)
    num_nodes = nv(lmo.graph)
    edges = collect(Graphs.edges(lmo.graph))
    n_ints = length(int_vars)
    output = zeros(ne)
    is_fixed = falses(ne) #whether an edge is set to 1 or zero
    included = UInt[] #indices of edges that are included
    num_included = 0
    weighted_edges = Tuple{Float64,UInt}[] #indices of edges that are not fixed to a certain value together with their weights
    for i in 1:n_ints #handle fixed edges
        if lb[i] ≈ 1
            is_fixed[int_vars[i]] = true
            num_included += 1
            output[int_vars[i]] = 1
            push!(included, int_vars[i])
        end
        if ub[i] ≈ 0
            is_fixed[int_vars[i]] = true
        end
    end
    for i in 1:ne #handle free edges
        if !is_fixed[i]
            push!(weighted_edges, (direction[i], i))
        end
    end
    #Initialize Union Find Structure
    uf = UnionFinder(num_nodes)
    for i in 1:num_included
        edge_index = included[i]
        u, v = Tuple(edges[edge_index])
        union!(uf, u, v)
    end
    #Sort weighted edges
    sort!(weighted_edges, by=first)
    #Iterate over edges until an MST is achieved
    track_fixed = 0
    for (_, edge_index) in weighted_edges
        num_included == (num_nodes - 1) && break
        u, v = Tuple(edges[edge_index])
        if find!(uf, u) != find!(uf, v)
            union!(uf, u, v)
            output[edge_index] = 1.0
            num_included += 1
        end
    end
    return output
end
