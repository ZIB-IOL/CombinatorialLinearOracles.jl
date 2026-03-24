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
Union-find find with path compression.
Used to detect cycles and connectivity in fixing checks.
"""
function uf_find!(parent::Vector{Int}, x::Int)
    while parent[x] != x
        parent[x] = parent[parent[x]]
        x = parent[x]
    end
    return x
end

"""
Union-find union.
Returns `false` when `a` and `b` are already connected (cycle).
"""
function uf_union!(parent::Vector{Int}, a::Int, b::Int)
    ra = uf_find!(parent, a)
    rb = uf_find!(parent, b)
    if ra == rb
        return false
    end
    parent[rb] = ra
    return true
end

"""
Minimum spanning tree LMO using Kruskal on the weighted graph.
Returns an incidence vector over `edges(g)`.
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
Bound-aware LMO for spanning trees.
Contracts forced edges, removes forbidden edges, and runs Kruskal
on the reduced graph, then lifts the solution to the original graph.
"""
function Boscia.bounded_compute_extreme_point(
    lmo::SpanningTreeLMO,
    direction,
    lb,
    ub,
    int_vars;
    kwargs...,
)
    N = length(direction)
    edges_iter = collect(Graphs.edges(lmo.graph))
    @assert length(edges_iter) == N

    # Contract all forced edges into components via union-find.
    # Connected nodes will have the same parent node at the end.
    parent = collect(1:Graphs.nv(lmo.graph))
    for (i, edge) in enumerate(edges_iter)
        if lb[i] ≈ 1
            @assert ub[i] ≈ 1
            uf_union!(parent, src(edge), dst(edge))
        end
    end
    # Count the number of connected components
    # which will be contracted to super nodes.
    comp_id = Dict{Int,Int}()
    comp = Vector{Int}(undef, Graphs.nv(lmo.graph))
    k = 0
    for vtx in 1:Graphs.nv(lmo.graph)
        root = uf_find!(parent, vtx)
        if !haskey(comp_id, root)
            k += 1
            comp_id[root] = k
        end
        comp[vtx] = comp_id[root]
    end

    # Initialize solution with all forced edges.
    v = spzeros(N)
    for i in 1:N
        if lb[i] ≈ 1
            v[i] = 1
        end
    end

    # Build reduced graph on components using the cheapest allowed
    # edge for each component pair. (Note that after contracted the graph
    # we might have multiple edges between the same super nodes.) 
    # Then run Kruskal on the reduced graph.
    if k > 1
        edge_choice = Dict{Tuple{Int,Int},Tuple{eltype(direction),Int}}()
        for (i, edge) in enumerate(edges_iter)
            # Edge is forbidden.
            if ub[i] ≈ 0
                continue
            end
            c1 = comp[src(edge)]
            c2 = comp[dst(edge)]
            # source and destination nodes lie in the same component,
            # so the edge can be ignored.
            if c1 == c2
                continue
            end
            # order independent key
            if c1 > c2
                c1, c2 = c2, c1
            end
            key = (c1, c2)
            w = direction[i]
            # If multiple edges connect the same super nodes
            # choose the cheapest one.
            if !haskey(edge_choice, key) || w < edge_choice[key][1]
                edge_choice[key] = (w, i)
            end
        end
        # Build reduced graph and weight matrix.
        reduced_graph = SimpleGraph(k)
        for key in keys(edge_choice)
            add_edge!(reduced_graph, key[1], key[2])
        end
        distmx = spzeros(k, k)
        for (key, (w, _)) in edge_choice
            distmx[key[1], key[2]] = w
            distmx[key[2], key[1]] = w
        end
        reduced_span = Graphs.kruskal_mst(reduced_graph, distmx)
        for edge in reduced_span
            c1 = src(edge)
            c2 = dst(edge)
            # order independent key
            if c1 > c2
                c1, c2 = c2, c1
            end
            idx = edge_choice[(c1, c2)][2]
            v[idx] = 1
        end
    end
    # Optional sanity check.
    @debug begin
        for i in 1:N
            if ub[i] ≈ 0
                @assert v[i] ≈ 0
            elseif lb[i] ≈ 1
                @assert v[i] ≈ 1
            end
        end
    end
    return v
end

"""
Lightweight feasibility check for a candidate `v`.
Enforces total edge count and singleton-cut constraints.
"""
function Boscia.is_simple_linear_feasible(lmo::SpanningTreeLMO, v)
    n = Graphs.nv(lmo.graph)
    if n == 0
        return true
    end
    total = sum(v)
    if abs(total - (n - 1)) > 1e-4
        return false
    end
    # detect cycles among edges that are (almost) fully selected
    parent = collect(1:n)
    for (idx, edge) in enumerate(edges(lmo.graph))
        if v[idx] < 1 - 1e-4
            continue
        end
        if !(uf_union!(parent, src(edge), dst(edge)))
            return false
        end
    end
    # singleton cut constraints: each vertex must have at least one incident edge
    degrees = zeros(eltype(v), n)
    for (idx, edge) in enumerate(edges(lmo.graph))
        if v[idx] ≈ 0
            continue
        end
        degrees[src(edge)] += v[idx]
        degrees[dst(edge)] += v[idx]
    end
    if minimum(degrees) < 1 - 1e-4
        return false
    end
    # ensure support is connected (prevents disjoint forests passing)
    parent = collect(1:n)
    for (idx, edge) in enumerate(edges(lmo.graph))
        if v[idx] <= 1e-4
            continue
        end
        uf_union!(parent, src(edge), dst(edge))
    end
    # All nodes should have a common root if the graph is connected.
    root = uf_find!(parent, 1)
    for vtx in 2:n
        if uf_find!(parent, vtx) != root
            return false
        end
    end
    return true
end

"""
Feasibility of bounds alone for spanning trees.
Returns `Boscia.OPTIMAL` if some spanning tree can satisfy the bounds.
"""
function Boscia.check_feasibility(
    lmo::SpanningTreeLMO,
    lb,
    ub,
    int_vars,
    n
)
    edges_iter = collect(Graphs.edges(lmo.graph))
    if n <= 1
        return Boscia.OPTIMAL
    end
    # The forced edges (lb=ub=1) must be acyclic.
    parent = collect(1:n)
    for (i, edge) in enumerate(edges_iter)
        if lb[i] ≈ 1
            if !uf_union!(parent, src(edge), dst(edge))
                @debug "Forced edges form a cycle"
                return Boscia.INFEASIBLE
            end
        end
    end
    # The graph must stay connected after removing forbidden edges.
    parent = collect(1:n)
    for (i, edge) in enumerate(edges_iter)
        if !(ub[i] ≈ 0)
            uf_union!(parent, src(edge), dst(edge))
        end
    end
    root = uf_find!(parent, 1)
    for vtx in 2:n
        if uf_find!(parent, vtx) != root
            @debug "Forbidden edges disconnect graph"
            return Boscia.INFEASIBLE
        end
    end
    return Boscia.OPTIMAL
end
