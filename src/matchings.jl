
"""
    MatchingLMO{G}(g::G)

Return an incidence vector v of the edges of `g` which has to be a `Graphs.AbstractGraph`, ordered as `edges(g)` for a minimum-weight matching.

Uses a reduction of the problem to minimum-weight perfect matching:
(see https://homepages.cwi.nl/~schaefer/ftp/pdf/masters-thesis.pdf section 1.5.1).
"""
struct MatchingLMO{G} <: FrankWolfe.LinearMinimizationOracle
    original_graph::G
    extended_graph::G
end

function MatchingLMO(original_graph::G) where {G}
    extended_graph = copy(original_graph)
    nvtx = nv(original_graph)
    add_vertices!(extended_graph, nvtx)
    for edge in edges(original_graph)
        add_edge!(extended_graph, src(edge) + nvtx, dst(edge) + nvtx)
    end
    for i in 1:nvtx
        add_edge!(extended_graph, i, i + nvtx)
    end
    return MatchingLMO{G}(original_graph, extended_graph)
end

function FrankWolfe.compute_extreme_point(
    lmo::MatchingLMO,
    direction::M;
    v=nothing,
    kwargs...,
) where {M}
    N = length(direction)
    if v === nothing
        v = spzeros(Int, N)
    else
        v .= 0
    end
    nvtx = nv(lmo.original_graph)
    w = Dict{edgetype(lmo.original_graph),eltype(direction)}()
    for (i, edge) in enumerate(edges(lmo.original_graph))
        w[edge] = direction[i]
        w[Edge(src(edge) + nvtx, dst(edge) + nvtx)] = direction[i]
    end

    for i in 1:nvtx
        w[Edge(i, i + nvtx)] = 0
    end

    match = GraphsMatching.minimum_weight_perfect_matching(lmo.extended_graph, w)

    K = length(match.mate)
    for (i, edge) in enumerate(edges(lmo.original_graph))
        for k in 1:K
            if match.mate[k] == src(edge) && dst(edge) == k
                v[i] = 1
            end
        end
    end
    return v
end

function Boscia.bounded_compute_extreme_point(
    lmo::MatchingLMO,
    direction,
    lb,
    ub,
    int_vars;
    kwargs...,
)
    # any entry i fixed to zero -> use a positive direction, ensuring the edge is not taken
    # any entry i fixed to one with neighbors (u, v) -> use positive direction for all other neighbors of u, of v
    corrected_direction = copy(direction)
    for (idx, edge) in enumerate(edges(lmo.original_graph))
        if ub[idx] ≈ 0
            @assert lb[idx] ≈ 0
            corrected_direction[idx] = 1
        elseif lb[idx] ≈ 1
            @assert ub[idx] ≈ 1
            (vtx1, vtx2) = Tuple(edge)
            # negative cost ensures the edge is taken, since no neighbor will be in the matching
            corrected_direction[idx] = -1
            for (idx2, e2) in enumerate(edges(lmo.original_graph))
                # check if e2 is adjacent to edge
                # we want one of the two nodes to be the same
                if xor(src(e2) in (vtx1, vtx2), dst(e2) in (vtx1, vtx2))
                    # we should not have incompatible edges fixed to one
                    @assert lb[idx2] ≈ 0 "incompatible edges $edge $e2"
                    corrected_direction[idx2] = 1
                end
            end
        end
    end
    v = FrankWolfe.compute_extreme_point(lmo, corrected_direction)
    @debug begin
        for idx in eachindex(direction)
            if ub[idx] ≈ 0
                @assert v[idx] ≈ 0
            elseif lb[idx] ≈ 1
                @assert v[idx] ≈ 1
            end
        end
    end
    return v
end

function Boscia.is_simple_linear_feasible(lmo::MatchingLMO, v)
    vertex_matching = zeros(Graphs.nv(lmo.original_graph))
    for (idx, edge) in enumerate(edges(lmo.original_graph))
        if v[idx] ≈ 0
            continue
        end
        vtx1, vtx2 = Tuple(edge)
        vertex_matching[vtx1] += v[idx]
        vertex_matching[vtx2] += v[idx]
    end
    # one vertex fractionally matched to multiple edges
    if maximum(vertex_matching) >= 1 + 1e-4
        return false
    end
    return true
end

"""
PerfectMatchingLMO{G}(g::Graphs)

Return the incidence vector of a minimum-weight perfect matching, `v` is ordered as `edges(g)`.
The constructor verifies that the graph admits a perfect matching.
"""
struct PerfectMatchingLMO{G} <: FrankWolfe.LinearMinimizationOracle
    graph::G
    function PerfectMatchingLMO(graph::G) where {G}
        @assert nv(graph) % 2 == 0
        return new{G}(graph)
    end
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
