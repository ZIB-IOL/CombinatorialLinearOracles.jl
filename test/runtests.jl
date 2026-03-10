import CombinatorialLinearOracles as CO
using Test
using Random
using SparseArrays
using LinearAlgebra
using Graphs
using GraphsMatching
using HiGHS
using StableRNGs
import FrankWolfe
import Boscia

rng = StableRNG(42)
Random.seed!(StableRNG(42), 42)

@testset "Perfect Matching LMO" begin
    N = Int(1e3)
    Random.seed!(4321)
    g = Graphs.complete_graph(N)
    iter = collect(Graphs.edges(g))
    M = length(iter)
    direction = randn(M)
    lmo = CO.PerfectMatchingLMO(g)
    v = FrankWolfe.compute_extreme_point(lmo, direction)
    tab = zeros(M)
    is_matching = true
    for i in 1:M
        if (v[i] == 1)
            is_matching = (tab[src(iter[i])] == 0 && tab[dst(iter[i])] == 0)
            if (!is_matching)
                break
            end
            tab[src(iter[i])] = 1
            tab[dst(iter[i])] = 1
        end
    end
    @test is_matching
end

@testset "Matching LMO" begin
    N = 50
    Random.seed!(9754)
    g = Graphs.complete_graph(N)
    iter = collect(Graphs.edges(g))
    M = length(iter)
    direction = randn(M)
    lmo = CO.MatchingLMO(g)
    v = FrankWolfe.compute_extreme_point(lmo, direction)
    @test Boscia.is_simple_linear_feasible(lmo, v)
    adj_mat = spzeros(M, M)
    for (i, edge) in enumerate(edges(g))
        adj_mat[src(edge), dst(edge)] = direction[i]
    end
    match_result = GraphsMatching.maximum_weight_matching(g, HiGHS.Optimizer, -adj_mat)
    v_sol = spzeros(M)
    K = length(match_result.mate)
    for k in 1:K
        for (i, edge) in enumerate(edges(g))
            if (match_result.mate[k] == src(edge) && dst(edge) == k)
                v_sol[i] = 1
            end
        end
    end
    @test v_sol == v
    v2 = FrankWolfe.compute_extreme_point(lmo, ones(M))
    @test norm(v2) == 0
    @test v == Boscia.bounded_compute_extreme_point(lmo, direction, zeros(M), ones(M), 1:M)
    @testset "Fix one entry to zero" begin
        for one_idx in SparseArrays.nonzeroinds(v)
            # upperbound one everywhere except one_idx fixed to zero
            v_fixed1 = Boscia.bounded_compute_extreme_point(
                lmo,
                direction,
                zeros(M),
                (1:M) .!= one_idx,
                1:M,
            )
            @test v_fixed1[one_idx] == 0
            @test Boscia.is_simple_linear_feasible(lmo, v_fixed1)
        end
    end
    @testset "Fix a single entry to one" begin
        for idx in rand(1:M, 100)
            # skip if entry already at one
            if v[idx] == 1
                continue
            end
            lb = (1:M) .== idx
            ub = ones(M)
            v_fixed2 = Boscia.bounded_compute_extreme_point(lmo, direction, lb, ub, 1:M)
            @test v_fixed2[idx] == 1
            @test Boscia.is_simple_linear_feasible(lmo, v_fixed2)
        end
    end
    @testset "Fix two entries to one" begin
        for (i1, e1) in enumerate(edges(g))
            for (i2, e2) in enumerate(edges(g))
                # lighter on computation
                if i1 ÷ 2 + i2 ÷ 2 > 0
                    continue
                end
                # non-adjacent edges
                if isempty(intersect(Tuple(e1), Tuple(e2)))
                    lb = zeros(M)
                    ub = ones(M)
                    lb[i1] = lb[i2] = 1
                    v_fixed3 = Boscia.bounded_compute_extreme_point(lmo, direction, lb, ub, 1:M)
                    @test v_fixed3[i1] == 1
                    @test v_fixed3[i2] == 1
                    @test Boscia.is_simple_linear_feasible(lmo, v_fixed3)
                end
            end
        end
    end
    # non-matching vector
    v_wrong = 1.0 * copy(v)
    idx1 = findfirst(==(Edge(1, 2)), collect(edges(g)))
    idx2 = findfirst(==(Edge(1, 3)), collect(edges(g)))
    v_wrong[idx1] = 0.75
    v_wrong[idx2] = 0.5
    @test !Boscia.is_simple_linear_feasible(lmo, v_wrong)
end


@testset "SpanningTreeLMO" begin
    N = 500
    Random.seed!(1645)
    g = Graphs.complete_graph(N)
    lmo = CO.SpanningTreeLMO(g)
    iter = collect(Graphs.edges(g))
    M = length(iter)
    @testset "Basic tree properties" begin
        direction = randn(M) .- 100
        v = FrankWolfe.compute_extreme_point(lmo, direction)
        tree = Vector{eltype(iter)}()
        for i in 1:M
            if (v[i] == 1)
                push!(tree, iter[i])
            end
        end
        @test Graphs.is_tree(SimpleGraphFromIterator(tree))
    end
    @testset "Test correctness for negative direction" begin
        direction = collect(-(1:M))
        v = FrankWolfe.compute_extreme_point(lmo, direction)
        @test dot(v, direction) < -4e-7
    end
end

@testset "Shortest path" begin
    n = 200
    Random.seed!(42)
    for _ in 1:10
        g = Graphs.random_orientation_dag(Graphs.complete_graph(n))
        src_node = 1
        dst_node = nv(g)
        while !has_path(g, src_node, dst_node) || src_node == dst_node
            src_node = rand(1:nv(g))
            dst_node = rand(1:nv(g))
        end
        lmo = CO.ShortestPathLMO(g, src_node, dst_node)
        for _ in 1:10
            direction = randn(ne(g))
            v = FrankWolfe.compute_extreme_point(lmo, direction)
            @test sum(v) >= 1
        end
    end
    @testset "Shortest path with negative costs" begin
        g = SimpleDiGraph(4)
        add_edge!(g, 1, 2)
        add_edge!(g, 1, 4)
        add_edge!(g, 2, 3)
        add_edge!(g, 3, 4)
        mat = adjacency_matrix(g)
        mat[3, 4] = -10
        lmo = CO.ShortestPathLMO(g, 1, 4)
        costs = ones(ne(g))
        idx = findfirst(==(Edge(3, 4)), collect(edges(g)))
        costs[idx] = -10
        v = FrankWolfe.compute_extreme_point(lmo, costs)
        @test sum(v) == 3
    end
end

@testset "Birkhoff Polytope" begin
    @testset "Continuous Problem" begin
        n = 4
        d = randn(rng, n, n)
        lmo = CO.BirkhoffLMO(n)
        x = ones(n, n) ./ n
        # test without fixings
        v_if = CO.compute_inface_extreme_point(lmo, d, x)
        v_fw = CO.compute_extreme_point(lmo, d)
        @test norm(v_fw - v_if) ≤ n * eps()
        fixed_col = 2
        fixed_row = 3
        # fix one transition and renormalize
        x2 = copy(x)
        x2[:, fixed_col] .= 0
        x2[fixed_row, :] .= 0
        x2[fixed_row, fixed_col] = 1
        x2 = x2 ./ sum(x2, dims=1)
        v_fixed = CO.compute_inface_extreme_point(lmo, d, x2)
        @test v_fixed[fixed_row, fixed_col] == 1
        # If matrix is already a vertex, away-step can give only itself
        @test norm(CO.compute_inface_extreme_point(lmo, d, v_fixed) - v_fixed) ≤ eps()
        # fixed a zero only
        x3 = copy(x)
        x3[4, 3] = 0
        # fixing zeros by creating a cycle 4->3->1->4->4
        x3[4, 4] += 1 / n
        x3[1, 4] -= 1 / n
        x3[1, 3] += 1 / n
        v_zero = CO.compute_inface_extreme_point(lmo, d, x3)
        @test v_zero[4, 3] == 0
        @test v_zero[1, 4] == 0
    end

    @testset "Integer Problem" begin
        n = 6
        # Create linear index in the matrix
        lin(i, j, n) = (j - 1) * n + i

        # 1) No fixings: FW vs in-face should match on a mixed-int oracle
        d = randn(rng, n, n)
        int_vars = [lin(1, 1, n), lin(2, 2, n), lin(3, 3, n), lin(4, 4, n)]  # only diagonals integer
        lmo = CO.BirkhoffLMO(n, int_vars)
        x = ones(n, n) ./ n
        v_if = CO.compute_inface_extreme_point(lmo, d, x)
        v_fw = CO.compute_extreme_point(lmo, d)
        @test norm(v_fw - v_if) ≤ n * eps()

        # 2) Fix one integer variable to 1.0 -> that (row,col) must be selected
        d = randn(rng, n, n)
        int_vars = [lin(2, 3, n), lin(3, 1, n), lin(1, 4, n)]  # a few scattered ints
        blmo = CO.BirkhoffLMO(n, int_vars)
        # choose c_idx within int_vars and force it to 1
        c_idx = 1
        fixed_var = int_vars[c_idx]
        j = ceil(Int, fixed_var / n)
        i = Int(fixed_var - n * (j - 1))
        # Update blmo after fixing
        Boscia.set_bound!(blmo, c_idx, 1.0, :greaterthan)
        Boscia.delete_bounds!(blmo, [])
        v_fw = CO.compute_extreme_point(blmo, d)
        @test v_fw[i, j] == 1.0
        # in-face should respect that fixing too
        x = ones(n, n) ./ n
        v_if = CO.compute_inface_extreme_point(blmo, d, x)
        @test v_if[i, j] == 1.0

        # 3) Interdiction: set upper bound 0 on an integer var that the assignment would otherwise pick.
        #    We construct d to strongly prefer that edge; the bound must force avoidance.
        d = zeros(n, n)
        hot_i, hot_j = 3, 2
        d[hot_i, hot_j] = -100.0  # make this edge extremely attractive
        int_vars = [lin(hot_i, hot_j, n), lin(1, 1, n), lin(2, 4, n)]
        blmo_block = CO.BirkhoffLMO(n, int_vars)
        c_idx_block = findfirst(==(lin(hot_i, hot_j, n)), int_vars)
        # Update the lmo_block
        Boscia.set_bound!(blmo_block, c_idx_block, 0.0, :lessthan)
        Boscia.delete_bounds!(blmo_block, [])

        v_block = CO.compute_extreme_point(blmo_block, d)
        @test v_block[hot_i, hot_j] == 0.0

        # sanity check: without the bound, the edge should be chosen
        blmo_free = CO.BirkhoffLMO(n, int_vars)
        v_free = CO.compute_extreme_point(blmo_free, d)
        @test v_free[hot_i, hot_j] == 1.0

        x3 = ones(n, n) ./ n
        v_if = CO.compute_inface_extreme_point(blmo_block, d, x3)
        @test v_if[hot_i, hot_j] == 0

        # 4) Vector interface consistency (matrix vs vector forms) on mixed-int oracle
        d = randn(rng, n, n)
        int_vars = [lin(1, 2, n), lin(2, 1, n), lin(3, 4, n)]
        blmo = CO.BirkhoffLMO(n, int_vars)
        # matrix form
        v_M = CO.compute_extreme_point(blmo, d)
        # vector form
        d_vec = vec(d)
        v_vec = CO.compute_extreme_point(blmo, d_vec)
        # turn vector result back to matrix to compare
        v_from_vec = reshape(v_vec, n, n)
        @test norm(v_M - v_from_vec) ≤ n * eps()

        # in-face vector vs matrix
        x = ones(n, n) ./ n
        v_M_if = CO.compute_inface_extreme_point(blmo, d, x)
        v_vec_if = CO.compute_inface_extreme_point(blmo, d_vec, vec(x))
        v_if_from_vec = reshape(v_vec_if, n, n)
        @test norm(v_M_if - v_if_from_vec) ≤ n * eps()

        # 5) feasibility test
        int_vars = [lin(1, 1, n), lin(2, 2, n)]
        blmo = CO.BirkhoffLMO(n, int_vars)
        @test Boscia.check_feasibility(blmo) == Boscia.OPTIMAL

        int_vars = [lin(1, 1, n)]
        blmo = CO.BirkhoffLMO(n, int_vars)
        Boscia.set_bound!(blmo, 1, 1.0, :greaterthan)  # lb = 1
        Boscia.set_bound!(blmo, 1, 0.0, :lessthan)    # ub = 0
        @test Boscia.check_feasibility(blmo) == Boscia.INFEASIBLE

        int_vars = [lin(2, 1, n), lin(2, 3, n)]
        blmo = CO.BirkhoffLMO(n, int_vars)
        Boscia.set_bound!(blmo, 1, 0.6, :greaterthan) # (2,1) lb = 0.6
        Boscia.set_bound!(blmo, 2, 0.6, :greaterthan) # (2,3) lb = 0.6
        @test Boscia.check_feasibility(blmo) == Boscia.INFEASIBLE

        int_vars = [lin(1, 2, n), lin(3, 2, n)]
        blmo = CO.BirkhoffLMO(n, int_vars)
        Boscia.set_bound!(blmo, 1, 0.7, :greaterthan) # (1,2) lb = 0.7
        Boscia.set_bound!(blmo, 2, 0.4, :greaterthan) # (3,2) lb = 0.4
        @test Boscia.check_feasibility(blmo) == Boscia.INFEASIBLE

        # 6) standard run test for Boscia
        n = 6
        p = randperm(rng, n)
        X_star = zeros(n, n)
        @inbounds for i in 1:n
            X_star[i, p[i]] = 1.0
        end
        x_star = vec(X_star)  # matches append_by_column = true

        # Quadratic objective: 0.5 * ||x - x_star||^2
        function f(x)
            return 0.5 * sum((x[i] - x_star[i])^2 for i in eachindex(x))
        end
        function grad!(storage, x)
            @. storage = x - x_star
        end

        blmo = CO.BirkhoffLMO(n, collect(1:n^2))
        x, _, result = Boscia.solve(f, grad!, blmo)
        X = reshape(x, n, n)

        @test sum(isapprox.(X, X_star, atol=1e-6, rtol=1e-2)) == n^2
        @test isapprox(f(x), f(result[:raw_solution]), atol=1e-6, rtol=1e-3)

        # DICG variant
        settings = Boscia.create_default_settings()
        settings.frank_wolfe[:variant] = Boscia.DecompositionInvariantConditionalGradient()

        x_dicg, _, result_dicg = Boscia.solve(f, grad!, blmo, settings=settings)
        X_dicg = reshape(x_dicg, n, n)

        @test sum(isapprox.(X_dicg, X_star, atol=1e-6, rtol=1e-2)) == n^2
        @test isapprox(f(x_dicg), f(result_dicg[:raw_solution]), atol=1e-6, rtol=1e-3)
    end
end
