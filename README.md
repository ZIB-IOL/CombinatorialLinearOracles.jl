# CombinatorialLinearOracles

[![Build Status](https://github.com/ZIB-IOL/CombinatorialLinearOracles.jl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ZIB-IOL/CombinatorialLinearOracles.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ZIB-IOL/CombinatorialLinearOracles.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ZIB-IOL/CombinatorialLinearOracles.jl)

This package implements linear minimization oracles which compute a minimizer of a linear function over a compact convex set.

CombinatorialLinearOracles is primarily a companion of [FrankWolfe.jl](https://github.com/ZIB-IOL/FrankWolfe.jl/) and implements several combinatorial linear minimization oracles, for instance for minimizing a linear function over a polytope defined by objects on graphs (spanning trees, matchings, ...).

CombinatorialLinearOracles also implements bounded linear minimization oracles (BLMO) for usage in the branch-and-bound of [Boscia.jl](https://github.com/ZIB-IOL/Boscia.jl).

## Installation

```julia
import Pkg
Pkg.add("https://github.com/ZIB-IOL/CombinatorialLinearOracles.jl")

import CombinatorialLinearOracles
```

## Usage

```julia
import FrankWolfe
import CombinatorialLinearOracles as CLO
using Graphs

g = complete_graph(5)
lmo = CLO.MatchingLMO(g)

direction = randn(Graphs.ne(g))
opt_matching = FrankWolfe.compute_extreme_point(lmo, direction)
```
