module CombinatorialLinearOracles

import FrankWolfe
using Graphs
using SparseArrays
using GraphsMatching
using Hungarian
using Boscia

include("matchings.jl")
include("spanning_tree.jl")
include("shortest_path.jl")
include("birkhoff_polytope.jl")

end
