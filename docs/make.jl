using CombinatorialLinearOracles
using Documenter

DocMeta.setdocmeta!(CombinatorialLinearOracles, :DocTestSetup, :(using CombinatorialLinearOracles); recursive=true)

makedocs(;
    modules=[CombinatorialLinearOracles],
    authors="Mathieu Besançon <mathieu.besancon@gmail.com> and contributors",
    repo="https://github.com/ZIB-IOL/CombinatorialLinearOracles.jl/blob/{commit}{path}#{line}",
    sitename="CombinatorialLinearOracles.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ZIB-IOL.github.io/CombinatorialLinearOracles.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ZIB-IOL/CombinatorialLinearOracles.jl",
    devbranch="main",
)
