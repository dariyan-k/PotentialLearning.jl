# Run this script:
#   $ julia --project=./ --threads=4
#   julia> include("fit-cnn-ace.jl")

using AtomsBase
using Unitful, UnitfulAtomic
using InteratomicPotentials 
using InteratomicBasisPotentials
using PotentialLearning
using LinearAlgebra 
using Flux
using Flux: flatten
using Optimization
using OptimizationOptimJL
using Random
include("utils/utils.jl")


# Load input parameters
args = ["experiment_path",      "a-Hfo2-300K-NVT-6000-NeuralACE/",
        "dataset_path",         "data/",
        "dataset_filename",     "a-Hfo2-300K-NVT-6000.extxyz",
        "energy_units",         "eV",
        "distance_units",       "Å",
        "random_seed",          "100",
        "n_train_sys",          "100",
        "n_test_sys",           "100",
        "n_red_desc",           "0", # No. of reduced descriptors. O: don't apply reduction
        "nn",                   "Chain(Dense(n_desc,8,relu),Dense(8,1))",
        "n_epochs",             "30",
        "n_batches",            "1",
        "optimiser",            "Adam(0.001)", # e.g. Adam(0.01) or BFGS()
        "n_body",               "3",
        "max_deg",              "3",
        "r0",                   "1.0",
        "rcutoff",              "5.0",
        "wL",                   "1.0",
        "csp",                  "1.0",
        "w_e",                  "0.01",
        "w_f",                  "1.0"]
args = length(ARGS) > 0 ? ARGS : args
input = get_input(args)


# Create experiment folder
path = input["experiment_path"]
run(`mkdir -p $path`)
@savecsv path input

# Fix random seed
if "random_seed" in keys(input)
    Random.seed!(input["random_seed"])
end

# Load dataset
ds_path = input["dataset_path"]*input["dataset_filename"] # dirname(@__DIR__)*"/data/"*input["dataset_filename"]
energy_units, distance_units = uparse(input["energy_units"]), uparse(input["distance_units"])
ds = load_data(ds_path, energy_units, distance_units)

# Split dataset
n_train, n_test = input["n_train_sys"], input["n_test_sys"]
conf_train, conf_test = split(ds, n_train, n_test)

# Start measuring learning time
learn_time = @elapsed begin #learn_time = 0.0

# Define ACE parameters
ace = ACE(species = unique(atomic_symbol(get_system(ds[1]))),
          body_order = input["n_body"],
          polynomial_degree = input["max_deg"],
          wL = input["wL"],
          csp = input["csp"],
          r0 = input["r0"],
          rcutoff = input["rcutoff"])
@savevar path ace

# Update training dataset by adding energy and force descriptors
println("Computing energy descriptors of training dataset...")
B_time = @elapsed e_descr_train = compute_local_descriptors(conf_train, ace, T = Float32)
println("Computing force descriptors of training dataset...")
dB_time = @elapsed f_descr_train = compute_force_descriptors(conf_train, ace, T = Float32)
GC.gc()
ds_train = DataSet(conf_train .+ e_descr_train .+ f_descr_train)
n_desc = length(e_descr_train[1][1])

# Dimension reduction of energy and force descriptors of training dataset
reduce_descriptors = input["n_red_desc"] > 0    
if reduce_descriptors
    n_desc = input["n_red_desc"]
    pca = PCAState(tol = n_desc)
    fit!(ds_train, pca)
    transform!(ds_train, pca)
end

# Define neural network model
nn = eval(Meta.parse(input["nn"])) # e.g. Chain(Dense(n_desc,8,Flux.leakyrelu), Dense(8,1))
nace = NNIAP(nn, ace)

# Learn
println("Learning energies and forces...")
w_e, w_f = input["w_e"], input["w_f"]
opt = eval(Meta.parse(input["optimiser"]))
n_epochs = input["n_epochs"]
#learn!(nace, ds_train, opt, n_epochs, loss, w_e, w_f)



# nn_conv = Chain(
#         Conv((3,), 1=>1, relu),
#         MaxPool((2,)), 
#         Conv((5,), 8=>16, relu),
#         MaxPool((2,)),
#         flatten,
#         Dense(64, 120, relu),
#         Dense(120, 10, relu),
#         Dense(10, 1)
#         )   


# include("utils/utils.jl")
# nn_conv = Chain(
#     Conv((3,), 1=>12, relu),
#     MaxPool((2,)),
#     # x -> reshape(x, :, :, 1)
#     Conv((2,), 12=>4, relu, pad=1),


#     )

# function LeNet5(; imgsize=(28,28,1), nclasses=10) 
#     out_conv_size = (imgsize[1]÷4 - 3, imgsize[2]÷4 - 3, 16)
    
#     return Chain(
#             Conv((5, 5), imgsize[end]=>6, relu),
#             MaxPool((2, 2)),
#             Conv((5, 5), 6=>16, relu),
#             MaxPool((2, 2)),
#             flatten,
#             Dense(prod(out_conv_size), 120, relu), 
#             Dense(120, 84, relu), 
#             Dense(84, nclasses)
#           )
# end

# include("utils/utils.jl")
# nn_conv = Chain(
#     Conv((3,), 1=>12, relu),
#     MaxPool((2,)),
#     flatten,
#     # x -> reshape(x, :, :, 1)
#     #Conv((2,), 12=>4, relu, pad=1),
#     #MaxPool((2,)),
#     #flatten,
#     Dense(144,1)
#     )

# include("utils/utils.jl")

# nn_conv = Chain(
#     Conv((2,2), 1=>7, relu, pad=1)#,
#     MaxPool((2, 2)),
#     )

# nn_conv = Chain(
#     Conv((2,2), 1=>1, relu, pad=1),
#     MaxPool((2, 2)),
#     Conv((2,2), 1=>2, relu, pad=1),
#     MaxPool((2, 2)),
#     flatten,
#     Dense(14,1)

# )

include("utils/utils.jl")
# nn_conv = Chain(
#     Conv((3,3), 1=>1, relu, pad=1),
#     Conv((3,3), 1=>1, relu, pad=1),
#     Dense(26,1)
# )

# nn_conv = Chain(
#     Conv((2,2), 1=>3, relu, pad=1),
#     MaxPool((2, 2)),
#     Conv((2,2), 3=>1, relu, pad=2),
#     x -> reshape(x, 64, 1,1,1),
#     Dense(64,1)
# )

nn_conv = Chain(
    Conv((2,2), 1=>2, leakyrelu, pad=1),
    Conv((2,2), 2=>1, leakyrelu, pad=1),
    Conv((5,5), 1=>1, leakyrelu, pad=1),
    Dense(26, 1)
)
       

nace_conv = NNIAP(nn_conv, ace)
learn!(nace_conv, ds_train, opt, n_epochs, loss, w_e, w_f)

@assert 0 == 2

end # end of "learn_time = @elapsed begin"

@savevar path Flux.params(nace_conv.nn)

# Post-process output: calculate metrics, create plots, and save results

# Update test dataset by adding energy and force descriptors
println("Computing energy descriptors of test dataset...")
e_descr_test = compute_local_descriptors(conf_test, ace, T = Float32)
println("Computing force descriptors of test dataset...")
f_descr_test = compute_force_descriptors(conf_test, ace, T = Float32)
GC.gc()
ds_test = DataSet(conf_test .+ e_descr_test .+ f_descr_test)

# Dimension reduction of energy and force descriptors of test dataset
if reduce_descriptors
    transform!(ds_test, pca)
end

# Get true and predicted values
e_train, f_train = get_all_energies(ds_train), get_all_forces(ds_train)
e_test, f_test = get_all_energies(ds_test), get_all_forces(ds_test)
e_train_pred, f_train_pred = get_all_energies(ds_train, nace_conv), get_all_forces(ds_train, nace_conv)
e_test_pred, f_test_pred = get_all_energies(ds_test, nace_conv), get_all_forces(ds_test, nace_conv)
@savevar path e_train
@savevar path e_train_pred
@savevar path f_train
@savevar path f_train_pred
@savevar path e_test
@savevar path e_test_pred
@savevar path f_test
@savevar path f_test_pred

# Compute metrics
metrics = get_metrics( e_train_pred, e_train, f_train_pred, f_train,
                       e_test_pred, e_test, f_test_pred, f_test,
                       B_time, dB_time, missing) #learn_time)
@savecsv path metrics

# Plot and save results
e_test_plot = plot_energy(e_test_pred, e_test)
@savefig path e_test_plot
f_test_plot = plot_forces(f_test_pred, f_test)
@savefig path f_test_plot
f_test_cos = plot_cos(f_test_pred, f_test)
@savefig path f_test_cos

# Adam(0.001)
# learn!(nace_conv_2, ds_train, Adam(0.001), 30, loss, w_e, w_f)