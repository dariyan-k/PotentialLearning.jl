using AtomsBase
using Unitful, UnitfulAtomic
using InteratomicPotentials 
using InteratomicBasisPotentials
using PotentialLearning
using LinearAlgebra
using Flux
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
        "n_train_sys",          "200",
        "n_test_sys",           "1800",
        "n_desc",               "26", # reduce descriptor dimension
        "nn",                   "Chain(Dense(n_desc,8,Flux.relu),Dense(8,1))",
        "n_epochs",             "10000",
        "n_batches",            "1",
        "optimiser",            "Adam(0.1)", # e.g. Adam(0.01) or BFGS()
        "n_body",               "3",
        "max_deg",              "3",
        "r0",                   "1.0",
        "rcutoff",              "5.0",
        "wL",                   "1.0",
        "csp",                  "1.0",
        "w_e",                  "1.0",
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
B_time = @elapsed e_descr_train = compute_local_descriptors(conf_train, ace)
println("Computing force descriptors of training dataset...")
dB_time = @elapsed f_descr_train = compute_force_descriptors(conf_train, ace)
GC.gc()
ds_train = DataSet(conf_train .+ e_descr_train .+ f_descr_train)

# Dimension reduction of energy and force descriptors of training dataset
reduce_descriptors = input["n_desc"] > 0
if reduce_descriptors 
    λ_l, W_l, m_l, lll, λ_f, W_f, m_f, fff = get_dim_red_pars(ds_train, input["n_desc"])
    e_descr_train, f_descr_train = reduce_desc(λ_l, W_l, m_l, lll, λ_f, W_f, m_f, fff)
end
n_desc = length(e_descr_train[1][1])
ds_train = DataSet(conf_train .+ e_descr_train .+ f_descr_train)

# Define neural network model
nn = eval(Meta.parse(input["nn"])) # e.g. Chain(Dense(n_desc,8,Flux.leakyrelu), Dense(8,1))
nace = NNIAP(nn, ace)

# Learn
println("Learning energies and forces...")
w_e, w_f = input["w_e"], input["w_f"]
opt = eval(Meta.parse(input["optimiser"]))
n_epochs = input["n_epochs"]
learn!(nace, ds_train, opt, n_epochs, loss, w_e, w_f)

end # end of "learn_time = @elapsed begin"

@savevar path Flux.params(nace.nn)

# Post-process output: calculate metrics, create plots, and save results

# Update test dataset by adding energy and force descriptors
println("Computing energy descriptors of test dataset...")
e_descr_test = compute_local_descriptors(conf_test, ace)
println("Computing force descriptors of test dataset...")
f_descr_test = compute_force_descriptors(conf_test, ace)
GC.gc()
ds_test = DataSet(conf_test .+ e_descr_test .+ f_descr_test)

# Dimension reduction of energy and force descriptors of test dataset
if reduce_descriptors
    lll = get_values.(get_local_descriptors.(ds_test))
    fff = get_values.(get_force_descriptors.(ds_test))
    e_descr_test, f_descr_test = reduce_desc(λ_l, W_l, m_l, lll, λ_f, W_f, m_f, fff)
end
ds_test = DataSet(conf_test .+ e_descr_test .+ f_descr_test)


# Get true and predicted values
e_train, f_train = get_all_energies(ds_train), get_all_forces(ds_train)
e_test, f_test = get_all_energies(ds_test), get_all_forces(ds_test)
e_train_pred, f_train_pred = get_all_energies(ds_train, nace), get_all_forces(ds_train, nace)
e_test_pred, f_test_pred = get_all_energies(ds_test, nace), get_all_forces(ds_test, nace)

@savevar path e_train
@savevar path e_train_pred
@savevar path f_train
@savevar path f_train_pred
@savevar path e_test
@savevar path e_test_pred
@savevar path f_test
@savevar path f_test_pred

metrics = get_metrics( e_train_pred, e_train, f_train_pred, f_train,
                       e_test_pred, e_test, f_test_pred, f_test,
                       B_time, dB_time, learn_time)
@savecsv path metrics

e_test_plot = plot_energy(e_test_pred, e_test)
@savefig path e_test_plot

f_test_plot = plot_forces(f_test_pred, f_test)
@savefig path f_test_plot

f_test_cos = plot_cos(f_test_pred, f_test)
@savefig path f_test_cos

