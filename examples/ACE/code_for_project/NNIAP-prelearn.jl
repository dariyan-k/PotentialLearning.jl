using Flux
using Optim

# Neural network interatomic potential
mutable struct NNIAP
    nn
    iap
end

# Neural network potential formulation using local descriptors to compute energy and forces
# See 10.1103/PhysRevLett.98.146401
#     https://fitsnap.github.io/Pytorch.html

function potential_energy(c::Configuration, nniap::NNIAP)
    Bs = get_values(get_local_descriptors(c))
    return sum([sum(nniap.nn(B_atom)) for B_atom in Bs])
end


# function potential_energy_global(c::Configuration, nniap::NNIAP)
#     B = sum(get_values(get_local_descriptors(c)))
#     return first(sum(nniap.nn(B)))
# end


# function potential_energy_local(c::Configuration, nniap::NNIAP)
#     Bs = get_values(get_local_descriptors(c))
#     return first([sum(nniap.nn(Bi)) for Bi in Bs])
# end



function force(c::Configuration, nniap::NNIAP)
    Bs = get_values(get_local_descriptors(c))
    dnndb = [first(gradient(x->sum(nniap.nn(x)), B_atom)) for B_atom in Bs]
    dbdr = get_values(get_force_descriptors(c))
    return [[-sum(dnndb .⋅ [dbdr[atom][coor]]) for coor in 1:3]
             for atom in 1:length(dbdr)]
end

# Neural network potential formulation using global descriptors to compute energy and forces
# See https://docs.google.com/presentation/d/1XI9zqF_nmSlHgDeFJqq2dxdVqiiWa4Z-WJKvj0TctEk/edit#slide=id.g169df3c161f_63_123

#function potential_energy(c::Configuration, nniap::NNIAP)
#    Bs = sum(get_values(get_local_descriptors(c)))
#    return sum(nniap.nn(Bs))
#end

#function force(c::Configuration, nniap::NNIAP)
#    B = sum(get_values(get_local_descriptors(c)))
#    dnndb = first(gradient(x->sum(nniap.nn(x)), B)) 
#    dbdr = get_values(get_force_descriptors(c))
#    return [[-dnndb ⋅ dbdr[atom][coor] for coor in 1:3]
#             for atom in 1:length(dbdr)]
#end


# Loss function ################################################################

# function loss(nn, iap, ds, w_e = 1, w_f = 1)
#     nniap = NNIAP(nn, iap)
#     es = get_all_energies(ds)
#     es_pred = get_all_energies(ds, nniap)
#     fs = get_all_forces(ds)
#     fs_pred = get_all_forces(ds, nniap)
#     return w_e * Flux.mse(es_pred, es) + w_f * Flux.mse(fs_pred, fs)
# end


# Auxiliary functions ##########################################################
function get_all_energies(ds::DataSet, nniap::NNIAP)
    return [potential_energy(ds[c], nniap) for c in 1:length(ds)]
end

# function get_all_energies_local(ds::DataSet, nniap::NNIAP)
#     return [potential_energy_local(ds[c], nniap) for c in 1:length(ds)]
# end

# function get_all_energies_global(ds::DataSet, nniap::NNIAP)
#     return [potential_energy_global(ds[c], nniap) for c in 1:length(ds)]
# end

function get_all_forces(ds::DataSet, nniap::NNIAP)
    return reduce(vcat,reduce(vcat,[force(ds[c], nniap)
                                    for c in 1:length(ds)]))
end

# NNIAP learning functions #####################################################

# Flux.jl training
# function learn!(nniap, ds, opt::Flux.Optimise.AbstractOptimiser, epochs, loss, w_e, w_f)
#     optim = Flux.setup(opt, nniap.nn)  # will store optimiser momentum, etc.
#     ∇loss(nn, iap, ds, w_e, w_f) = gradient((nn) -> loss(nn, iap, ds, w_e, w_f), nn)
#     losses = []
#     for epoch in 1:epochs
#         # Compute gradient with current parameters and update model
#         grads = ∇loss(nniap.nn, nniap.iap, ds, w_e, w_f)
#         Flux.update!(optim, nniap.nn, grads[1])
#         # Logging
#         curr_loss = loss(nniap.nn, nniap.iap, ds, w_e, w_f)
#         push!(losses, curr_loss)
#         println(curr_loss)
#     end
# end

function loss(nn, iap, ds, w_e = 1, w_f = 1)
    nniap = NNIAP(nn, iap)
    e = 0.0; f = 0.0
    if w_e > 0.0
        es, es_pred = get_all_energies(ds), get_all_energies(ds, nniap)
        e = w_e * Flux.mse(es_pred, es)
    end
    if w_f > 0.0
        fs, fs_pred = get_all_forces(ds), get_all_forces(ds, nniap)
        f = w_f * Flux.mse(fs_pred, fs)
    end    
    return e + f
end

function prelearn!(nace, ds_train, opt, n_epochs, loss)
    nace_e = NNIAP(deepcopy(nace.nn), nace.iap)
    nace_f = NNIAP(deepcopy(nace.nn), nace.iap)
    t_e = @spawn learn!(nace_e, ds_train, opt, n_epochs, loss, 1, 0)
    t_f = @spawn learn!(nace_f, ds_train, opt, n_epochs, loss, 0, 1)
    t_e = learn!(nace, ds_train, opt, n_epochs, loss, 1, 0)
    t_f = learn!(nace_e, ds_train, opt, n_epochs, loss, 0, 1)
    @sync begin
        wait(t_e)
        wait(t_f)
    end
    ps, re = Flux.destructure(nace.nn)
    ps = (Flux.params(nace_e.nn) .+ Flux.params(nace_f.nn)) ./ 2
    nace.nn = re(ps)

    ps_e = Flux.params(nace_e.nn)
    ps_f = Flux.params(nace_f.nn)
    ps = []

    for i in 1:length(ps_e)
       append!(ps, (ps_e[i] .+ ps_f[i]) ./ 2)
    end

    nace.nn = re(ps)
end


function learn!(nniap, ds, opt::Flux.Optimise.AbstractOptimiser, epochs, loss, w_e, w_f)
   optim = Flux.setup(opt, nniap.nn)  # will store optimiser momentum, etc.
   ∇loss(nn, iap, ds, w_e, w_f) = gradient((nn) -> loss(nn, iap, ds, w_e, w_f), nn)
   losses = []
    for epoch in 1:epochs
        # Compute gradient with current parameters and update model
        grads = ∇loss(nniap.nn, nniap.iap, ds, w_e, w_f)
        Flux.update!(optim, nniap.nn, grads[1])
        # Logging
        curr_loss = loss(nniap.nn, nniap.iap, ds, w_e, w_f)
        push!(losses, curr_loss)
        println(curr_loss)
        println("epoch: ", epoch)
    end
end


# Optimization.jl training
function learn!(nniap, ds, opt::Optim.FirstOrderOptimizer, maxiters, loss, w_e, w_f)
    ps, re = Flux.destructure(nniap.nn)
    batchloss(ps, p) = loss(re(ps), nniap.iap, ds, w_e, w_f)
    ∇bacthloss = OptimizationFunction(batchloss, Optimization.AutoForwardDiff()) # Optimization.AutoZygote()
    prob = OptimizationProblem(∇bacthloss, ps, []) # prob = remake(prob,u0=sol.minimizer)
    cb = function (p, l) println("Loss BFGS: $l"); return false end
    sol = solve(prob, opt, maxiters=maxiters, callback = cb)
    ps = sol.u
    nn = re(ps)
    nniap.nn = nn
    #copyto!(nniap.nn, nn)
    #global nniap = NNIAP(nn, nniap.iap) # TODO: improve this
end



