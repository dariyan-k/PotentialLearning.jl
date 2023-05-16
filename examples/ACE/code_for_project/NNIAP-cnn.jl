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
    res(M) = reshape(M, (26, 1, 1,1))
    s = sum([sum(nniap.nn(res(B_atom))) for B_atom in Bs])
    #@assert 0 == 1
    return s
end

# function force(c::Configuration, nniap::NNIAP)
#     Bs = get_values(get_local_descriptors(c))
#     res(M) = reshape(M, (26, 1, 1))
#     dnndb = [first(gradient(x->sum(nniap.nn(x)), res(B_atom))) for B_atom in Bs]
#     dbdr = get_values(get_force_descriptors(c))
#     return [[-sum(dnndb .⋅ [dbdr[atom][coor]]) for coor in 1:3]
#              for atom in 1:length(dbdr)]
# end

# function potential_energy(c::Configuration, nniap::NNIAP)
#     Bs = get_values(get_local_descriptors(c))
#     return sum([sum(nniap.nn(B_atom)) for B_atom in Bs])
# end

function force(c::Configuration, nniap::NNIAP)
    Bs = get_values(get_local_descriptors(c))
    res(M) = reshape(M, (26, 1, 1,1))
    dnndb = [first(gradient(x->sum(nniap.nn(x)), res(B_atom))) for B_atom in Bs]
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
function loss(nn, iap, ds, w_e = 1, w_f = 1)
    nniap = NNIAP(nn, iap)
    es = get_all_energies(ds)
    es_pred = get_all_energies(ds, nniap)
    fs = get_all_forces(ds) 
    fs_pred = get_all_forces(ds, nniap)
    result = w_e * Flux.mse(es_pred, es) + w_f * Flux.mse(fs_pred, fs)
    return  result
end


# Auxiliary functions ##########################################################
function get_all_energies(ds::DataSet, nniap::NNIAP)
    return [potential_energy(ds[c], nniap) for c in 1:length(ds)]
end

function get_all_forces(ds::DataSet, nniap::NNIAP)
    return reduce(vcat,reduce(vcat,[force(ds[c], nniap)
                                    for c in 1:length(ds)]))
end

function batch_and_shuffle(data, num_batches) # new
    # Shuffle the data
    shuffle!(data)

    # Calculate the number of batches
    batch_size = ceil(Int, length(data) / num_batches)
    # Create the batches
    batches = [data[(i-1)*batch_size+1:min(i*batch_size, end)] for i in 1:num_batches]

    return batches
end

# NNIAP learning functions #####################################################

# Flux.jl training
function learn!(nniap, ds, opt::Flux.Optimise.AbstractOptimiser, epochs, loss, w_e, w_f)
    #n_batches = 1
    optim = Flux.setup(opt, nniap.nn)  # will store optimiser momentum, etc.
    ∇loss(nn, iap, ds, w_e, w_f) = gradient((nn) -> loss(nn, iap, ds, w_e, w_f), nn)
    losses = []

    #batch_lists = batch_and_shuffle(collect(1:length(ds_train)), n_batches)
    #batch_list_len = length(batch_lists)

    for epoch in 1:epochs
        # Compute gradient with current parameters and update model

        #batch_index = mod(epoch, batch_list_len) + 1 
        #ds_batch = ds[batch_lists[batch_index]]


        grads = ∇loss(nniap.nn, nniap.iap, ds, w_e, w_f)
        Flux.update!(optim, nniap.nn, grads[1])
        # Logging
        curr_loss = loss(nniap.nn, nniap.iap, ds, w_e, w_f)
        push!(losses, curr_loss)
        println(curr_loss)
        println("epoch ", epoch)
    end

    println("min loss: ", minimum(losses))
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



