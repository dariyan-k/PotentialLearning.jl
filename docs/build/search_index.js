var documenterSearchIndex = {"docs":
[{"location":"functions.html#Functions","page":"Functions","title":"Functions","text":"","category":"section"},{"location":"functions.html","page":"Functions","title":"Functions","text":"Modules = [PotentialLearning]","category":"page"},{"location":"functions.html#PotentialLearning.GaN","page":"Functions","title":"PotentialLearning.GaN","text":"GaN Potential\nSee https://iopscience.iop.org/article/10.1088/1361-648X/ab6cbe\n\n\n\n\n\n","category":"type"},{"location":"functions.html#PotentialLearning.GaN-Tuple{Dict}","page":"Functions","title":"PotentialLearning.GaN","text":"GaN(params::Dict)\n\nCreates \n\n\n\n\n\n","category":"method"},{"location":"functions.html#PotentialLearning.LennardJones","page":"Functions","title":"PotentialLearning.LennardJones","text":"Lennard-Jones Potential\n\n\n\n\n\n","category":"type"},{"location":"functions.html#PotentialLearning.SNAP_LAMMPS","page":"Functions","title":"PotentialLearning.SNAP_LAMMPS","text":"Wrapper of the SNAP implementation of LAMMPS, built with LAMMPS.jl\nMathematical formulation: A. P. Thompson et al.\n                          http://dx.doi.org/10.1016/j.jcp.2014.12.018\n\n\n\n\n\n","category":"type"},{"location":"functions.html#PotentialLearning.SNAP_LAMMPS-Tuple{Dict}","page":"Functions","title":"PotentialLearning.SNAP_LAMMPS","text":"SNAP_LAMMPS(params::Dict)\n\nCreation of a SNAP_LAMMPS instance based on the configuration parameters\n\n\n\n\n\n","category":"method"},{"location":"functions.html#PotentialLearning.calc_A-Tuple{String, SNAP_LAMMPS}","page":"Functions","title":"PotentialLearning.calc_A","text":"calc_A(path::String, p::SNAP_LAMMPS)\n\nCalculation of the A matrix of SNAP (Eq. 13, 10.1016/j.jcp.2014.12.018)\n\n\n\n\n\n","category":"method"},{"location":"functions.html#PotentialLearning.error-Tuple{Vector{Float64}, Any, SNAP_LAMMPS}","page":"Functions","title":"PotentialLearning.error","text":"error(β::Vector{Float64}, p, s::SNAP_LAMMPS)\n\nError function to perform the learning process (Eq. 14, 10.1016/j.jcp.2014.12.018)\n\n\n\n\n\n","category":"method"},{"location":"functions.html#PotentialLearning.learn-Tuple{PotentialLearning.Potential, Vector{Float64}, Dict}","page":"Functions","title":"PotentialLearning.learn","text":"learn(p::Potential, dft_training_data::Vector{Float64}, params::Dict)\n\nFit the potentials, forces, and stresses against the DFT data using the configuration parameters.\n\n\n\n\n\n","category":"method"},{"location":"functions.html#PotentialLearning.load_conf_params-Tuple{String}","page":"Functions","title":"PotentialLearning.load_conf_params","text":"load_conf_params(path::String)\n\nLoad configuration parameters\n\n\n\n\n\n","category":"method"},{"location":"functions.html#PotentialLearning.load_dft_data-Tuple{Dict}","page":"Functions","title":"PotentialLearning.load_dft_data","text":"load_dft_data(params::Dict)\n\nLoad DFT data \n\n\n\n\n\n","category":"method"},{"location":"functions.html#PotentialLearning.load_positions_per_conf-Tuple{String, Int64, Int64, Int64}","page":"Functions","title":"PotentialLearning.load_positions_per_conf","text":"load_positions_per_conf(path::String, no_atoms_per_conf::Int64,\n                             no_conf_init::Int64, no_conf_end::Int64)\n\nLoad atomic positions per configuration\n\n\n\n\n\n","category":"method"},{"location":"functions.html#PotentialLearning.potential_energy-Tuple{Dict, Int64, PotentialLearning.Potential}","page":"Functions","title":"PotentialLearning.potential_energy","text":"potential_energy(params::Dict, j::Int64, p::Potential)\n\nCalculation of the potential energy of a particular atomic configuration (j). This calculation requires accessing the SNAP implementation of LAMMPS.\n\n\n\n\n\n","category":"method"},{"location":"functions.html#PotentialLearning.potential_energy-Tuple{Vector{StaticArrays.SVector{3, Float64}}, Float64, PotentialLearning.Potential}","page":"Functions","title":"PotentialLearning.potential_energy","text":"potential_energy(atomic_positions::Vector{Position}, rcut::Float64, p::Potential)\n\nCalculation of the potential energy of a particular atomic configuration. It is based on the atomic positions of the configuration, the rcut, and a particular potential.\n\n\n\n\n\n","category":"method"},{"location":"functions.html#PotentialLearning.validate-Tuple{PotentialLearning.Potential, Vector{Float64}, Dict}","page":"Functions","title":"PotentialLearning.validate","text":"validate(p::Potential, dft_validation_data::Vector{Float64}, params::Dict)\n\nValidate trained potentials, forces, and stresses.\n\n\n\n\n\n","category":"method"},{"location":"index.html#[WIP]-PotentialLearning.jl:-The-Julia-Library-of-Molecular-Dynamics-Potentials","page":"Home","title":"[WIP] PotentialLearning.jl: The Julia Library of Molecular Dynamics Potentials","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"An Open Source library for active training and fast calculation of molecular dynamics potentials for atomistic simulations of materials. ","category":"page"},{"location":"index.html#Features-under-development","page":"Home","title":"Features under development","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"Surrogate DFT data generation\nGallium nitride model\nIntegration with GalacticOptim.jl to perform the optimization process\nIntegration with LAMMPS.jl to access the SNAP implementation of LAMMPS\nImplementation of a pure Julia version of SNAP\nGPU implementation using KernelAbstractions.jl","category":"page"},{"location":"index.html#Installation-instructions","page":"Home","title":"Installation instructions","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"To install PotentialLearning.jl in Julia follow the next steps:","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"Type julia in your terminal and press ]\n] add PotentialLearning.jl","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"Note: this package is not currenlty registered","category":"page"},{"location":"index.html#How-to-setup-and-run-your-experiment","page":"Home","title":"How to setup and run your experiment","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"Load configuration parameters, DFT data, and potential.","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"    path = \"../examples/GaN-SNAP-LAMMPS/\"\n    params = load_conf_params(path)\n    \n    dft_training_data, dft_validation_data = load_dft_data(params)\n    \n    snap = SNAP_LAMMPS(params)","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"Fit the potentials, forces, and stresses against the DFT data using the configuration parameters.","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"    learn(snap, dft_training_data, params)\n","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"Validate trained potentials, forces, and stresses","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"    rel_error = validate(snap, dft_validation_data, params)\n    ","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"(Image: Build Status) (Image: Coverage)","category":"page"}]
}
