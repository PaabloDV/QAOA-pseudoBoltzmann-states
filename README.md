# QAOA pseudo-Boltzmann-states
Supplementary material for "QAOA pseudo-Boltzmann states" https://arxiv.org/pdf/2201.03358.pdf 

This repository contains the code to reproduce all the results shown in the manuscript.

- Modules:
    - [mympi.py](./mympi.py) : allows to run the numerical simulations in parallel making use of the Message Passing Interface (MPI) with the MPI for Python package, or of the multiprocessing python package.
    - [problem_generator.py](./problem_generator.py) : generates random combinatorial optimization problems with given settings. In particular, it contains functions to generate Max-Cut, QUBO, and Random Ising problems embedded in Erdös-Renyi and r-regular random graphs.  
    - [qaoa.py](./qaoa.py) : simulates a single-layer QAOA circuit as described in "QAOA pseudo-Boltzmann states". It includes functions to optimize the variational angles, and to compute some observables of the output states after the single-layer QAOA ansatz.

- Scrips to run the numerical simulations and save the data in pickle files. 

    They let you specify several arguments of the numerical experiment in `mainjob()` such as the optimization problem ('QUBO','MAXCUT', or 'Random Ising') and its number of variables n, the type of graph ('Gnm', or 'RRG') and its argument (density for Erdös-Renyi graphs, and degree for r-regular graphs), the number of samples in the experiment, or the objective function used in the optimization of the variational parameters ('Energy' or 'GS-Probability'). You can also use MPI parallelization (`mpi=True`), or parallelize it by the multiprocessing python package setting `mpi=False` and choosing the number of workers. 
    - [exp_PseudoBoltzmannStates_observables.py](./exp_PseudoBoltzmannStates_observables.py) : experiment to compute observables of the output states after the single-layer QAOA circuit with optimal angles, such as the angles, the enhancement of the ground state amplitude probability, or the effective temperature of the pseudo-Boltzmann state.
    - [exp_PseudoBoltzmannStates_average.py](./exp_PseudoBoltzmannStates_average.py) : experiment to compute the average of the amplitude probabilities after the single-layer QAOA circuit with optimal angles over many problem replicas. To compare instances we normalize the energy between 0 and 1, and we divide the spectrum into several intervals.
    - [exp_PseudoBoltzmannStates_VSgamma.py](./exp_PseudoBoltzmannStates_VSgamma.py) and [exp_PseudoBoltzmannStates_VStheta.py](./exp_PseudoBoltzmannStates_VStheta.py) : experiment to compute the evolution of the observables after the single-layer QAOA circuit as we change the QAOA angle gamma/theta respectively. We keep the other QAOA angle theta/gamma fixed at its optimum value.
    - [exp_covariancesEH.py](./exp_covariancesEH.py) : experiment to compute the average of the covariances in the H-E distribution such as explained in the section "Derivation of pseudo-Boltzmann states" of the manuscript. We fit the distribution to a sum over gaussians from we calculate such covariance.
    - [exp_mcmc_comparison.py](./exp_mcmc_comparison.py) : experiment to compare the effective temperature of single-layer QAOA pseudo-Boltzmann states with optimal angles with the theoretical limit of MCMC methods given by R.Eldan et al. https://arxiv.org/pdf/2007.08200.pdf .

- Notebooks to plot the results from pickle files (obtained from the previous python scripts):
    - [Pseudo_Boltzmannstates_Fig2.ipynb](./Pseudo_Boltzmannstates_Fig2.ipynb) : reproduce the results in Figure 2 (data from "exp_PseudoBoltzmannStates_average.py").
    - [Pseudo_Boltzmannstates_Fig3.ipynb](./Pseudo_Boltzmannstates_Fig3.ipynb) : reproduce the results in Figure 3 (data from "exp_PseudoBoltzmannStates_observables.py", "exp_PseudoBoltzmannStates_VSgamma.py", and "exp_PseudoBoltzmannStates_VStheta.py").
    - [Pseudo_Boltzmannstates_Fig4_covEH.ipynb](./Pseudo_Boltzmannstates_Fig4_covEH.ipynb) : reproduce the results in Figure 4 (data from "exp_covariancesEH.py").
    - [Pseudo_Boltzmannstates_Trends.ipynb](./Pseudo_Boltzmannstates_Trends.ipynb) : reproduce the results shown in the section II.B of the Supplementary material (data from "exp_PseudoBoltzmannStates_observables.py").
    - [Pseudo_Boltzmannstates_GaussianMixture.ipynb](./Pseudo_Boltzmannstates_GaussianMixture.ipynb) : in this notebook we numerically compute the probability distribution p(H, E, x) over eigenenergies E and Hamming distances H from eigenstates to the reference x. We fit this distribution to a sum of one or two gaussians.
    - [QAOAvsMCMC.ipynb](./QAOAvsMCMC.ipynb) : reproduce the results in Figure 5 and the section II.C of the Supplementary material (data from "exp_mcmc_comparison.py").

- We also include an additional notebook [QAOA_pseudoBoltzmann_states_withqiskit.ipynb](./QAOA_pseudoBoltzmann_states_withqiskit.ipynb). Here we replicate the pseudo-Boltzmann states produced by a single-layer QAOA circuit and reported in the manuscript, but using Qiskit instead of [qaoa.py](./qaoa.py). 

Simulation data are available upon reasonable request.


