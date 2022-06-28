# **********************************
# EXPERIMENT TO COMPARE THE EFFECTIVE TEMPERATURE OF SINGLE-LAYER QAOA PSEUDO-BOLTZMANN STATES WITH OPTIMAL ANGLES
# WITH THE THEORETICAL LIMIT OF MCMC METHODS GIVEN BY R.Eldan et al. https://arxiv.org/pdf/2007.08200.pdf 
# **********************************

import pandas as pd
import networkx as nx
import pickle
import time
from qaoa import *
from mympi import *
from problem_generator import *
import os

def experiment(data):
    starttime = time.time()
    N, sigma, sample, ptname, graph_type, graph_argument, objective_function, seed_sequence = data

    np.random.seed(seed_sequence.generate_state(10))
    E, J = generate_interaction(N, ptname, graph_type, graph_argument)

    # Optimal angles
    θ, γ = optimization(N, E, cost_function=objective_function)

    # Running QAOA
    ψ2 = QAOA(θ, γ, N, E)

    # Enhancement
    gsprob = GS_probability(ψ2, E)
    enhan = Enhancement(N, gsprob, E)

    # Effective temperature
    β, org, score = BoltzmannFit(ψ2, E)

    E -= np.min(E)  # To avoid overflows when computing thermal distribution
    J = J - np.diag(np.diag(J))
    Jnorm = np.linalg.norm(J, ord=2)
    βmc = 1.0 / Jnorm
    mcprob = GS_probability_mc(βmc, E)

    #Ground state probability computed from the QAOA effective temperature
    qaoaprobbeta = GS_probability_mc(β, E)

    finishtime = time.time() - starttime
    print(f"Ran job with {N} qubits in {finishtime} seconds")
    return {
        "NºQubits": N,
        "Graph": graph_type,
        "Density": graph_argument,
        "Sigma": sigma,
        "Problem type": ptname,
        "Cost function": objective_function,
        "Theta Op": θ,
        "Gamma Op": γ,
        "GS probability": gsprob,
        "Enhancement": enhan,
        "Beta full energies": β,
        "Org full energies": org,
        "Score full energies": score,
        "Sample": sample,
        "Jnorm": Jnorm,
        "Beta MC": βmc,
        "GS probability MC": mcprob,
        "GS probability QAOA beta": qaoaprobbeta,
        "Time": finishtime,
    }


def add_seeds(work_items):
    """Precompute random seeds and add them to the job descriptions, to
    ensure that parallel jobs do not generate the same data."""
    generator = np.random.SeedSequence()
    return [
        work_item + (sequence,)
        for work_item, sequence in zip(work_items, generator.spawn(len(work_items)))
    ]


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def main_job(
    mpi=False,
    workers=1,
    sample_size=1000,
    nmin=4,
    nmax=20,
    problem_types=["QUBO", "SK"],
    graphs=[("Gnm", 1)],
    objective_function='GS-Probability',
    sigma=1.0,
    outputfile=None,
    overwrite=False,
):
    for ptname in problem_types:
        for graph_type, graph_arguments in graphs:
            if outputfile is None:
                outputfile = f"expts/mcmc_comparison-{ptname}_nq{nmin}-{nmax}_nsamples{sample_size}_{graph_type}{graph_arguments}_{objective_function}.pkl"
            if not overwrite and os.path.exists(outputfile):
                print(
                    f"Output {outputfile} already exists, {worker_id(mpi)} does not have to do anything."
                )
                continue
            work_to_do = add_seeds(
                [
                    (N, sigma, sample, ptname, graph_type, graph_arguments,objective_function)
                    for N in range(nmin, nmax + 1, 2)
                    for sample in range(sample_size)
                ]
            )
            print(f"Launching workers for {outputfile}")
            if mpi:
                raw_data = mpi_split_job(experiment, work_to_do, root_works=False)
            else:
                raw_data = workers_split_job(
                    workers, experiment, work_to_do, root_works=False
                )
            if am_I_root_worker(mpi):
                # Save data as DataFrame only if we are the process that collects
                # the data (raw_data != None)
                try:
                    ensure_dir(outputfile)
                    raw_dataframe = pd.DataFrame(raw_data)
                    print(f"Writing to file {outputfile}")
                    with open(outputfile, "wb") as f:
                        pickle.dump(raw_dataframe, f)
                except:
                    print(f"Error when saving file {outputfile}")


if __name__ == "__main__":
    main_job(workers=1, nmin=4, nmax=24, sample_size=500, graphs=[("Gnm", 1.0)],objective_function='Energy',problem_types=["SK"])