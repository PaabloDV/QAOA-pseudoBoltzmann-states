# **********************************
# EXPERIMENT TO COMPUTE THE EVOLUTION OF THE OBSERVABLES AFTER SINGLE-LAYER QAOA
# AS WE CHANGE THE QAOA ANGLE GAMMA. WE KEEP THE QAOA ANGLE THETA FIXED AT ITS OPTIMUM VALUE.
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
    θ, _ = optimization(N, E, cost_function=objective_function)
    γ_array = np.linspace(0.001, 0.6, 31)

    ψ2_array = []
    gsprob_array = []
    enhan_array = []
    β_array = []
    org_array = []
    score_array = []
    for γ in γ_array:  
        # Running QAOA
        ψ2 = QAOA(θ, γ, N, E)
        ψ2_array.append(ψ2)
        # Enhancement
        gsprob = GS_probability(ψ2, E)
        gsprob_array.append(gsprob)
        enhan_array.append(Enhancement(N, gsprob, E))
        # Effective temperature
        β, org, score = BoltzmannFit(ψ2, E)
        β_array.append(β)
        org_array.append(org)
        score_array.append(score)

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
        "Gamma": γ_array,
        "GS probability": gsprob_array,
        "Enhancement": enhan_array,
        "Beta full energies": β_array,
        "Org full energies": org_array,
        "Score full energies": score_array,
        "Sample": sample,
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
    problem_types=["QUBO", "Random_Ising", "MAXCUT"],
    graphs=[("Gnm", 1)],
    objective_function='GS-Probability',
    sigma=1.0,
    outputfile=None,
    overwrite=False,
):
    for ptname in problem_types:
        for graph_type, graph_arguments in graphs:
            outputfile = f"expts/exp_VSgamma-{ptname}_nq{nmin}-{nmax}_nsamples{sample_size}_{graph_type}{graph_arguments}_{objective_function}.pkl"
            
            if not overwrite and os.path.exists(outputfile):
                print(
                    f"Output {outputfile} already exists, {worker_id(mpi)} does not have to do anything."
                )
                continue

            work_to_do = add_seeds(
                [
                    (N, sigma, sample, ptname, graph_type, graph_arguments, objective_function)
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
    main_job(mpi=False,workers=25, nmin=6, nmax=22, sample_size=500, graphs=[("Gnm", 0.9)],problem_types=["QUBO"],objective_function='Energy')

