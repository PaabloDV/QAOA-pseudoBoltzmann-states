# **********************************
# EXPERIMENT TO COMPUTE THE AVERAGE OF THE AMPLITUDE PROBABILITIES AFTER SINGLE-LAYER QAOA
# WITH OPTIMAL ANGLES OVER MANY PROBLEM REPLICAS. TO COMPARE INSTANCES WE NORMALIZE THE ENERGY AND 
# WE DIVIDE THE SPECTRUM INTO SEVERAL INTERVALS. 
# **********************************

import pandas as pd
import networkx as nx
import pickle
import time
from qaoa import *
from mympi import *
from problem_generator import *
from multiprocessing import Pool
import os

def normalized_energy(E):
    "Compute the normalized energy"
    Emax = np.max(E)
    ndxmax = np.where(E == Emax)[0]
    Emin = np.min(E)
    ndxmin = np.where(E == Emin)[0]
    normvalue = Emax-Emin 
    NE = [(e-Emin)/normvalue for e in E]
    return NE, Emax, Emin, ndxmax, ndxmin, normvalue

def bins_amplitudes(ψ2,NE,nintervals=100):
    average_amplitude=np.zeros(nintervals)
    for j in range(nintervals):
        ndx = np.where(np.array([0 if ne >= (j/nintervals) and ne <= ((j+1)/nintervals) else 1 for ne in NE]) == 0)[0]
        if len(ndx) != 0:
            average_amplitude[j] += np.sum(ψ2[ndx])/len(ndx)
    return average_amplitude

def experiment(data):
    starttime = time.time()
    N, sigma, sample, ptname, graph_type, graph_argument, objective_function, seed_sequence = data

    np.random.seed(seed_sequence.generate_state(10))
    E, J = generate_interaction(N, ptname, graph_type, graph_argument)
    NE, Emax, Emin, ndxmax, ndxmin, normvalue  = normalized_energy(E)
    
    # Optimal angles
    θ, γ = optimization(N, E, cost_function=objective_function)

    # Running QAOA
    ψ2 = QAOA(θ, γ, N, E)

    # Binning the amplitude so that we save an array of len=nintervals with the average amplitude in each interval
    average_ψ2 = bins_amplitudes(ψ2,NE,nintervals=100)

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
        "Average Amplitude": average_ψ2,
        "Normalization factor": normvalue,
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
    sample_size=500,
    nmin=4,
    nmax=20,
    problem_types=["QUBO", "Random_Ising", "MAXCUT"],
    graphs=[("Gnm", 0.9)],
    objective_function='GS-Probability',
    sigma=1.0,
    outputfile=None,
    overwrite=False,
):
    for ptname in problem_types:
        for graph_type, graph_arguments in graphs:
            outputfile = f"expts/exp_PseudoBoltzmannStatesAverage-{ptname}_nq{nmin}-{nmax}_nsamples{sample_size}_{graph_type}{graph_arguments}_{objective_function}.pkl"

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
    main_job(mpi=False,workers=25, nmin=14, nmax=14,sample_size=1000, problem_types=["QUBO","Random_Ising","MAXCUT"], objective_function='Energy')
