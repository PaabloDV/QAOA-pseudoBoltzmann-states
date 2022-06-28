# **********************************
# EXPERIMENT TO COMPUTE THE AVERAGE OF THE COVARIANCES IN THE H-E DISTRIBUTION.
# WE FIT THE DISTRIBUTION TO A SUM OVER GAUSSIANS FROM WE CALCULATE SUCH COVARIANCE.
# **********************************

import pandas as pd
import networkx as nx
import pickle
import time
from qaoa import *
from problem_generator import *
from mympi import *
from sklearn import mixture
import os

# The BayesianGaussianMixture method over all eigenstates can give rise to many warnings messages when it does not 
# reach full convergence. You can avoid printing those messages by uncommenting the following two lines.
import warnings
warnings.filterwarnings("ignore")


def normEnergy(E):
    "Compute the normalized energy"
    Emax = np.max(E)
    ndxmax = np.where(E == Emax)[0]
    Emin = np.min(E)
    ndxmin = np.where(E == Emin)[0]
    NE = [(e-Emin)/(Emax-Emin) for e in E]
    return Emax, ndxmax, Emin, ndxmin, NE

def Hamming(y):
    "Calculate the Hamming distances between y and every string of bits"
    N = len(y)
    bits = all_bit_strings(N)
    x = (bits-0.5)*2
    y = (y-0.5)*2
    return 1/2*(N-(x.T@y))

def HE_cov(ndxx,E,N,problem='QUBO'):
    "Calculate the covariance between energy levels and the Hamming distance of ndxx to those energy levels. We use a Variational Bayesian" 
    "estimation of the Gaussian mixtures to derive the covariance of the distribution."

    if problem=='MAXCUT':
        ngaussians=2
    if problem=='Random_Ising':
        ngaussians=2
    if problem=='QUBO':
        ngaussians=1

    bits = all_bit_strings(N)
    hm = Hamming(bits[:,ndxx])
    X = np.empty((np.size(E),2))
    X[:, 0] = E
    X[:, 1] = hm
    #Fit to gaussian Mixture
    initializations=3 #Number of maximum repetitions in case of no full convergence
    step=1
    for i in range(initializations):
        dpgmm = mixture.BayesianGaussianMixture(n_components=ngaussians,random_state=10*(i+1), covariance_type="full",init_params='random').fit(X)
        if dpgmm.converged_ or step==initializations:
            if ngaussians==1:
                covariance = dpgmm.covariances_[0][0,1]
            elif ngaussians==2: 
                #We only save sigmaEH^+ (see supplementary material). The subspace corresponds to the lowest mean.
                if dpgmm.means_[0][1]<=dpgmm.means_[1][1]:
                    covariance = dpgmm.covariances_[0][0,1]
                else:
                    covariance = dpgmm.covariances_[1][0,1]
            break
        else:
            step+=1
    return covariance,E[ndxx]

def average_result_covar(N,E,sample,nintervals=100,problem='QUBO'):
    "This function distributes the eigenstates in 100 intervals of the normalized energy, and then calculate the covariance between the energy levels and the Hamming distance to those energy levels." 
    "We use a Variational Bayesian estimation of the Gaussian mixtures to derive the covariance of the distribution."

    covariance_samples = []
    energy_intervals_samples = []
    samples = []
    _,_,_,_,NE = normEnergy(E)
    for j in range(nintervals):
        ndx = np.where(np.array([0 if ne >= (j/nintervals) and ne <= ((j+1)/nintervals) else 1 for ne in NE]) == 0)[0]

        for index in ndx:
            starttime_=time.time()
            sigmaEH,ex = HE_cov(index,E,N,problem=problem)
            #print(f'Time one eigenstate: {time.time()-starttime_}s')

            covariance_samples.append(sigmaEH)
            energy_intervals_samples.append(j/nintervals)
            samples.append(sample)

    return samples,covariance_samples,energy_intervals_samples

def experiment(data):
    starttime = time.time()
    N, sigma, sample, ptname, graph_type, graph_argument, seed_sequence = data

    np.random.seed(seed_sequence.generate_state(10))
    E, J = generate_interaction(N, ptname, graph_type, graph_argument)
    samples,covariance_samples,energy_intervals_samples = average_result_covar(N,E,sample,nintervals=100,problem=ptname)

    finishtime = time.time() - starttime
    print(f"Ran job with {N} qubits in {finishtime} seconds")
    return {
        "NºQubits": N,
        "Graph": graph_type,
        "Density": graph_argument,
        "Sigma": sigma,
        "Problem type": ptname,
        "Covariances": covariance_samples,
        "Normalized energy": energy_intervals_samples,
        "Sample": samples,
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
    sigma=1.0,
    outputfile=None,
    overwrite=False,
):
    for ptname in problem_types:
        for graph_type, graph_arguments in graphs:
            outputfile = f"expts/exp_covarianceEH2-{ptname}_nq{nmin}-{nmax}_nsamples{sample_size}_{graph_type}{graph_arguments}.pkl"
            
            if not overwrite and os.path.exists(outputfile):
                print(
                    f"Output {outputfile} already exists, {worker_id(mpi)} does not have to do anything."
                )
                continue
            work_to_do = add_seeds(
                [
                    (N, sigma, sample, ptname, graph_type, graph_arguments)
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
    #main_job(workers=100, nmin=14, nmax=14, sample_size=100, graphs=[("Gnm",1)],problem_types=["Random_Ising"])
    main_job(mpi=True, workers=100, nmin=14, nmax=14, sample_size=100, graphs=[("Gnm",1)],problem_types=["Random_Ising"])

