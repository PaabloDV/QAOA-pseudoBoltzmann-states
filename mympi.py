import time


def split_list(data, N):
    L = len(data)
    return [data[n::N] for n in range(0, N)]


def am_I_root_worker(mpi=False):
    return worker_id(mpi) == 0


def worker_id(mpi=False):
    if mpi:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        return comm.Get_rank()
    else:
        return 0


def mpi_split_job(function, data, root_works=False):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    if rank != 0:
        groups = None
    elif root_works:
        print(f"Root received {len(data)} jobs, to be divided into {size} groups")
        groups = split_list(data, size)
    else:
        print(f"Root received {len(data)} jobs, to be divided into {size-1} groups")
        groups = [[]] + split_list(data, size - 1)
    data = comm.scatter(groups, root=0)
    print(f"Worker {rank} executing {len(data)} tasks.")
    starttime = time.time()
    try:
        data = [function(item) for item in data]
    except Exception as e:
        print(f"Exception received by worker {rank}:\n{e}\nForegoing payload.")
        data = []
    data = comm.gather(data, root=0)
    finishtime = time.time() - starttime
    print(f"Worker {rank} finished after {finishtime}s")
    if rank == 0:
        return sum(data, [])
    else:
        return None


def workers_split_job(workers, function, data, root_works=False):
    print(f"Root received {len(data)} jobs, to be divided into {workers} groups")
    if workers > 1:
        from multiprocessing import Pool

        with Pool(processes=workers) as pool:
            data = pool.map(function, data)

    else:
        data = [function(item) for item in data]
    return data


if __name__ == "__main__":
    import mpi4py

    print("Hello")

    def compute(item):
        return f"value: {item}"

    data = list(range(32))
    mpi_split_job(compute, data, root_works=True)
    mpi_split_job(compute, data, root_works=False)
