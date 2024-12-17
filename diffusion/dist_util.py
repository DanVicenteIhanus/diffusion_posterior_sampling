"""
Helpers for distributed training.
- CUDA, MPS and 
"""

import io
import os
import socket

import blobfile as bf
from mpi4py import MPI
import torch as th
import torch.distributed as dist

# Enable CPU fallback for unsupported MPS operations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3

def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    if th.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}"
        backend = "nccl"
    else:
        backend = "gloo"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    
    comm = MPI.COMM_WORLD
    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)

    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, init_method="env://")

def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.backends.mps.is_available():
        return th.device("mps")
    elif th.cuda.is_available():
        return th.device("cuda")
    return th.device("cpu")

def safe_all_gather(tensor_list, tensor):
    """
    Safely perform all_gather operation with MPS support.
    - If we use MPS then we just return the tensor.
    """
    if th.backends.mps.is_available():
        tensor_list[0].copy_(tensor)
        return tensor_list
    else:
        return dist.all_gather(tensor_list, tensor)

def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    chunk_size = 2 ** 30  # MPI has a relatively small size limit
    if MPI.COMM_WORLD.Get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        num_chunks = len(data) // chunk_size
        if len(data) % chunk_size:
            num_chunks += 1
        MPI.COMM_WORLD.bcast(num_chunks)
        for i in range(0, len(data), chunk_size):
            MPI.COMM_WORLD.bcast(data[i : i + chunk_size])
    else:
        num_chunks = MPI.COMM_WORLD.bcast(None)
        data = bytes()
        for _ in range(num_chunks):
            data += MPI.COMM_WORLD.bcast(None)

    map_location = kwargs.get('map_location', 'cpu')
    if isinstance(map_location, str):
        map_location = th.device(map_location)
    
    def custom_load(storage, location):
        return storage

    kwargs['map_location'] = custom_load

    try:
        state_dict = th.load(io.BytesIO(data), weights_only=False, **kwargs)
        return state_dict
    except Exception as e:
        print(f"Loading with custom_load failed: {e}")
        return th.load(io.BytesIO(data), map_location='cpu')

def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            if th.backends.mps.is_available():
                # Move to CPU for sync, then back to MPS
                cpu_p = p.cpu()
                dist.broadcast(cpu_p.float(), 0)
                p.copy_(cpu_p.to('mps'))
            else:
                dist.broadcast(p.float(), 0)

def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()