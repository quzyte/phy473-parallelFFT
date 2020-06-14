from time import time
import numpy as np
from numpy.fft import fftfreq, fft, ifft, irfft2, rfft2
from mpi4py import MPI
import comm_tools

N = 2**7
comm = MPI.COMM_WORLD
num_processes = comm.Get_size()
rank = comm.Get_rank()
Np = N // num_processes

X = np.mgrid[:N, :N, :N].astype(float)*2*np.pi/N
U = np.sin(X[0])*np.cos(X[1])*np.cos(X[2])

def print_(*args, comm_arg=MPI.COMM_WORLD, checkRank=0, **kwargs):
    """function to print arguments only in one core (default rank=0 in that comm group)"""
    if comm_arg.Get_rank() == checkRank:
        print(*args, **kwargs)

try:
    from pyfftw.interfaces.numpy_fft import fft, ifft, irfft2, rfft2
    import pyfftw
    pyfftw.interfaces.cache.enable()
    fftw_status = 'Using pyfftw'
except ImportError:
    fftw_status = 'Using numpy.fft'

# print_(fftw_status)
# print_('real space data shape: ', U.shape)
# print_('k-space data shape: ', U_hat.shape)
# print_('cores: ', num_processes)


def fftn_mpi(u):
    # do rfft along z axis and fft along y axis
    temp1 = rfft2(u, axes=(1, 2))
    # realign arrays along xz axes
    temp1 = comm_tools.redistribute(temp1, 0, 1)
    # do fft along x axis
    fu = fft(temp1, axis=0)
    # arrays now distributed along y axis
    return fu

def ifftn_mpi(fu):
    # do ifft along x axis
    temp2 = ifft(fu, axis=0)
    # realign arrays along yz axes
    temp2 = comm_tools.redistribute(temp2, 1, 0)
    # do ifft along y axis, and irfft along z axis
    u = irfft2(temp2, axes=(1, 2))
    # arrays now distributed along x axis
    return u


# distribute array into slab shape
Udist = comm_tools.distribute(U, axis=0) if rank == 0 else comm_tools.distribute(axis=0)
print(f'-- Rank : {comm.Get_rank()} -- \n Array size: {Udist.shape}\n')

# fft-ifft code
t0 = time()
U_hat_dist = fftn_mpi(Udist)
t1 = time()
U_approx_dist = ifftn_mpi(U_hat_dist)
t2 = time()

#recombine
U_approx = comm_tools.accumulate(U_approx_dist, axis=0)


k1 = comm.allreduce(np.sum(Udist*Udist))
k2 = comm.allreduce(np.sum(U_approx_dist*U_approx_dist))
dt1 = comm.allreduce(t1-t0, op=MPI.MAX)
dt2 = comm.allreduce(t2-t1, op=MPI.MAX)

print_('\nTesting Results:')
print_("Duration: Forward= ", round(dt1, 3), 's  Inverse= ', round(dt2, 3), 's')
print_('Norm before = ', k1)
print_('Norm after = ', k2)
print_(f'Difference = {(k1-k2)}')
if rank == 0:
    print_(f'Returned array same as initial : {np.allclose(U, U_approx)}')
