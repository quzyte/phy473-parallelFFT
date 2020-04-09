from time import time
import numpy as np
from numpy.fft import fftfreq, fft, ifft, irfft2, rfft2
from mpi4py import MPI

N = 2**8
comm = MPI.COMM_WORLD
num_processes = comm.Get_size()
rank = comm.Get_rank()
Np = N // num_processes

X = np.mgrid[rank*Np:(rank+1)*Np, :N, :N].astype(float)*2*np.pi/N
U = np.empty((Np, N, N))
U_hat = np.empty((N, Np, N//2+1), dtype=complex)
U_approx = np.empty((Np, N, N))
Uc_hat = np.empty((N, Np, N//2+1), dtype=complex)
Uc_hatT = np.empty((Np, N, N//2+1), dtype=complex)

U = np.sin(X[0])*np.cos(X[1])*np.cos(X[2])

try:
    from pyfftw.interfaces.numpy_fft import fft, ifft, irfft2, rfft2
    import pyfftw
    pyfftw.interfaces.cache.enable()
    fftw_status = 'Using pyfftw'
except ImportError:
    fftw_status = 'Using numpy.fft'
if rank == 0:
    print(fftw_status)
    print('real space data shape: ', U.shape)
    print('k-space data shape: ', U_hat.shape)
    print('cores: ', num_processes)


def fftn_mpi(u, fu):
    Uc_hatT[:] = rfft2(u, axes=(1, 2))
    fu[:] = np.moveaxis(Uc_hatT.reshape(Np, num_processes, Np, N//2+1), 1, 0).reshape(fu.shape)
    comm.Alltoall(MPI.IN_PLACE, [fu, MPI.DOUBLE_COMPLEX])
    fu[:] = fft(fu, axis=0)
    return fu

def ifftn_mpi(fu, u):
    Uc_hat[:] = ifft(fu, axis=0)
    comm.Alltoall(MPI.IN_PLACE, [Uc_hat, MPI.DOUBLE_COMPLEX])
    Uc_hatT[:] = np.rollaxis(Uc_hat.reshape((num_processes, Np, Np, N//2+1)), 1).reshape(Uc_hatT.shape)
    u[:] = irfft2(Uc_hatT, axes=(1, 2))
    return u


t0 = time()
U_hat = fftn_mpi(U, U_hat)
t1 = time()
U_approx = ifftn_mpi(U_hat, U)
t2 = time()


k1 = comm.reduce(np.sum(U*U))
k2 = comm.reduce(np.sum(U_approx*U_approx))
dt1 = comm.reduce(t1-t0, op=MPI.MAX)
dt2 = comm.reduce(t2-t1, op=MPI.MAX)
if rank == 0:
    print('\nTesting Results:')
    print("Longest Thread Duration: Forward= ", round(dt1, 3), 's  Inverse= ', round(dt2, 3), 's')
    print('Norm before = ', k1)
    print('Norm after = ', k2)
    print(f'Difference = {(k1-k2):.2e}')
