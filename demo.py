from __future__ import print_function, division
from time import time
from numpy import *
from numpy.fft import fftfreq, fft, ifft, irfft2, rfft2
from mpi4py import MPI

try:
    from pyfftw.interfaces.numpy_fft import fft, ifft, irfft2, rfft2
    import pyfftw
    pyfftw.interfaces.cache.enable()
    # print('pyFFTW Import Success')

except ImportError:
    print('Not Importing pyFFTW')

nu = 0.000625
N = 2**2
comm = MPI.COMM_WORLD
num_processes = comm.Get_size()
rank = comm.Get_rank()
if rank == 0:
    print('cores: ', num_processes)

# print('rank: ', rank)

Np = N // num_processes
X = mgrid[rank*Np:(rank+1)*Np, :N, :N].astype(float)*2*pi/N
if rank == 0:
    print('X shape: ', X.shape)
U = empty((Np, N, N))
U_hat = empty((N, Np, N//2+1), dtype=complex)
U_approx = empty((Np, N, N))
Uc_hat = empty((N, Np, N//2+1), dtype=complex)
Uc_hatT = empty((Np, N, N//2+1), dtype=complex)

def fftn_mpi(u, fu):
    if rank == 0:
        print('u.shape: ', u.shape)
        print('fu.shape initially: ', fu.shape)
    Uc_hatT[:] = rfft2(u, axes=(1, 2))
    if rank == 0:
        print('Uc_hatT.shape: ', Uc_hatT.shape)
    fu[:] = rollaxis(Uc_hatT.reshape(Np, num_processes, Np, N//2+1), 1).reshape(fu.shape)
    if rank == 0:
        print('fu.shape middle: ', fu.shape)
    comm.Alltoall(MPI.IN_PLACE, [fu, MPI.DOUBLE_COMPLEX])
    fu[:] = fft(fu, axis=0)
    if rank == 0:
        print('fu.shape end: ', fu.shape)
    print('core',rank,' value of fu: ',fu);
    return fu

def ifftn_mpi(fu, u):
    Uc_hat[:] = ifft(fu, axis=0)
    comm.Alltoall(MPI.IN_PLACE, [Uc_hat, MPI.DOUBLE_COMPLEX])
    Uc_hatT[:] = rollaxis(Uc_hat.reshape((num_processes, Np, Np, N//2+1)), 1).reshape(Uc_hatT.shape)
    u[:] = irfft2(Uc_hatT, axes=(1, 2))
    return u

U = sin(X[0])*cos(X[1])*cos(X[2])

t0 = time()
U_hat = fftn_mpi(U, U_hat)
t1 = time()
U_approx = ifftn_mpi(U_hat, U)
t2 = time()

# k1 = comm.reduce(0.5*sum(U*U)*(1./N)**3, op=MPI.SUM)
# k2 = comm.reduce(0.5*sum(U_approx*U_approx)*(1./N)**3)

k11 = 0.5 * sum(U*U)*(1./N)**3
k1 = comm.reduce(k11)
k22 = 0.5 * sum(U_approx*U_approx)*(1/N)**3
k2 = comm.reduce(k22)

if rank == 0:
    print("Times = ", t1-t0, t2-t1)
    print('Norm before = ', k1)
    print('Norm after = ', k2)
    print('Difference = {:e}'.format(k1-k2))
