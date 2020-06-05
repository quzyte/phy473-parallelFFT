from time import time
import numpy as np
from numpy.fft import fftfreq, fft, ifft, irfft2, rfft2, rfft, irfft
from mpi4py import MPI

N = 2**8
comm = MPI.COMM_WORLD
num_processes = comm.Get_size()
rank = comm.Get_rank()
Np = N // num_processes

P1 = 2
P2 = num_processes // P1
N1 = N//P1
N2 = N//P2

# comm groups
commxz = comm.Split(rank/P1)
commxy = comm.Split(rank%P1)

xzrank = commxz.Get_rank()
xyrank = commxy.Get_rank()

# split into N1 x N2 pieces
X = np.mgrid[xzrank*N1:(xzrank+1)*N1, xyrank*N2:(xyrank+1)*N2, :N].astype(float)*2*np.pi/N
Uc_hat_z = np.empty((N1, N2, N//2+1), dtype=complex)
Uc_hat_z2 = np.empty((N1, N2, N//2), dtype=complex)
Uc_hat_x = np.empty((N, N2, N1//2), dtype=complex)
Uc_hat_xr = np.empty((N, N2, N1//2), dtype=complex)
Uc_hat_y = np.empty((N2, N, N1//2), dtype=complex)
U = np.empty((N1, N2, N))
U_hat = np.empty((N2, N, N1//2), dtype=complex)
U_approx = np.empty((N1, N2, N))

U = np.sin(X[0])*np.cos(X[1])*np.cos(X[2])

def print_(*args, comm_arg=MPI.COMM_WORLD, checkRank=0, **kwargs):
    """function to print arguments only in one core (default rank=0 in that comm group)"""
    if comm_arg.Get_rank() == checkRank:
        print(*args, **kwargs)


try:
    from pyfftw.interfaces.numpy_fft import fft, ifft, irfft2, rfft2, rfft, irfft
    import pyfftw
    pyfftw.interfaces.cache.enable()
    fftw_status = 'Using pyfftw'
except ImportError:
    fftw_status = 'Using numpy.fft'

print_(fftw_status)
print_('real space data shape: ', U.shape)
print_('k-space data shape: ', U_hat.shape)
print_('cores: ', num_processes)


def fftn_mpi(u, fu):
    Uc_hat_z[:] = rfft(u, axis=2)
    Uc_hat_x[:] = np.moveaxis(Uc_hat_z[:, :, :-1].reshape((N1, N2, P1, N1//2)), 2, 0).reshape(Uc_hat_x.shape)
    commxz.Alltoall([Uc_hat_x, MPI.DOUBLE_COMPLEX], [Uc_hat_xr, MPI.DOUBLE_COMPLEX])
    Uc_hat_x[:] = fft(Uc_hat_xr, axis=0)
    # Uc_hat_y[:] = np.moveaxis(Uc_hat_x.reshape((P2, N2, N2, N1/2)), 2, 1).reshape(Uc_hat_y.shape)
    commxy.Alltoall([Uc_hat_x, MPI.DOUBLE_COMPLEX], [Uc_hat_xr, MPI.DOUBLE_COMPLEX])
    Uc_hat_y[:] = np.moveaxis(Uc_hat_xr.reshape((P2, N2, N2, N1//2)), 1, 0).reshape(Uc_hat_y.shape)
    fu[:] = fft(Uc_hat_y, axis=1)
    return fu

def ifftn_mpi(fu, u):
    Uc_hat_y[:] = ifft(fu, axis=1)
    Uc_hat_xr[:] = np.moveaxis(Uc_hat_y.reshape((N2, P2, N2, N1//2)), 1, 0).reshape(Uc_hat_xr.shape)
    commxy.Alltoall([Uc_hat_xr, MPI.DOUBLE_COMPLEX], [Uc_hat_x, MPI.DOUBLE_COMPLEX])
    Uc_hat_xr[:] = ifft(Uc_hat_x, axis=0)
    commxz.Alltoall([Uc_hat_xr, MPI.DOUBLE_COMPLEX], [Uc_hat_x, MPI.DOUBLE_COMPLEX])
    Uc_hat_z2[:] = np.moveaxis(Uc_hat_x.reshape((P1, N1, N2, N1//2)), 0, 2).reshape(Uc_hat_z2.shape)
    u = irfft(Uc_hat_z, axis=2)
    return u


t0 = time()
U_hat = fftn_mpi(U, U_hat)
t1 = time()
U_approx[:] = ifftn_mpi(U_hat, U)
t2 = time()


k1 = comm.reduce(np.sum(U*U))
k2 = comm.reduce(np.sum(U_approx*U_approx))
dt1 = comm.reduce(t1-t0, op=MPI.MAX)
dt2 = comm.reduce(t2-t1, op=MPI.MAX)


if rank == 0:
    print_('\nTesting Results:')
    print_("Longest Thread Duration: Forward= ", round(dt1, 3), 's  Inverse= ', round(dt2, 3), 's')
    print_('Norm before = ', k1)
    print_('Norm after = ', k2)
    print_(f'Difference = {(k1-k2)}')
