from numpy import *
from numpy . fft import fftfreq , fft , ifft , \
    irfft2 , rfft2 , irfftn , rfftn
from mpi4py import MPI
try :
    from pyfftw . interfaces . numpy_fft \
    import fft , ifft , irfft2 , rfft2 , \
    irfftn , rfftn
except ImportError :
    pass # Rely on numpy . fft r o u t i n e s
# Get some MPI
comm = MPI.COMM_WORLD
num_processes = comm.Get_size ()
rank = comm.Get_rank ()

M = 6
N = 2**M
L = 2*pi
X = mgrid[:N,:N,:N].astype(float)*L/N

Nf = N/2+1
kx = ky = fftfreq(N,1./N).astype(int)
kz = kx[:Nf].copy();
kz[-1] *= -1
K= array(meshgrid(kx,ky,kz,indexing='ij'),dtype=int)
K2 = sum(K*K, 0,dtype=int)
K_over_K2 = K.astype(float) / where( K2==0,1,K2).astype(float)

U = empty((3,N,N,N),dtype=float)
U_hat = empty((3,N,N,Nf),dtype=complex)
P = empty((N,N,N),dtype=float)
P_hat = empty((N,N,Nf),dtype=complex)
curl = empty((3,N,N,N),dtype=float)

def fftn_mpi(u,fu):
    """ FFT of u in three directions . """
    if num_processes == 1 :
        fu[:] = rfftn(u,axes=(0,1,2))
    return fu
def ifftn_mpi(fu,u) :
    """ Inverse FFT of fu in three
    directions . """
    if num_processes == 1 :
        u[:] = irfftn( fu , axes =( 0 ,1 , 2 ) )
    return u
# Usage
# print(U)
# print(U_hat)
# U_hat = fftn_mpi (U , U_hat )
# U = ifftn_mpi ( U_hat , U )
