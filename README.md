# phy473-parallelFFT
3D Multidimensional FFTs in parallel using MPI in Python3

# Install Required libraries: 
pip install numpy mpi4y <br/>
sudo apt install libopenmpi-dev
and some others

# Usage
mpi run -np (number of core) python3 demo_slab.py <br/>
eg: mpi run -np 8 python3 demo_slab.py

or for FFT using pencil decomposition:
eg: mpi run -np 8 python3 demo_pencil.py

