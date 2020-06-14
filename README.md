# phy473-parallelFFT
3D Multidimensional FFTs in parallel using MPI in Python3  
  
# Install Required libraries: 
pip install numpy mpi4py  
sudo apt install libopenmpi-dev  
and some others  
  
# Usage
mpirun -np &lt;number_of_cores&gt; python3 &lt;filename&gt;
  
For FFT using slab decomposition:  
eg: mpirun -np 7 python3 slabw.py

For FFT using pencil decomposition:  
eg: mpirun -np 6 python3 pencilw.py

