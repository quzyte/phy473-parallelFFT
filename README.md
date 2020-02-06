# phy473-parallelFFT
# Install Required libraries: 
pip install numpy mpi4y && 
sudo apt install libopenmpi-dev

# Usage
mpi run -np (number of core) python TG.py
eg: mpi run -np 8 python TG.py
