## Run with mpirun -np <your_desired_number_of_cores> python3 <this_file_name>
## This baby can handle any number of cores -- odd, even, Mersenne prime, Graham's number ...
## Just make sure the length of the array is larger than <your_desired_number_of_cores>

from copy import deepcopy
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

a = [[0, 1, 2, 3, 4, 5],
     [6, 7, 8, 9, 10, 11],
     [12, 13, 14, 15, 16, 17],
     [18, 19, 20, 21, 22, 23],
     [24, 25, 26, 27, 28, 29]]


grid = np.mgrid[0:3, 0:4, 0:5]
b = grid[0]*20+grid[1]*5+grid[2]
# array([[[ 0,  1,  2,  3,  4],
#         [ 5,  6,  7,  8,  9],
#         [10, 11, 12, 13, 14],
#         [15, 16, 17, 18, 19]],
# 
#        [[20, 21, 22, 23, 24],
#         [25, 26, 27, 28, 29],
#         [30, 31, 32, 33, 34],
#         [35, 36, 37, 38, 39]],
# 
#        [[40, 41, 42, 43, 44],
#         [45, 46, 47, 48, 49],
#         [50, 51, 52, 53, 54],
#         [55, 56, 57, 58, 59]]])

## simple test case (for understanding):
A = np.array(a)
## complex test case (for testing):
# A = np.array(b)

## the dimensions to swap; try different combinations - (0, 1), (1, 0), (1, 2), (2, 1) ... as long as the array has sufficient number of dimensions
dimPre, dimPost = 0, 1
lenPre, lenPost = A.shape[dimPre], A.shape[dimPost]


## compute split indices and sizes (format: (x, y, z) where x:begin_index, y:block_size, z:end_index)
indsPre = [( lenPre//size*i + min(i, lenPre%size), lenPre//size + (i < lenPre%size), lenPre//size*(i+1) + min(i+1, lenPre%size) ) for i in range(size)]
indsPost= [( lenPost//size*i + min(i, lenPost%size), lenPost//size + (i < lenPost%size), lenPost//size*(i+1) + min(i+1, lenPost%size) ) for i in range(size)]


## prepare data blocks for transfer
shapePre = list(A.shape)
shapePre[dimPre] = indsPre[rank][1]
shapePost = list(A.shape)
shapePost[dimPost] = indsPost[rank][1]
copy_indices = [slice(A.shape[i]) for i in range(A.ndim)]
copy_indices[dimPre] = slice(indsPre[rank][0], indsPre[rank][2])
APre = np.copy(A[tuple(copy_indices)])
APost = np.empty(shapePost, dtype=int)


## compute the datatypes for sending/receiving
sendSizes = deepcopy([deepcopy(shapePre) for _ in range(size)])
sendSubSizes = deepcopy(sendSizes)
sendStarts = [ [0] * APre.ndim for _ in range(size) ]
sendTypes = [[] for _ in range(size)]

recvSizes = deepcopy([deepcopy(shapePost) for _ in range(size)])
recvSubSizes = deepcopy(recvSizes)
recvStarts = [ [0] * APost.ndim for _ in range(size) ]
recvTypes = [[] for _ in range(size)]
for i in range(size):
    # for sending data, i.e., transfer from process rank to i
    sendSubSizes[i][dimPre], sendSubSizes[i][dimPost] = indsPre[rank][1], indsPost[i][1]
    sendStarts[i][dimPost] = indsPost[i][0]
    sendTypes[i] = MPI.INT64_T.Create_subarray(sendSizes[i], sendSubSizes[i], sendStarts[i]).Commit()
    # for receiving data, i.e., transfer from process i to rank
    recvSubSizes[i][dimPre], recvSubSizes[i][dimPost] = indsPre[i][1], indsPost[rank][1]
    recvStarts[i][dimPre] = indsPre[i][0]
    recvTypes[i] = MPI.INT64_T.Create_subarray(recvSizes[i], recvSubSizes[i], recvStarts[i]).Commit()


## execute the call routine
comm.Alltoallw([APre, [1]*size, [0]*size, sendTypes], [APost, [1]*size, [0]*size, recvTypes])

## print(f' -- Rank: {rank} -- \nInds Pre: \n{indsPre} \nInds Post: \n{indsPost} \n\nA Pre: \n{APre} \n\nSend sizes: {sendSubSizes} \nRecv sizes: {recvSubSizes} \nSend Starts: {sendStarts} \nRecv Starts: {recvStarts} \n\nA Post: \n{APost} \n\n\n\n')
print(f' -- Rank: {rank} -- \nBefore: \n{APre} \nAfter: \n{APost}\n\n\n')
