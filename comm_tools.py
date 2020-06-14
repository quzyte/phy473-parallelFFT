import numpy as np
from copy import deepcopy
from numpy.fft import fft, ifft, rfft, irfft, rfft2, irfft2, fftfreq
from mpi4py import MPI

def getSplit(lendim, P):
    '''
    Function to return the split indices, and block sizes when a dimension of length lendim is 
    divided into P blocks. (P need not divide lendim)
    --
    Arguments:
        lendim : Length of the dimension to be split (integer)
        P : Number of parts for the dimension to be broken into
    Returns:
        split : a list of length P containing (start index, block size, end index) of all blocks
    Examples:
         - getSplit(4, 4) returns [(0, 1, 1), (1, 1, 2), (2, 1, 3), (3, 1, 4)]
         - getSplit(6, 4) returns [(0, 2, 2), (2, 2, 4), (4, 1, 5), (5, 1, 6)]

    Note: Not tested when P > lendim
    '''
    split = [( lendim//P*i + min(i, lendim%P), lendim//P + (i < lendim%P), lendim//P*(i+1) + min(i+1, lendim%P) ) for i in range(P)]
    return split



def distribute(A=np.empty(0), axis=0, comm=MPI.COMM_WORLD, root=0):
    '''
    Function to distribute numpy array (almost) equally among all nodes
    along an axis using MPI AlltoallW. Returns distributed array.
    --
    Arguments:
        A : The array to distribute (for the source node; optional dummy argument for other nodes)
        axis : The axis along which array is to be distributed (default 0).
        comm : MPI Communicator object (optional; default MPI.COMM_WORLD)
        root : Rank of source node to distribute array from (default 0)
    Returns:
        ADist : Distributed array.
    Examples:
         - if A has shape [5, 5, 5] and comm.Get_size() is 3, 
            the arrays returned, respectively, have shapes
            [2, 5, 5], [2, 5, 5], [1, 5, 5]
         - if A has shape [2, 5, 5], comm.Get_size() is 3 and axis=1,
            the arrays returned, respectively, have shapes
            [2, 2, 5], [2, 2, 5], [2, 1, 5]

    Note: Not tested when comm.Get_size() > A.shape[axis]
    '''
    P = comm.Get_size()
    rank = comm.Get_rank()
    dtype, shape = None, None
    if rank == root:
        assert A.shape is not (0,)
        dtype = A.dtype
        shape = list(A.shape)
    dtype = comm.bcast(dtype, root=root)
    shape = comm.bcast(shape, root=root)
    inds = getSplit(shape[axis], P)
    shapeDist = list(shape)
    shapeDist[axis] = inds[rank][1]
    dtypeMPI = MPI._typedict[np.dtype(dtype).char]  # corresponding MPI datatype


    ## compute the datatypes for sending/receiving
    sendStarts = [[0]*len(shape) for _ in range(P)]
    sendTypes = [[] for _ in range(P)]

    for i in range(P):
        # for sending data, i.e., transfer from process root to i
        shapei = list(shape)
        shapei[axis] = inds[i][1]
        sendStarts[i][axis] = inds[i][0]
        sendTypes[i] = dtypeMPI.Create_subarray(shape, shapei, sendStarts[i]).Commit()

    # for receiving data, i.e., transfer from process root to rank
    recvType = dtypeMPI.Create_subarray(shapeDist, shapeDist, [0]*len(shape)).Commit()

    ## execute the call routine
    ADist = np.empty(shapeDist, dtype)
    comm.Alltoallw([A, [rank==root]*P, [0]*P, sendTypes], [ADist, [0]*(root)+[1]+[0]*(P-1-root), [0]*P, [recvType]*P])

    return ADist



def accumulate(ADist, axis=0, comm=MPI.COMM_WORLD, root=0):
    '''
    Function to accumulate numpy array distributed among all nodes along 
    an axis using MPI Alltoallw. Returns accumulated array 
    in root process, and empty numpy placeholder at others.
    --
    Arguments:
        ADist : The distributed array
        axis : The axis along which array is distributed (default 0)
        comm : MPI Communicator object (optional; default MPI.COMM_WORLD)
        root : Rank of source node to accumulate array to (default 0)
    Returns:
        A : Accumulated array in root node and empty numpy placeholder in others
    Examples:
         - if axis=1, and the distributed arrays have shapes
            [2, 2, 5], [2, 2, 5], [2, 1, 5]
            then accumulated array has shape [2, 5, 5]
         - if the distributed arrays have shapes
            [2, 5, 5], [2, 5, 5], [1, 5, 5]
            then accumulated array has shape [5, 5, 5]
    '''
    P = comm.Get_size()
    rank = comm.Get_rank()
    shapeDist = list(ADist.shape)
    sizesDist = comm.allgather(shapeDist[axis])
    shape = list(shapeDist)
    shape[axis] = sum(sizesDist)
    inds = [ (0, sizesDist[0], sizesDist[0]) ] * P
    for i in range(1,P):
        inds[i] = (inds[i-1][2], sizesDist[i], inds[i-1][2]+sizesDist[i])

    dtypeMPI = MPI._typedict[ADist.dtype.char]  # corresponding MPI datatype


    ## compute the datatypes for sending/receiving
    recvStarts = [[0]*len(shape) for _ in range(P)]
    recvTypes = [[] for _ in range(P)]

    for i in range(P):
        # for receiving data, i.e., transfer from process i to root
        shapei = list(shape)
        shapei[axis] = inds[i][1]
        recvStarts[i][axis] = inds[i][0]
        recvTypes[i] = dtypeMPI.Create_subarray(shape, shapei, recvStarts[i]).Commit()

    # for sending data, i.e., transfer from process rank to root
    sendType = dtypeMPI.Create_subarray(shapeDist, shapeDist, [0]*len(shapeDist)).Commit()

    ## execute the call routine
    A = np.empty_like(ADist, shape=shape) if rank==root else np.empty(0)
    comm.Alltoallw([ADist, [0]*(root)+[1]+[0]*(P-1-root), [0]*P, [sendType]*P], [A, [rank==root]*P, [0]*P, recvTypes])

    return A



def redistribute(APre, axisPre=0, axisPost=1, comm=MPI.COMM_WORLD):
    '''
    Function to redistribute numpy array (almost) equally 
    among all nodes. Array is initally distributed along axisPre and 
    this function returns the array redistributed along axisPost.
    --
    Arguments:
        APre : The array to redistribute (initially distributed along axisPre)
        axisPre : The axis along which array is distributed (default 0)
        axisPost : The axis along which array is to be redistributed (default 1).
        comm : MPI Communicator object (default MPI.COMM_WORLD)
    Returns:
        APost : Redistributed array (distributed along axisPost)
    Examples:
         - if axisPre is 0, axisPost = 1, comm.Get_size() is 3,
            and initially arrays have shapes
            [2, 8, 6], [2, 8, 6], [1, 8, 6], then
            redistributed arrays will have shapes
            [5, 3, 6], [5, 3, 6], [5, 2, 6]

    Note: Not tested when comm.Get_size() > Apre.shape[axisPost]
    '''
    P = comm.Get_size()
    rank = comm.Get_rank()

    shapePre = list(APre.shape)
    sizesPre = comm.allgather(shapePre[axisPre])
    shape = list(shapePre)
    shape[axisPre] = sum(sizesPre)
    indsPre = [ (0, sizesPre[0], sizesPre[0]) ] * P
    for i in range(1,P):
        indsPre[i] = (indsPre[i-1][2], sizesPre[i], indsPre[i-1][2]+sizesPre[i])

    shapePost = list(shape)
    indsPost = getSplit(shape[axisPost], P)
    shapePost[axisPost] = indsPost[rank][1]

    dtypeMPI = MPI._typedict[APre.dtype.char]  # corresponding MPI datatype
    APost = np.zeros(shapePost, dtype=APre.dtype)


    ## compute the datatypes for sending/receiving
    sendSizes = deepcopy([deepcopy(shapePre) for _ in range(P)])
    sendSubSizes = deepcopy(sendSizes)
    sendStarts = [ [0] * APre.ndim for _ in range(P) ]
    sendTypes = [[] for _ in range(P)]

    recvSizes = deepcopy([deepcopy(shapePost) for _ in range(P)])
    recvSubSizes = deepcopy(recvSizes)
    recvStarts = [ [0] * APost.ndim for _ in range(P) ]
    recvTypes = [[] for _ in range(P)]

    for i in range(P):
        # for sending data, i.e., transfer from process rank to i
        sendSubSizes[i][axisPre], sendSubSizes[i][axisPost] = indsPre[rank][1], indsPost[i][1]
        sendStarts[i][axisPost] = indsPost[i][0]
        sendTypes[i] = dtypeMPI.Create_subarray(sendSizes[i], sendSubSizes[i], sendStarts[i]).Commit()

        # for receiving data, i.e., transfer from process i to rank
        recvSubSizes[i][axisPre], recvSubSizes[i][axisPost] = indsPre[i][1], indsPost[rank][1]
        recvStarts[i][axisPre] = indsPre[i][0]
        recvTypes[i] = dtypeMPI.Create_subarray(recvSizes[i], recvSubSizes[i], recvStarts[i]).Commit()

    ## execute the call routine
    comm.Alltoallw([APre, [1]*P, [0]*P, sendTypes], [APost, [1]*P, [0]*P, recvTypes])

    return APost
