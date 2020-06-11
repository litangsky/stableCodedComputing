# -*- coding: utf-8 -*-
##########################################################################################
# Code for complex Vandemonde method
# 1) check the computation time in worker nodes and decoding time in master node
# 2) check the accuracy of decoding (MSE) Note: We did not add noise in AWS so MSE should be much lower than we add the noise

# The dimension of matrix is matrix A (dim t*(b*k)) and vector x (dim t)
# We want to trail it multiple times. Therefore we consider a "matrix" x (dim t*sample)
# The number of worker nodes is n
# The threshold is k
##########################################################################################
# Note: The decoding time and computation time are very small if we experienment matrix A times vector x.
#       To garantee we could have a more accurate decoding/computation time, we experiment
#       matrix A times a "matrix" x (dim t * sample) and get decoding/computation time.
#       Finally we output the actual decoding/computation time by dividing "sample".
##########################################################################################

from matVecMulAWS import genRandMat, pickWorkers, getCompVandEncodingMat, encodeMat, getMultResult, getMSE
from matVecMulAWS import threadSleep, transmitSubMat2Workers, transmitVector2Workers, transmitStragglersInfo, getIsStraggler, receiveMatrixFromMaster, receiveVectorFromMaster, sendRstToMaster, getResultFromWorkers

from mpi4py import MPI
import numpy as np

# Change to True for more accurate timing, sacrificing performance
barrier = True    
straggling_time = 5

#matrix A (size (s*qu)*t) and vector x(length t)
n = 31 # number of total workers 
k = 29 # number of active workers(threshold)
b = 680 # row block size
t = 28000 # number of rows of matrix A(length of x)
q = 1 # embedding matrix size
sample = 700 # the number of trails (number of columns of "matrix" x)

comm = MPI.COMM_WORLD
if comm.size != n + 1:
    print("The number of MPI processes mismatches the number of workers.")
    comm.Abort(1)
    
if comm.rank == 0:
    # Master
    print("Matrix-vector multiplication: Complex Vandemonde method")
    print("n=%d workers, k=%d treashold, perform %d trials" % (n, k, sample))
    print("matrix A row=%d, column=%d" % (t, b*k))
    
    # Randomly choose k out of n workers, the rest of workers are staggelers
    workersSet = pickWorkers(k, n)
    
    # Transmit the straggler info to master node.
    transmitStragglersInfo(comm, n, workersSet)
    
    # Create random matrices and vectors
    print("Master starts to generate matrix")
    A = genRandMat(-50, 50, [t, k*b*q])
    x = genRandMat(-50, 50, [t, sample])
    print("Master finishes generating matrix")
    
    # Get the encoding matrix(encoding A only)
    G = getCompVandEncodingMat(k, n)
    
    # Encode A
    Aencode = encodeMat(A, G, b, k, n, q)
    
    print("Master starts to transmit assignments to workers")
    transmitSubMat2Workers(comm, Aencode, n)
    transmitVector2Workers(comm, x, n)
    print("Master finishes transmitting assignments to workers")

    # Optionally wait for all workers to receive their submatrices, for more accurate timing
    if barrier:
        comm.Barrier()

    # Master node decodes the result        
    Cdecoded = getResultFromWorkers(comm, n, k, G, b, sample, q)
    C = getMultResult(A, x, b, sample, k, q)
    
    # Compare the result (MSE)   
    getMSE(C, Cdecoded)
    
else:
    # Determine whether the worker node is a straggler
    isStraggler = getIsStraggler(comm, k, n)
    
    # Receive the submatrix and vector from master node
    print("Slave %d starts to receive the info from master node"%comm.rank)
    matFromMaster = receiveMatrixFromMaster(comm, (t, b*q), np.complex128)
    vecFromMaster = receiveVectorFromMaster(comm, (t, sample))
    print("Slave %d finish receiving the info"%comm.rank)

    if isStraggler:
        threadSleep(straggling_time)

    #This is to prevent segmentation fault for large matrices
    if barrier:
        comm.Barrier()    
    
    # Send the result to master node
    print("Slave %d starts to compute and send the result back"%comm.rank)
    sendRstToMaster(comm, matFromMaster, vecFromMaster)
    print("Slave %d finishes computing and sending the result back"%comm.rank)



