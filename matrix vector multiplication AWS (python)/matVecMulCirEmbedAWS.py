# -*- coding: utf-8 -*-
##########################################################################################
# Code for proposed Vandermonde embeded by circulant matrix method in AWS
# 1) check the computation time in worker nodes and decoding time in master node
# 2) check the accuracy of decoding (MSE) Note: We did not add noise in AWS

# The dimension of matrix is matrix A (dim t*(b*k*q)) and vector x (dim t)
# We want to trail it multiple times. Therefore we consider a "matrix" x (dim t*sample)
# The number of worker nodes is n
# The threshold is k
##########################################################################################
# Note: The decoding time and computation time are very small if we experienment matrix A times vector x.
#       To garantee we could have a more accurate decoding/computation time, we experiment
#       matrix A times a "matrix" x (dim t * sample) and get decoding/computation time.
#       Finally we output the actual decoding/computation time by dividing "sample".
##########################################################################################
from matVecMulAWS import genRandMat, pickWorkers, encodeMatForCirEmbed, getMultResult, getMSE, createFakeSymbol, getCirEncodingMat
from matVecMulAWS import threadSleep, transmitSubMat2Workers, transmitVector2Workers, transmitStragglersInfo, getIsStraggler, receiveMatrixFromMaster, receiveVectorFromMaster, sendRstToMaster, getResultFromWorkersCirEmbed

from mpi4py import MPI
import numpy as np

# Change to True for more accurate timing, sacrificing performance
barrier = True    
straggling_time = 5

#matrix A (dim t*(b*k*q)) and vector x (dim t)
n = 31 # number of total workers 
k = 29 # number of active workers(threshold)
b = 22 # column block size of A
t = 28000 # number of rows of matrix A(length of x)
q = 31 # embedding matrix size
sample = 700 # the number of trails (number of columns of "matrix" x)

comm = MPI.COMM_WORLD
if comm.size != n + 1:
    print("The number of MPI processes mismatches the number of workers.")
    comm.Abort(1)
    
if comm.rank == 0:
    # Master
    print("Matrix-vector multiplication: Proposed Vandermonde embeded by circulant matrix method")
    print("n=%d workers, k=%d treashold, perform %d trials" % (n, k, sample))
    print("matrix A row=%d, column=%d" % (t, b*k*(q-1)))
    
    # Randomly choose k out of n workers, the rest of workers are staggelers
    workersSet = pickWorkers(k, n)
    
    # Transmit the straggler info to master node.
    transmitStragglersInfo(comm, n, workersSet)
    
    # Create random matrices and vectors
    print("Master starts to generate matrix")
    A = genRandMat(-50, 50, [t, k*b*q])
    A = createFakeSymbol(A, k, b, q)
    
    x = genRandMat(-50, 50, [t, sample])
    print("Master finishes generating matrix")
    
    # Get the encoding matrix for A
    # Note: circulant embed method will return encoding matrix G 
    # and B which is the block diagnoal matrix obtained by eigenvalue decomposition 
    G, B = getCirEncodingMat(k, n, q)
    print("Master finishes generating generator matrix")

    # Encode A
    Aencode = encodeMatForCirEmbed(A, G, b, k, n, q)
    
    print("Master starts to transmit assignments to workers")
    transmitSubMat2Workers(comm, Aencode, n)
    transmitVector2Workers(comm, x, n)
    print("Master finishes transmitting assignments to workers")

    # Optionally wait for all workers to receive their submatrices, for more accurate timing
    if barrier:
        comm.Barrier()
     
    # Master node decodes the result
    Cdecoded = getResultFromWorkersCirEmbed(comm, n, k, B, b, sample, q)
    C = getMultResult(A, x, b, sample, k, q)
    
    # Compare the result (MSE)
    getMSE(C, Cdecoded)
   
else:
    # Worker node
    # Determine whether the worker node is a straggler
    isStraggler = getIsStraggler(comm, k, n)
    
    # Receive the submatrix and vector from mastelsr node
    print("Slave %d starts to receive the info from master node"%comm.rank)
    matFromMaster = receiveMatrixFromMaster(comm, (t, b*q), np.float64)
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

