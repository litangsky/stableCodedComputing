# -*- coding: utf-8 -*-
##########################################################################################
# Code for proposed method (Vandermonde matrix embedded by rotation matrix) in AWS
# 1) check the computation time in worker nodes and decoding time in master node
# 2) check the accuracy of decoding (MSE) Note: We did not add noise in AWS

# The dimension of matrix is matrix A (dim t*(kA*bA*q)) and matrix B (dim t*(kB*bB*q))
# The number of worker nodes is n
# The threshold is k = kA*kB
##########################################################################################

from mpi4py import MPI
from matMatMulAWS import pickWorkers, transmitStragglersInfo, randMatGenerator, getRotEmbedEncodingMatOfA, getRotEmbedEncodingMatOfB, getEncodingMat, encodeMat, threadSleep, sendRstToMaster
from matMatMulAWS import transmitSubMatOfA2Workers, transmitSubMatOfB2Workers, getResultFromWorkers, getMultResult, getMSE, getIsStraggler, receiveMatrixOfAFromMaster, receiveMatrixOfBFromMaster

import numpy as np
#########################################################
#########################################################
# matrix A (dim t*(kA*bA*q)) and matrix B (dim t*(kB*bB*q))
n = 31 # number of total workers 
kA = 4 # 1/kA storage fraction of A
kB = 7 # 1/kB storage fraction of B
bA = 1000 # col block size of A
bB = 600 # col block size of B
q = 2 # embedding matrix size
t = 14000 # size of row (A and B)
k = kA*kB # recovery threshold
#######################################################

#######################################################
comm = MPI.COMM_WORLD
barrier = True    
straggling_time = 10

if comm.size != n + 1:
    print("The number of MPI processes mismatches the number of workers.")
    comm.Abort(1)
    
if comm.rank == 0:
    # Master
    print("Matrix-matrix multiplication: Proposed Vandermonde matrix embedded by rotation matrix")
    print("n=%d workers, k=%d treashold" % (n, kA*kB))
    print("matrix A row=%d, column=%d, matrix B row=%d, column=%d" % (t, kA*q*bA, t, kB*q*bB))    

    # Randomly choose u out of n workers, the rest of workers are staggelers
    workers = pickWorkers(k, n)

    # broadcast the stragglers information to all of workers
    transmitStragglersInfo(comm, n, workers)
    
    # Create random matrices and vectors
    A = randMatGenerator(-50, 50, [t, bA*kA*q])
    B = randMatGenerator(-50, 50, [t, bB*kB*q])

    # Get the encoding matrix(encoding A only)
    GA = getRotEmbedEncodingMatOfA(kA, kB, n)
    GB = getRotEmbedEncodingMatOfB(kA, kB, n)
    G = getEncodingMat(GA, GB, k, n, q)

    # Encode A and B
    print("Master starts to encode")
    Aencode = encodeMat(A, GA, bA, kA, n, q)
    Bencode = encodeMat(B, GB, bB, kB, n, q) 
    print("Master finishes encoding")

    # Transmit corresponding matrix and vector to each worker nodes
    print("Master starts to transmit assignments to workers")
    transmitSubMatOfA2Workers(comm, Aencode, n, bA)
    transmitSubMatOfB2Workers(comm, Bencode, n, bB)
    print("Master finishes transmitting assignments to workers")
    
    # Optionally wait for all workers to receive their submatrices, for more accurate timing
    if barrier:
        comm.Barrier()

    # Master node decodes the result        
    Cdecoded = getResultFromWorkers(comm, n, k, G, bA, bB, kA, kB, q)

    C = getMultResult(A, B, bA, bB, kA, kB, q)
    
    # Compare the result (MSE)       
    getMSE(C, Cdecoded)
    
else:
    # Determine whether the worker node is a straggler
    isStraggler = getIsStraggler(comm, k, n)
    
    # Receive the submatrix and vector from master node
    print("Worker %d starts to receive matrix from masters"%comm.rank)
    matOfAFromMaster = receiveMatrixOfAFromMaster(comm, (t, bA*q), np.float64)
    matOfBFromMaster = receiveMatrixOfBFromMaster(comm, (t, bB*q), np.float64)
    print("Worker %d finish receiving matrix from masters"%comm.rank)

    if isStraggler:
        threadSleep(straggling_time)

    #This is to prevent segmentation fault for large matrices
    if barrier:
        comm.Barrier()  
        
    # Send the result to master node
    sendRstToMaster(comm, matOfAFromMaster, matOfBFromMaster, bA, bB, q)

