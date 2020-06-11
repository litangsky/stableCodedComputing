# -*- coding: utf-8 -*-
##########################################################################################
# Code for proposed Vandermonde embeded by rotation matrix method
# 1) check the average/worst condition number for encoding matrix
# 2) check the accuracy of decoding (MSE) when adding the Gaussian noise to the worker node


# The dimension of matrix is matrix A (dim t*(b*k)) and vector x (dim t)
# We want to trail it multiple times. Therefore we consider a "matrix" x (dim t*sample)
# The number of worker nodes is n
# The threshold is k
##########################################################################################

from matVecMul import addGaussianNoise, getRotEmbedEncodingMat, genRandMat, getMultResult, pickWorkers, transmitSubMat2Workers, computeWorker, decodeResult, getMSE, findWorkerSet2WorstCond, encodeMat
import numpy as np

#########################################################
# matrix A (dim t*(b*k*q)) and vector x (dim t)
# We want to run the trail multiple times to get the more accurate result then we consider a matrix-like vector x (size t*sample)
n = 31 # number of total workers
k = 29  # number of active workers
b = 34 # column block size of A
t = 2800 # number of rows of matrix A(length of x)
q = 2 # embedding matrix size
sample = 1 # the number of trails (number of columns of "matrix" x)
SNR = 80 # SNR of Gaussian noise adding to the matrix in slave node
#########################################################
print("Matrix-vector multiplication: Proposed Vandermonde embeded by rotation matrix method")
print("n=%d workers, k=%d treashold, SNR=%d db noise, perform %d trials" % (n, k, SNR, sample))
print("matrix A row=%d, column=%d" % (t, b*k*q))

# Get the encoding matrix for A
G = getRotEmbedEncodingMat(k, n)

# Create random matrices and vector 
A = genRandMat(-50, 50, [t, k*b*q])
x = genRandMat(-50, 50, [t, sample])

# Get the correct result
C = getMultResult(A, x, b, sample, k, q)

# Encode A to Aencode 
Aencode = encodeMat(A, G, b, k, n, q)   
 
# Randomly pick k out n worker nodes. Rest are stagglers
#workers = pickWorkers(k, n)
# Find the set of non-staggler worker nodes corresponding to the worst condition number of encoding matrix
workersSet = findWorkerSet2WorstCond(G, k, n, q)

# 1) Transmit the submatrix to worker nodes. 
# 2) Add noise to matrix in worker nodes
# 3) Worker nodes compute their assignments
Cencode = [None] * k
matsTransmit = transmitSubMat2Workers(Aencode, n, b, q)

vecTransmitNoise = addGaussianNoise(x, SNR)
for i in range(k):
    matTransmitNoise = addGaussianNoise(matsTransmit[workersSet[i]], SNR)
    Cencode[i] = computeWorker(matTransmitNoise, vecTransmitNoise)

# Master node collects the result and decode it        
CencodeArray = np.asarray(Cencode)
Cresult = decodeResult(CencodeArray, n, G, workersSet, b, sample, q)

# Compare the result (MSE)
getMSE(C, Cresult)