# -*- coding: utf-8 -*-
##########################################################################################
# Code for proposed Vandermonde embeded by circulant matrix method
# 1) check the average/worst condition number for encoding matrix
# 2) check the accuracy of decoding (MSE) when adding the Gaussian noise to the worker node

# Note: since the encoding matrix in circulant matrix method is not full rank
# checking the conditio number is different from other methods()
# We check the worst and average condition number of the full rank block matrix of B
# 1) B is the block diagnoal matrix obtained by eigenvalue decomposition of encoding matrix G 
# 2) only the first block matrix in B is singular
# function "checkCondNumberForCirEmbed" is to check the worst and average condition number

# The dimension of matrix is matrix A (dim t*(b*k*q)) and vector x (dim t)
# We want to trail it multiple times. Therefore we consider a "matrix" x (dim t*sample)
# The number of worker nodes is n
# The threshold is k
##########################################################################################

from matVecMul import addGaussianNoise, checkCondNumberForCirEmbed, getCirEncodingMat, genRandMat, getMultResult, encodeMat, pickWorkers, transmitSubMat2Workers, computeWorker, decodeResult, getMSE, decodeResultFreqCirEmbed, createFakeSymbol, encodeMatForCirEmbed
import numpy as np

#########################################################
# matrix A (dim t*(b*k*q)) and vector x (dim t)
# We want to run the trail multiple times to get the more accurate result then we consider a matrix-like vector x (size t*sample)
n = 31 # number of total workers 
k = 29 # number of active workers(threshold)
b = 22 # column block size of A
t = 2800 # number of rows of matrix A(length of x)
q = 31 # embedding matrix size
sample = 1 # the number of trails (number of columns of "matrix" x)
SNR = 80 # SNR of Gaussian noise adding to the matrix in slave node
#########################################################
print("Matrix-vector multiplication: Proposed Vandermonde embeded by circulant matrix method")
print("n=%d workers, k=%d treashold, SNR=%d db noise, perform %d trials" % (n, k, SNR, sample))
print("matrix A row=%d, column=%d" % (t, b*k*(q-1)))
# Get the encoding matrix for A
# Note: circulant embed method will return encoding matrix G 
# and B which is the block diagnoal matrix obtained by eigenvalue decomposition of encoding matrix G 
G, B = getCirEncodingMat(k, n, q)

checkCondNumberForCirEmbed(B, q, k, n)

# Create random matrices and vector 
A = genRandMat(-50, 50, [t, k*b*q])
A = createFakeSymbol(A, k, b, q)

x = genRandMat(-50, 50, [t, sample])

# Get the correct result
C = getMultResult(A, x, b, sample, k, q)

# Encode A to Aencode
Aencode = encodeMatForCirEmbed(A, G, b, k, n, q)

# Randomly pick k out n worker nodes. Rest are stagglers
workersSet = pickWorkers(k, n)

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
Cencode = np.asarray(Cencode)
Cresult = decodeResultFreqCirEmbed(Cencode, n, B, workersSet, b, sample, q)

# Compare the result (MSE)
getMSE(C, Cresult)