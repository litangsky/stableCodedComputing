# -*- coding: utf-8 -*-
##########################################################################################
# Code for proposed method (Vandermonde matrix embedded by rotation matrix)
# 1) check the average/worst condition number for encoding matrix
# 2) check the accuracy of decoding (MSE) when adding the Gaussian noise (SNR) to the worker node

# The dimension of matrix is matrix A (dim t*(kA*bA*q)) and matrix B (dim t*(kB*bB*q))
# The number of worker nodes is n
# The threshold is k = kA*kB
##########################################################################################

from matMatMul import addGaussianNoise, getRotEmbedEncodingMatOfA, getRotEmbedEncodingMatOfB, getEncodingMat, randMatGenerator, getMultResult, encodeMat, pickWorkers, transmitSubMat2Workers, computeWorker, reshapeReceivedInfo, decodeResult, getMSE, findWorkerSet2WorstCond
import numpy as np
#########################################################
#matrix A (dim t*(kA*bA*q)) and matrix B (dim t*(kB*bB*q))
n = 11 # number of total workers 
kA = 4 # 1/kA storage fraction of A
kB = 2 # 1/kB storage fraction of B
bA = 100 # col block size of A
bB = 100 # col block size of B
q = 2 # embedding matrix size
t = 400 # size of row (A and B)
k = kA*kB # recovery threshold
SNR = 1000000 # SNR of Gaussian noise adding to the matrix in slave node
#######################################################
print("Matrix-matrix multiplication: Proposed Vandermonde matrix embedded by rotation matrix")
print("n=%d workers, k=%d treashold, SNR=%d db noise" % (n, kA*kB, SNR))
print("matrix A row=%d, column=%d, matrix B row=%d, column=%d" % (t, kA*q*bA, t, kB*q*bB))

# Get the encoding matrix for A and B
GA = getRotEmbedEncodingMatOfA(kA, kB, n, q)
GB = getRotEmbedEncodingMatOfB(kA, kB, n, q)
G = getEncodingMat(GA, GB, k, n, q)

# Create random matrices
A = randMatGenerator(0, 10000, [t, bA*kA*q])
B = randMatGenerator(0, 10, [t, bB*kB*q])
entryUpperBound1 = 10000
entryUpperBound2 = 10
A1 = np.random.randint(0, entryUpperBound1, [t, (int)(bA*kA*q/2)])
A2 = np.random.randint(0, entryUpperBound2, [t, (int)(bA*kA*q/2)])
A = np.concatenate((A1, A2), axis = 1)

B1 = np.random.randint(0, entryUpperBound1, [t, (int)(bB*kB*q/2)])
B2 = np.random.randint(0, entryUpperBound2, [t, (int)(bB*kB/2)])
B = np.concatenate((B1, B2), axis = 1)

A1 = np.ones([(int)(t/2), bA*kA*q]) * entryUpperBound1
A1 = A1.astype(np.int)
#A2 = np.random.randint(0, entryUpperBound2, [t, (int)(bA*uA/2)])
A2 = np.ones([(int)(t/2), bA*kA*q])
A2 = A2.astype(np.int)
A = np.concatenate((A1, A2), axis = 0)

#B1 = np.random.randint(0, entryUpperBound1, [t, (int)(bB*uB/2)])
B1 = np.ones([(int)(t/2), bB*kB*q])
B1 = B1.astype(np.int)
#B2 = np.random.randint(0, entryUpperBound2, [t, (int)(bB*uB/2)])
B2 = np.ones([(int)(t/2), bB*kB*q]) * entryUpperBound1
B2 = B2.astype(np.int)

B = np.concatenate((B1, B2), axis = 0)
#A = np.random.uniform(0, 1, [t, bA*kA*q])
#B = np.random.uniform(0, 1, [t, bB*kB*q])
#A = np.ones([t, bA*kA*q]) * 0.0055555
#B = np.ones([t, bB*kB*q]) * 0.0055555

# Get the correct result
C = getMultResult(A, B, bA, bB, kA, kB, q)

# Encode A and B to Aencode and Bencode
Aencode = encodeMat(A, GA, bA, kA, n, q)
Bencode = encodeMat(B, GB, bB, kB, n, q) 

# Randomly pick k out n worker nodes. Rest are stagglers
#workers = pickWorkers(k, n)
# Find the set of non-staggler worker nodes corresponding to the worst condition number of encoding matrix
workers = findWorkerSet2WorstCond(G, k, n, q)

# Transmit the submatrix to worker nodes and worker nodes compute their assignments
Cencode = [None] * k
matsTransmitOfA = transmitSubMat2Workers(Aencode, n, bA)
matsTransmitOfB = transmitSubMat2Workers(Bencode, n, bB)
for i in range(k):
    matsTransmitOfANoise = addGaussianNoise(matsTransmitOfA[workers[i]], SNR)
    matsTransmitOfBNoise = addGaussianNoise(matsTransmitOfB[workers[i]], SNR)
    Cencode[i] = computeWorker(matsTransmitOfANoise, matsTransmitOfBNoise, bA, bB, q)

# Master node collects the result and decode it
CencodeReshape = reshapeReceivedInfo(Cencode, bA, bB, kA, kB, n, q)
Cresult = decodeResult(CencodeReshape, n, G, workers, q)

# Compare the result (MSE)
getMSE(C, Cresult)
