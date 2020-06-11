# -*- coding: utf-8 -*-
########################################################################################################
# Code for Random khatri-rao-product codes method
# 1) check the average/worst condition number for encoding matrix
# 2) check the accuracy of decoding (MSE) when adding the Gaussian noise to the worker node

# Note: this paper is a random method. Therefore we try this method #numTrails time and find the best encoding matrix

# Random khatri-rao-product codes for numericallystable distributed matrix multiplication.
# In 2019 57th Annual Allerton Conference on Communication, Control, and Computing (Allerton),
# Subramaniam, A. M., Heidarzadeh, A., and Narayanan, K. R.

# The dimension of matrix is matrix A (dim t*(kA*bA)) and matrix B (dim t*(kB*bB)) (q=1)
# The number of worker nodes is n
# The threshold is k = kA*kB
########################################################################################################

from matMatMul import addGaussianNoise, getRandomKarRaoEncodingMat, getEncodingMat, randMatGenerator, getMultResult, encodeMat, pickWorkers, transmitSubMat2Workers, computeWorker, reshapeReceivedInfo, decodeResult, getMSE, findWorkerSet2WorstCond

#########################################################
#matrix A (dim t*(kA*bA*q)) and matrix B (dim t*(kB*bB*q))
n = 31 # number of total workers 
kA = 4  # 1/kA cache fraction of A
kB = 7 # 1/kB cache fraction of B
bA = 200 # col block size of A
bB = 120 # col block size of B
q = 1 # embedding matrix size
t = 1400 # size of row (A and B)
k = kA*kB # recovery threshold
SNR = 80 # SNR of Gaussian noise adding to the matrix in slave node
numTrails = 200 # we try #numTrails to find the best cond number 

#######################################################
print("Matrix-matrix multiplication: Random khatri-rao-product codes method")
print("n=%d workers, k=%d treashold, SNR=%d db noise, number of trials to get the best encoding matrix=%d" % (n, kA*kB, SNR, numTrails))
print("matrix A row=%d, column=%d, matrix B row=%d, column=%d" % (t, kA*q*bA, t, kB*q*bB))

# Get the encoding matrix for A and B
GA, GB = getRandomKarRaoEncodingMat(kA, kB, n, numTrails)
G = getEncodingMat(GA, GB, k, n, q)

# Create random matrices 
A = randMatGenerator(-50, 50, [t, bA*kA*q])
B = randMatGenerator(-50, 50, [t, bB*kB*q])

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
