# -*- coding: utf-8 -*-

##########################################################################################
# Code for complex Vandemonde method (complex polynomial codes)
# 1) check the average/worst condition number for encoding matrix
# 2) check the accuracy of decoding (MSE) when adding the Gaussian noise to the worker node

# Polynomial codes: an optimal design for high-dimensional coded matrix multiplication.
# In Proc. of Adv. in Neural Inf. Proc. Sys. (NIPS), pp. 4403–4413, 2017.
# Yu, Q., Maddah-Ali, M. A., and Avestimehr, A. S.

# The dimension of matrix is matrix A (dim t*(sA*uA)) and matrix B (dim t*(sB*uB)) (q=1)
# The number of worker nodes is n
# The threshold is k = kA*kB
##########################################################################################


from matMatMul import addGaussianNoise, getCompVandEncodingMatOfA, getCompVandEncodingMatOfB, getEncodingMat, randMatGenerator, getMultResult, encodeMat, pickWorkers, transmitSubMat2Workers, computeWorker, reshapeReceivedInfo, decodeResult, getMSE, findWorkerSet2WorstCond

#########################################################
#matrix A (dim t*(kA*q*bA)) and matrix B (dim t*(kA*q*bA))
n = 31 # number of total workers 
kA = 4  # 1/kA storage fraction of A
kB = 7 # 1/kB storage fraction of B
bA = 200 # col block size of A
bB = 120 # col block size of B
q = 1 # embedding matrix size
t = 1400 # number of rows (A and B)
k = kA*kB # recovery threshold
SNR = 80 # SNR of Gaussian noise adding to the matrix in slave node
#######################################################
print("Matrix-matrix multiplication: complex Vandemonde method (complex polynomial codes)")
print("n=%d workers, k=%d treashold, SNR=%d db noise" % (n, kA*kB, SNR))
print("matrix A row=%d, column=%d, matrix B row=%d, column=%d" % (t, kA*q*bA, t, kB*q*bB))
# Get the encoding matrix for A and B
GA = getCompVandEncodingMatOfA(kA, kB, n)
GB = getCompVandEncodingMatOfB(kA, kB, n)
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