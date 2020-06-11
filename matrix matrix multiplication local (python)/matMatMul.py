# -*- coding: utf-8 -*-

#from mpi4py import MPI
from numpy.linalg import inv
from numpy.linalg import matrix_power
from numpy.linalg import cond
from itertools import combinations 

import numpy as np
import random
import sys

# Generate a random matrix with dimension dim
def randMatGenerator(low, high, dim):
    mat = np.random.randint(low, high, dim)
    return mat

# Random pick k worker nodes from worker set. Rest of them are stragglers
def pickWorkers(threshold, numWorkers):
    workers = random.sample(set(range(numWorkers)), threshold)
    return workers

# add noise to matrix
def addGaussianNoise(mat, SNR):
    noisePower = 10**(-SNR/10)
    noise = np.random.randn(mat.shape[0], mat.shape[1]) * np.sqrt(noisePower)
    return mat + noise

# encode orginal matrix by generator matrix          
def encodeMat(mat, generatorMat, blockSize, u, numWorkers, embedSize):
    matSplit = np.split(mat, u*embedSize, axis=1)
    row = mat.shape[0]
    matEncoded = [None] * numWorkers 
    for j in range(numWorkers):
        encodeTemp = np.zeros([row, blockSize*embedSize], dtype=generatorMat.dtype)
        for k in range(embedSize):
            encodeTemp[:,k*blockSize:k*blockSize+blockSize] = sum([matSplit[i]*generatorMat[i][j*embedSize+k] for i in range(u*embedSize)])
        matEncoded[j] = encodeTemp
    return matEncoded

# transmit the corresponding encoded matrix to workers
def transmitSubMat2Workers(matEncoded, numWorkers, blockSize):
    matsTransmit = [None] * numWorkers
    for i in range(numWorkers):
        matsTransmit[i] = matEncoded[i]
    return matsTransmit

def reshapeReceivedInfo(matReceived, blockSizeA, blockSizeB, uA, uB, numWorkers, embedSize):
    square = embedSize * embedSize
    matReshape = np.zeros((blockSizeA*blockSizeB,square*uA*uB), dtype=matReceived[0].dtype)
    for i in range(uA*uB):
        matReshape[:, i*square:i*square+square] = matReceived[i]
    return matReshape

# decode the result 
# Note that here we only use trivial decoding algorithm for all of schemes
def decodeResult(matReceived, numWorkers, generatorMat, workersSet, embedSize): 
    threshold = len(workersSet)
    square = embedSize * embedSize
    G2Workers = np.zeros((threshold*square, threshold*square), dtype=generatorMat.dtype)
    for i in range(threshold):
        worker = workersSet[i]
        G2Workers[:, i*square:i*square+square] = generatorMat[:, worker*square:worker*square+square]
    # directly inverse G in time domain to get result
    Cdecode = np.matmul(matReceived, inv(G2Workers))
    return Cdecode
    
# Worker node computes its assignment
def computeWorker(A, B, sA, sB, embedSize):
    C = np.zeros((sA*sB,embedSize*embedSize), dtype=A.dtype)
    for i in range(embedSize):
        Atemp = A[:, i*sA:i*sA+sA]
        for j in range(embedSize):
            Btemp = B[:, j*sB:j*sB+sB]
            Ctemp = np.matmul(Atemp.T, Btemp).ravel()
            Ctemp1 = np.reshape(Ctemp, (sA*sB, 1))
            C[:, i*embedSize+j:i*embedSize+j+1] = Ctemp1
    return C

# Get the correct result. Will compare it with the decoded result finally.
def getMultResult(A, B, blockSizeA, blockSizeB, uA, uB, embedSize):
    C = np.matmul(A.T, B)
    #reshape C
    square = embedSize * embedSize
    Creshape = np.zeros([blockSizeA*blockSizeB, uA*uB*square])
    for i in range(uA*embedSize):
        for j in range(uB*embedSize):
            Creshape[:, embedSize*uB*i+j:embedSize*uB*i+j+1] = np.reshape(C[i*blockSizeA:i*blockSizeA+blockSizeA, j*blockSizeB:j*blockSizeB+blockSizeB], (blockSizeA*blockSizeB,1))
    return Creshape

# Based on encoding matrix A and encoding matrix of B, get the encoding matrix G
# G will be used in decoding and checking the average/worst condition number 
def getEncodingMat(GA, GB, threshold, numWorkers, embedSize):
    square = embedSize * embedSize
    G = np.zeros((threshold*square, numWorkers*square), dtype=GA.dtype)
    for i in range(numWorkers):
        G[:, i*square:i*square+square] = np.kron(GA[:,i*embedSize:i*embedSize+embedSize], GB[:,i*embedSize:i*embedSize+embedSize])
    return G

# Find the the of worker nodes that correspond to the worst cond number
# and report the worst and average condition number
def findWorkerSet2WorstCond(generatorMat, threshold, numWorkers, embedSize):
    comb = list(combinations(range(numWorkers), threshold))
    condCollect = []
    for i in range(len(comb)):
        square = embedSize * embedSize
        G2Workers = np.zeros((threshold*square, threshold*square), dtype=generatorMat.dtype)
        for j in range(threshold):
            worker = comb[i][j]
            G2Workers[:, j*square:j*square+square] = generatorMat[:, worker*square:worker*square+square]
        condCollect.append(cond(G2Workers))
    print("Average condition number is %f"%np.mean(condCollect))
    worstCond = np.max(condCollect)
    print("Worst condition number is %f"%worstCond)
    workerSet2WorstCond = comb[condCollect.index(worstCond)]
    return workerSet2WorstCond

# Find the the of worker nodes that correspond to the worst cond number
def findWorstCond(generatorMat, threshold, numWorkers, embedSize):
    comb = list(combinations(range(numWorkers), threshold))
    condCollect = []
    for i in range(len(comb)):
        square = embedSize * embedSize
        G2Workers = np.zeros((threshold*square, threshold*square), dtype=generatorMat.dtype)
        for j in range(threshold):
            worker = comb[i][j]
            G2Workers[:, j*square:j*square+square] = generatorMat[:, worker*square:worker*square+square]
        condCollect.append(cond(G2Workers))
    worstCond = np.max(condCollect)
    return worstCond

# Compare the correct result to the decoded result
def getMSE(orginalRst, decodeRst):
    mse = np.sum(abs((orginalRst - decodeRst)**2))/np.sum(abs(orginalRst**2))
    print("MSE is %s"%mse)
    return mse

#############################################################################################################
# The following function is to create encoding matrix for different methods

# Encoding matrix of matrix A in Vandermonde embedded by rotation matrix scheme
def getRotEmbedEncodingMatOfA(uA, uB, numWorkers, embedSize):
    alpha = np.exp(2*1j*np.pi/numWorkers) #n-th root of unity
    rotMat = [[np.real(alpha), -np.imag(alpha)], [np.imag(alpha), np.real(alpha)]]
    GA = np.zeros((uA*embedSize, numWorkers*embedSize)) #generator matrix in time domain: a vandmond like matrix
    for i in range(uA):
        for j in range(numWorkers):
            #generator matrix in time domain, G(i,j)= rotMat^(i*j)
            GA[i*embedSize:i*embedSize+embedSize, j*embedSize:j*embedSize+embedSize] = matrix_power(rotMat, uB*i*j)
    return GA

# Encoding matrix of matrix B in Vandermonde embedded by rotation matrix scheme
def getRotEmbedEncodingMatOfB(uA, uB, numWorkers, embedSize):
    alpha = np.exp(2*1j*np.pi/numWorkers) #n-th root of unity
    rotMat = [[np.real(alpha), -np.imag(alpha)], [np.imag(alpha), np.real(alpha)]]
    GB = np.zeros((uB*embedSize, numWorkers*embedSize))
    for i in range(uB):
        for j in range(numWorkers):
            GB[i*embedSize:i*embedSize+embedSize, j*embedSize:j*embedSize+embedSize] = matrix_power(rotMat, i*j) 
    return GB    

# Encoding matrix of matrix A in the paper
# Numerically stable polynomially coded computing. 
# [Online] Available at: https://arxiv.org/abs/1903.08326, 2019.
# Fahim, M. and Cadambe, V. R.
def getVVEncodingMatOfA(uA, uB, numWorkers):
    GA = np.zeros((uA, numWorkers)) #generator matrix in time domain: a vandmond like matrix
    for i in range(uA):
        for j in range(numWorkers):
            #generator matrix in time domain, G(i,j)= rotMat^(i*j)
            chev = np.cos((j*2+1)*np.pi/(2*numWorkers))
            GA[i, j] = np.cos((i+1)*np.arccos(chev))
    return GA

# Encoding matrix of matrix B in the paper
# Numerically stable polynomially coded computing. 
# [Online] Available at: https://arxiv.org/abs/1903.08326, 2019.
# Fahim, M. and Cadambe, V. R.
def getVVEncodingMatOfB(uA, uB, numWorkers):
    GB = np.zeros((uB, numWorkers))
    for i in range(uB):
        for j in range(numWorkers):
            chev = np.cos((j*2+1)*np.pi/(2*numWorkers))
            GB[i, j] = np.cos((i+1)*uA*np.arccos(chev))
    return GB    

# Encoding matrix of matrix A of real polynomial code 
# Polynomial codes: an optimal design for high-dimensional coded matrix multiplication.
# In Proc. of Adv. in Neural Inf. Proc. Sys. (NIPS), pp. 4403–4413, 2017.
# Yu, Q., Maddah-Ali, M. A., and Avestimehr, A. S.
def getRealVandEncodingMatOfA(uA, uB, numWorkers):
    GA = np.zeros((uA, numWorkers)) #generator matrix in time domain: a vandmond like matrix
    evl = np.linspace(-1, 1, numWorkers)
    for i in range(uA):
        for j in range(numWorkers):
            #generator matrix in time domain, G(i,j)= rotMat^(i*j)
            GA[i, j] = evl[j]**(i*uB)
    return GA

# Encoding matrix of matrix B of real polynomial code 
# Polynomial codes: an optimal design for high-dimensional coded matrix multiplication.
# In Proc. of Adv. in Neural Inf. Proc. Sys. (NIPS), pp. 4403–4413, 2017.
# Yu, Q., Maddah-Ali, M. A., and Avestimehr, A. S.
def getRealVandEncodingMatOfB(uA, uB, numWorkers):
    GB = np.zeros((uB, numWorkers))
    evl = np.linspace(-1, 1, numWorkers)
    for i in range(uB):
        for j in range(numWorkers):
            GB[i, j] = evl[j]**i
    return GB    

# Encoding matrix of matrix A of complex polynomial code 
# Polynomial codes: an optimal design for high-dimensional coded matrix multiplication.
# In Proc. of Adv. in Neural Inf. Proc. Sys. (NIPS), pp. 4403–4413, 2017.
# Yu, Q., Maddah-Ali, M. A., and Avestimehr, A. S.
def getCompVandEncodingMatOfA(uA, uB, numWorkers):
    alpha = np.exp(2*1j*np.pi/numWorkers) #n-th root of unity
    GA = np.zeros((uA, numWorkers), dtype=complex) #generator matrix in time domain: a vandmond like matrix
    for i in range(uA):
        for j in range(numWorkers):
            #generator matrix in time domain, G(i,j)= rotMat^(i*j)
            GA[i, j] = alpha**(i*j*uB)
    return GA

# Encoding matrix of matrix B of complex polynomial code 
# Polynomial codes: an optimal design for high-dimensional coded matrix multiplication.
# In Proc. of Adv. in Neural Inf. Proc. Sys. (NIPS), pp. 4403–4413, 2017.
# Yu, Q., Maddah-Ali, M. A., and Avestimehr, A. S.
def getCompVandEncodingMatOfB(uA, uB, numWorkers):
    alpha = np.exp(2*1j*np.pi/numWorkers) #n-th root of unity
    GB = np.zeros((uB, numWorkers), dtype=complex)
    for i in range(uB):
        for j in range(numWorkers):
            GB[i, j] = alpha**(i*j)
    return GB

# Encoding matrix of matrix A and B of randomKarRao method
# Random khatri-rao-product codes for numericallystable distributed matrix multiplication.
# In 2019 57th Annual Allerton Conference on Communication, Control, and Computing (Allerton),
# Subramaniam, A. M., Heidarzadeh, A., and Narayanan, K. R.
# Note: this paper is a random method. Therefore we try this method #numTrails time and find the best encoding matrix
def getRandomKarRaoEncodingMat(uA, uB, numWorkers, numTrails):
    candGA = np.zeros((uA, numWorkers))
    candGB = np.zeros((uB, numWorkers))

    worstCond = sys.float_info.max
    threshold = uA*uB
    GA = np.zeros((uA, numWorkers))
    GB = np.zeros((uB, numWorkers))
    for i in range(uB):
        GA[:, uA*i:uA*i+uA] = np.eye(uA)
        temp = np.zeros((uB, uA))
        temp[i, :] = 1
        GB[:, uA*i:uA*i+uA] = temp

    #for trail in range(numTrails):
    for trail in range(numTrails):
        GA[:, threshold:numWorkers] = np.random.random([uA, numWorkers-threshold])
        GB[:, threshold:numWorkers] = np.random.random([uB, numWorkers-threshold])

    G = getEncodingMat(GA, GB, threshold, numWorkers, 1)
    curWorstCond = findWorstCond(G, threshold, numWorkers, 1)
    if curWorstCond < worstCond:
        candGA = GA
        candGB = GB
    return candGA, candGB
    
