# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 12:50:44 2020

@author: tl
"""
from numpy.linalg import matrix_power
from numpy.linalg import inv
from numpy.linalg import cond
from itertools import combinations 
from scipy.linalg import circulant

import scipy as sy
import numpy as np
import random as rd

# Generate a random matrix with dimension dim
def genRandMat(low, high, dim):
    mat = np.random.randint(low, high, dim).astype(np.float64)
    return mat

# Random pick k worker nodes from worker set. Rest of them are stragglers
def pickWorkers(threshold, numWorkers):
    workersSet = rd.sample(set(range(numWorkers)), threshold)
    return workersSet


# add noise to matrix
def addGaussianNoise(mat, SNR):
    noisePower = 10**(-SNR/10)
    noise = np.random.randn(mat.shape[0], mat.shape[1]) * np.sqrt(noisePower)
    return mat + noise
   
# encode matrix in the master node          
def encodeMat(mat, generator, blockSize, threshold, numWorkers, embedMatSize):
    matSplit = np.split(mat, threshold*embedMatSize, axis=1)
    row = mat.shape[0]
    matEncoded = [None] * numWorkers 
    for j in range(numWorkers):
        encodeTemp = np.zeros([row, embedMatSize*blockSize], dtype=generator.dtype)
        for k in range(embedMatSize):
            encodeTemp[:, k*blockSize:k*blockSize+blockSize] = sum([matSplit[i]*generator[i][embedMatSize*j+k] for i in range(embedMatSize*threshold)])
        matEncoded[j] = encodeTemp
    return matEncoded

# transmit the corresponding encoded matrix to workers
def transmitSubMat2Workers(matEncoded, numWorkers, blockSize, embedMatSize):
    matsTransmit = [None] * numWorkers
    for i in range(numWorkers):
        matsTransmit[i] = matEncoded[i]
    return matsTransmit

# decode the matrix multiplication result based on the result from worker nodes
def decodeResult(matReceived, numWorkers, generator, workersSet, blockSize, sample, embedSize):
    threshold = len(workersSet)
    matReceived = np.reshape(matReceived.T,(blockSize*sample,threshold*embedSize), order='F')
    G2Workers = np.zeros((threshold*embedSize, threshold*embedSize), dtype=generator.dtype)
    for i in range(threshold):
        worker = workersSet[i]
        G2Workers[:, i*embedSize:i*embedSize+embedSize] = generator[:, worker*embedSize:worker*embedSize+embedSize]
    # directly inverse G in time domain to get result
    Cdecode = np.matmul(matReceived, inv(G2Workers))
    #Cdecode = np.linalg.solve(G2Workers, matReceived.T)
    return Cdecode

# computation in worker nodes
def computeWorker(A, B):
    C = np.matmul(A.T, B)
    return C

# Get the correct reuslt based on A and x we randomly generate 
def getMultResult(A, B, blockSize, sample, threshold, embedSize):
    ASplit = np.split(A, threshold*embedSize, axis=1)
    C = []
    for i in range(threshold*embedSize):
        C.append(np.matmul(ASplit[i].T, B).ravel())
    C = np.asarray(C).T
    return C

# Find the set of worker nodes that are not stagglers
# This set of worker nodes correspond to the worst condition number of corresponding encoding matrix
def findWorkerSet2WorstCond(generatorMat, threshold, numWorkers, embedSize):
    comb = list(combinations(range(numWorkers), threshold))
    condCollect = []
    for i in range(len(comb)):
        G2Workers = np.zeros((threshold*embedSize, threshold*embedSize), dtype=generatorMat.dtype)
        for j in range(threshold):
            worker = comb[i][j]
            G2Workers[:, j*embedSize:j*embedSize+embedSize] = generatorMat[:, worker*embedSize:worker*embedSize+embedSize]
        condCollect.append(cond(G2Workers))
    print("Average condition number is %f"%np.mean(condCollect))
    worstCond = np.max(condCollect)
    print("Worst condition number is %f"%worstCond)
    workerSet2WorstCond = comb[condCollect.index(worstCond)]
    return workerSet2WorstCond

# check the worst and average condition number for circulant embed method
# Note: since the encoding matrix in circulant matrix method is not full rank
# checking the conditio number is different from other methods()
# We check the worst and average condition number of the full rank block matrix of B
# 1) B is the block diagnoal matrix obtained by eigenvalue decomposition of encoding matrix G 
# 2) only the first block matrix in B is singular
def checkCondNumberForCirEmbed(B, embedSize, threshold, numWorkers):
    comb = list(combinations(range(numWorkers), threshold))
    for i in range(len(comb)):
        condCollect = []
        for j in range(1, embedSize):
            Bsub = B[j*threshold:j*threshold+threshold, j*numWorkers:j*numWorkers+numWorkers]
            Bsub = Bsub[:, comb[i]]
            condCollect.append(cond(Bsub))
    print("Average condition number is %f"%np.mean(condCollect))
    print("Worst condition number is %f"%np.max(condCollect))    

# Compare the correct result to the decoded result    
def getMSE(orginalRst, decodeRst):
    mse = np.sum(abs((orginalRst - decodeRst)**2))/np.sum(abs(decodeRst**2))
    print("MSE is %s"%mse)
    return mse

#############################################################################################################
# The following function is to create encoding matrix for different methods
    
# Encoding matrix in Vandermonde embedded by rotation matrix scheme
def getRotEmbedEncodingMat(threshold, numWorkers):
    alpha = np.exp(2*1j*np.pi/numWorkers) #n-th root of unity
    rotMat = [[np.real(alpha), -np.imag(alpha)], [np.imag(alpha), np.real(alpha)]]
    G = np.zeros((2*threshold, 2*numWorkers)) #generator matrix in time domain: a vandmond like matrix
    for i in range(threshold):
        for j in range(numWorkers):
            #generator matrix in time domain, G(i,j)= rotMat^(i*j)
            G[2*i:2*i+2, 2*j:2*j+2] = matrix_power(rotMat, i*j) 
    return G

# Encoding matrix in real Vendermonde method 
def getRealVandEncodingMat(activeWorkers, totalWorkers):
    G = np.zeros((activeWorkers, totalWorkers)) #generator matrix in time domain: a vandmond like matrix
    evl = np.linspace(-1, 1, totalWorkers)
    for i in range(activeWorkers):
        for j in range(totalWorkers):
            #generator matrix in time domain, G(i,j)= rotMat^(i*j)
            G[i, j] = evl[j]**(i)
    return G

# Encoding matrix in complex Vendermonde method 
def getCompVandEncodingMat(activeWorkers, totalWorkers):
    alpha = np.exp(2*1j*np.pi/totalWorkers) #n-th root of unity
    G = np.zeros((activeWorkers, totalWorkers), dtype=complex) #generator matrix in time domain: a vandmond like matrix
    for i in range(activeWorkers):
        for j in range(totalWorkers):
            #generator matrix in time domain, G(i,j)= rotMat^(i*j)
            G[i, j] = alpha**(i*j)
    return G

# Encoding matrix in Vandermonde matrix embedding by circulant matrix method 
def getCirEncodingMat(threshold, numWorkers, embedSize):
    # embedding matrix is a circulant matrix cirMat
    genCol = np.zeros(embedSize)
    genCol[embedSize-1] = 1
    cirMat = circulant(genCol).T
    
    G = np.zeros((threshold*embedSize, numWorkers*embedSize)) #generator matrix in time domain: a vandmond like matrix
    B = np.zeros((threshold*embedSize, numWorkers*embedSize), dtype=complex) #generator matrix in freq domain

    for i in range(threshold):
        for j in range(numWorkers):
            #generator matrix in time domain, G(i,j)= cirMat^(i*j)
            cir = matrix_power(cirMat, i*j) 
            G[i*embedSize:i*embedSize+embedSize, j*embedSize:j*embedSize+embedSize] = cir 
        
            #generator matrix in freq domain, B(i,j) = fft(G(i,j))
            head = cir[0,:]
            headFeq = sy.fft(head)
            B[i*embedSize:i*embedSize+embedSize, j*embedSize:j*embedSize+embedSize] = np.diag(headFeq)
    
    ##permute matrix B into D(block diagnoal matrix)
    permCol = embedSize*np.mod(range(embedSize*numWorkers), numWorkers) + [np.floor(x/numWorkers) for x in range(embedSize*numWorkers)]
    permRow = embedSize*np.mod(range(embedSize*threshold), threshold) + [np.floor(x/threshold) for x in range(embedSize*threshold)]
    permCol = permCol.astype(int)
    permRow = permRow.astype(int)
    B = B[permRow, :]
    B = B[:, permCol]
    return G, B

##########################################################################################
# The following three functions are only for circulant embedding method
# preprocessing the matrix, only used in circulant embedding methrod
def createFakeSymbol(mat, threshold, blockSize, embedSize):
    row = mat.shape[0]
    for i in range(threshold):
        temp = np.zeros((row, blockSize))
        for j in range(embedSize-1):
            temp = temp + mat[:, i*blockSize*embedSize+j*blockSize:i*blockSize*embedSize+j*blockSize+blockSize]
        mat[:,(i+1)*blockSize*embedSize-blockSize:(i+1)*blockSize*embedSize] = -temp
    return mat


def encodeMatForCirEmbed(mat, generator, blockSize, threshold, numWorkers, embedMatSize):
    matSplit = np.split(mat, threshold*embedMatSize, axis=1)
    row = mat.shape[0]
    matEncoded = [None] * numWorkers 
    for j in range(numWorkers):
        encodeTemp = np.zeros([row, embedMatSize*blockSize], dtype=generator.dtype)
        for k in range(embedMatSize):
            nonZeroEle = np.nonzero(generator[:, embedMatSize*j+k])
            encodeTemp[:, k*blockSize:k*blockSize+blockSize] = sum([matSplit[nonZeroEle[0][i]] for i in range(threshold)])
        matEncoded[j] = encodeTemp
    return matEncoded

# decoding algorithm for circulant embeding method. The decoding algorithm over frequency domain
def decodeResultFreqCirEmbed(matReceived, numWorkers, generator, workersSet, blockSize, sample, embedSize):
    threshold = len(workersSet)
    #matReceived = np.reshape(matReceived.T,(blockSize*sample,threshold*embedSize), order='F').T
    matReceived = np.reshape(matReceived,(threshold*embedSize,blockSize*sample))
    matRecFreq = np.zeros((threshold*embedSize, blockSize*sample), dtype=complex) #iDFT each block of received codeword(c_rec)
    
    for i in range(threshold):
        matRecFreq[i*embedSize:i*embedSize+embedSize, :] = np.fft.fft(matReceived[i*embedSize:i*embedSize+embedSize,:], axis=0)        
    
    permMatEncodeFreqPerm = embedSize*np.mod(range(embedSize*threshold), threshold) + [np.floor(x/threshold) for x in range(embedSize*threshold)]   
    permMatEncodeFreqPerm = permMatEncodeFreqPerm.astype(int)
    matRecFreqPerm = matRecFreq[permMatEncodeFreqPerm]

    matDecodeFreqPerm = np.zeros((threshold*embedSize,blockSize*sample), dtype=complex)

    for i in range(1,embedSize):
        blockdiagMat = generator[i*threshold:i*threshold+threshold, i*numWorkers:i*numWorkers+numWorkers]
        blockdiagMat = blockdiagMat[:,workersSet]
        blockdiagMat = blockdiagMat.astype(complex)
        #block codeword times inverse of one of block diagonal matrices
        matDecodeFreqPerm[i*threshold:i*threshold+threshold, :] =  np.matmul(inv(blockdiagMat).T, matRecFreqPerm[i*threshold:i*threshold+threshold, :])
    
    ##permute C_fp_d to C_f_d, recover C by C_f_d(by FFT)
    permMatDecodeFreq = threshold*np.mod(range(embedSize*threshold), embedSize) + [np.floor(x/embedSize) for x in range(embedSize*threshold)]
    permMatDecodeFreq = permMatDecodeFreq.astype(int)
    matDecodeFreq = matDecodeFreqPerm[permMatDecodeFreq]; #permute A_fp_d to A_f_d
    matDecode = np.zeros((threshold*embedSize, blockSize*sample))
    
    for i in range(threshold):
        matDecode[i*embedSize:i*embedSize+embedSize,:] = np.fft.ifft(matDecodeFreq[i*embedSize:i*embedSize+embedSize,:], axis=0).real    

    matDecode = matDecode.T
    return matDecode
