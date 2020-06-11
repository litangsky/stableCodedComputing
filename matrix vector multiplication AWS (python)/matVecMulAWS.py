# -*- coding: utf-8 -*-

from numpy.linalg import matrix_power
from numpy.linalg import inv
from scipy.linalg import circulant
from mpi4py import MPI

import numpy as np
import scipy as sy
import random as rd
import time

# Generate a random matrix with dimension dim
def genRandMat(low, high, dim):
    mat = np.random.randint(low, high, dim).astype(np.float64)
    return mat

# Random pick k worker nodes from worker set. Rest of them are stragglers
def pickWorkers(threshold, numWorkers):
    workersSet = rd.sample(set(range(numWorkers)), threshold)
    return workersSet

# Decode the matrix multiplication result based on the result from worker nodes
def decodeResult(matReceived, numWorkers, generator, workersSet, blockSize, sample, embedSize):
    activeWorkers = len(workersSet)
    matReceived = np.reshape(matReceived.T,(blockSize*sample,activeWorkers*embedSize), order='F')
    G2Workers = np.zeros((activeWorkers*embedSize, activeWorkers*embedSize), dtype=generator.dtype)
    for i in range(activeWorkers):
        worker = workersSet[i]
        G2Workers[:, i*embedSize:i*embedSize+embedSize] = generator[:, worker*embedSize:worker*embedSize+embedSize]
    # directly inverse G in time domain to get result
    decodeStartTime = time.time()
    Cdecode = np.matmul(matReceived, inv(G2Workers))
    decodeEndTime = time.time()
    print("Decoding time is %f"%((decodeEndTime - decodeStartTime)/sample))
    return Cdecode

# computation in worker nodes
def computeWorker(A, B):
    C = np.matmul(A.T, B)
    return C

# Take the matrix multiplication ATB
def getMultResult(A, B, blockSize, sample, threshold, embedSize):
    ASplit = np.split(A, threshold*embedSize, axis=1)
    C = []
    for i in range(threshold*embedSize):
        C.append(np.matmul(ASplit[i].T, B).ravel())
    C = np.asarray(C).T
    return C

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
def getRealVandEncodingMat(threshold, numWorkers):
    G = np.zeros((threshold, numWorkers)) #generator matrix in time domain: a vandmond like matrix
    evl = np.linspace(-1, 1, numWorkers)
    for i in range(threshold):
        for j in range(numWorkers):
            #generator matrix in time domain, G(i,j)= rotMat^(i*j)
            G[i, j] = evl[j]**(i)
    return G

# Encoding matrix in complex Vendermonde method 
def getCompVandEncodingMat(threshold, numWorkers):
    alpha = np.exp(2*1j*np.pi/numWorkers) #n-th root of unity
    G = np.zeros((threshold, numWorkers), dtype=complex) #generator matrix in time domain: a vandmond like matrix
    for i in range(threshold):
        for j in range(numWorkers):
            #generator matrix in time domain, G(i,j)= rotMat^(i*j)
            G[i, j] = alpha**(i*j)
    return G
    
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


##########################################################################################
# The following functions are only for circulant embedding method
# preprocessing the matrix, only used in circulant embedding methrod
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

    decTimeStart = time.time()
    for i in range(1,embedSize):
        blockdiagMat = generator[i*threshold:i*threshold+threshold, i*numWorkers:i*numWorkers+numWorkers]
        blockdiagMat = blockdiagMat[:,workersSet]
        blockdiagMat = blockdiagMat.astype(complex)
        #%block codeword times inverse of one of block diagonal matrices
        matDecodeFreqPerm[i*threshold:i*threshold+threshold, :] =  np.matmul(inv(blockdiagMat).T, matRecFreqPerm[i*threshold:i*threshold+threshold, :])
    decTimeEnd = time.time()
    decTime = decTimeEnd - decTimeStart  
    
    ##permute C_fp_d to C_f_d, recover C by C_f_d(by FFT)
    permMatDecodeFreq = threshold*np.mod(range(embedSize*threshold), embedSize) + [np.floor(x/embedSize) for x in range(embedSize*threshold)]
    permMatDecodeFreq = permMatDecodeFreq.astype(int)
    matDecodeFreq = matDecodeFreqPerm[permMatDecodeFreq]; #permute A_fp_d to A_f_d
    matDecode = np.zeros((threshold*embedSize, blockSize*sample))
    
    for i in range(threshold):
        matDecode[i*embedSize:i*embedSize+embedSize,:] = np.fft.ifft(matDecodeFreq[i*embedSize:i*embedSize+embedSize,:], axis=0).real         

    print("Decoding time is %f"%(decTime/sample))
    matDecode = matDecode.T
    return matDecode

# Take the preprossing to the matrix
def createFakeSymbol(mat, threshold, blockSize, embedSize):
    row = mat.shape[0]
    for i in range(threshold):
        temp = np.zeros((row, blockSize))
        for j in range(embedSize-1):
            temp = temp + mat[:, i*blockSize*embedSize+j*blockSize:i*blockSize*embedSize+j*blockSize+blockSize]
        mat[:,(i+1)*blockSize*embedSize-blockSize:(i+1)*blockSize*embedSize] = -temp
    return mat

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

################################################################################
# The following functions are for AWS
# Worker node does (stragglerTime)sec unnecessary work 
def threadSleep(stragglerTime):
    time.sleep(stragglerTime)
    
# transmit the corresponding encoded matrix to workers
def transmitSubMat2Workers(comm, matEncoded, numWorkers):
    matsTransmit = [None] * numWorkers        
    for i in range(numWorkers):
        if matEncoded[0].dtype == np.float64:
            matsTransmit[i] = comm.Isend([matEncoded[i],MPI.DOUBLE], dest=i+1, tag=15)
        else:
            matsTransmit[i] = comm.Isend([matEncoded[i],MPI.COMPLEX], dest=i+1, tag=15)
    MPI.Request.Waitall(matsTransmit)

# transmit the vector to workers
def transmitVector2Workers(comm, vector, numWorkers):
    vectorTransmit = [None] * numWorkers
    for i in range(numWorkers):
        vectorTransmit[i] = comm.Isend([vector, MPI.DOUBLE], dest=i+1, tag=16)
    MPI.Request.Waitall(vectorTransmit)

# transmit the staggler information to each master node. Master node will decide if it is a stagglers
def transmitStragglersInfo(comm, numWorkers, workersSet):
    for i in range(len(workersSet)):
        for j in range(numWorkers):
            comm.send(workersSet[i], dest=j+1, tag=7)

# get straggler information from master node
def getIsStraggler(comm, threshold, numWorkers):
    isStraggler = True
    for i in range(threshold):
        activeID = comm.recv(source=0, tag=7)
        if activeID == comm.rank:
            isStraggler = False
    return isStraggler

# worker node receives the sub-matrix from master node
def receiveMatrixFromMaster(comm, dim, matType):
    matReceive = np.empty_like(np.matrix([[0]*dim[1] for i in range(dim[0])])).astype(matType)
    r = comm.Irecv(matReceive, source=0, tag=15)
    r.wait()
    return matReceive

# worker node receives the vector from master node
def receiveVectorFromMaster(comm, dim):
    vecReceive = np.empty_like(np.matrix([[0]*dim[1] for i in range(dim[0])])).astype(np.float64)
    r = comm.Irecv(vecReceive, source=0, tag=16)
    r.wait()
    return vecReceive

# master node collects time that worker nodes spend on matrix computing and report the average computation time 
def getComputeTimeFromWorkers(comm, numWorkers, threshold, sample):
    #Receive computation time from workers
	computeTime = np.array([[0] for i in range(numWorkers)]).astype(np.float64)
	for i in range(numWorkers):
		computeTime[i] = comm.recv(source=i+1, tag=50)
	print("The average worker computation time is: %f" %(np.sum(computeTime)/(numWorkers*sample)))

# the worker node computes its assignment and return the computation result to master node
def sendRstToMaster(comm, mat, vec):
    computeStartTime = time.time()
    rstToMaster = computeWorker(mat, vec)
    print("Slave %d finish computing"%comm.rank)
    computeEndTime = time.time()
    sendRst = comm.Isend(rstToMaster, dest=0, tag=31)
    sendRst.Wait()
    sendTime = comm.send(computeEndTime-computeStartTime, dest=0, tag=50)

# the master node receives the computation result from first #threhols fastest worker node and decode the result 
def getResultFromWorkers(comm, numWorkers, threshold, generator, blockSize, sample, embedSize):
    rstDict = []
    for i in range(numWorkers):
        rstDict.append(np.zeros((blockSize*embedSize, sample), dtype=generator.dtype))
       
    rstReceive = [None] * numWorkers
    for i in range(numWorkers):
        if generator.dtype == np.float64:
            rstReceive[i] = comm.Irecv([rstDict[i], MPI.DOUBLE], source=i+1, tag=31)
        else:
            rstReceive[i] = comm.Irecv([rstDict[i], MPI.COMPLEX], source=i+1, tag=31)            

    Cencode = [None] * threshold
    workersSet = []
    for i in range(threshold):
        j = MPI.Request.Waitany(rstReceive)
        workersSet.append(j)
        Cencode[i] = rstDict[j]

    CencodeArray = np.asarray(Cencode)

    Cresult = decodeResult(CencodeArray, numWorkers, generator, workersSet, blockSize, sample, embedSize)
    
    getComputeTimeFromWorkers(comm, numWorkers, threshold, sample)

    return Cresult

# the master node receives the computation result from first #threhols fastest worker node and decode the result 
# only for circulant embedding method
def getResultFromWorkersCirEmbed(comm, numWorkers, threshold, generatorFreq, blockSize, sample, embedSize):
    rstDict = []
    print("received shape %d, %d"%(blockSize*embedSize, sample))
    for i in range(numWorkers):
        rstDict.append(np.zeros((blockSize*embedSize, sample), dtype=float))
       
    rstReceive = [None] * numWorkers
    for i in range(numWorkers):
        rstReceive[i] = comm.Irecv([rstDict[i], MPI.DOUBLE], source=i+1, tag=31)         

    Cencode = [None] * threshold
    workersSet = []
    for i in range(threshold):
        j = MPI.Request.Waitany(rstReceive)
        workersSet.append(j)
        Cencode[i] = rstDict[j]

    CencodeArray = np.asarray(Cencode)

    Cresult = decodeResultFreqCirEmbed(CencodeArray, numWorkers, generatorFreq, workersSet, blockSize, sample, embedSize)
    
    getComputeTimeFromWorkers(comm, numWorkers, threshold, sample)

    return Cresult
