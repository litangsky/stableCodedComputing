# -*- coding: utf-8 -*-
#from mpi4py import MPI
from mpi4py import MPI
from numpy.linalg import inv
from numpy.linalg import matrix_power
from numpy.linalg import cond
from itertools import combinations 

import numpy as np
import random
import time
import sys

# Worker node does (stragglerTime)sec unnecessary work 
def threadSleep(stragglerTime):
    time.sleep(stragglerTime)

# Generate a random matrix with dimension dim
def randMatGenerator(low, high, dim):
    mat = np.random.randint(low, high, dim).astype(np.float64)
    return mat

# Random pick k worker nodes from worker set. Rest of them are stragglers
def pickWorkers(threshold, numWorkers):
    workers = random.sample(set(range(numWorkers)), threshold)
    return workers

# encode orginal matrix by generator matrix          
def encodeMat(mat, generator, blockSize, u, numWorkers, embedSize):
    matSplit = np.split(mat, u*embedSize, axis=1)
    row = mat.shape[0]
    matEncoded = [None] * numWorkers 
    for j in range(numWorkers):
        encodeTemp = np.zeros([row, blockSize*embedSize], dtype=generator.dtype)
        for k in range(embedSize):
            encodeTemp[:,k*blockSize:k*blockSize+blockSize] = sum([matSplit[i]*generator[i][j*embedSize+k] for i in range(u*embedSize)])
        matEncoded[j] = encodeTemp
    return matEncoded

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
    decodeStartTime = time.time()
    Cdecode = np.matmul(matReceived, inv(G2Workers))
    decodeEndTime = time.time()
    print("Decoding time is %f"%(decodeEndTime - decodeStartTime))
    return Cdecode

# Worker node computes its assignment    
def computeWorker(A, B, sA, sB, embedMatSize):
    C = np.zeros((sA*sB,embedMatSize*embedMatSize), dtype=A.dtype)
    for i in range(embedMatSize):
        Atemp = A[:, i*sA:i*sA+sA]
        for j in range(embedMatSize):
            Btemp = B[:, j*sB:j*sB+sB]
            Ctemp = np.matmul(Atemp.T, Btemp).ravel()
            Ctemp1 = np.reshape(Ctemp, (sA*sB, 1))
            C[:, i*embedMatSize+j:i*embedMatSize+j+1] = Ctemp1
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
# G will be used in decoding
def getEncodingMat(GA, GB, threshold, numWorkers, embedSize):
    square = embedSize * embedSize
    G = np.zeros((threshold*square, numWorkers*square), dtype=GA.dtype)
    for i in range(numWorkers):
        G[:, i*square:i*square+square] = np.kron(GA[:,i*embedSize:i*embedSize+embedSize], GB[:,i*embedSize:i*embedSize+embedSize])
    return G

def reshapeReceivedInfo(matReceived, blockSizeA, blockSizeB, uA, uB, totalWorkers, embedMatSize):
    square = embedMatSize * embedMatSize
    matReshape = np.zeros((blockSizeA*blockSizeB,square*uA*uB), dtype=matReceived[0].dtype)
    for i in range(uA*uB):
        matReshape[:, i*square:i*square+square] = matReceived[i]
    return matReshape

# Compare the correct result to the decoded result
def getMSE(orginalRst, decodeRst):
    mse = np.sum(abs((orginalRst - decodeRst)**2))/np.sum(abs(decodeRst**2))
    print("MSE is %s"%mse)
    return mse

#############################################################################################################
# The following function is to create encoding matrix for different methods
    
# Encoding matrix of matrix A in Vandermonde embedded by rotation matrix scheme
def getRotEmbedEncodingMatOfA(uA, uB, numWorkers):
    alpha = np.exp(2*1j*np.pi/numWorkers) #n-th root of unity
    rotMat = [[np.real(alpha), -np.imag(alpha)], [np.imag(alpha), np.real(alpha)]]
    GA = np.zeros((uA*2, numWorkers*2)) #generator matrix in time domain: a vandmond like matrix
    for i in range(uA):
        for j in range(numWorkers):
            #generator matrix in time domain, G(i,j)= rotMat^(i*j)
            GA[i*2:i*2+2, j*2:j*2+2] = matrix_power(rotMat, uB*i*j)
    return GA

# Encoding matrix of matrix B in Vandermonde embedded by rotation matrix scheme
def getRotEmbedEncodingMatOfB(uA, uB, numWorkers):
    alpha = np.exp(2*1j*np.pi/numWorkers) #n-th root of unity
    rotMat = [[np.real(alpha), -np.imag(alpha)], [np.imag(alpha), np.real(alpha)]]
    GB = np.zeros((uB*2, numWorkers*2))
    for i in range(uB):
        for j in range(numWorkers):
            GB[i*2:i*2+2, j*2:j*2+2] = matrix_power(rotMat, i*j) 
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

#############################################################################################################
# The following function is to create encoding matrix for different methods
# transmit the staggler information to each master node. Master node will decide if it is a stagglers
def transmitStragglersInfo(comm, totalWorkers, workers):
    for i in range(len(workers)):
        for j in range(totalWorkers):
            comm.send(workers[i], dest=j+1, tag=7)

# transmit the corresponding encoded matrix A to workers
def transmitSubMatOfA2Workers(comm, matEncoded, totalWorkers, blockSize):
    matsTransmit = [None] * totalWorkers
    for i in range(totalWorkers):
        if matEncoded[0].dtype == np.float64:
            matsTransmit[i] = comm.Isend([matEncoded[i], MPI.DOUBLE], dest=i+1, tag=15)
        else:
            matsTransmit[i] = comm.Isend([matEncoded[i], MPI.COMPLEX], dest=i+1, tag=15)            
    MPI.Request.Waitall(matsTransmit)

# transmit the corresponding encoded matrix B to workers
def transmitSubMatOfB2Workers(comm, matEncoded, totalWorkers, blockSize):
    matsTransmit = [None] * totalWorkers
    for i in range(totalWorkers):
        if matEncoded[0].dtype == np.float64:
            matsTransmit[i] = comm.Isend([matEncoded[i], MPI.DOUBLE], dest=i+1, tag=16)
        else:
            matsTransmit[i] = comm.Isend([matEncoded[i], MPI.COMPLEX], dest=i+1, tag=16)    
    MPI.Request.Waitall(matsTransmit)

# get straggler information from master node
def getIsStraggler(comm, activeWorker, totalWorkers):
    isStraggler = True
    for i in range(activeWorker):
        activeID = comm.recv(source=0, tag=7)
        if activeID == comm.rank:
            isStraggler = False
    return isStraggler

# worker receives encoded matrix A from master
def receiveMatrixOfAFromMaster(comm, dim, receivedtype):
    matReceive = np.empty_like(np.matrix([[0]*dim[1] for i in range(dim[0])])).astype(receivedtype)
    r = comm.Irecv(matReceive, source=0, tag=15)
    r.wait()
    return matReceive

# worker receives encoded matrix B from master
def receiveMatrixOfBFromMaster(comm, dim, receivedtype):
    matReceive = np.empty_like(np.matrix([[0]*dim[1] for i in range(dim[0])])).astype(receivedtype)
    r = comm.Irecv(matReceive, source=0, tag=16)
    r.wait()
    return matReceive

# the worker node computes its assignment and return the computation result to master node
def sendRstToMaster(comm, matA, matB, blockSizeA, blockSizeB, embedMatSize):
    computeStartTime = time.time()
    rstToMaster = computeWorker(matA, matB, blockSizeA, blockSizeB, embedMatSize)
    computeEndTime = time.time()
    sendRst = comm.Isend(rstToMaster, dest=0, tag=31)
    sendRst.Wait()
    
    sendTime = comm.send(computeEndTime-computeStartTime, dest=0, tag=50)

# the master node receives the computation result from first #threhols fastest worker node and decode the result 
def getResultFromWorkers(comm, numWorkers, threshold, generator, blockSizeA, blockSizeB, uA, uB, embedSize):
    rstDict = []
    for i in range(numWorkers):
        rstDict.append(np.zeros((blockSizeA*blockSizeB, embedSize*embedSize), dtype=generator.dtype))
       
    rstReceive = [None] * numWorkers
    for i in range(numWorkers):
        if generator.dtype == np.float64:
            rstReceive[i] = comm.Irecv([rstDict[i], MPI.DOUBLE], source=i+1, tag=31)
        else:
            rstReceive[i] = comm.Irecv([rstDict[i], MPI.COMPLEX], source=i+1, tag=31)
        
    Cencode = [None] * threshold
    workers = []
    for i in range(threshold):
        j = MPI.Request.Waitany(rstReceive)
        workers.append(j)
        Cencode[i] = rstDict[j]
    CencodeReshape = reshapeReceivedInfo(Cencode, blockSizeA, blockSizeB, uA, uB, numWorkers, embedSize)
    Cresult = decodeResult(CencodeReshape, numWorkers, generator, workers, embedSize)
    
    getComputeTimeFromWorkers(comm, numWorkers, threshold)

    return Cresult

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

# master node collects time that worker nodes spend on matrix computing and report the average computation time 
def getComputeTimeFromWorkers(comm, totalWorkers, activeWorkers):
    #Receive computation time from workers
	computeTime = np.array([[0] for i in range(totalWorkers)]).astype(np.float64)
	for i in range(totalWorkers):
		computeTime[i] = comm.recv(source=i+1, tag=50)
	print("The average worker computation time is: %f" %(np.sum(computeTime)/totalWorkers))