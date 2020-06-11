The code submission contains the following parts.

1. matrix vector multiplication local (python)
   It provides a version of code for distributed matrix-vector multiplication methods that can run in local machine. It can check the worst/average condition number, normalized MSE under given SNR for the following four methods.
   1) matVecMulCirEmbed: proposed circulant matrix embeding method
   2) matVecMulRotEmbed: proposed rotation matrix embeding method  
   3) matVecMulRealVand: real Vandermonde method
   4) matVecMulCompVand: complex Vandermonde method   

   input parameters:
   #matrix A (dim t*(k*b*q)) and vector x (dim t)
      n: number of total workers 
      k: number of active workers(threshold)
      b: column block size of A
      t: number of rows of matrix A(length of x)
      q: embedding matrix size
      sample: the number of trails (number of columns of "matrix" x)
      SNR: SNR of Gaussian noise adding to the matrix and vector in the worker node
   
    output:
      1) size of A and x
      2) worst/average condition number
      3) normalized MSE under given SNR(corresponds to the set of workers whose encoding matrix has worst condition number)

2. matrix vector multiplication AWS (python)
    It provides a version of code for distributed matrix-vector multiplication methods that can run in AWS. It can check the average computation time and decoding time, and normalized MSE (we did not add noise in AWS version but you can) for the following four methods.
    1) matVecMulCirEmbedAWS: proposed circulant matrix embeding method
    2) matVecMulRotEmbedAWS: proposed rotation matrix embeding method  
    3) matVecMulRealVandAWS: real Vandermonde method
    4) matVecMulCompVandAWS: complex Vandermonde method

   input parameters:
   #matrix A (dim t*(k*b*q)) and vector x (dim t)
      n: number of total workers 
      k: number of active workers(threshold)
      b: column block size of A
      t: number of rows of matrix A(length of x)
      q: embedding matrix size
      sample: the number of trails (number of columns of "matrix" x)

   output:
      1) size of A and x
      2) average computation time of worker nodes
      3) decoding time in the master node
      4) normalized MSE (we did not add noise in AWS version but you can. MSE in AWS only indicates we get the correct answer)

##########################################################################################
   Note: The decoding time and computation time are very small if we experienment matrix A times vector x.
         To garantee we could have a more accurate decoding/computation time, we experiment matrix A times a "matrix" x (dim t * sample) and get decoding/computation time.
         Finally we output the actual decoding/computation time by dividing "sample".
##########################################################################################

3. matrix matrix multiplication local (python)
   It provides a version of code for distributed matrix-matrix multiplication methods that can run in local machine. It can check the worst/average condition number, normalized MSE under given SNR for the following four methods.
   1) matMatMulRotEmbed: proposed rotation matrix embeding method  
   2) matMatMulRealVand: real polynomial code method
   3) matMatMulCompVand: complex polynomial code method   
   4) matMatMulRandKarRao: Random khatri-rao-product codes method
   5) matMatMulVV: Fahim-Cadambe codes method

   input parameters:
   #matrix A (dim t*(kA*bA*q)) and matrix B (dim t*(kB*bB*q))
   n: number of total workers 
   kA: 1/kA storage fraction of A
   kB: 1/kB storage fraction of B
   bA: col block size of A
   bB: col block size of B
   q: embedding matrix size
   t: size of row (A and B)
   k = kA*kB: recovery threshold
   SNR: SNR of Gaussian noise adding to the matrix in slave node
   
    output:
      1) size of A and B
      2) worst/average condition number
      3) normalized MSE under given SNR(corresponds to the set of workers whose encoding matrix has worst condition number)

4. matrix matrix multiplication AWS (python)
    It provides a version of code for distributed matrix-matrix multiplication methods that can run in AWS. It can check the average computation time and decoding time, and normalized MSE (we did not add noise in AWS version but you can) for the following four methods.
   1) matMatMulRotEmbedAWS: proposed rotation matrix embeding method  
   2) matMatMulRealVandAWS: real polynomial code method
   3) matMatMulCompVandAWS: complex polynomial code method   
   4) matMatMulRandKarRaoAWS: Random khatri-rao-product codes method
   5) matMatMulVVAWS: Fahim-Cadambe codes method

   input parameters:
   #matrix A (dim t*(k*b*q)) and vector x (dim t)
      n: number of total workers 
      kA: 1/kA storage fraction of A
      kB: 1/kB storage fraction of B
      bA: col block size of A
      bB: col block size of B
      q: embedding matrix size
      t: size of row (A and B)
      k = kA*kB: recovery threshold
      sample: the number of trails (number of columns of "matrix" x)

   output:
      1) size of A and B
      2) average computation time of worker nodes
      3) decoding time in the master node
      4) normalized MSE (we did not add noise in AWS version but you can. MSE in AWS only indicates we get the correct answer)

5. generate MSE curve
     It provides the matlab code to generate SNR-MSE cur in Fig.1 and Fig. 2 in the paper. All of experiments are obtained by running the codes in matrix vector multiplication local (python) and matrix matrix multiplication local (python). 
