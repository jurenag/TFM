import numpy as np
import copy
from test_read_and_format import file_to_numpyarray_test, reformat_density_matrices, realArray_to_complexArray




def every_matrix_real_to_complex(density_matrices, N):
    '''This function takes:
    - density_matrices: (tridimensional numpy array) A set of density matrices as returned by 
    reformat_density_matrices(), which is a tridimensional numpy array of np.shape(...)[0] matrices.
    - N: (integer) Dimension of the hilbert space to which the density matrices in density_matrices belong.
    
    This function performs realArray_to_complexArray() over each matrix, and it returns the tridimensional numpy 
    array of np.shape(...)[0] complex square matrices.'''
    
    if type(density_matrices)!=type(np.array([])) or density_matrices.ndim!=3:
        print('every_matrix_real_to_complex(), Err1')
    if np.shape(density_matrices)[1]!=N or np.shape(density_matrices)[2]!=2*N:
        print('every_matrix_real_to_complex(), Err2')
        
    number_of_matrices = np.shape(density_matrices)[0]
    output_array = np.empty((number_of_matrices,N,N), dtype=complex)
    for k in range(number_of_matrices):
        output_array[k,:,:] = realArray_to_complexArray(density_matrices[k,:,:],N)
    return output_array
    
    
    
    
def commuter(a, b):
    return b, a
    
    
    
 
def partial_transpose(sqc_matrix, N, transpose_first_subsystem=True):
    '''This function takes:
    - sqc_matrix: (bidimensional square complex numpy array) Matrix representation of the density matrix
    - N: (integer) Dimension of the tensor product space
    - transpose_first_subsystem: (boolean) Determines whether the partial transposition is performed over the
    first subsystem or not.

    This function returns its partial transpose (just in case the state is actually separable. Otherwise, the 
    partial transposition is not well defined and the function returns a matrix which has nothing to do with 
    partial transposition, therefore losing the positivity which characterizes a density matrix).
    
    WARNING: THIS FUNCTION ONLY WORKS FOR H2XH2, I.E. THE TENSOR PRODUCT OF TWO QUBITS.'''
    
    if type(sqc_matrix)!=type(np.array([])) or np.ndim(sqc_matrix)!=2 or type(N)!=type(1):
        print('partial_transpose(), Err1')
        return -1
    if np.shape(sqc_matrix)[0]!=N or np.shape(sqc_matrix)[1]!=N:
        print('partial_transpose(), Err2')
        return -1
    
    if transpose_first_subsystem==True:
        sqc_matrix[0,2], sqc_matrix[2,0] = commuter(sqc_matrix[0,2], sqc_matrix[2,0])
        sqc_matrix[0,3], sqc_matrix[2,1] = commuter(sqc_matrix[0,3], sqc_matrix[2,1])
        sqc_matrix[1,2], sqc_matrix[3,0] = commuter(sqc_matrix[1,2], sqc_matrix[3,0])
        sqc_matrix[1,3], sqc_matrix[3,1] = commuter(sqc_matrix[1,3], sqc_matrix[3,1])
    else:
        sqc_matrix[0,1], sqc_matrix[1,0] = commuter(sqc_matrix[0,1], sqc_matrix[1,0])
        sqc_matrix[0,3], sqc_matrix[1,2] = commuter(sqc_matrix[0,3], sqc_matrix[1,2])
        sqc_matrix[2,1], sqc_matrix[3,0] = commuter(sqc_matrix[2,1], sqc_matrix[3,0])
        sqc_matrix[2,3], sqc_matrix[3,2] = commuter(sqc_matrix[2,3], sqc_matrix[3,2])
    return sqc_matrix
    
    
    
    
def negativity(hermitian_matrix, N, transpose_first=True):
    '''This function takes:
    - hermitian_matrix: (bidimensional square complex array) The matrix representation of an hermitian operator. 
    (Remember that such operators have real eigenvalues.)
    - N: (integer) Dimension of the tensor product space.
    
    This function returns the negativity of hermitian_matrix, i.e. the sum of the absolute values of its partial 
    transpose negative eigenvalues.
    
    WARNING: Since we call partial_transpose() within the body of this function, this function is still only
    applicable to the case of H2xH2, i.e. N=4.'''
    
    if type(hermitian_matrix)!=type(np.array([])) or np.ndim(hermitian_matrix)!=2 or type(N)!=type(1):
        print('negativity(), Err1')
    if np.shape(hermitian_matrix)[0]!=np.shape(hermitian_matrix)[1]:
        print('negativity(), Err2')
        
    aux = copy.deepcopy(hermitian_matrix)
    aux = partial_transpose(aux, N, transpose_first_subsystem=transpose_first)
    eigenvalues, _ = np.linalg.eigh(aux, UPLO='L')
    negativity = 0.0
    for i in range(np.shape(eigenvalues)[0]):
        if eigenvalues[i]<0.0:
            negativity += np.abs(eigenvalues[i])
    return negativity
    
    
    
    
def negativity_of_every_density_matrix(density_matrices, N, tolerance, transpose_first_subsystem=True):
    '''This function takes a tridimensional array of density matrices which are formatted
    as they are returned from every_matrix_real_to_complex(), i.e. it is a tridimensional numpy
    array which represents a collection of complex square arrays (NxN). This function  
    computes the negativity of every matrix and returns a unidimensional array with such negativities,
    such that its k-th entry match the negativity of the density_matrices[k,:,:]. It also receives
    a real scalar tolerance, so that if |negativity(density_matrices[k,:,:])|<tolerance, then 
    such negativity is taken as null.'''
    if type(density_matrices)!=type(np.array([])) or type(N)!=type(1) or type(tolerance)!=type(0.0):
        print('negativity_of_every_density_matrix(), Err1')
        return -1
    if density_matrices.ndim!=3:
        print('negativity_of_every_density_matrix(), Err2')
        return -1
    if np.shape(density_matrices)[1]!=N or np.shape(density_matrices)[2]!=N:
        print('negativity_of_every_density_matrix(), Err3')
        return -1
    if density_matrices.dtype!=np.empty((1,1), dtype=complex).dtype:
        print('negativity_of_every_density_matrix(), Err4')
        return -1


    number_of_matrices = np.shape(density_matrices)[0]
    aux = np.empty((np.shape(density_matrices)[1],np.shape(density_matrices)[2]), dtype=complex)
    negativities = np.empty((number_of_matrices,), dtype=float)
    for k in range(number_of_matrices):
        aux = density_matrices[k,:,:]
        aux_negativity = negativity(aux, N, transpose_first=transpose_first_subsystem)
        if np.abs(aux_negativity)<tolerance:
            aux_negativity = 0.0
        negativities[k] = aux_negativity
    return negativities
    
    
    
    
def calculate_negativity_distribution(filepath, N, tolerance, transpose_first_subsystem=True, skipped_rows=0):
    '''This function takes:
    - filepath: (string) path to a file which contains the density matrices formatted as required
    by file_to_numpyarraytest(). 
    - N: (integer) Dimension of the tensor product space
    - transpose_first_subsystem: (boolean) Determines whether the transposition is performed over the first or 
    the second subsystem. 
    
    This function returns the average negativity of the density matrices stored in filepath, its standard 
    deviation and a unidimensional array, negativities[:], so that negativities[k] stores the negativity of the 
    k-th density matrix in filepath.
    
    WARNING: Since this function calls negativity_of_every_density_matrix(), which in turn calls negativity(),
    which in turn calls partial_transpose(), this function is still only callable for H2xH2.'''
    
    if type(filepath)!=type('') or type(N)!=type(1) or type(transpose_first_subsystem)!=type(True):
        print('calculate_negativity_distribution(), Err1')
    
    density_matrices = reformat_density_matrices(file_to_numpyarray_test(filepath, N, rows_to_skip=skipped_rows))
    density_matrices = every_matrix_real_to_complex(density_matrices, N)
    negativities = negativity_of_every_density_matrix(density_matrices, N, tolerance, transpose_first_subsystem=True)
    
    return np.average(negativities), np.std(negativities), negativities




def trace(N, DM):
    '''This function takes:
    - N: (integer) The dimension of the Hilbert space to which the received density matrix belongs.
    - DM: (bidimensional square complex numpy array) Matrix representation of the density matrix whose
    trace we want to evaluate.'''

    if np.ndim(DM)!=2 or np.shape(DM)[0]!=N or np.shape(DM)[1]!=N:
        print("trace(), Err1")
        return -1

    trace = 0.0
    for i in range(N):
        trace += DM[i,i]
    return trace




def purity(N, DM):
    '''This function takes:
    - N: (integer) The dimension of the Hilbert space to which the received density matrix belongs.
    - DM: (bidimensional square complex numpy array) Matrix representation of the density matrix whose
    purity we want to evaluate.

    This function returns the purity of DM, which is nothing but the trace of the squared operator, i.e.
    purity(DM)=Tr(DM*DM).'''

    if np.ndim(DM)!=2 or np.shape(DM)[0]!=N or np.shape(DM)[1]!=N:
        print("purity(), Err1")
        return -1

    return trace(N, np.matmul(DM, DM))