import pandas as pd
import numpy as np




def file_to_numpyarray_test(filepath, N, rows_to_skip=0):
    '''This function takes:
    - filepath: (string) The path to a data file, relative to the current working directory (=cwd). Such
    file which must represent a (bidimensional) table of real numbers whose rows length is homogeneous. 
    - N: (integer): Dimension of the hilbert space to wich the density matrices in filepath must belong. In other
    words, 2*N*N must match the number of columns within each row of the file whose path is filepath. 
    
    The function takes such table and transform it into a numpy array whose entries type is float. The function 
    returns such array.'''
    
    if type(filepath)!=type(''):
        print('file_to_numpyarray_test(), Err1')
        return -1
    if type(N)!=type(1):
        print('file_to_numpyarray_test(), Err2')
        return -1
    if N<1:
        print('file_to_numpyarray_test(), Err3')
        return -1

    #The data files provided by D.M. have one tabulator at the end of each row, which makes pd.read_table() to think that 
    #the data file has one additional column whose entries are NaN. To prevent this from happening, we must specify usecols=range(first_n_columns)
    output_array = pd.read_table(filepath, sep='\t', header=None, index_col=None, usecols=range(2*N*N), skiprows=rows_to_skip)
    return np.array(output_array, dtype=float)
    
    
    
    
def extended_DM_to_formatted_DM(extended_DM, rows=4, columns=8):
    '''This function is internal to reformat_density_matrices().'''
    if type(extended_DM)!=type(np.array([])):
        print("extended_DM_to_formatted_DM(), Err1")
        return -1
    if np.shape(extended_DM)!=np.shape(np.ones(columns*rows)):
        print("extended_DM_to_formatted_DM(), Err2")
        return -1
    matrix = np.empty((rows,columns))
    for i in range(rows):
        for j in range(columns):
            matrix[i,j] = extended_DM[(i*columns)+j]
    return matrix
    
    
    
    
def reformat_density_matrices(input_array, rows=4, columns=8):
    '''This function takes:
    - input_array: (bidimensional numpy array) This array must contain one density matrix per row. The first 
    <columns> entres match the first row of the density matrix which encompass <rows> complex entries. The first 
    <columns> entries of the j-th row match the first row of the j-th density matrix, which encompass <rows> 
    complex entires. The following <rows> entries of the j-th row are the second row of the j-th density matrix and
    so on. 
    - rows (resp. columns): (integer) Number of rows (resp. columns) of real entries (i.e. shape of the unfolded
    matrix. By unfolding a complex matrix I mean taking a NxN complex matrix and splitting its real and imaginary
    parts to get a Nx2N matrix in the most 'natural' way.)
    
    This function returns a tridimensional numpy array, output_array[k,i,j], so that output_array contains the 
    information regarding one density matrix for a fixed k. The length of output_array along the second 
    (resp. third) axis is 4(=rows) (resp. 8(=columns)), whereas its length along the first axis match the number 
    of density matrices stored in it.'''
    
    if type(input_array)!=type(np.array([])):
        print('reformat_density_matrices(), Err1')
        return -1
    if np.ndim(input_array)!=2:
        print('reformat_density_matrices(), Err2')
        return -1
    if np.shape(input_array)[1]!=columns*rows:
        print('reformat_density_matrices(), Err3')
        return -1

    #Compute number_of_matrices as the number of rows in input_array
    number_of_matrices = np.shape(input_array)[0]
    output_array = np.empty((number_of_matrices,rows,columns))
    for k in range(number_of_matrices):
        output_array[k,:,:] = extended_DM_to_formatted_DM(input_array[k,:],rows,columns)[:,:]
    return output_array
    
    
    
    
def realArray_to_complexArray(input_array, N):
    '''This function takes:
    - input_array: (bidimensional real numpy array) Its number of columns must double its number of rows (i.e. 
    a real bidimensional array Nx2N).
    - N: (integer) Dimension of the hilbert space.
    
    This function assembles input_array into a bidimensional square complex array, output_array, NxN, according 
    to the following criteria:
    Re(output_array[i,j]) = input_array[i,2j-1]      Im(output_array[i,j]) = input_array[i,2j]     i,j=1,..,N.'''
    
    if type(input_array)!=type(np.array([])) or type(N)!=type(1):
        print('realArray_to_complexArray(), Err1')
        return -1
    if np.shape(input_array)!=np.shape(np.ones((N,2*N))):
        print('realArray_to_complexArray(), Err2')
        return -1

    output_array = np.empty((N,N), dtype=complex)
    for l in range(N):
        for m in range(N):
            output_array[l,m] = input_array[l,2*m] +input_array[l,(2*m)+1]*1j
    return output_array
