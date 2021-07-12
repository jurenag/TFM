import pandas as pd
import numpy as np




def file_to_numpyarray(filepath, N, label, take_redundancy, rows_to_skip=0):
    '''This function takes: 
    - filepath: (string) The path to a data file which must represent a (bidimensional) table of real numbers
    whose rows length is homogeneous. 
    - N: (integer) Dimension of the Hilbert space to which density matrices in filepath belong. In other words,
    2N*N must match the number of columns within each row.
    - label: (integer) It is appended as the last element of each row of the output array.
    - take_redundancy: (boolean) If take_redundancy==True, then every entry of each row is read (remember that 
    each row represents a matrix). Otherwise, the function only takes (for each row), those entries which are not 
    redundant. This redundancy is due to the fact that each matrix (i.e. each row) is hermitic.
    
    The function takes such table and transform it into a numpy array whose entries type is float. It appends one
    more scalar entry (label) as the last column of every row. The function returns the resulting array.'''
    
    if type(filepath)!=type('') or type(N)!=type(1) or type(label)!=type(1.0) or type(take_redundancy)!=type(True):
        print('file_to_numpyarray(), Err1')
        return -1
    if N<1:
        print('file_to_numpyarray(), Err2')
        return -1
    
    #The data files provided by D.M. have one tabulator at the end of each row, which makes pd.read_table() think that 
    #the data file has one additional column whose entries are NaN. To prevent this from happening, we must specify 
    #usecols=range(2N*N), if take_redundancy==True, or usecols=useful_cols(N,True) otherwise
    if take_redundancy==True:
        output_array = pd.read_table(filepath, sep='\t', header=None, index_col=None, usecols=range(2*N*N), skiprows=rows_to_skip)
    else:
        output_array = pd.read_table(filepath, sep='\t', header=None, index_col=None, usecols=useful_cols(N,True), skiprows=rows_to_skip) 
    output_array = np.array(output_array, dtype=float)
    number_of_matrices = np.shape(output_array)[0]
    label_column = label*np.ones((number_of_matrices,1),dtype=float)
    output_array = np.concatenate((output_array,label_column), axis=1)
    return output_array
    
    
    
    
def useful_cols(N, upper_half=True):
    '''This function takes:
    - N: (integer) The dimenson of the Hilbert space.
    - upper_half: (boolean)
    
    This function returns a tuple which contains the column indices which match the matrix elements of an hermitian
    operator which are not redundant. This function is highly dependent of the format used for the density matrices
    in the files supplied by D.M. We are assuming that, in agreement with such files, the NxN complex entries which
    encompass an hermitian operator over a N-dimensional hilbert space are flattened into one row, and then each 
    element is broken up into two real entries. With this method, each density matrix is stored in one row of 
    2(N^2) columns, so that the first 2N entries match the N complex entries of the first row of the density 
    matrix, the subsequent 2N entries match the N complex entries of the second row of the density matrix and so on.
    If upper_half==True, then the function returns the indices matching the matrix elements over the main diagonal.
    The function returns the indices of the matrix elements under the main diagonal otherwise.'''
    
    if type(N)!=type(1) or type(upper_half)!=type(True):
        print('useful_cols(), Err1')
        return -1
    if N<1:
        print('useful_cols(), Err2')
        return -1

    #For this unfolded matrix (NxN -> Nx2N) the index conditions of main diagonal are:
    #2i=j or 2i=j+1. There are two conditions because the main diagonal now unfolds in two
    #diagonals. The first of those (2i=j) is the imaginary part of the original diagonal,
    #whereas the second of those (2i=j+1) is the real diagonal. The condition for the upper
    #half matrix elements is 2i<j, and the condition for the lower half matrix elements 
    #is 2i>j+1. For the programming-like indexing, where the arrays elements start at 0, 
    #these conditions transform into:
    #2(i+1)=j+1 (imaginary diag.) or 2(i+1)=j+2 (real diag.) for the diagonals
    #2(i+1)<j+1 for the upper triangle half, and 2(i+1)>j+2 for the lower triangle half.
    result_list = []
    cont = 0
    if upper_half==True:
        for i in range(N):
            for j in range(2*N):
                if j>((2*(i+1))-1) or j==((2*(i+1))-2):
                    result_list.append(cont)
                cont = cont+1
        return result_list
    for i in range(N):
        for j in range(2*N):
            if j<((2*(i+1))-2) or j==((2*(i+1))-2):
                result_list.append(cont)
            cont = cont+1
    return result_list
    
    
    
    
def concatenate_entangled_and_separable_arrays(array1, array2, shuffle):
    '''This function takes:
    - array1 (resp. array2): (bidimensional numpy array) Both arrays must have the same number of columns. 
    - shuffle: (boolean) Determines whether the function shuffles the concatenation of arrays along the 0-th
    axis before returning it.
    
    This function returns the concatenation of both arrays along the 0-th axis, i.e. stack the rows of array1 
    over the rows of array2. If shuffle==True, the concatenation of arrays is shuffled along the 0-th axis before
    being returned.'''
    
    if type(array1)!=type(np.array([])) or type(array2)!=type(np.array([])) or type(shuffle)!=type(True):
        print('concatenate_entangled_and_separable_arrays(), Err1')
        return -1
    if np.ndim(array1)!=2 or np.ndim(array2)!=2:
        print('concatenate_entangled_and_separable_arrays(), Err2')
        return -1
    #The following condition ensures that both arrays store density matrices of the same size.
    #The number of density matrices (i.e. the number of rows in the arrays) might differ from
    #one another.
    if np.shape(array1)[1]!=np.shape(array2)[1]:
        print('concatenate_entangled_and_separable_arrays(), Err3')
        return -1

    concatenated_array = np.concatenate((array1,array2),axis=0) 
    #np.random.shuffle() always shuffles along the first axis. This is exactly what I need.   
    if shuffle==True:
        np.random.shuffle(concatenated_array)
    return concatenated_array
    
    
    
    
def split_array_train_test(array_to_split, fraction):
    '''This function takes:
    - array_to_split: (bidimensional numpy array)
    - fraction: (real scalar) Real number in [0,1].
    
    The function splits the array along the 0-th axis (i.e. the 'frontier' of separation is parallel to a row) so 
    that a fraction fraction of rows stays in the bigger array (1-fraction if fraction <0.5), and a fraction 
    (1-fraction) stays in the smaller array. The function returns both arrays in that order, i.e. the bigger one 
    and the smaller one.'''
    
    if type(fraction)!=type(0.0) or type(array_to_split)!=type(np.array([])):
        print('split_array(), Err1')
        return -1
    if fraction<0.0 or fraction>1.0:
        print('split_array(), Err2')
        return -1
    if np.ndim(array_to_split)!=2:
        print('split_array(), Err3')
        return -1

    if fraction <0.5:
        fraction = 1.0-fraction
    N = int(np.ceil(np.shape(array_to_split)[0]*fraction))
    split_arrays = np.split(array_to_split, (N,), 0)
    return split_arrays[0], split_arrays[1]
    
    
    
    
def split_array_input_label(array_to_split):
    '''This function takes:
    - array_to_split: (bidimensional numpy array) Must have more than one column.
    
    The function splits the array along the 1-st axis (i.e. the frontier of separation is parallel to
    a column) so that the first split array is array_to_split without its last column, and the second
    split array is the last column of array_to_split. The function returns both arrays in such order.
    Furthermore, the function returns the shape of the first array without its first element. Typically,
    when the split arrays are used as the input and goal values of a neural network, such 
    np.shape(split_arrays[0])[1:] is the input shape of the neural network, since different data samples of
    the network input distribute along the 0-th axis of split_arrays[0].'''
    
    if type(array_to_split)!=type(np.array([])) or np.ndim(array_to_split)!=2:
        print('split_array_input_label(), Err1')
        return -1

    split_arrays = np.split(array_to_split, ((np.shape(array_to_split)[1])-1,), 1)
    return split_arrays[0], split_arrays[1], np.shape(split_arrays[0])[1:]
