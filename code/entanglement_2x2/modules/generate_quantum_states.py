import numpy as np
from numpy.random import Generator, PCG64
from scipy import constants as sp
from physical_tests import negativity, purity




def random_int(min, max, rng):
    '''This function takes:
    - min (resp. max): (integer) Minimum inclusive (maximum exclusive) integer value that can be returned by 
    this function.
    - rng: (a random generator, as initialized by rng = Generator(PCG64(int(seed))). Here, seed is some integer 
    number.
    
    This function takes an already initialized random generator rng and samples a random integer in the interval
    [min,max).'''
    
    return int(np.floor(min+(rng.random()*(max-min))))
    
    
    
    
def generate_random_pure_density_matrix(N, rng):
    '''This function takes:
    - N: (integer) Dimension of the hilbert space whose states we want to sample. For example, for the case of 
    the tensor product space H_a otimes H_b, where H_a refers to the Hilbert space of to one qu-a-it 
    (for example, H_2 (H_3) for the qubit (qutrit)), the dimension of H_a otimes H_b is a*b=N. In that case, a 
    density matrix (=DM) (pure or mixed) is a NxN matrix actuating over H_a otimes H_b. 
    - rng: (a random generator, as initialized by rng = Generator(PCG64(int(seed))). Here, seed is some integer.
    
    This function returns a random pure quantum state (i.e. a NxN square complex numpy array) 
    which belongs to the Hilbert space of dimension N. The state is generated according to the
    previously explained algorithm.
    
    NOTE_1: This function does not exploits the symmetry condition induced by the hermiticity of the DMs. If 
    anytime you use it for N so big that efficiency becomes an issue, then you might modify 
    generate_random_pure_density_matrix() so as to take advantage of such symmetry.
    
    NOTE_2: It is VITAL that this function receives rng and it is not initialized within this function body 
    (i.e. one initialization per function call). If that was the case, we would have strong correlations even if 
    the integer passed to Generator() as seed is time.time(), since I expect to call this function multiple times 
    within the same second. That would result in constantly initializing rng with the same seed and hence, 
    sampling the same density matrices a lot of times (strong correlations in the DMs sample set).'''
    
    modules = rng.random((N,))
    phases = rng.random((N,))
    #rng.random() returns a random number in (0,1). In order to get one random number in (0,2\pi) we must multiply
    #rng.random() by 2\pi. The scipy library implements scientific constants. Namely, scipy.constants.pi gives \pi.
    phases = 2*sp.pi*phases
    #Initializing density_matrix as complex is essential
    density_matrix = np.empty((N,N), dtype=complex)
    for i in range(N):
        for k in range(N):
            density_matrix[i,k] = modules[i]*modules[k]*np.exp(1.j*(phases[i]-phases[k]))
    #Compute density_matrix normalization
    norm = 0.0
    for i in range(N):
        norm += np.power(modules[i], 2)
    return density_matrix/norm
    



def generate_random_mixed_density_matrix(N, L, rng):
    '''This function takes:
    - N: (integer) The dimension of the hilbert space to which the returned density matrix belongs.
    - L: (integer) Number of elements in the convex linear combination.
    - rng: (a random generator, as initialized by rng = Generator(PCG64(int(seed))). Here, seed is some integer.
    
    This function returns a NxN complex square numpy array which is the matrix representation of a random mixed (L)
    quantum state.'''
    
    #I will calculate the desired DM as a sum. So, I have to initialize it to the zero matrix.
    DM_holder = np.zeros((N,N), dtype=complex)
    coefficients = rng.random((L,))
    #Normalize coefficients
    coefficients = coefficients/np.sum(coefficients)
    for i in range(L):
        aux = generate_random_pure_density_matrix(N, rng)
        DM_holder[:,:] = DM_holder[:,:] + coefficients[i]*aux[:,:]
    return DM_holder




def prepend_whatever_distribution(filepath, mean, std, name):
    '''This function takes:
    - filepath: (string) Filepath to the file where we want to pre-append the statistical data.
    - mean (resp. std): (real scalar) Arithmetic mean (sample std) of the samples contained in filepath.
    - name: (string) Name of the random magnitud whose mean and std we want to prepend to filepath.
    
    This function opens filepath, reads its content, and overwrite it in the following way. It writes
    the mean and std as the first line, then appends all of the previous content. 

    NOTE: It is essential that filepath is not already opened.
    
    WARNING: This funcion is highly inefficient. However, it can be used for not so big files.'''

    file = open(filepath, 'r+')
    content = file.read()
    file.seek(0, 0)
    file.write("#The "+name+" mean of the density matrices stored in this file is "+str(mean)+"; Its standard deviation is "+str(std)+"\n")
    file.write(content)
    file.close()
    return 




def write_DM_to_file(output_file, DM, N, preflattening_needed=True):
    '''This function takes:
    - output_file: (file variable as given by open(filepath))
    - DM: (complex numpy array) If preflattening_needed=False, then I expect it to be the matrix representation of
    an hermitian operator which has been flattened according to numpy.ndarray.flatten(). Otherwise, I expect
    it to be a bidimensional square complex (hermitian) array.
    - N: (integer) Dimension of the hilbert space to which DM belongs.
    - preflattening_needed: (boolean) Dictates whether the DM needs to be flattened or not.
    
    This function writes the density matrix given by DM into an output file, output_file, according to the usual
    format (flat DM). It is IMPORTANT to note that this function adds a line feed ('\n') after the flat DM.'''
    
    if type(DM)!=type(np.array([])) or type(N)!=type(1) or type(preflattening_needed)!=type(True):
        print('write_DM_to_file(), Err1')
        return -1
    
    if(preflattening_needed==True):
        DM = np.ndarray.flatten(DM)
        
    for k in range(N*N):
        output_file.write(str(DM[k].real)+'\t'+str(DM[k].imag)+'\t')
    output_file.write('\n')
    
    return
    
    
    
    
def generate_M_random_DMs_file(N, M, filepath, rng, pure=True, L_min=2, L_max=10):
    '''This function takes:
    - N: (integer) Dimension of the hilbert space whose states we want to sample.
    - M: (integer) The number of pure random density matrices (belonging to such space) that are going to be 
    generated.
    - filepath: (string) The path to the output file, relative to the current working directory (=cwd). 
    - rng: (a random generator, as initialized by rng = Generator(PCG64(int(seed))). Here, seed is some integer 
    number.
    - pure: (boolean) Determines whether the generated random quantum states are pure (pure==True) or mixed
    (pure==False).
    - L_min (resp. L_max): (integer) This parameter only makes a difference if pure==False. It determins the 
    minimum (maximum) random number of added terms in the linear convex combination that yields a mixed random 
    state. If you want this function to generate M random mixed density matrices with exactly L terms in the
    linear combination, then you might give L_min=L, L_max=L.
    
    This function generates M random DMs of H (dim(H)=N) and writes them down to the output file, whose path is
    given by filepath. Such writing process is performed according to the usual format, i.e. one density matrix 
    per row, so that each row matches the flattened density matrix as given by numpy.ndarray.flatten(). If pure==True,
    then this function returns the arithmetic mean of the negativity evaluated over every density matrix that is 
    written down to filepath. It also returns its standard deviation. Both of the data is pre-appended as the first 
    line of the file which ends up storing the generated density matrices. If pure==False, then, in addition to the 
    specified behaviour, the function also returns the purity mean and the purity std. Both purity values are also 
    pre-appended to the output file if pure==False. For a similar reason as in NOTE_2 in 
    generate_random_pure_density_matrix() documentation, it is essential that this functon receives the already 
    initialized random generator rng. I.e. rng is not initialized with every call of this function. This is important 
    since I may need to call this function multiple times consecutively. In that case, and if M is small enough, the 
    generation of the whole file might take so short that the seed for the  initialization of rng in subsequent calls 
    (if rng was initialized within this function body) of this function is the same. This would lead to two DMs files 
    with exactly the same DMs.'''
    
    if type(N)!=type(1) or type(M)!=type(1) or type(filepath)!=type(''):
        print('generate_M_random_pure_DMs_file(), Err1')
        return -1
    
    output_file = open(filepath, mode='w')
    output_file.write('#Hilbert space dimension: '+str(N)+'\n')
    negativities = []
    if pure==True:
        output_file.write('#This file contains '+str(M)+' random pure density matrices\n')
        for i in range(M):
            dm_holder = generate_random_pure_density_matrix(N, rng)
            write_DM_to_file(output_file, dm_holder, N, preflattening_needed=True)
            negativities.append(negativity(dm_holder,  N))
        output_file.close()
        negativities = np.array(negativities)
        negativity_mean = np.mean(negativities); negativity_std = np.std(negativities)
        prepend_whatever_distribution(filepath, negativity_mean, negativity_std, 'negativity')
        return negativity_mean, negativity_stdgenerate_M_random
    else:
        Ls = []
        purities = []
        output_file.write('#This file contains '+str(M)+' random mixed density matrices with L in['+str(L_min)+','+str(L_max)+') number of terms in its convex linear combination\n')
        for i in range(M):
            L = random_int(L_min, L_max, rng)
            dm_holder = generate_random_mixed_density_matrix(N, L, rng)
            write_DM_to_file(output_file, dm_holder, N, preflattening_needed=True)
            negativities.append(negativity(dm_holder, N))
            Ls.append(L)
            purities.append(purity(N, dm_holder))
        output_file.close()
        negativities = np.array(negativities)
        negativity_mean = np.mean(negativities); negativity_std = np.std(negativities)
        Ls = np.array(Ls)
        L_mean = np.mean(Ls); L_std = np.std(Ls)
        purities = np.array(purities)
        purity_mean = np.mean(purities); purity_std = np.std(purities)
        prepend_whatever_distribution(filepath, purity_mean, purity_std, 'purity')
        prepend_whatever_distribution(filepath, L_mean, L_std, 'L')
        prepend_whatever_distribution(filepath, negativity_mean, negativity_std, 'negativity')
        return negativity_mean, negativity_std, L_mean, L_std, purity_mean, purity_std
#Function Check: This produces what it is intended to produce




def belonging_subinterval_index(neg_part, negativity, howManySubintervals):
    '''This function takes:
    - neg_part: (unidimensional numpy real array) Negativity partition as initialized by 
    generate_random_pure_DMs_files_negativitywise()
    - negativity: A value of negativity
    - howManySubintervals: The number of negativity subintervals according to the negativity partition neg_part
    
    This function returns the subinterval of neg_part which negativity belongs to. To do so, I assume the following
    labelling for the partition subintervals. The subinterval (neg_part[i],neg_part[i+1]) is the i-th subinterval.
    For example, for howManySubintervals=5 and the usual maximum value of negativity (0.5), we have that 
    neg_party=array([0.0,0.1,0.2,0.3,0.4,0.5]). So that (neg_part[0],neg_part[1])=(0.0,0.1) is the 0-th subinterval.
    On the other hand, (neg_part[4],neg_part[5]) is the 4-th subinterval. Since subintervals start counting at
    zero, there is actually five subintervals as determined by howManySubintervals=5. In the end, this function
    is doing a search task using the binary search method. 
    
    NOTE_1: If negativity happens to be an element of neg_part, i.e. negativity matches one of the subintervals 
    boundaries, then I consider it to belong to the subinterval whose lower limit matches negativity. For example, 
    in the previous example, negativity=0.2 would belong to (0.2,0.3), so the function would return 2.'''
    
    #I do not take the test for negativity since typically type(negativity)=<class numpy.float64>!=type(1.0)=<class float>
    if type(howManySubintervals)!=type(1) or np.shape(neg_part)!=(howManySubintervals+1,) or type(neg_part)!=type(np.array([])):
        print('belonging_subinterval_index(), Err1')
        return -1
    if negativity<neg_part[0] or negativity>neg_part[howManySubintervals]:
        print('belonging_subinterval_index(), Err2')
        return -1
    
    #Just in case negativity belongs to neg_part
    if negativity in neg_part:
        #I consider range(howManySubintervals+1) just in case negativity==max_neg.
        for i in range(howManySubintervals+1):
            if negativity==neg_part[i]:
                return i
    
    lower_marker = 0
    #neg_part has actually howManySubintervals+1 elements
    upper_marker = howManySubintervals
    current_marker = int(np.floor(howManySubintervals/2))
    found = False
    while found==False:
        #Note that, per construction of neg_part, both of these logical conditions cannot be true simultaneously
        if neg_part[current_marker]>negativity:
            upper_marker = current_marker
            current_marker -= (upper_marker-lower_marker)/2
            current_marker = int(np.floor(current_marker))
        elif neg_part[current_marker+1]<negativity:
            lower_marker = current_marker
            current_marker += (upper_marker-lower_marker)/2
            current_marker = int(np.floor(current_marker))
        else:
            found = True
    return current_marker




def open_every_file(filepaths, openmode='w'):
    '''This function takes:
    - filepaths: (list of strings) Holds the filepaths to the files which must be opened.
    - openmode: (string) This string is passed to open() as mode.
    This function opens every file whose path is contained in filepaths. The function returns a list of file 
    variables, files_v, where every file is allocated, so that the file whose path is filepahts[i] has been 
    opened into files_v[i].
    '''
    
    if type(filepaths)!=type([]):
        print('set_every_file(), Err1')
        
    files_v = []
    for i in range(len(filepaths)):
        files_v.append(open(filepaths[i], mode=openmode))
    return files_v




def close_every_file(files_v):
    '''Analogous to open_every_file(), but for closing them. This function now receives a list of files variables,
    files_v.'''
    
    if type(files_v)!=type([]):
        print('close_every_file(), Err1')
    
    for i in range(len(files_v)):
        files_v[i].close()
    return 




def write_headers_pure(output_file, M, N, neg_subinterval_tuple):
    output_file.write('#This file stores '+str(M)+' random pure density matrices which belong to a hilbert space of dim='+str(N)+'\n')
    output_file.write('#Every density matrix has a negativity in '+str(neg_subinterval_tuple)+'\n')
    return
    
    
    
    
def write_headers_mixed(output_file, M, N, neg_subinterval_tuple, L_min, L_max):
    output_file.write('#This file stores '+str(M)+' random mixed density matrices which belong to a hilbert space of dim='+str(N)+' with L in['+str(L_min)+','+str(L_max)+')\n')
    output_file.write('#Every density matrix has a negativity in '+str(neg_subinterval_tuple)+'\n')
    return
    
    
    
    
def generate_random_DMs_files_negativitywise(neg_resolution, M, N, filepath_root, rng, max_neg = 0.5, transpose_first_subsystem=True, pure=True, L_min=2, L_max=10):
    '''This function takes:
    - neg_resolution: (real scalar) Step taken to discretize the negativity interval [0,max_neg=0.5] into intervals.
    - M: (integer) Number of DMs to generate within each negativity interval.
    - N: (integer) Dimension of the hilbert space.
    - rng: Random number generator as given by rng = Generator(PCG64(int(seed))
    - filepath_root: (string) This string is used in order to craft the filepaths for every output file. This
    function will output DMs to a number of output files, each of which associated with a certain negativity
    subinterval. Schematically, such craft could be done so that the actual output files filepath are crafted as
    filepath_root+str([neg_int_min,neg_int_max])+'.txt', where we are using neg_int_min (resp. neg_int_max) to
    refer to the lower (upper) limit of the negativity subinterval whose DMs will be stored in such file.
    - max_neg: (real scalar) Maximum value of negativity. It is needed to generate the negativity partition).
    - transpose_first_subsystem: (boolean) Determines whether the partial transpose (which is performed in order
    to compute the negativity) is performed over the first subsystem or not.
    - pure: (boolean) Determines whether the generated DMs are pure (pure==True) or mixed (pure==False)
    - L_min (resp. L_max): (integer) Determines the minimum inclusive (maximum exclusive) number of terms added
    to the convex linear combination which yields the mixed density matrix.
    
    This function takes the biggest negativity resolution which is smaller than neg_resolution but allows the
    interval [0,max_neg] to be split up into an integer number of negativity subintervals. This gives a negativity
    partition, neg_part[i], so that (neg_part[i],neg_part[i+1]) gives the i-th negativity subinterval. If 
    pure==True, this function iteratively calls generate_random_pure_density_matrix(), evaluate the negativity of 
    the returned DM by calling negativity(), then flattens it and write it into the corresponding output file. 
    According to the previously given explanation, such filepath could be:
    filepath_root+'_str((neg_part[i],neg_part[i+1]))+'.txt'. Calling this function results in max_neg/neg_resolution
    files, each of which stores M density matrix according to their negativity. The arithmetic mean of the purities 
    of the DMs that were stored in a certain file is written down to such file as a header. Such header also includes 
    the negativity std of the same set of DMs samples. Furthermure, for pure==True this function returns two 
    unidimensional arrays. The first of them stores the negativities arithmetic means for each file. The second of 
    them stores the std's. If pure==False, then instead of calling generate...(), this function iteratively calls 
    generate_random_mixed_density_matrix(). In this case, the headers of the generated files contain information not 
    only about the negativity distribution of the DMs stored in the same file, but also its purity. In this case, the 
    function returns four unidimensional arrays. The first (resp. last) two arrays store the negativity (purity) means 
    and std of the DMs of each file.'''
    
    if type(neg_resolution)!=type(1.0) or type(M)!=type(1) or type(N)!=type(1) or type(filepath_root)!=type('') or type(max_neg)!=type(1.0):
        print('generate_random_pure_DMs_files_negativitywise(), Err1')
        return -1
    if M<1:
        print('generate_random_pure_DMs_files_negativitywise(), Err2')
        return -1
    

    #The negativity is, per definition, positive. Therefore it is safe to assume that its minimum value is 0.
    howManySubintervals = int(np.ceil(max_neg/neg_resolution))
    neg_resolution = max_neg/howManySubintervals
    #Now neg_resolution holds a real value such that max_neg/neg_resolution is integer.
    
    #Generate the negativity partition
    neg_part = np.linspace(0, max_neg, num=howManySubintervals+1, endpoint=True)
    
    #Now that we have neg_part, we can generate the list of filepaths and open them
    filepaths = []
    for i in range(howManySubintervals):
        filepaths.append(filepath_root+'_'+str((neg_part[i],neg_part[i+1]))+'.txt')
    files_v = open_every_file(filepaths, openmode='w')
    
    #Write some informative headers in each file
    if pure==True:
        for i in range(howManySubintervals):
            write_headers_pure(files_v[i], M, N, (neg_part[i],neg_part[i+1]))
    else:
        for i in range(howManySubintervals):
            write_headers_mixed(files_v[i], M, N, (neg_part[i],neg_part[i+1]), L_min, L_max)
    
    #Generate boolean array to check which subinterval has already been fully populated up to M DMs.
    files_are_full = np.empty((howManySubintervals,), dtype=bool)
    #Initialize every entry to False
    files_are_full[:] = False
    
    #Generate an integer counter to keep track of how many DMs have already been added to each subinterval.
    #Initialize it to zero in every subinterval.
    files_DMs_cont = np.zeros((howManySubintervals,), dtype=int)
    
    #Python implements the product between boolean variables as the algebraic product between integers with 
    #the bijection True=1 and False=0. Therefore, the only way that the product of every files_are_full[i] is
    #1 is that every entry is actually 1 (True), which means that every file has already been filled with M DMs.
    negativities = np.empty((howManySubintervals, M))
    negativity_mean = np.empty((howManySubintervals,), dtype=float)
    negativity_std = np.empty((howManySubintervals,), dtype=float)
    if pure==True:
        while np.prod(files_are_full)!=1:
            DM_holder = generate_random_pure_density_matrix(N, rng)  
            negativity_holder = negativity(DM_holder, N, transpose_first=transpose_first_subsystem)
            index_holder = belonging_subinterval_index(neg_part, negativity_holder, howManySubintervals)
            #Condtional statement not to keep writing DMs to those output files where we have already reached
            #the desired number of DMs
            if files_are_full[index_holder]==True:
                continue
            else:
                #write_DM_to_file takes DM_holder by reference and flattens it. negativity() is fed a bidimensional
                #numpy complex array. Hence, it is less time consuming to compute the negativity first, then 
                #flatten it and write it down to the output file.
                negativities[index_holder, files_DMs_cont[index_holder]] = negativity(DM_holder, N)
                write_DM_to_file(files_v[index_holder], DM_holder, N, preflattening_needed=True)
                files_DMs_cont[index_holder] += 1
                if files_DMs_cont[index_holder]==M:
                    files_are_full[index_holder]=True
        close_every_file(files_v)
        for i in range(howManySubintervals):
            negativity_mean[i] = np.mean(negativities[i,:])
            negativity_std[i] = np.std(negativities[i,:])
            prepend_whatever_distribution(filepaths[i], negativity_mean[i], negativity_std[i], 'negativity')
        return negativity_mean, negativity_std
    else:
        Ls = np.empty((howManySubintervals, M), dtype=int)
        purities = np.empty((howManySubintervals, M), dtype=complex)
        while np.prod(files_are_full)!=1:
            L = random_int(L_min, L_max, rng)
            DM_holder = generate_random_mixed_density_matrix(N, L, rng)
            negativity_holder = negativity(DM_holder, N, transpose_first=transpose_first_subsystem)
            index_holder = belonging_subinterval_index(neg_part, negativity_holder, howManySubintervals)
            if files_are_full[index_holder]==True:
                continue
            else:
                #Only compute purity if the DM has been accepted
                negativities[index_holder, files_DMs_cont[index_holder]] = negativity(DM_holder, N)
                Ls[index_holder, files_DMs_cont[index_holder]] = L
                purities[index_holder, files_DMs_cont[index_holder]] = purity(N, DM_holder)
                write_DM_to_file(files_v[index_holder], DM_holder, N, preflattening_needed=True)
                files_DMs_cont[index_holder] += 1
                if files_DMs_cont[index_holder]==M:
                    files_are_full[index_holder]=True
        close_every_file(files_v)
        L_mean = np.empty((howManySubintervals,), dtype=float)
        L_std = np.empty((howManySubintervals,), dtype=float)
        purity_mean = np.empty((howManySubintervals,), dtype=float)
        purity_std = np.empty((howManySubintervals,), dtype=float)
        for i in range(howManySubintervals):
            negativity_mean[i] = np.mean(negativities[i,:])
            negativity_std[i] = np.std(negativities[i,:])
            L_mean[i] = np.mean(Ls[i,:])
            L_std[i] = np.std(Ls[i,:])
            purity_mean[i] = np.mean(purities[i,:])
            purity_std[i] = np.std(purities[i,:])
            prepend_whatever_distribution(filepaths[i], purity_mean[i], purity_std[i], 'purity')
            prepend_whatever_distribution(filepaths[i], L_mean[i], L_std[i], 'L')
            prepend_whatever_distribution(filepaths[i], negativity_mean[i], negativity_std[i], 'negativity')
        return negativity_mean, negativity_std, L_mean, L_std, purity_mean, purity_std