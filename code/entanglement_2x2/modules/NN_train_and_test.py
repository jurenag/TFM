import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from NN_read_and_format import file_to_numpyarray, concatenate_entangled_and_separable_arrays, split_array_train_test, split_array_input_label




def create_ANN(nn, actHL, actLast, loss_f, input_shape, opt='rmsprop'):
    '''This function takes:
    - nn: (list of integers) Must have 1 or more integers. They are the number of neurons in each hidden layer.
    - actHL (resp. lastAct): (string) Name of the activation function in the hidden layers (resp. output layer).
    - loss_f: (string) Name of the loss function to use.
    - input_shupe: (tuple) input shape of the samples to feed the Neural network.
    - opt: (string referring to tf.keras.optimizers object) It is set to 'rmsprop' by default. See other options 
    in 
    
    This function creates a keras.Sequential() instance, model, with the specified characteristics and returns it.'''
    
    if type(nn)!=type([]) or type(actHL)!=type('') or type(actLast)!=type('') or type(loss_f)!=type('') or type(input_shape)!=type(()):
        print('fit_this_ANN(), Err1')
        return -1
    if len(nn)<1:
        print('fit_this_ANN(), Err2')
        return -1
  

    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for i in range(len(nn)-1):
        model.add(keras.layers.Dense(nn[i], activation=actHL,))
    model.add(keras.layers.Dense(nn[len(nn)-1], activation=actLast))
    #See documentation for keras.Sequential.compile() in https://keras.io/api/models/model_training_apis/
    model.compile(optimizer=opt, loss=loss_f)
    return model
    
    
    
  
def train_ANN(model, x_train, y_train, nepochs, BS, verb, use_val_data = False, x_val=None, y_val=None):
    if type(x_train)!=type(np.array([])) or type(y_train)!=type(np.array([])) or type(nepochs)!=type(1) or type(BS)!=type(1) or type(verb)!=type(1):
        print('train_ANN(), Err1')
        return -1 

    if use_val_data == True:
        if type(x_val)!=type(np.array([])) or type(y_val)!=type(np.array([])):
            print('train_ANN(), Err2')
            return -1
        return model.fit(x_train, y_train, batch_size=BS, epochs=nepochs, verbose=verb, validation_data=(x_val,y_val))
    else:
        return model.fit(x_train, y_train, batch_size=BS, epochs=nepochs, verbose=verb)
    
    
    
    
def elementary_test(true_value, predicted_value, tolerance):
    '''This function takes:
    - true_value: (integer).
    - predicted_value: (integer).
    - tolerance: (real scalar).
    
    This function returns 1 if |true_value -predicted_value|<tolerance, and 0 otherwise.'''
    
    if type(tolerance)!=type(1.0):
        print('elementary_test(), Err1')
        return -1
    
    if abs(true_value-predicted_value)<tolerance:
        return 1
    else:
        return 0
    
    
    
    
def multiple_test(model, x_test, y_test, tolerance):
    '''This function takes:
    - model: (keras.Sequential object/instance).
    - x_test: (numpy array)  It is a (k+1)-dimensional, where k is the dimension of the input received by model. 
    x_test is nothing but an array of different input samples, so that running along the 0-th axis gives different 
    samples of the model input.
    - y_test: (bidimensional numpy array) It must have one column. It is an array of outputs which is analogous to 
    x_test, in the sense that running along its 0-th axis provides different samples of the desired output. Both 
    arrays (x_test and y_test) must have the same length along the 0-th axis. 
    - tolerance: (real scalar) It is passed to elementary_test() as tolerance. 
    
    NOTE_1: This function is only useful for testing trained models with real scalar output.'''
    
    if type(x_test)!=type(np.array([])) or type(y_test)!=type(np.array([])) or type(tolerance)!=type(1.0):
        print('multiple_test(), Err1')
        return -1
    if np.ndim(y_test)!=2:
        print('multiple_test(), Err2')
        return -1
    if np.shape(x_test)[0]!=np.shape(y_test)[0]:
        print('multiple_test(), Err3')
        return -1

    predictions_array = model.predict(x_test)
    result = np.empty((np.shape(y_test)[0],))
    for i in range(np.shape(y_test)[0]):
        result[i] = elementary_test(y_test[i,0],predictions_array[i,0],tolerance)
    return result
    
    
    
    
#ASR stands for average success rate, and ASRSTD is for average success rate standard deviation
def save_results(filepath, loss, standard_deviation, ASR, ASRSTD, additional_tests_taken=False, ASR2=None, ASRSTD2=None, use_val_data=False, val_loss=None, val_loss_std=None, openmode='a'):
    
    if type(filepath)!=type('') or type(loss)!=type(np.array([])) or type(standard_deviation)!=type(np.array([])):
        print('save_results(), Err1')
        return -1
    if np.ndim(loss)!=1 or np.ndim(standard_deviation)!=1:
        print('save_results(), Err2')
        return -1
    if len(loss)!=len(standard_deviation):
        print('save_results(), Err3')
        return -1

    #Assess whether validation data is available.
    if use_val_data == True:
        if type(val_loss)!=type(np.array([])) or type(val_loss_std)!=type(np.array([])) or np.ndim(val_loss)!=1 or np.ndim(val_loss_std)!=1 or len(val_loss)!=len(val_loss_std):
            print('save_results(), Err4')

        #Up to here, It is clear that loss and standard_deviation have the same (required) unidimensional shape. It is also clear that both
        #val_loss and val_loss_std have the same shape. Now let us check whether the four arrays have the same shape:
        if len(loss)!=len(val_loss):
            print('save_results(), Err5')
    
    output_file = open(filepath, mode=openmode)
    output_file.write('#Sucess rate averaged over every simulation and over every sample in the test set: '+str(ASR*100)+'%\n')
    output_file.write('#Sample standard deviation for averaged success rate: '+str(ASRSTD*100)+'%\n')
    if additional_tests_taken == True:
        output_file.write('#Same average success rate for supplementary tests: '+str(ASR2)+'%\n')
        output_file.write('#Sample STD for averaged success rate in supplementary tests: '+str(ASRSTD2)+'%\n')

    if use_val_data == True:
        output_file.write('#Epoch\tLoss\tLoss sample STD\tVal. Loss\tV.L. sample STD\n')
        for i in range(len(loss)):
            output_file.write('%d\t%f\t%f\t%f\t%f\n' % (i+1,loss[i],standard_deviation[i],val_loss[i],val_loss_std[i]))
    else:
        output_file.write('#Epoch\tLoss\tSample STD\n')
        for i in range(len(loss)):
            output_file.write('%d\t%f\t%f\n' % (i+1,loss[i],standard_deviation[i]))

    output_file.close()
    return
    
    
    
    
def write_headers(outFilePath, N, howManyTimes, first_filepath, first_type, second_filepath, second_type, architecture, nepochs, fraction, actHL, actLast, loss_f, BS, tolerance, optimizer, openmode='w', usedEarlyStopping=True, metric='val_loss', epochs_patience=0, min_improvement=0):
    output_file = open(outFilePath, mode=openmode)
    output_file.write('#Tensor product hilbert space dimension: '+str(N)+ '; Number of simulations: '+str(howManyTimes)+';\n')
    output_file.write('#'+first_type+' DMs were read from: '+first_filepath+'; '+second_type+' DMs were read from: '+second_filepath+';\n')
    output_file.write('#Architecture of the MLP: '+str(architecture)+'; Number of epochs: '+str(nepochs)+'; Fraction of DMs used for training: '+str(fraction)+';\n')
    output_file.write('#Activation function in the hidden layers: '+actHL+'; Activation function in the output layer: '+actLast+'; Loss function: '+loss_f+';\n')
    output_file.write('#Optimizer: '+optimizer+'; Batch size: '+str(BS)+'; Test tolerance: '+str(tolerance)+';\n')
    if usedEarlyStopping==True:
        output_file.write('#tf.Keras.callbacks.EarlyStopping was used with: metric:'+str(metric)+'; Epochs patience:'+str(epochs_patience)+'; Minimum improvement:'+str(min_improvement)+';\n')
    output_file.close()
    return




def correct_array_length(array, desired_length, new_entries=0.0):
    '''This function takes:
    - array: (unidimensional numpy array of scalars).
    - desired_length: (integer) The length of the array returned by this function.

    Optional parameters:
    - new_entries: (float) It is set to 0.0 by default. It makes a difference only if desired_length is
    greater than the length of the given array, array. It is the value of the entries that are added to
    array in order to make it as long as desired_length.

    Say original_length = np.shape(array)[0]. Then, if original_shape is less than desired-length, this 
    function concatenates array with another unidimensional float array. All of the entries of the 
    appended array are equal to new_entries. The length of the appended array is tuned so that the 
    resulting array has length equal to desired_length. Otherwise, if original_length is greather than 
    desired_length, then a number original_length-desired_length of entries (at the end of array) are split
    away from array.
    '''

    if np.ndim(array)!=1:
        print('correct_array_length(), Error 1')
        return -1 

    original_length = np.shape(array)[0]
    if original_length == desired_length:
        return array
    #The case original_length>desired_length has already been discarded in the first conditional statement.
    elif original_length<desired_length:
        aux_array = np.ones((desired_length-original_length,), dtype=type(new_entries))
        aux_array = new_entries*aux_array
        return np.concatenate((array, aux_array), axis=0)
    else:
        array, _ = np.split(array, (desired_length,), axis=0)
        return array




def first_null_position(array):
    '''This function takes:
    - array: (unidimensional numpy array of dtype==int) 
    
    This function sweeps the array from beggining (array[0]), position by position, until a null entry
    is found. If the funtion gets to array[i]==0, then the function returns i. If no null entry is found,
    the function returns -1.'''

    if np.ndim(array)!=1:
        print('Not allowed ndim for array. Returning -2.')
        return -2

    for i in range(len(array)):
        if array[i]==0:
            return i

    return -1      
    
    
    
    
def binaryOutput_formatData_trainNN_averageLoss_averageTestResults_and_writeResults(N, howManyTimes, first_filepath, \
first_type, second_filepath, second_type, architecture, max_nepochs, fraction, actHL, actLast, loss_f, BS, take_redundancy=False, optimizer='rmsprop', \
perform_additional_tests=False, first_test_filepath=None, second_test_filepath=None, outFilePath=None, tolerance=1e-4, \
use_validation_data=False, trigger_early_stopping=False, metric_to_monitor=None, epochs_patience=10, min_improvement=1e-3, \
monitor_mode='min', baseline=None, recover_best_configuration=True, first_label=0.0, second_label=1.0, shuffle=True, rts=None, \
verb=0, snitch_every=1):
    '''The COMPULSORY PARAMETERS that this function takes are:
    - N: (integer) Dimension of the total hilbert space associated to the studied quantum system. For the
    case of two qubits, such dimension is 2x2=4. It can be seen as the side length of the studied density matrices.
    - howManyTimes: (integer) The number of times that the experiment (create ANN, train ANN and measure loss) is 
    repeated. 
    - first_filepath (resp. second filepath): (string) Path to the file which contain the first type (resp. second 
    type) of quantum density matrices. Such density matrices must be written rows-wise. I.e. one matrix per row 
    as it should be read by file_to_numpyarray() (See such function definition for more info). 
    -architecture: (list of integers) Its length matches the {number of hidden layers plus one} (the output layer). The 
    i-th element is the number of neurons in the i-th layer. 
    - max_nepochs: (integer) Number of training epochs. In case trigger_early_stopping==True, max_nepochs is the 
    maximum number of epochs that the training will last if it is not earlier interrupted by the keras callback
    tf.Keras.callbacks.EarlyStopping
    - fraction: (real scalar) Number in [0.5,1] which match the fraction of density matrices which are used in the
    training set.
    - actHL: (resp. actLast) (string) Name of the activation function in the hidden (resp. output) layers. 
    - loss_f: (string) Name of the loss function. 
    - BS: (integer) Batch size used for training, i.e. number of training samples used to perform the update of the 
    network parameters once.
    - first_type (resp. second_type): (string) It is passed to write_headers as first_type (second_type). For example,
    a common choice is first_type='separable' and second_type='maximally entangled'.
    
    The OPTIONAL PARAMETERS that this function takes are:
    - optimizer: (string referring to a tf.keras.optimizers object) It is set to 'rmsprop' by default. See more options in
    keras documentation.
    - perform_additional_tests: (boolean) Whether to perform additional tests over the trained networks. It is set to
    False by default. In such case, the data passed to first_test_filepath and second_test_filepath is ignored.
    - first_test_filepath (resp. second_test_filepath): (list of strings) It is set to None by default. Otherwise,
    they are the paths to the files which contain the first type (resp. second type) of density matrices (with the 
    usual format) which are to be used in order to perform additonal tests once the ANN has been trained. These 
    tests are performed IN ADDITION TO the original test that is performed using a fraction fraction of density 
    matrices stored in first_filepath and second_filepath. It is necessary that the number of strings in first_test_filepath
    matches the number of strings in second_test_filepath. In such case, if L is the length of both lists, then L
    additional tests are performed, so that in the k-th test, the first type of density matrices are read from
    first_test_filepath[k-1], and the second type of density matrices are read from second_test_filepath[k-1].
    - outFilePath: (string) Path of the file which will store the results given by the function. It is set to 
    None by default. In such case, the results are not written down to any file. 
    - tolerance: (real scalar) Maximum difference between the goal value and the output value which can be thrown 
    by the NN and still be considered as a correct answer by the NN. It is set to 1e-4 by default.
    - use_validation_data: (boolean) It is set to False by default. Otherwise, the training is done with 
    validation data. In such case, the validation data set is the test data, i.e. a fraction (1-fraction) of the
    DMs in a shuffled concatenation of the DMs in first_filepath and second_filepath.
    - trigger_early_stopping: (boolean) It is set to False by default. If this parameter is set to True, then
    the training (model.fit(...)) is carried out using the callback tf.keras.callbacks.EarlyStopping. The following
    three optional parameters are the arguments required by this callback. Further information about this callback
    is available in tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
    - metric_to_monitor: (string) The magnitude to monitor after every epoch. Keras set this parameter to 'val_loss',
    which is the validation loss given by model.fit(...).history['val_loss'] on the specified validation data set. In
    this function, metric_to_monitor is also set to 'val_loss' by default. Note that this is not compatible with
    use_validation_data==False. I.e. such default value must have been replaced upon this function call unless we specify
    use_validation_data==True and provide the validation data set.
    - epochs_patience: (integer) It is set to 10 by default. This parameter is passed to EarlyStopping as patience. It
    fixes how many epochs the training should go on without showing any improvement in metric_to_monitor. In order to
    assess whether an improvement took place or not, EarlyStopping uses min_delta.
    - min_improvement: (scalar) It is set to 1e-3 by default. It is passed to EarlyStopping as min_delta. If the variation
    in metric_to_monitor is less than min_improvement, then it is considered that there was no improvement in that epoch.
    - recover_best_configuration: (boolean) It is set to True by default. It is passed to EarlyStopping as restore_best_weights. 
    If EarlyStopping actually stopped the training before reaching max_nepochs, then the model returned by model.fit(...) is 
    the one that resulted in the best value for metric_to_monitor between all of the model updates during the last epochs_patience
    epochs.
    - monitor_mode: (string) It is passed to EarlyStopping as mode. It can take one of these values: \{'min', 'max', 'auto'\}.
    It is set to 'min' by default. 'min' means that are improvements are made if there is a negative variation in metric_to_monitor
    that is greater than min_improvement in absolute value. 'max' means that the goal is to let metric_to_monitor increase. In
    'auto' mode, the mode is inferred.
    - baseline: (scalar) It is set to None by default. It is passed to EarlyStopping as baseline.
    - first_label (resp. second_label): (scalar) Goal values which the ANN must output for the first type (resp. 
    second type) of DMs. They are set to separable_label=0.0 and entangled_label=1.0 by default. Please
    note that these labels must be set so that they are reachable by the network output. For example, for the 
    common choice of sigmoid activation function in the last neuron, it is vital that both labels fall in the
    interval [0,1] since sigmoid(x)\in[0,1].
    - shuffle: (boolean) Determines whether the concatenation of the first and second type of DMs is 
    shuffled before being split into training set and test set. It is set to True by default. See 
    concatenate_entangled_and_separable_arrays() for more info. 
    - rts: (tuple of 2+2*len(separable_test_filepath) integers) The integers in this tuple are given to 
    file_to_numpyarray as rows_to_skip. In this case, rts[0] (resp. rts[1]) are the rows to skip when reading 
    the file containing the first (resp. second) type of DMs used for the network training. rts[2] 
    (resp. rts[3]) are the rows to skip when reading the files that store the first (resp. second) type of DMs
    used to perform the first additional test over the already trained network. rts[4] and rts[5] are used
    to perform the second additional test and so on.
    - verb: (integer) It is passed to model.fit as verbose. 
    - snitch_every: (integer) The function will print the progress everytime the experiment has been completed 
    i*snitch_every times, with i a positive integer.

    FUNCTION RETURNS: The function averages the loss function over howManyTimes experiments and returns the 
    average of the evolution of the loss function as a unidimensional array of shape (nepochs,). It also computes 
    its sample standard deviation and returns it as a second output, as a unidimensional numpy array of 
    shape (nepochs,). The function returns two more scalar arguments, which are the test success rate, which is 
    averaged over every test sample and over every simulation; and its sample standard deviation. If 
    separable_test_filepath (and entangled_test_filepath) are given, then the function returns two more lists of
    scalars. The first list stores the test success rate, whereas the second list stores its standard deviation, 
    but this time averaged over every DMs found in such files. If use_validation_data==True, then the 
    function returns two more unidimensional arrays of shape (nepochs,) which are the history of the loss
    tested over the validation data and its sample std. If an outFilePath is given, then the results are saved
    to the file whose path is outFilePath.'''

    if trigger_early_stopping == True and metric_to_monitor == 'val_loss':
        if use_validation_data == False:
            print('Validation data must be used if trigger_early_stopping==True and metric_to_monitor=val_loss. Returning -1')
            return -1

    #If separable_test_filepath is None or entangled_test_filepath is None, then no additional test will be performed.
    if perform_additional_tests==False:
        first_test_filepath=[]
        second_test_filepath=[]
    #Later on we perform loops for i in range(len(separable_test_filepath)). If separable_test_filepath was set to None,
    #then such loop will be performed for i in range(len([])), i.e. for i in range(0), which is equivalent to no loop at all.
    else:
        if len(first_test_filepath)!=len(second_test_filepath):
            print('Err, Returning -1.')
            return -1

    #Setting rts to None means skipping no lines in any input data file.
    if rts==None:
        rts = list(np.zeros(((2+(2*len(first_test_filepath))),), dtype=int))

    #If rts was different to None, then it is not still ensured that it has the required length.
    if len(rts)!= 2+(2*len(first_test_filepath)):
        print('Not allowed length for rts. Returning -1.')
        return -1

    #Craft the train and the test data sets out of the input files.
    first_array = file_to_numpyarray(first_filepath, N, first_label, take_redundancy=take_redundancy, rows_to_skip=rts[0])
    #WARNING: Since in this case (2x2) we are working with density matrices, file_to_numpyarray()
    #reads 2*N*N real entries.
    second_array = file_to_numpyarray(second_filepath, N, second_label, take_redundancy=take_redundancy, rows_to_skip=rts[1])
    whole_array = concatenate_entangled_and_separable_arrays(first_array, second_array, shuffle)
    train_array, test_array = split_array_train_test(whole_array, fraction)
    x_train, y_train, input_shape = split_array_input_label(train_array)
    x_test, y_test, _ = split_array_input_label(test_array)

    #Craft and group the test data sets used for additional tests over the trained networks.
    x_test_2 = []
    y_test_2 = []
    for i in range(len(first_test_filepath)):
        first_array = file_to_numpyarray(first_test_filepath[i], N, first_label, take_redundancy=take_redundancy, rows_to_skip=rts[2*(i+1)])
        second_array = file_to_numpyarray(second_test_filepath[i], N, second_label, take_redundancy=take_redundancy, rows_to_skip=rts[(2*(i+1))+1])
        whole_array = concatenate_entangled_and_separable_arrays(first_array, second_array, shuffle)
        aux_x, aux_y, _ = split_array_input_label(whole_array)
        x_test_2.append(aux_x)
        y_test_2.append(aux_y)

    #The following unidimensional array counts how many simulations reached a certain epoch. For example, if
    #every simulation reached the 22-nd epoch, then reached_this_epoch[21]=howManyTimes
    loss_history = np.zeros((max_nepochs,),dtype=float)
    loss_history_squared = np.zeros((max_nepochs,),dtype=float)
    test_results = np.zeros((np.shape(y_test)[0],))
        
    if trigger_early_stopping==True:
        reached_this_epoch = np.zeros((max_nepochs,), dtype=int)
        my_callbacks = [tf.keras.callbacks.EarlyStopping(monitor=metric_to_monitor, min_delta=min_improvement, \
        patience=epochs_patience, verbose=0, mode=monitor_mode, baseline=baseline, restore_best_weights=recover_best_configuration)]
    else:
        #In order to regularize further calculus, I generalize reached_this_epoch to the case of trigger_early_stopping==False
        #just by setting every entry in reached_this_epoch to howManyTimes
        reached_this_epoch = np.ones((max_nepochs,), dtype=int)
        reached_this_epoch *= howManyTimes
    
    #Assuming the length of every array in y_test_2 is the same. Note that this assumption is not general. This will only work
    #if the number of data inputs in each test file is the same.
    if first_test_filepath==[]:
        test_results_2 = np.array([])
    else:
        test_results_2 = np.zeros((len(first_test_filepath), np.shape(y_test_2[0])[0],))

    if use_validation_data==True:
        val_loss_history = np.zeros((max_nepochs,),dtype=float)
        val_loss_history_squared = np.zeros((max_nepochs,),dtype=float)
        for i in range(howManyTimes):
            model = create_ANN(architecture, actHL, actLast, loss_f, input_shape, opt=optimizer)
            if trigger_early_stopping == False:
                #In this case, every simulation entails max_nepochs epochs, therefore there is no need to correct the loss length
                aux = model.fit(x_train, y_train, batch_size=BS, epochs=max_nepochs, verbose=verb, validation_data=(x_test,y_test))
                aux_2 = np.array(aux.history['loss'])
                aux_3 = np.array(aux.history['val_loss'])
            else:
                #In this case, after each training, the loss and val_loss history may have a length shorter than max_nepochs.
                aux = model.fit(x_train, y_train, batch_size=BS, epochs=max_nepochs, verbose=verb, callbacks=my_callbacks, validation_data=(x_test, y_test))
                aux_2 = np.array(aux.history['loss'])
                actual_length = np.shape(aux_2)[0]
                aux_3 = np.array(aux.history['val_loss'])
                aux_4 = np.ones((actual_length,), dtype=int)
                aux_2 = correct_array_length(aux_2, max_nepochs, new_entries=0.0)
                aux_3 = correct_array_length(aux_3, max_nepochs, new_entries=0.0)
                aux_4 = correct_array_length(aux_4, max_nepochs, new_entries=0)
                #Count the number of times that a training reached a certain epoch
                reached_this_epoch += aux_4
        
            #Once the loss and the val_loss history is formatted suitably ((max_nepochs,)) regardless the truth value of
            #trigger_early_stopping, I add up the results to the average.
            loss_history += aux_2
            loss_history_squared += np.power(aux_2, 2)
            val_loss_history += aux_3
            val_loss_history_squared += np.power(aux_3, 2)

            #Test the trained network
            test_results = test_results + multiple_test(model, x_test, y_test, tolerance)
            for j in range(len(first_test_filepath)):
                test_results_2[j,:] = test_results_2[j,:] + multiple_test(model, x_test_2[j], y_test_2[j], tolerance)            
          
            #Verbose
            if i%snitch_every==0:
                print('Progress: ', 100*(i+1)/howManyTimes, '%') 
  
    else:
        #In this case, no validation loss is studied.
        for i in range(howManyTimes):
            model = create_ANN(architecture, actHL, actLast, loss_f, input_shape, opt=optimizer)
            if trigger_early_stopping == False:
                #In this case, every simulation entails max_nepochs epochs, therefore there is no need to correct the loss length
                aux = model.fit(x_train, y_train, batch_size=BS, epochs=max_nepochs, verbose=verb)
                aux_2 = np.array(aux.history['loss'])
            else:
                #In this case, after each training, the loss and val_loss history may have a length shorter than max_nepochs.
                aux = model.fit(x_train, y_train, batch_size=BS, epochs=max_nepochs, verbose=verb, callbacks=my_callbacks)
                aux_2 = np.array(aux.history['loss'])
                actual_length = np.shape(aux_2)[0]
                aux_4 = np.ones((actual_length,), dtype=int)
                aux_2 = correct_array_length(aux_2, max_nepochs, new_entries=0.0)
                aux_4 = correct_array_length(aux_4, max_nepochs, new_entries=0)
                reached_this_epoch += aux_4
        
            #Once the loss is formatted suitably ((max_nepochs,)) regardless the truth value of
            #trigger_early_stopping, I add up the results to the average.
            loss_history += aux_2
            loss_history_squared += np.power(aux_2, 2)

            #Test the trained network
            test_results = test_results + multiple_test(model, x_test, y_test, tolerance)
            for j in range(len(first_test_filepath)):
                test_results_2[j,:] = test_results_2[j,:] + multiple_test(model, x_test_2[j], y_test_2[j], tolerance)            
          
            #Verbose
            if i%snitch_every==0:
                print('Progress: ', 100*(i+1)/howManyTimes, '%') 

    #If trigger_early_stopping==True and max_nepochs is too big compared to the usual number of epochs that it takes the NN to reach a
    #plateau for the monitored metric, then it is probable that there are some entries at the end of loss_history, val_loss_history and
    #reached_this_epoch that are null. These entries may cause division_by_zero errors when normalizing the results. Therefore, we must
    #correct the length of such arrays so that those null entries are erased from the arrays.
    pos = first_null_position(reached_this_epoch)
    if pos != -1:
        #pos == -1 means that there was no null entry in reached_this_epoch.
        reached_this_epoch = correct_array_length(reached_this_epoch, pos)
        loss_history = correct_array_length(loss_history, pos)
        loss_history_squared = correct_array_length(loss_history_squared, pos)
        if use_validation_data == True:
            val_loss_history = correct_array_length(val_loss_history, pos)
            val_loss_history_squared = correct_array_length(val_loss_history_squared, pos)

    #Normalize results and compute std's
    loss_history = loss_history/reached_this_epoch  #The data for each epoch is normalized according to how many times such epoch was reached.
    loss_history_squared = loss_history_squared/reached_this_epoch
    std = np.sqrt((loss_history_squared-np.power(loss_history,2))/reached_this_epoch)
    #The sample std of a random variable which is, per definition, the average over multiple independent realizations of the same elementary random variable
    #is the std of such elementary random variable divided by the square root of the number of independent realizations, in this case, howManyTimes. (*ref)

    if use_validation_data == True:
        val_loss_history = val_loss_history/reached_this_epoch
        val_loss_history_squared = val_loss_history_squared/reached_this_epoch
        val_loss_std = np.sqrt((val_loss_history_squared-np.power(val_loss_history,2))/reached_this_epoch)

    #Tests are always performed regardless early stopping, therefore they should be normalized to HowManyTimes.
    test_results = test_results/howManyTimes
    howManyTestSamples = np.shape(test_results)[0]
    test_results_2 = test_results_2/howManyTimes

    if first_test_filepath==[]:
        howManyTestSamples2 = 0
    else:
        howManyTestSamples2 = np.shape(test_results_2)[1]

    average_success_rate = np.mean(test_results)
    #(*ref) In these cases, if I consider the elementary random variable to be the mean of the result of a multiple tests over the same
    # single input sample (i.e. 1+0+0+1+1+0+0+1+1+0/howManyTimes, where 1 means correct labeling of the ANN and 0 means incorrect labeling),
    #then I'm summing over howManyTestSamples. Therefore, the std of the average success rate is the std of the success rate divided
    #by sqrt(howManyTestSamples)
    average_success_rate_std = np.sqrt((np.mean(np.power(test_results, 2))-np.power(np.mean(test_results), 2))/howManyTestSamples)
    average_success_rate_2 = []
    average_success_rate_std_2 = []
    for i in range(len(first_test_filepath)):
        average_success_rate_2.append(np.mean(test_results_2[i,:])) 
        #No need to worry about dividing by zero in the following line since, if separable_test_filepath==[], then the body of this loop is
        #performed not even once. 
        aux = np.sqrt((np.mean(np.power(test_results_2[i,:], 2))-np.power(np.mean(test_results_2[i,:]), 2))/howManyTestSamples2)
        average_success_rate_std_2.append(aux)
        


     #SIGUE CORRIGIENDO POR AQUÍ. HAZ UNA VERSIÓN MÁS GENERAL DE SAVE RESULTS Y WRITE HEADERS
     # CUANDO VUELVAS A NN_train_and_test.py PUEDES BORRAR LA FUNCIÓN train_ANN, en esta función no se usa y puedes sustituir 
     # #las nuveas funciones q escibas para escribir resultados y headers por las antiguas 

    if use_validation_data == True:  
        if outFilePath!=None:  
            write_headers(outFilePath, N, howManyTimes, first_filepath, first_type, second_filepath, second_type, architecture, max_nepochs, fraction, actHL, actLast, loss_f, BS, tolerance, optimizer, usedEarlyStopping=trigger_early_stopping, metric=metric_to_monitor, epochs_patience=epochs_patience, min_improvement=min_improvement)     
            save_results(outFilePath, loss_history, std, average_success_rate, average_success_rate_std, additional_tests_taken=perform_additional_tests, ASR2=average_success_rate_2, ASRSTD2=average_success_rate_std_2, use_val_data=True, val_loss=val_loss_history, val_loss_std=val_loss_std)
        return loss_history, std, average_success_rate, average_success_rate_std, average_success_rate_2, average_success_rate_std_2, val_loss_history, val_loss_std, reached_this_epoch, np.shape(loss_history)[0]
    else:
        if outFilePath!=None:  
            write_headers(outFilePath, N, howManyTimes, first_filepath, first_type, second_filepath, second_type, architecture, max_nepochs, fraction, actHL, actLast, loss_f, BS, tolerance, optimizer, usedEarlyStopping=trigger_early_stopping, metric=metric_to_monitor, epochs_patience=epochs_patience, min_improvement=min_improvement)
            save_results(outFilePath, loss_history, std, average_success_rate, average_success_rate_std, additional_tests_taken=perform_additional_tests, ASR2=average_success_rate_2, ASRSTD2=average_success_rate_std_2, use_val_data=False, val_loss=None, val_loss_std=None)
        return loss_history, std, average_success_rate, average_success_rate_std, average_success_rate_2, average_success_rate_std_2, reached_this_epoch, np.shape(loss_history)[0]
    
    

	
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
