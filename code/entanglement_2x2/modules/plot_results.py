import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt




def none_unidimensional_np_array(length):
    '''This function takes:
    - length: (integer) Number of entries in the resulting array.

    This function returns a unidimensional numpy array with length entries. Every entry is equal to None.'''

    result = []
    for i in range(length):
        result.append(None)
    return np.array(result)




def plot(filepaths, rows_to_skip, xycolumns=(0,1), stdColumn=None, plotStd=None, separator='\t', err_every=None, labels=None, showGrid=False, save_data=None, show_fig=True, xlim=None, ylim=None, xlabel=None, ylabel=None, labelsfontsize=12, ticksfontsize=16, legendncol=1, legendfontsize=16):
    '''This function takes:
    - filepaths: (list of strings) Each string is the path to a file that stores data which is going to be plotted.
    - rows_to_skip: (tuple of len(filepaths) integer) rows_to_skip[i] is the number of rows to skip at the header of the file given by filepaths[i].
    - xycolumns: (bidimensional array of integers of dimensions (len(filepaths)x2)) xycolumns[i,0] (resp. xycolumns[i,1]) is the number of the column
     that contains the x-values (y-values) for the i-th filepath in filepaths.
    - stdColumn: (tuple of len(filepaths) integers) stdColumn[i] is the number of the column that contains the std for the y-values.
    - plotStd: (list of booleans) If plotStds[i]==True, then the data in filepahts[i] is plotted using pyplot.errobar.
    - separator: (string) It is passed to pandas.read_table as sep. It is set to '\t' (tabulator separation) by default.
    - labels: (list of strings) labels[i] is the plot label for the data contained in filepaths[i].
    - showGrid: (boolean) If showGrid==True, then plt.grid() is called before showing and/or saving the output figure.
    - save_data: (tuple of two elements) If save_data!=None, then it must be a tuple of two elements, where the first one is the path to save the
    output figure, whereas the second element is the format. I.e. it is a string that is passed to plt.savefig as format. I.e. format='png'.
    - show_fig: (boolean) Whether to call plt.show() once everything has been plotted, or not.
    '''

    howManyFiles = len(filepaths)

    if plotStd == None:
        #This gives an array full of False
        plotStd = np.zeros((howManyFiles,), dtype=bool)

    if err_every==None:
        err_every = np.ones((howManyFiles,), dtype=int)

    if labels == None:
        labels = none_unidimensional_np_array(howManyFiles)

    
    plt.figure()
    plt.xlabel(xlabel, fontsize=labelsfontsize)
    plt.ylabel(ylabel, fontsize=labelsfontsize)
    if xlim!=None:
        plt.xlim(xlim)
    if ylim!=None:
        plt.ylim(ylim)
    
    for i in range(howManyFiles):
        if plotStd[i]==True:
            useful_columns_holder = (xycolumns[i,0], xycolumns[i,1], stdColumn[i])
            #A backslash at the end of a line tells Python to extend the current logical line over across to the next physical line.
            data_holder = np.array( \
            pd.read_table(filepaths[i], sep=separator, header=None, index_col=None, usecols=useful_columns_holder, skiprows=rows_to_skip[i]) \
        )
            plt.errorbar(data_holder[:,0], data_holder[:,1], yerr=data_holder[:,2], label=labels[i], errorevery=err_every[i])
        else:
            useful_columns_holder = (xycolumns[i,0], xycolumns[i,1])
            data_holder = np.array( \
            pd.read_table(filepaths[i], sep=separator, header=None, index_col=None, usecols=useful_columns_holder, skiprows=rows_to_skip[i]) \
        )
            plt.plot(data_holder[:,xycolumns[i,0]], data_holder[:,xycolumns[i,1]], yerr=data_holder[:,stdColumn[i]], label=labels[i])
        
    plt.xticks(fontsize=ticksfontsize)
    plt.yticks(fontsize=ticksfontsize)

    plt.legend(loc='best', fontsize=legendfontsize, ncol=legendncol)
    if showGrid==True:
        plt.grid()

    plt.tight_layout()

    if save_data!=None:
        plt.savefig(save_data[0], format=save_data[1])

    if show_fig==True:
        plt.show()

    return 



def ASR_grouped_bar_chart(filepath, bar_width, inbetweenwidth=None, separator='\t', ylim=None, legend_columns=None, showGrid=False, save_data=None, show_fig=True, ylabel=None, title=None, legendfontsize=12, xlabelfontsize=12, ylabelfontsize=12):
    '''This function takes:
    - filepath: (string) Path to the file that contains a table of the averaged success rates (ASR). Its format must be the following. The first
    column is a label column. For example '(0.0,0.1)', '(0.1,0.2)' etc. The rest of the columns store the ASR obtained after the training. For 
    example, the second column stores the ASRs obtained when the training was performed using DMs with negativity in (0.1,0.2).
    - bar_width: (scalar) Width of each bar
    - inbetweenwidth: (scalar) Width between two subsequent labels in the x-axis. If no value is given to inbetweendith, then it is set to be 
    the number of columns in filepath times the bar_width, what means that the last bar from one x-label is one bar width away from the first bar
    of the subsequent x-label no matter the value taken by bar_width.
    - separator: (string) It is passed to pandas.read_table as sep.
    - ylim: (tuple of two scalars) y-range used for the plot.
    - legends_columns: (integer) Number of columns used to arrange the legend.
    - showGrid: (boolean) Wether to show or not to show the grid in the plot.
    - save_data: (tuple of two elements) If save_data!=None, then it must be a tuple of two elements, where the first one is the path to save the
    output figure, whereas the second element is the format. I.e. it is a string that is passed to plt.savefig as format. I.e. format='png'.
    - show_fig: (boolean) Whether to call plt.show() once everything has been plotted, or not.
    - ylabel: (string) Label for the y-axis.
    - title: (string) Title that is shown above the plot.
    '''

    data_holder = pd.read_table(filepath, sep='\t', header=0, index_col=0)

    #Assuming the labels are the first column of the data file
    labels = data_holder.columns
    xticks_labels = data_holder.index

    #In this case, the number of columns equals the number of rows. This is due to the nature of our problem. Since we are performing the training
    #for each negativity interval, and then performing the tests over every negativity interval.
    nColumns = len(data_holder.columns)

    #ASRs stores the table data as a bidimensional numpy array without its column names or row indices.        
    ASRs = np.array(data_holder)    

    if inbetweenwidth==None:
        inbetweenwidth = bar_width*nColumns

    x = np.linspace(0, len(labels)*inbetweenwidth, num=len(labels))      #The label locations

    aux = (float(nColumns -1))/2
    bar_lims = np.linspace(-1.0*aux, aux, num=nColumns)

    fig, ax = plt.subplots()
    for i in range(nColumns):
        #rects.append(ax.bar(x+(bar_lims[i]*bar_width)+(bar_width/2.0), ASRs[:,i], bar_width, label=labels[i]))
        ax.bar(x+(bar_lims[i]*bar_width)+(bar_width/2.0), ASRs[:,i], bar_width, label=labels[i])

    if ylabel!=None:
        ax.set_ylabel(ylabel, fontsize=ylabelfontsize)
    if title!=None:
        ax.set_title(title)
    
    ax.set_xticks(x)
    ax.set_xticklabels(xticks_labels, fontsize=xlabelfontsize)

    if legend_columns!=None:
        ax.legend(loc='best', ncol=legend_columns, fontsize=legendfontsize)
    else:
        ax.legend(loc='best')

    if ylim!=None:
        ax.set_ylim(ylim)
    if showGrid!=False:
        ax.grid()

    fig.tight_layout()

    if save_data!=None:
        plt.savefig(save_data[0], format=save_data[1])

    if show_fig==True:
        plt.show()

    return
            