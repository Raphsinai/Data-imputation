import impyute
import numpy as np
import random
import openpyxl


def read_csv(filename : str) -> np.array:
    """
    Returns a Numpy array of the evaluated data

    input:
        filename: string of the path to the file to read

    output:
        names: names of each column
        np.array: read data
    """
    with open(filename, 'r') as f:
        lines = f.read().split()
        f.close()
    names   = [eval(cpg_name) for cpg_name in lines[0].split(',')]
    data    = [line.split(',')[1:] for line in lines[1:]]
    for l_index, line in enumerate(data):
        for v_index, value in enumerate(line):
            try:
                data[l_index][v_index] = float(value)
            except ValueError:
                data[l_index][v_index] = np.nan
    return names, np.array(data)

def get_non_nan(array : np.ndarray) -> np.array:
    ## TODO ##
    #
    # Get columns or rows with least nans and exclude those samples
    #
    return

def get_small_nan(array : np.ndarray, min_n : int = 5, square : bool = True, use_mult : bool =False) -> np.array:
    height, width = array.shape
    if use_mult:
        if square:
            mult_height = []
            mult_width = []
        n_height, n_width = min_n, min_n
        for x in range(n_height, height+1):
            if height%x==0:
                if square:
                    mult_height.append(x)
                else:
                    n_height = x
                    break
        for x in range(n_width, width+1):
            if width%x==0:
                if square:
                    mult_width.append(x)
                else:
                    n_width = x
                    break
        if square:
            common_mult = set(mult_width).intersection(set(mult_height))
            if len(common_mult) != 0:
                n_width, n_height = list(common_mult)[0], list(common_mult)[0]
            else:
                n_width, n_height = mult_width[0], mult_height[0]
    else:
        n_width, n_height = min_n, min_n
    isnan       = np.isnan(array)
    segments    = []
    n_segments  = []
    rows_cols   = []
    for i in range(1, height//n_height+1):
        for j in range(1, width//n_width+1):
            nan_arr = isnan[((i-1)*n_height+1):(i*n_height+1), ((j-1)*n_width+1):(j*n_width+1)]
            arr = array[((i-1)*n_height+1):(i*n_height+1), ((j-1)*n_width+1):(j*n_width+1)]
            rows_cols.append(((((i-1)*n_height+1), ((j-1)*n_width+1)), ((i*n_height+1), (j*n_width+1))))
            segments.append(arr)
            n_segments.append(np.count_nonzero(nan_arr))
    m = min(n_segments)
    indexes = []
    for index, n in enumerate(n_segments):
        if n == m:
            indexes.append(index)
    return len(indexes), [rows_cols[index] for index in indexes], [segments[index] for index in indexes]

def save_to_sheet(array : np.ndarray, sheet, names : list = None, sample_start : int = 1):
    if names != None:
        names = ['Sample #']+names
        sheet.append(names)
    for n, row in enumerate(array):
        l_row = list(row)
        for i, value in enumerate(np.isnan(row)):
            if value:
                l_row[i] = 'NA'
        row = [f'Sample #{n+sample_start}']+l_row
        sheet.append(row)
    return sheet
    

def test_all(array : np.ndarray, names : list = None, sample_start : int = 1) -> list:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = 'Original'
    sheet = save_to_sheet(array, sheet, names, sample_start)
    workbook.create_sheet(title='MICE Imputation')
    sheet = workbook['MICE Imputation']
    imputed = impyute.mice(array)
    sheet = save_to_sheet(imputed, sheet, names, sample_start)
    workbook.create_sheet(title='k=3 Nearest Neighbours Imputation')
    sheet = workbook['k=3 Nearest Neighbours Imputation']
    imputed = impyute.fast_knn(array, k=3)
    sheet = save_to_sheet(imputed, sheet, names, sample_start)
    workbook.create_sheet(title='Expectation Maximisation Imputation')
    sheet = workbook['Expectation Maximisation Imputation']
    imputed = impyute.em(array)
    sheet = save_to_sheet(imputed, sheet, names, sample_start)
    workbook.create_sheet(title='Moving Window (Interpolation) Imputation Transposed')
    sheet = workbook['Moving Window (Interpolation) Imputation Transposed']
    imputed = impyute.moving_window(array.transpose())
    sheet = save_to_sheet(imputed.transpose(), sheet, names, sample_start)
    workbook.create_sheet(title='Moving Window (Interpolation) Imputation')
    sheet = workbook['Moving Window (Interpolation) Imputation']
    imputed = impyute.moving_window(array)
    sheet = save_to_sheet(imputed, sheet, names, sample_start)
    workbook.save('Test.xlsx')

    
    
    

if __name__ == '__main__':
    np.set_printoptions(linewidth=1000)
    names, array = read_csv('sample_data_raf.csv')
    smallest_nan = get_small_nan(array, min_n=12)
    
    test_all(smallest_nan[2][0], names[smallest_nan[1][0][0][1]:smallest_nan[1][0][1][1]], smallest_nan[1][0][0][0])