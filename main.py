import impyute
import numpy as np
import random
import openpyxl
from openpyxl.styles.fills import PatternFill


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

def get_non_nan(array : np.ndarray, start_n : int = 0, col_n : int = 5) -> np.array:
    cols = array.transpose()
    cols_nan = np.isnan(cols)
    n_cols_nan = [np.count_nonzero(col) for col in cols_nan]
    m = sorted(n_cols_nan)[start_n:start_n+col_n]
    col_indicies = []
    row_indicies = [i for i in range(array.shape[0])]
    new_cols = []
    for i, value in enumerate(n_cols_nan):
            if value in m:
                col_indicies.append(i)
                new_cols.append(cols[i])
    new_rows = np.array(new_cols).transpose()
    new_array = new_rows.copy()
    deleted = 0
    for i, row in enumerate(np.isnan(new_rows)):
        if np.count_nonzero(row) > 0:
            new_array = np.delete(new_array, i-deleted, axis=0)
            row_indicies.remove(i)
            deleted += 1
    return row_indicies, col_indicies, new_array

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

def save_to_sheet(array : np.ndarray, sheet, names : list = None, samples : list = [], highlight : list = []):
    """
    Save numpy array to an excel sheet with specified highlighted nan coordinates
    """
    # If names of columns are provided add them to the first row
    if names != None:
        names = ['Sample #']+names
        sheet.append(names)
    # For each row replace nan with 'NA' string
    for n, row in enumerate(array):
        l_row = list(row)
        for i, value in enumerate(np.isnan(row)):
            if value:
                l_row[i] = 'NA'
        row = [f'Sample #{samples[n]}']+l_row
        sheet.append(row)
    for cell in highlight:
        sheet[chr(65+cell[0]+1)+str(cell[1]+2)].fill = PatternFill(start_color='ebe834', end_color='ebe834', fill_type='solid')
    return sheet
    

def randints(max : int, n_rand : int, seed : int = 25) -> list:
    nums = [None for _ in range(n_rand)]
    for i in range(n_rand):
        random.seed(seed+i)
        nums[i] = round(random.random()*max)
    return nums


def test_all(array : np.ndarray, n_nan : int = 1, names : list = None, samples : list = [], filename : str = 'Test', seed : int = 25) -> list:
    """
    Test all relevant imputation methods and save them to a spreadsheet
    """
    print(array.shape)
    height, width = array.shape
    col_indicies = randints(width, n_nan, seed)
    row_indicies = randints(height, n_nan, seed+n_nan)
    na_indicies = [(col_indicies[i], row_indicies[i]) for i in range(n_nan)]
    # Create workbook to store data
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = 'Original'
    sheet = save_to_sheet(array, sheet, names, samples, na_indicies)
    for index in na_indicies:
        array[index[1], index[0]] = np.nan
    workbook.create_sheet(title='MICE Imputation')
    sheet = workbook['MICE Imputation']
    imputed = impyute.mice(array)
    sheet = save_to_sheet(imputed, sheet, names, samples, na_indicies)
    workbook.create_sheet(title='k=3 Nearest Neighbours Imputation')
    sheet = workbook['k=3 Nearest Neighbours Imputation']
    imputed = impyute.fast_knn(array, k=3)
    sheet = save_to_sheet(imputed, sheet, names, samples, na_indicies)
    workbook.create_sheet(title='Expectation Maximisation Imputation')
    sheet = workbook['Expectation Maximisation Imputation']
    imputed = impyute.em(array)
    sheet = save_to_sheet(imputed, sheet, names, samples, na_indicies)
    workbook.create_sheet(title='Moving Window (Interpolation) Imputation Transposed')
    sheet = workbook['Moving Window (Interpolation) Imputation Transposed']
    imputed = impyute.moving_window(array.transpose())
    sheet = save_to_sheet(imputed.transpose(), sheet, names, samples, na_indicies)
    workbook.create_sheet(title='Moving Window (Interpolation) Imputation')
    sheet = workbook['Moving Window (Interpolation) Imputation']
    imputed = impyute.moving_window(array)
    sheet = save_to_sheet(imputed, sheet, names, samples, na_indicies)
    workbook.save(f'{filename}.xlsx')

    
    
    

if __name__ == '__main__':
    np.set_printoptions(linewidth=1000)
    names, array = read_csv('sample_data_raf.csv')
    # smallest_nan = get_small_nan(array, min_n=12)
    no_nan = get_non_nan(array, 10)

    print(no_nan[0], no_nan[1])
    
    test_all(no_nan[2], 2, [names[i] for i in no_nan[1]], no_nan[0],seed=40)