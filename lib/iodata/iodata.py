__all__ = ['load_csv', 'LoadCSVError', 'save_csv', 'SaveCSVError']

import csv
import numpy as np

class LoadCSVError(Exception):
    """
    CSV reading error
    """
    pass

class SaveCSVError(Exception):
    """
    CSV saving error
    """
    pass

def load_csv(fn, delim='|'):
    """
    Function that loads a csv file
    Input: fn, [delim]
    Output: x: np.array with the time data
            y: np.array with the time series data
            header: header of the csv file
            title: the column 0,0 of the csv file
    """
    f = open(fn, 'r')
    csvr = csv.reader(f, delimiter=delim)
    try:
        content = [row for row in csvr]
        nc = len(content[0])
        title = content[0][0]
        headerv = [content[i][0] for i in range(1, len(content))]
        headerh = content[0][1::]
        y = np.array([content[i][1::] for i in range(1, len(headerv) + 1)],
                     dtype=np.float)
    except:
        raise LoadCSVError
    else:
        return headerh, y, headerv, title

def save_csv(fn, x, y, header, title="", delim='|'):
    """
    Creates a CSV file
    """
    rows = [[]]
    rows[0].append(title)
    rows[0] += [elem for elem in x]
    for i, elem in enumerate(header):
        rows.append([elem] + [serie for serie in y[i]])
    try:
        csvwriter = csv.writer(file(fn, "w"), delimiter=delim)
        csvwriter.writerows(rows)
    except:
        raise SaveCSVErrror
