# by Huaiyu TAN

import numpy as np
def is_dd(M: list):
    p = 0
    for i in range(len(M)):
        if np.sum(np.abs(M[i]))-np.abs(M[i][i]) >= np.abs(M[i][i]):
            p += 1
    if p == 0:
        print("The Matrix is diagonally dominant !")
    else:
        print("The Matrix is not diagonally dominant !")
    return p

M = np.array([[3,2,0],[-4,-20,-6],[7,1,9]])
p = is_dd(M)
